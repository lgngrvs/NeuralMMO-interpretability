"""Fisher Linear Discriminant diagnostic for cluster similarity.

For each pair of clusters, measures the effective dimensionality of their
separation: how many directions must you project out before they become
indistinguishable? If the answer is small (1-3 out of hundreds), the clusters
are nearly identical in a large shared subspace.

Also computes a feature-regression subspace analysis: projects activations
into the subspace that predicts behavioral features, and checks whether
clusters that are far apart in activation space are close in the
behaviorally-relevant subspace.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

# Reuse feature computation from analyze_activations
from analyze_activations import (
    FEATURE_NAMES,
    _is_alive,
    compute_features,
)


def load_analysis_results(cluster_dir):
    """Load cluster labels and models from a previous analysis run."""
    cluster_dir = Path(cluster_dir)
    label_records = np.load(cluster_dir / "cluster_labels.npy")
    # Columns: env_id, agent_id, step, label
    return label_records


def load_activations_for_labels(data_dir, label_records, subsample_rate):
    """Load activation records matching the label records.

    Returns (activations, features, labels) arrays aligned by index.
    """
    data_dir = Path(data_dir)
    json_files = list(data_dir.rglob("activations.json"))
    if not json_files:
        print(f"No activations.json found under {data_dir}")
        sys.exit(1)

    json_path = json_files[0]
    print(f"Loading {json_path.name} ({json_path.stat().st_size / 1024 / 1024:.0f} MB)...",
          flush=True)
    with open(json_path, "rb") as f:
        all_records = orjson.loads(f.read())
    print(f"  {len(all_records)} records loaded")

    # Build lookup from label_records: (env_id, agent_id, step) -> label
    label_lookup = {}
    for row in label_records:
        key = (int(row[0]), int(row[1]), int(row[2]))
        label_lookup[key] = int(row[3])

    # Match records to labels
    matched_records = []
    matched_labels = []
    for r in tqdm(all_records, desc="Matching records to labels"):
        if not _is_alive(r):
            continue
        key = (r["env_id"], r["agent_id"], r["step"])
        if key in label_lookup:
            matched_records.append(r)
            matched_labels.append(label_lookup[key])

    print(f"  Matched {len(matched_records)} / {len(label_lookup)} labeled records")
    del all_records

    activations, features, metadata = compute_features(matched_records)
    labels = np.array(matched_labels, dtype=int)
    return activations, features, labels


def fisher_discriminant_dimensionality(act_a, act_b, variance_threshold=0.95):
    """Compute the effective discriminant dimensionality between two clusters.

    Finds the directions that separate the clusters by solving the generalized
    eigenvalue problem for Fisher's criterion. Returns the number of dimensions
    needed to explain `variance_threshold` of the between-class separation,
    plus the full eigenvalue spectrum.

    For two clusters the between-class scatter is rank-1, so we instead measure
    how much of the total separability is concentrated in each direction by
    projecting onto eigenvectors of S_W^{-1} and checking separation along each.

    Returns:
        n_dims: number of dimensions needed to capture variance_threshold of
                the classifier's discriminating power
        eigenvalues: full spectrum of S_W^{-1} S_B (sorted descending)
        accuracy_curve: classifier accuracy after removing top-k discriminant
                        directions, for k = 0, 1, 2, ...
    """
    from sklearn.decomposition import PCA
    from scipy.linalg import eigh

    pooled = np.vstack([act_a, act_b])
    n_a, n_b = len(act_a), len(act_b)
    n_total = n_a + n_b
    d = pooled.shape[1]

    # When n_samples < n_dims, S_W is rank-deficient. Reduce with PCA.
    # Cap at min(n_a, n_b) / 2 to avoid overfitting in the LOO classifier.
    # The smaller cluster is the bottleneck for generalization.
    max_pca_dims = min(d, max(2, min(n_a, n_b) // 2))
    if max_pca_dims < d:
        pca = PCA(n_components=max_pca_dims)
        pooled = pca.fit_transform(pooled)
        act_a_r = pooled[:n_a]
        act_b_r = pooled[n_a:]
    else:
        act_a_r = act_a
        act_b_r = act_b

    mean_a = act_a_r.mean(axis=0)
    mean_b = act_b_r.mean(axis=0)
    global_mean = pooled.mean(axis=0)

    # Between-class scatter
    d_a = (mean_a - global_mean).reshape(-1, 1)
    d_b = (mean_b - global_mean).reshape(-1, 1)
    S_B = n_a * (d_a @ d_a.T) + n_b * (d_b @ d_b.T)

    # Within-class scatter
    centered_a = act_a_r - mean_a
    centered_b = act_b_r - mean_b
    S_W = centered_a.T @ centered_a + centered_b.T @ centered_b

    # Regularize S_W for numerical stability
    reg = 1e-4 * np.trace(S_W) / S_W.shape[0]
    S_W += reg * np.eye(S_W.shape[0])

    # Solve generalized eigenvalue problem: S_B v = λ S_W v
    eigenvalues, eigenvectors = eigh(S_B, S_W)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # For two classes, S_B is rank 1, so only the first eigenvalue matters.
    # But we compute the accuracy curve by iteratively projecting out the top
    # discriminant directions and measuring linear classifier accuracy.
    max_dims = min(30, len(eigenvalues))
    accuracy_curve = np.zeros(max_dims + 1)

    # Labels: 0 for A, 1 for B
    labels = np.array([0] * n_a + [1] * n_b)

    for k in range(max_dims + 1):
        if k == 0:
            projected = pooled.copy()
        else:
            # Project out top-k discriminant directions
            V_k = eigenvectors[:, :k]
            projection = V_k @ V_k.T
            projected = pooled - pooled @ projection

        # Leave-one-out cross-validated nearest-centroid classifier
        # (avoids overfitting when n_samples << n_dims)
        correct = 0
        for i in range(n_total):
            # Compute centroids excluding point i
            if i < n_a:
                mean_a_loo = (projected[:n_a].sum(axis=0) - projected[i]) / (n_a - 1)
                mean_b_loo = projected[n_a:].mean(axis=0)
            else:
                mean_a_loo = projected[:n_a].mean(axis=0)
                mean_b_loo = (projected[n_a:].sum(axis=0) - projected[i]) / (n_b - 1)
            dist_a = np.linalg.norm(projected[i] - mean_a_loo)
            dist_b = np.linalg.norm(projected[i] - mean_b_loo)
            pred = 0 if dist_a < dist_b else 1
            if pred == labels[i]:
                correct += 1
        accuracy_curve[k] = correct / n_total

    # Number of dims to reach threshold: where accuracy drops below
    # (1 - variance_threshold) * (acc_0 - 0.5) + 0.5
    baseline_acc = accuracy_curve[0]
    target_acc = (1 - variance_threshold) * (baseline_acc - 0.5) + 0.5
    n_dims = 0
    for k in range(1, len(accuracy_curve)):
        if accuracy_curve[k] <= target_acc:
            n_dims = k
            break
    else:
        n_dims = max_dims

    return n_dims, eigenvalues[:max_dims], accuracy_curve


def feature_subspace_analysis(activations, features, labels):
    """Project activations into feature-predictive subspace, compare cluster distances.

    Fits a linear regression activations -> features, then compares pairwise
    cluster distances in the full space vs. the feature-predictive subspace.
    """
    # Fit linear regression: features = activations @ W + b
    # Add bias column
    X = np.hstack([activations, np.ones((len(activations), 1))])
    # Least squares solve
    W, residuals, rank, sv = np.linalg.lstsq(X, features, rcond=None)
    W_nobias = W[:-1]  # (d, n_features)

    # R² per feature
    predictions = X @ W
    ss_res = ((features - predictions) ** 2).sum(axis=0)
    ss_tot = ((features - features.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1)

    # Project activations into column space of W_nobias
    # Q is an orthonormal basis for the feature-predictive subspace
    Q, _ = np.linalg.qr(W_nobias)
    n_feat_dims = W_nobias.shape[1]
    Q = Q[:, :n_feat_dims]
    projected = activations @ Q  # (N, n_features)

    # Pairwise cluster distances: full space vs projected
    unique_labels = sorted(set(labels) - {-1})
    centroids_full = {}
    centroids_proj = {}
    for lab in unique_labels:
        mask = labels == lab
        centroids_full[lab] = activations[mask].mean(axis=0)
        centroids_proj[lab] = projected[mask].mean(axis=0)

    n_clusters = len(unique_labels)
    dist_full = np.zeros((n_clusters, n_clusters))
    dist_proj = np.zeros((n_clusters, n_clusters))
    for i, la in enumerate(unique_labels):
        for j, lb in enumerate(unique_labels):
            dist_full[i, j] = np.linalg.norm(centroids_full[la] - centroids_full[lb])
            dist_proj[i, j] = np.linalg.norm(centroids_proj[la] - centroids_proj[lb])

    return {
        "r2_per_feature": r2,
        "feature_names": FEATURE_NAMES,
        "cluster_labels": unique_labels,
        "dist_full": dist_full,
        "dist_proj": dist_proj,
        "n_feature_dims": n_feat_dims,
        "Q": Q,
    }


def mahalanobis_cosine_similarity(activations, labels):
    """Compute pairwise Mahalanobis cosine similarity between cluster centroids.

    Whitens the activation space using the pooled within-class covariance,
    then computes cosine similarity between cluster centroid displacements
    (mean_i - global_mean) in the whitened space. This accounts for correlated
    and differently-scaled dimensions.

    Returns dict with similarity matrix, whitened displacement vectors, and
    top activation dimensions contributing to each cluster's identity.
    """
    from sklearn.decomposition import PCA

    unique_labels = sorted(set(labels) - {-1})
    n_clusters = len(unique_labels)
    global_mean = activations.mean(axis=0)

    # Pooled within-class covariance
    S_W = np.zeros((activations.shape[1], activations.shape[1]))
    cluster_masks = {}
    for lab in unique_labels:
        mask = labels == lab
        cluster_masks[lab] = mask
        centered = activations[mask] - activations[mask].mean(axis=0)
        S_W += centered.T @ centered

    # Whitening: need S_W^{-1/2}. Use PCA to handle rank deficiency.
    # Eigendecompose S_W, invert the nonzero eigenvalues.
    n_samples = sum(m.sum() for m in cluster_masks.values())
    max_rank = min(activations.shape[1], n_samples - n_clusters)
    pca = PCA(n_components=min(max_rank, activations.shape[1]))
    pca.fit(activations)  # fit on all data to get stable basis

    # Whitening transform: project into PCA space, scale by 1/sqrt(eigenvalue)
    components = pca.components_  # (n_components, d)
    explained_var = pca.explained_variance_  # (n_components,)
    # Regularize to avoid dividing by near-zero variance
    reg = 1e-6 * explained_var.max()
    whitening_scale = 1.0 / np.sqrt(explained_var + reg)

    # Compute displacements in whitened space
    displacements = {}
    displacements_raw = {}
    for lab in unique_labels:
        centroid = activations[cluster_masks[lab]].mean(axis=0)
        delta = centroid - global_mean
        displacements_raw[lab] = delta
        # Project to PCA space and whiten
        delta_pca = components @ delta  # (n_components,)
        delta_whitened = delta_pca * whitening_scale
        displacements[lab] = delta_whitened

    # Cosine similarity matrix in whitened space
    cosine_sim = np.zeros((n_clusters, n_clusters))
    for i, la in enumerate(unique_labels):
        for j, lb in enumerate(unique_labels):
            da = displacements[la]
            db = displacements[lb]
            norm_a = np.linalg.norm(da)
            norm_b = np.linalg.norm(db)
            if norm_a < 1e-10 or norm_b < 1e-10:
                cosine_sim[i, j] = 0.0
            else:
                cosine_sim[i, j] = np.dot(da, db) / (norm_a * norm_b)

    # Also compute regular (non-whitened) cosine similarity for comparison
    cosine_sim_raw = np.zeros((n_clusters, n_clusters))
    for i, la in enumerate(unique_labels):
        for j, lb in enumerate(unique_labels):
            da = displacements_raw[la]
            db = displacements_raw[lb]
            norm_a = np.linalg.norm(da)
            norm_b = np.linalg.norm(db)
            if norm_a < 1e-10 or norm_b < 1e-10:
                cosine_sim_raw[i, j] = 0.0
            else:
                cosine_sim_raw[i, j] = np.dot(da, db) / (norm_a * norm_b)

    # Top activation dimensions per cluster: which neurons contribute most
    # to the whitened displacement?
    top_dims_per_cluster = {}
    for lab in unique_labels:
        # Map whitened displacement back to original space for interpretability
        delta_whitened = displacements[lab]
        # Contribution of each original dimension: sum of |whitened_pca_j * component_j_i|
        # across PCA components j, for original dimension i
        contrib = np.abs((delta_whitened[:, None] * components)).sum(axis=0)  # (d,)
        top_idx = np.argsort(contrib)[::-1][:20]
        top_dims_per_cluster[lab] = list(zip(top_idx.tolist(),
                                             contrib[top_idx].tolist()))

    return {
        "cluster_labels": unique_labels,
        "cosine_sim_whitened": cosine_sim,
        "cosine_sim_raw": cosine_sim_raw,
        "top_dims_per_cluster": top_dims_per_cluster,
        "displacements_whitened": displacements,
    }


def null_hypothesis_test(activations, labels, n_random=200, seed=42):
    """Test whether real clusters differ from random subsets of the same size.

    Draws random "fake clusters" from the full dataset (same sizes as the real
    small clusters), then computes the same statistics (whitened cosine similarity,
    feature-subspace distance, centroid distance) for random pairs. Compares the
    real cluster statistics to the null distribution.
    """
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(seed)
    unique_labels = sorted(set(labels) - {-1})
    global_mean = activations.mean(axis=0)

    # Identify small clusters (not the largest one)
    cluster_sizes = {lab: (labels == lab).sum() for lab in unique_labels}
    largest = max(cluster_sizes, key=cluster_sizes.get)
    small_labels = [lab for lab in unique_labels if lab != largest]
    small_sizes = [cluster_sizes[lab] for lab in small_labels]

    # Whitening setup (same as mahalanobis_cosine_similarity)
    pca = PCA(n_components=min(activations.shape[1], len(activations) - len(unique_labels)))
    pca.fit(activations)
    components = pca.components_
    explained_var = pca.explained_variance_
    reg = 1e-6 * explained_var.max()
    whitening_scale = 1.0 / np.sqrt(explained_var + reg)

    def whiten_displacement(centroid):
        delta = centroid - global_mean
        delta_pca = components @ delta
        return delta_pca * whitening_scale

    # Real cluster stats
    real_displacements = {}
    real_centroids = {}
    for lab in small_labels:
        mask = labels == lab
        real_centroids[lab] = activations[mask].mean(axis=0)
        real_displacements[lab] = whiten_displacement(real_centroids[lab])

    # Real pairwise whitened cosine between small clusters
    real_cosines = []
    for i, la in enumerate(small_labels):
        for j, lb in enumerate(small_labels):
            if j <= i:
                continue
            da, db = real_displacements[la], real_displacements[lb]
            cos = np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db))
            real_cosines.append(cos)

    # Real distances from global mean (whitened norm)
    real_norms = [np.linalg.norm(real_displacements[lab]) for lab in small_labels]

    # Null distribution: draw random subsets, compute same stats
    null_cosines = []
    null_norms = []

    # Draw n_random random subsets for each small cluster size
    typical_size = int(np.mean(small_sizes))
    random_displacements = []
    for i in range(n_random):
        idx = rng.choice(len(activations), size=typical_size, replace=False)
        centroid = activations[idx].mean(axis=0)
        disp = whiten_displacement(centroid)
        random_displacements.append(disp)
        null_norms.append(np.linalg.norm(disp))

    # Pairwise cosines between random subsets (sample pairs)
    n_pairs = min(5000, n_random * (n_random - 1) // 2)
    for _ in range(n_pairs):
        i, j = rng.choice(n_random, size=2, replace=False)
        da, db = random_displacements[i], random_displacements[j]
        na, nb = np.linalg.norm(da), np.linalg.norm(db)
        if na > 1e-10 and nb > 1e-10:
            null_cosines.append(np.dot(da, db) / (na * nb))

    null_cosines = np.array(null_cosines)
    null_norms = np.array(null_norms)

    return {
        "small_labels": small_labels,
        "real_cosines": np.array(real_cosines),
        "null_cosines": null_cosines,
        "real_norms": np.array(real_norms),
        "null_norms": null_norms,
        "typical_size": typical_size,
        "n_random": n_random,
    }


def generate_report(pairwise_results, feature_results, mahal_results, null_results, output_dir):
    """Generate text report and plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    cluster_labels = feature_results["cluster_labels"]

    # --- Text report ---
    report_path = os.path.join(output_dir, "fisher_report.txt")
    with open(report_path, "w") as f:
        f.write("Fisher Discriminant Analysis: Cluster Separability\n")
        f.write("=" * 60 + "\n\n")

        # Feature regression quality
        f.write("Feature Regression R² (activations → behavioral features):\n")
        for name, r2 in zip(feature_results["feature_names"],
                            feature_results["r2_per_feature"]):
            f.write(f"  {name:25s}  R²={r2:.3f}\n")
        f.write(f"\n  Feature-predictive subspace: {feature_results['n_feature_dims']} dims\n\n")

        # Distance comparison
        f.write("Pairwise Centroid Distances:\n")
        f.write(f"  {'':>10s}")
        for lab in cluster_labels:
            f.write(f"  C{lab:>5d}")
        f.write("\n")

        f.write("  Full space:\n")
        for i, la in enumerate(cluster_labels):
            f.write(f"  C{la:<8d}")
            for j in range(len(cluster_labels)):
                f.write(f"  {feature_results['dist_full'][i, j]:6.1f}")
            f.write("\n")

        f.write("\n  Feature subspace:\n")
        for i, la in enumerate(cluster_labels):
            f.write(f"  C{la:<8d}")
            for j in range(len(cluster_labels)):
                f.write(f"  {feature_results['dist_proj'][i, j]:6.1f}")
            f.write("\n")

        # Ratio: how much of the full-space distance is in the feature subspace?
        f.write("\n  Ratio (feature / full):\n")
        for i, la in enumerate(cluster_labels):
            f.write(f"  C{la:<8d}")
            for j in range(len(cluster_labels)):
                d_full = feature_results['dist_full'][i, j]
                d_proj = feature_results['dist_proj'][i, j]
                ratio = d_proj / d_full if d_full > 1e-8 else 0
                f.write(f"  {ratio:6.2f}")
            f.write("\n")

        # Pairwise Fisher results
        f.write("\n\nPairwise Fisher Discriminant Dimensionality:\n")
        f.write("-" * 60 + "\n")
        for (la, lb), res in sorted(pairwise_results.items()):
            f.write(f"\nC{la} vs C{lb}:\n")
            f.write(f"  Dims to separate (95%): {res['n_dims']}\n")
            f.write(f"  Baseline accuracy:      {res['accuracy_curve'][0]:.3f}\n")
            f.write(f"  Top eigenvalues:        ")
            top_eigs = res['eigenvalues'][:5]
            f.write("  ".join(f"{e:.4f}" for e in top_eigs))
            f.write("\n")
            f.write(f"  Accuracy after removing k dims:\n")
            for k in range(min(8, len(res['accuracy_curve']))):
                bar = "█" * int(res['accuracy_curve'][k] * 40)
                f.write(f"    k={k:2d}: {res['accuracy_curve'][k]:.3f}  {bar}\n")

        # Mahalanobis cosine similarity
        f.write("\n\nMahalanobis Cosine Similarity (whitened displacement vectors):\n")
        f.write("=" * 60 + "\n")
        f.write("  Cosine similarity of (cluster_mean - global_mean) in whitened space.\n")
        f.write("  +1 = clusters deviate from the mean in the same direction.\n")
        f.write("   0 = orthogonal deviations.  -1 = opposite directions.\n\n")

        m_labels = mahal_results["cluster_labels"]
        f.write(f"  {'Whitened':>10s}")
        for lab in m_labels:
            f.write(f"  C{lab:>5d}")
        f.write("\n")
        for i, la in enumerate(m_labels):
            f.write(f"  C{la:<8d}")
            for j in range(len(m_labels)):
                f.write(f"  {mahal_results['cosine_sim_whitened'][i, j]:+6.3f}")
            f.write("\n")

        f.write(f"\n  {'Raw':>10s}")
        for lab in m_labels:
            f.write(f"  C{lab:>5d}")
        f.write("\n")
        for i, la in enumerate(m_labels):
            f.write(f"  C{la:<8d}")
            for j in range(len(m_labels)):
                f.write(f"  {mahal_results['cosine_sim_raw'][i, j]:+6.3f}")
            f.write("\n")

        # Top activation dimensions per cluster
        f.write("\n\nTop Activation Dimensions per Cluster (whitened displacement):\n")
        f.write("-" * 60 + "\n")
        for lab in m_labels:
            f.write(f"\nC{lab}:\n")
            f.write(f"  {'Rank':>4s}  {'Dim':>5s}  {'Contrib':>8s}\n")
            for rank, (dim, contrib) in enumerate(mahal_results["top_dims_per_cluster"][lab][:10]):
                f.write(f"  {rank+1:4d}  {dim:5d}  {contrib:8.3f}\n")

        # Null hypothesis test results
        f.write("\n\nNull Hypothesis Test: Are Clusters Different from Random Subsets?\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Drew {null_results['n_random']} random subsets of size "
                f"{null_results['typical_size']} from the full dataset.\n\n")

        f.write("  Whitened cosine similarity (pairwise):\n")
        rc = null_results['real_cosines']
        nc = null_results['null_cosines']
        f.write(f"    Real clusters (C{','.join(str(l) for l in null_results['small_labels'])}):\n")
        f.write(f"      mean={rc.mean():+.4f}  std={rc.std():.4f}  "
                f"range=[{rc.min():+.4f}, {rc.max():+.4f}]\n")
        f.write(f"    Random subsets:\n")
        f.write(f"      mean={nc.mean():+.4f}  std={nc.std():.4f}  "
                f"range=[{nc.min():+.4f}, {nc.max():+.4f}]\n")
        # p-value: fraction of null cosines <= real mean
        p_cosine = (nc <= rc.mean()).mean()
        f.write(f"    p-value (real mean <= null): {p_cosine:.4f}\n\n")

        f.write("  Whitened displacement norm (distance from global mean):\n")
        rn = null_results['real_norms']
        nn = null_results['null_norms']
        f.write(f"    Real clusters:  mean={rn.mean():.3f}  std={rn.std():.3f}  "
                f"range=[{rn.min():.3f}, {rn.max():.3f}]\n")
        f.write(f"    Random subsets: mean={nn.mean():.3f}  std={nn.std():.3f}  "
                f"range=[{nn.min():.3f}, {nn.max():.3f}]\n")
        # How many standard deviations above/below are real norms?
        z_norm = (rn.mean() - nn.mean()) / nn.std() if nn.std() > 0 else 0
        f.write(f"    z-score (real vs null): {z_norm:+.2f}\n")
        p_norm = (nn >= rn.mean()).mean()
        f.write(f"    p-value (real norm >= null): {p_norm:.4f}\n")

        if rn.mean() > nn.mean() + 2 * nn.std():
            f.write("\n  >> Real clusters are FARTHER from the mean than random subsets.\n")
            f.write("     These are genuine outlier groups, not random noise.\n")
        elif rn.mean() < nn.mean() - 2 * nn.std():
            f.write("\n  >> Real clusters are CLOSER to the mean than random subsets.\n")
            f.write("     Clusters may be arbitrary partitions of the dense core.\n")
        else:
            f.write("\n  >> Real cluster norms are WITHIN the null range.\n")
            f.write("     Cannot distinguish from random subsets by displacement alone.\n")

    print(f"  Wrote {report_path}")

    # --- Plot 1: Accuracy curves ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    pairs = sorted(pairwise_results.keys())
    for idx, (la, lb) in enumerate(pairs):
        if idx >= len(axes):
            break
        ax = axes[idx]
        res = pairwise_results[(la, lb)]
        curve = res['accuracy_curve']
        ax.plot(range(len(curve)), curve, 'o-', markersize=3, linewidth=1.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
        ax.set_title(f"C{la} vs C{lb} (n_dims={res['n_dims']})", fontsize=10)
        ax.set_xlabel("Dims removed")
        ax.set_ylabel("Classifier accuracy")
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)
    # Hide unused axes
    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Classifier Accuracy After Removing Top-k Discriminant Directions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_curves.png")
    fig.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {acc_path}")

    # --- Plot 2: Distance comparison heatmaps ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    n = len(cluster_labels)
    tick_labels = [f"C{l}" for l in cluster_labels]

    im1 = ax1.imshow(feature_results['dist_full'], cmap='viridis')
    ax1.set_xticks(range(n)); ax1.set_xticklabels(tick_labels)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(tick_labels)
    ax1.set_title("Full Activation Space\nCentroid Distance")
    fig.colorbar(im1, ax=ax1)
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f"{feature_results['dist_full'][i,j]:.1f}",
                    ha='center', va='center', fontsize=7,
                    color='white' if feature_results['dist_full'][i,j] > feature_results['dist_full'].max()*0.5 else 'black')

    im2 = ax2.imshow(feature_results['dist_proj'], cmap='viridis')
    ax2.set_xticks(range(n)); ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(tick_labels)
    ax2.set_title(f"Feature Subspace ({feature_results['n_feature_dims']}D)\nCentroid Distance")
    fig.colorbar(im2, ax=ax2)
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{feature_results['dist_proj'][i,j]:.1f}",
                    ha='center', va='center', fontsize=7,
                    color='white' if feature_results['dist_proj'][i,j] > feature_results['dist_proj'].max()*0.5 else 'black')

    # Ratio
    d_full = feature_results['dist_full']
    d_proj = feature_results['dist_proj']
    with np.errstate(invalid='ignore'):
        ratio = np.where(d_full > 1e-8, d_proj / d_full, 0)
    im3 = ax3.imshow(ratio, cmap='RdYlGn', vmin=0, vmax=1)
    ax3.set_xticks(range(n)); ax3.set_xticklabels(tick_labels)
    ax3.set_yticks(range(n)); ax3.set_yticklabels(tick_labels)
    ax3.set_title("Ratio (Feature / Full)\n1.0 = all separation is behavioral")
    fig.colorbar(im3, ax=ax3)
    for i in range(n):
        for j in range(n):
            ax3.text(j, i, f"{ratio[i,j]:.2f}",
                    ha='center', va='center', fontsize=7)

    fig.suptitle("Cluster Distance: Full Space vs Feature-Predictive Subspace",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    dist_path = os.path.join(output_dir, "distance_comparison.png")
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {dist_path}")

    # --- Plot 3: Eigenvalue spectra ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for (la, lb), res in sorted(pairwise_results.items()):
        eigs = res['eigenvalues']
        # Normalize so they sum to 1 for comparability
        eig_sum = eigs.sum()
        if eig_sum > 1e-10:
            eigs_norm = eigs / eig_sum
        else:
            eigs_norm = eigs
        ax.plot(range(len(eigs_norm)), eigs_norm, 'o-', markersize=3,
                label=f"C{la} vs C{lb}", linewidth=1.5)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Fraction of total discriminability")
    ax.set_title("Fisher Discriminant Eigenvalue Spectra\n(steeper = fewer dimensions needed to separate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 15)
    eig_path = os.path.join(output_dir, "eigenvalue_spectra.png")
    fig.savefig(eig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {eig_path}")

    # --- Plot 4: Mahalanobis cosine similarity heatmaps ---
    m_labels = mahal_results["cluster_labels"]
    n_m = len(m_labels)
    m_tick_labels = [f"C{l}" for l in m_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    im1 = ax1.imshow(mahal_results['cosine_sim_raw'], cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks(range(n_m)); ax1.set_xticklabels(m_tick_labels)
    ax1.set_yticks(range(n_m)); ax1.set_yticklabels(m_tick_labels)
    ax1.set_title("Raw Cosine Similarity\n(centroid displacements)")
    fig.colorbar(im1, ax=ax1)
    for i in range(n_m):
        for j in range(n_m):
            ax1.text(j, i, f"{mahal_results['cosine_sim_raw'][i,j]:+.2f}",
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(mahal_results['cosine_sim_raw'][i,j]) > 0.6 else 'black')

    im2 = ax2.imshow(mahal_results['cosine_sim_whitened'], cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xticks(range(n_m)); ax2.set_xticklabels(m_tick_labels)
    ax2.set_yticks(range(n_m)); ax2.set_yticklabels(m_tick_labels)
    ax2.set_title("Mahalanobis Cosine Similarity\n(whitened displacements)")
    fig.colorbar(im2, ax=ax2)
    for i in range(n_m):
        for j in range(n_m):
            ax2.text(j, i, f"{mahal_results['cosine_sim_whitened'][i,j]:+.2f}",
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(mahal_results['cosine_sim_whitened'][i,j]) > 0.6 else 'black')

    fig.suptitle("Cluster Displacement Cosine Similarity\n(+1 = same deviation direction from global mean)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    mahal_path = os.path.join(output_dir, "mahalanobis_cosine.png")
    fig.savefig(mahal_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {mahal_path}")

    # --- Plot 5: Null hypothesis comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Cosine similarity distributions
    ax1.hist(null_results['null_cosines'], bins=50, density=True, alpha=0.6,
             color='gray', label=f"Random pairs (n={len(null_results['null_cosines'])})")
    for i, cos in enumerate(null_results['real_cosines']):
        sl = null_results['small_labels']
        # Generate pair label from index
        pair_idx = 0
        for a in range(len(sl)):
            for b in range(a+1, len(sl)):
                if pair_idx == i:
                    pair_label = f"C{sl[a]} vs C{sl[b]}"
                pair_idx += 1
        ax1.axvline(cos, color=f"C{i}", linewidth=2, label=f"{pair_label}: {cos:+.3f}")
    ax1.set_xlabel("Whitened cosine similarity")
    ax1.set_ylabel("Density")
    ax1.set_title("Pairwise Cosine Similarity:\nReal Clusters vs Random Subsets")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Displacement norm distributions
    ax2.hist(null_results['null_norms'], bins=50, density=True, alpha=0.6,
             color='gray', label=f"Random subsets (n={null_results['n_random']})")
    for i, (lab, norm) in enumerate(zip(null_results['small_labels'],
                                         null_results['real_norms'])):
        ax2.axvline(norm, color=f"C{i}", linewidth=2, label=f"C{lab}: {norm:.2f}")
    ax2.set_xlabel("Whitened displacement norm")
    ax2.set_ylabel("Density")
    ax2.set_title("Distance from Global Mean:\nReal Clusters vs Random Subsets")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Null Hypothesis Test: Are Clusters Different from Random Subsets?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    null_path = os.path.join(output_dir, "null_hypothesis.png")
    fig.savefig(null_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {null_path}")

    import shutil
    if shutil.which("imgcat"):
        import subprocess
        for img in [acc_path, dist_path, eig_path, mahal_path, null_path]:
            subprocess.run(["imgcat", img], check=False)


def main():
    parser = argparse.ArgumentParser(
        description="Fisher Linear Discriminant diagnostic for cluster similarity")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing activation data")
    parser.add_argument("--cluster-dir", type=str, required=True,
                        help="Directory containing analysis results (cluster_labels.npy, etc.)")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Subsample rate used in original analysis (for record matching)")
    parser.add_argument("--variance-threshold", type=float, default=0.95,
                        help="Fraction of separability to explain (default: 0.95)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <cluster-dir>/fisher/)")
    args = parser.parse_args()

    cluster_dir = Path(args.cluster_dir)
    label_records = load_analysis_results(cluster_dir)
    activations, features, labels = load_activations_for_labels(
        args.data_dir, label_records, args.subsample)

    unique_labels = sorted(set(labels) - {-1})
    print(f"\nClusters: {unique_labels}")
    print(f"Activation dim: {activations.shape[1]}")

    # Filter out noise
    non_noise = labels != -1
    activations_clean = activations[non_noise]
    features_clean = features[non_noise]
    labels_clean = labels[non_noise]

    # Pairwise Fisher discriminant
    print(f"\nComputing pairwise Fisher discriminants...")
    pairwise_results = {}
    for la, lb in tqdm(list(combinations(unique_labels, 2)), desc="Cluster pairs"):
        mask_a = labels_clean == la
        mask_b = labels_clean == lb
        act_a = activations_clean[mask_a]
        act_b = activations_clean[mask_b]
        n_dims, eigenvalues, accuracy_curve = fisher_discriminant_dimensionality(
            act_a, act_b, variance_threshold=args.variance_threshold)
        pairwise_results[(la, lb)] = {
            'n_dims': n_dims,
            'eigenvalues': eigenvalues,
            'accuracy_curve': accuracy_curve,
        }

    # Feature subspace analysis
    print(f"\nComputing feature subspace analysis...")
    feature_results = feature_subspace_analysis(
        activations_clean, features_clean, labels_clean)

    # Mahalanobis cosine similarity
    print(f"\nComputing Mahalanobis cosine similarity...")
    mahal_results = mahalanobis_cosine_similarity(activations_clean, labels_clean)

    # Null hypothesis test
    print(f"\nRunning null hypothesis test...")
    null_results = null_hypothesis_test(activations_clean, labels_clean)

    # Generate report
    output_dir = args.output_dir or str(cluster_dir / "fisher")
    print(f"\nWriting outputs to {output_dir}/")
    generate_report(pairwise_results, feature_results, mahal_results, null_results, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
