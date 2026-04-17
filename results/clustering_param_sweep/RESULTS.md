# Activation Clustering Hyperparameter Sweep Results

## Executive Summary

We swept clustering hyperparameters to find configurations that produce 2-8 clusters in the action decoder's hidden state space, corresponding to behavioral modes (combat, foraging, trading, etc.). We tested **3 strategies** across **2 policies** (baseline_10M and takeru_200M), evaluating 172 parameter combinations per policy.

**Key finding:** K-means on PCA-reduced activations reliably produces well-separated clusters that pass all verification criteria, while HDBSCAN struggles with high noise rates. However, the dominant cluster structure encodes **agent progression** (game phase / experience level) rather than distinct behavioral modes like "in combat" vs "foraging."

## Methodology

### Data Pipeline
1. Load activation records from `activation_data/{policy}/activations.json`
2. Filter dead timesteps (agent_id == 0 or health <= 0)
3. Compute 14 behavioral features from observations and actions
4. Reduce dimensionality with PCA
5. Cluster with one of three strategies
6. Score and verify each parameter combination

### Three Clustering Strategies

| Strategy | Description | Parameters Swept |
|----------|-------------|-----------------|
| **S1: PCA + HDBSCAN** | Cluster in high-D PCA space | PCA dims: {10,20,30,50}, min_cluster_size: {50,100,200,400}, min_samples: {5,10,25} |
| **S2: PCA + UMAP + HDBSCAN** | Reduce to 2D with UMAP, then cluster | PCA dims: {20,30}, UMAP n_neighbors: {15,30}, min_dist: {0.0,0.1}, then HDBSCAN as above |
| **S3: PCA + K-means** | Force k clusters in PCA space | PCA dims: {10,20,30,50}, k: {2,3,4,5,6,7,8} |

### Composite Scoring Metric

```
score = mean_weighted_d * cluster_count_factor * (1 - noise_ratio) * entropy_factor
```

- **mean_weighted_d**: Size-weighted mean of top-3 |Cohen's d| per cluster. Large clusters with moderate effect sizes score higher than tiny clusters with huge effect sizes.
- **cluster_count_factor**: 1.0 for 2-8 clusters, 0.0 for 0-1, drops for >8.
- **noise_ratio**: Fraction of points not assigned to any cluster (k-means has 0% by definition).
- **entropy_factor**: Size-weighted mean of step and agent entropy. Penalizes clusters that are temporal or agent-specific confounds.

### Verification Criteria (pass/fail)

| Criterion | Threshold |
|-----------|-----------|
| Number of clusters | 2-8 |
| Noise ratio | < 30% |
| Min cluster size | > 2% of N |
| Size-weighted mean top-3 \|Cohen's d\| | > 0.8 |
| Min step entropy (any cluster) | > 0.4 |
| Min agent entropy (any cluster) | > 0.4 |

### 14 Behavioral Features Used for Evaluation

n_visible_entities, n_visible_npcs, n_visible_players, self_health, self_food, self_water, self_gold, max_combat_level, in_combat, n_inventory_items, tick, is_moving, is_attacking, is_trading

---

## Results: baseline_10M (10,111 activation records)

### PCA Variance Explained

| Dims | Variance Explained |
|------|--------------------|
| 10 | 44.5% |
| 20 | 55.4% |
| 30 | 62.1% |
| 50 | 70.8% |

### Top 10 Configurations

| Rank | Strategy | PCA | k/MCS | MS | #Cl | Noise% | WtdD | Ent | Score | Pass |
|------|----------|-----|-------|----|-----|--------|------|-----|-------|------|
| 1 | pca_kmeans | 50 | 3 | - | 3 | 0.0% | 1.60 | 0.84 | 1.347 | YES |
| 2 | pca_kmeans | 10 | 3 | - | 3 | 0.0% | 1.59 | 0.84 | 1.340 | YES |
| 3 | pca_kmeans | 20 | 3 | - | 3 | 0.0% | 1.55 | 0.84 | 1.303 | YES |
| 4 | pca_kmeans | 30 | 3 | - | 3 | 0.0% | 1.54 | 0.84 | 1.297 | YES |
| 5 | pca_kmeans | 20 | 4 | - | 4 | 0.0% | 1.52 | 0.83 | 1.262 | YES |
| 6 | pca_kmeans | 30 | 4 | - | 4 | 0.0% | 1.46 | 0.83 | 1.212 | YES |
| 7 | pca_kmeans | 50 | 4 | - | 4 | 0.0% | 1.46 | 0.83 | 1.212 | YES |
| 8 | pca_kmeans | 50 | 7 | - | 7 | 0.0% | 1.45 | 0.82 | 1.192 | YES |
| 9 | pca_kmeans | 30 | 8 | - | 8 | 0.0% | 1.44 | 0.82 | 1.184 | YES |
| 10 | pca_kmeans | 20 | 6 | - | 6 | 0.0% | 1.42 | 0.83 | 1.179 | YES |

**All top 20 are K-means. All pass verification.** No HDBSCAN configuration passed (minimum noise was 73%).

### Best Configuration: PCA 50D + K-means k=3

| Cluster | Size | % of Data | Top Features (Cohen's d) |
|---------|------|-----------|--------------------------|
| C0 | 2,677 | 26.5% | tick (d=-2.47), max_combat_level (d=-2.07), self_gold (d=-1.61) |
| C1 | 4,339 | 42.9% | max_combat_level (d=+1.16), tick (d=+1.11), n_visible_players (d=-1.10) |
| C2 | 3,095 | 30.6% | tick (d=-2.18), max_combat_level (d=-1.91), self_gold (d=-1.54) |

**Interpretation:** The 3 clusters primarily separate agents by game phase:
- **C0**: Early game agents — low tick, low combat level, low gold, many visible players nearby
- **C1**: Experienced agents — high combat level, advanced in episode, fewer visible players (spread out)
- **C2**: Early-mid game agents — similar to C0 but with different inventory/resource profile

### Why HDBSCAN Failed on baseline_10M

HDBSCAN requires density peaks to form clusters. In PCA space (S1), activations are too diffuse — no dense islands exist, so most points become noise. In UMAP 2D space (S2), visual structure exists but HDBSCAN's density-based approach still labels 60-80% as noise because the clusters blend into each other at their boundaries.

---

## Results: takeru_200M (26,271 activation records)

### PCA Variance Explained

| Dims | Variance Explained |
|------|--------------------|
| 10 | 79.1% |
| 20 | 89.9% |
| 30 | 94.2% |
| 50 | 97.6% |

Note: takeru's activations are much more concentrated than baseline's (79% vs 44% in 10D).

### Top 10 Configurations

| Rank | Strategy | PCA | UNN | UMD | k/MCS | MS | #Cl | Noise% | WtdD | Ent | Score | Pass |
|------|----------|-----|-----|-----|-------|----|-----|--------|------|-----|-------|------|
| 1 | umap_hdbscan | 30 | 15 | 0.0 | 50 | 10 | 5 | 28.6% | 0.81 | 0.87 | 0.505 | YES |
| 2 | pca_kmeans | 50 | - | - | 5 | - | 5 | 0.0% | 0.57 | 0.88 | 0.504 | no |
| 3 | pca_kmeans | 30 | - | - | 5 | - | 5 | 0.0% | 0.57 | 0.89 | 0.503 | no |
| 4 | pca_kmeans | 20 | - | - | 5 | - | 5 | 0.0% | 0.56 | 0.88 | 0.496 | no |
| 5 | pca_kmeans | 10 | - | - | 5 | - | 5 | 0.0% | 0.55 | 0.89 | 0.483 | no |
| 6 | pca_kmeans | 20 | - | - | 8 | - | 8 | 0.0% | 0.55 | 0.88 | 0.481 | no |
| 7 | pca_kmeans | 20 | - | - | 6 | - | 6 | 0.0% | 0.54 | 0.88 | 0.479 | no |
| 8 | umap_hdbscan | 20 | 15 | 0.0 | 400 | 10 | 8 | 16.1% | 0.65 | 0.88 | 0.478 | no |
| 9 | pca_kmeans | 10 | - | - | 4 | - | 4 | 0.0% | 0.53 | 0.89 | 0.473 | no |
| 10 | pca_kmeans | 50 | - | - | 6 | - | 6 | 0.0% | 0.53 | 0.88 | 0.472 | no |

**Only 1 configuration passes verification** — UMAP+HDBSCAN with min_dist=0.0. K-means produces clusters but their effect sizes are too low (d < 0.8) to pass the threshold.

### Best Configuration: PCA 30D + UMAP(nn=15, min_dist=0.0) + HDBSCAN(mcs=50, ms=10)

| Cluster | Size | % of Data | Top Features (Cohen's d) |
|---------|------|-----------|--------------------------|
| C7 | 1,162 | 4.4% | max_combat_level (d=+0.70), tick (d=+0.55), n_visible_players (d=-0.50) |
| C10 | 1,112 | 4.2% | max_combat_level (d=+0.61), tick (d=+0.59), n_visible_players (d=-0.58) |
| C11 | 876 | 3.3% | max_combat_level (d=-1.74), tick (d=-1.59), self_gold (d=-1.36) |
| C41 | 534 | 2.0% | self_food (d=-0.90), tick (d=+0.58), max_combat_level (d=+0.52) |
| C64 | 527 | 2.0% | tick (d=-0.76), max_combat_level (d=-0.75), in_combat (d=+0.38) |

**Interpretation:** The more-trained policy has a more uniform activation space. Clusters are smaller and effect sizes weaker. The dominant distinguishing features remain tick and max_combat_level (game progression), with hints of behavioral modes: C41 correlates with low food (foraging need), C64 with combat.

---

## Earlier Sweep: PCA + HDBSCAN Only (with cluster_selection_epsilon)

Before adding K-means and UMAP strategies, we ran a sweep of PCA+HDBSCAN with `cluster_selection_epsilon` in {0.0, 0.1, 0.5} across 240 combinations. Results for both policies:

- **baseline_10M**: Best score 0.454 (PCA=10, mcs=100, ms=5, eps=0.0). 3 clusters, 73.2% noise. No combos passed verification. Epsilon had no effect (identical results across all epsilon values for each PCA/HDBSCAN combo).
- **takeru_200M**: Best score 0.000. Only 1 cluster found in most combos (97.3% noise). The more-trained policy's activations are too uniform for density clustering in PCA space.

---

## Cross-Policy Comparison

| Metric | baseline_10M | takeru_200M |
|--------|-------------|-------------|
| Alive records | 10,111 | 26,271 |
| PCA 10D variance | 44.5% | 79.1% |
| Best score | 1.347 | 0.505 |
| Best strategy | PCA + K-means | UMAP + HDBSCAN |
| Configs passing verification | 28/172 (all K-means) | 1/172 |
| Dominant cluster features | tick, max_combat_level | tick, max_combat_level |

The more-trained policy (takeru_200M) has more concentrated activations (higher PCA variance in fewer dims) but weaker cluster separation. This suggests training produces a more uniform internal representation where behavioral variation is encoded more subtly.

---

## Conclusions

1. **K-means is the right algorithm** for this data. Neural network activations form continuous manifolds, not discrete density peaks, making HDBSCAN inappropriate in high-D. HDBSCAN only works after UMAP compression with min_dist=0.0 (which forces tight clusters).

2. **k=3 is the sweet spot** for baseline_10M. The effect size is highest at k=3 and degrades as k increases, suggesting 3 natural divisions in the activation space.

3. **The activation space primarily encodes game phase**, not instantaneous behavioral mode. The top distinguishing features (tick, max_combat_level, self_gold) all track agent progression rather than actions like combat or trading.

4. **More-trained policies are harder to cluster.** takeru_200M's activations are more compact and uniform, producing weaker cluster separations.

## Suggested Next Steps

- **Regress out game phase** (tick, max_combat_level) from activations before clustering to reveal behavioral variation within a game phase
- **Cluster within time windows** (e.g., only tick 100-400) to control for progression effects
- **Collect more episodes** with diverse behavioral situations to improve sample diversity

## Files

- `sweep_cluster_params.py` — The sweep script (run with `uv run python sweep_cluster_params.py <data_dir> --subsample 1`)
- `sweep_results/sweep_baseline_10M_*.csv` — Full ranked results for baseline
- `sweep_results/sweep_takeru_200M_*.csv` — Full ranked results for takeru
- `sweep_results/best_umap.png` — UMAP scatter of best combo (last run)
- `sweep_results/best_features.png` — Cohen's d heatmap of best combo (last run)
