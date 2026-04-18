# LSTM Cluster Probes Log

## Key Findings
- **LSTM clusters are partially decodable from action-decoder activations**, but NOT fully linearly separable (accuracy ~69-71%, well above 42% shuffle baseline, but below the 90% threshold for "linearly decodable").
- Cluster 0 is well-separated (F1=0.83), Cluster 2 is moderate (F1=0.75), but **Cluster 1 is poorly distinguished** (F1=0.38) -- the probe confuses it heavily with Cluster 2.
- PCA(50) slightly outperforms raw 256-dim (70.9% vs 68.9%), suggesting some noise dimensions hurt the raw probe.
- LDA achieves 70.2% accuracy -- comparable to logistic regression, confirming the linear separability ceiling.
- **Probe weight directions are nearly orthogonal to PCA variance directions** (max cosine sim = 0.11). The cluster-separating directions are NOT aligned with principal variance -- they live in low-variance subspaces. This is a strong finding: the information is encoded, but not in the directions of maximum activation variance.

## Detailed Log

### Step 1: Data loading and joining
**Action:** Loaded action-decoder cache (36,501 records, 256-dim) and LSTM cluster labels (170,775 records). Joined on (env_id, agent_id, step).
**Result:** 25,920 matched samples. Label distribution: {0: 7295, 1: 6595, 2: 12030}. Class 2 is majority (~46%).

### Step 2: Train/test split
**Action:** Trajectory-based split (by unique env_id, agent_id pairs), 80/20, random_state=42.
**Result:** 42 unique trajectories -> 33 train (19,873 samples), 9 test (6,047 samples). Label distributions are roughly proportional.

### Step 3: Logistic regression probes
**Action:** Trained multinomial logistic regression on (a) raw 256-dim, (b) PCA-50, (c) shuffle baseline. StandardScaler applied.
**Result:**
- Raw 256-dim: accuracy=0.689, macro F1=0.652. Cluster 0: F1=0.83, Cluster 1: F1=0.38, Cluster 2: F1=0.75.
- PCA-50: accuracy=0.709, macro F1=0.664. Slight improvement. PCA explains 86.8% of variance.
- Shuffle baseline: accuracy=0.424, macro F1=0.233. Confirms probes are learning real signal.

### Step 4: LDA projection
**Action:** Fit LDA(n_components=2) on scaled activations, projected full dataset.
**Result:** LDA classification accuracy=0.702. The 2D projection shows partial but incomplete separation -- Clusters 0 and 2 separate well along LD1, but Cluster 1 overlaps heavily with Cluster 2.

### Step 5: Probe direction analysis
**Action:** Computed cosine similarity between logistic regression weight vectors (3x256) and top-20 PCA directions.
**Result:** Maximum |cosine similarity| is only 0.11 across all (probe, PCA) pairs. The cluster-discriminating directions are nearly orthogonal to principal variance directions. This means the behavioral mode information is encoded in low-variance dimensions that PCA would discard if too aggressive.
