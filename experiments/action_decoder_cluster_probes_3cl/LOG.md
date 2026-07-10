# Action-Decoder 3-Cluster Probe Log

## Key Findings
- **All 3 clusters are linearly separable**: 97.5% accuracy (macro F1=0.951) with logistic regression on raw 256-dim activations
- Adding the third cluster (fragile food-stressed mode in PCA dims 21-30) drops accuracy only ~1.7pp from the 2-cluster baseline (99.2% -> 97.5%)
- Cluster 1 (smallest, n=3017) is the hardest to classify: F1=0.888 vs 0.980/0.983 for C0/C2
- PCA-50 probe performs nearly as well (96.97%), and LDA achieves 96.66%
- Probe weight vectors are distributed across many PCA directions (max |cos_sim| = 0.27), meaning cluster boundaries are not aligned with single variance axes
- Shuffle baseline at 69.1% (majority-class rate ~74%) confirms probes are learning real structure

## Detailed Log

### Step 1: Generate 3-cluster labels
**Action:** PCA(30) -> T-SNE 3D (perplexity=50) -> HDBSCAN(min_cluster_size=200, min_samples=10) on 7,314 subsampled points; KNN(k=5) propagation to all 36,501.
**Result:** 3 clusters as expected: C0=6,362, C1=3,017, C2=27,122 (full dataset). 14.6% noise in subsampled data. Matches expected cluster sizes (C0~1252, C1~571, C2~4420 subsampled).

### Step 2: Train/test split
**Action:** Trajectory-based 80/20 split (42 trajectories -> 33 train, 9 test).
**Result:** Train=29,595, Test=6,906. Label distributions roughly proportional.

### Step 3: Logistic regression probes
**Action:** Three probes: raw 256-dim, PCA-50, shuffle baseline.
**Result:**
- Raw 256-dim: 97.52% acc, macro F1=0.9506
- PCA-50: 96.97% acc, macro F1=0.9358
- Shuffle: 69.11% acc, macro F1=0.2757
- Per-class (raw): C0 F1=0.980, C1 F1=0.888, C2 F1=0.983

### Step 4: LDA projection
**Action:** LDA(n_components=2) for visualization and classification.
**Result:** LDA accuracy 96.66%. 2D projection shows clear separation of all 3 clusters.

### Step 5: Probe weight vs PCA analysis
**Action:** Cosine similarity of 3 probe weight vectors against top-20 PCA directions.
**Result:** Max |cos_sim| = 0.27 (C0 weight vs PC3). Probe directions are not aligned with individual PCA components — cluster structure is distributed across the representation space.
