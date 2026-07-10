# Action-Decoder Cluster Probes Log

## Key Findings

- **Action-decoder T-SNE/HDBSCAN clusters are almost perfectly linearly separable** in the same 256-dim action-decoder space: **99.2% accuracy** (macro F1 = 0.989). This is dramatically higher than the LSTM-cluster probe (70.9%), confirming the nonlinear dimensionality reduction (PCA->T-SNE->HDBSCAN) found structure that is already nearly linear.
- Only **2 clusters** emerged from HDBSCAN (not 3 as in prior experiments), with 1.5% noise. C0 = 6,302 points (17%), C1 = 30,199 points (83%). The 3-cluster result from Experiments 2-3 appears sensitive to exact T-SNE/HDBSCAN parameters.
- **Cross-representation probe**: LSTM cell-state -> AD clusters achieves **88.9% accuracy** (macro F1 = 0.844). The AD cluster boundary is also largely readable from LSTM state, though less perfectly.
- **Probe-PCA alignment is moderate** (max |cosine sim| = 0.257 on PC3), significantly higher than the LSTM cluster probe (max 0.11). The AD-cluster discriminating direction has more overlap with variance directions, consistent with the cluster being more "linear."
- PCA-50 accuracy (99.1%) is nearly identical to raw 256-dim (99.2%), and LDA achieves 98.8% -- the boundary is capturable by a single linear discriminant.

## Detailed Log

### Step 1: Generate action-decoder clusters
**Action:** Loaded yaofeng_200M action-decoder activations (36,501 x 256). Subsampled at rate 5 per trajectory -> 7,314 points. PCA(20) -> T-SNE 3D (perp=50) -> HDBSCAN (mcs=200, ms=3). KNN(k=5) propagated labels to all 36,501 records.
**Result:** 2 clusters (not 3). C0: 1,272 subsample / 6,302 full. C1: 5,931 subsample / 30,199 full. Noise: 1.5%. PCA 20D explains 78.1% variance. T-SNE took 73s.

### Step 2: Train/test split
**Action:** Trajectory-based (env_id, agent_id) GroupShuffleSplit, 80/20, random_state=42.
**Result:** 42 trajectories total. Train: 29,595 (33 traj). Test: 6,906 (9 traj). Class imbalance: C0 ~23% / C1 ~77% in both splits.

### Step 3: Logistic regression probes
**Action:** Multinomial logistic regression on raw 256-dim, PCA-50, and shuffle baseline.
**Result:**
- Raw 256-dim: **accuracy 99.2%**, macro F1 0.989. C0: P=0.989, R=0.978, F1=0.983. C1: P=0.993, R=0.997, F1=0.995.
- PCA-50 (86.5% var): accuracy 99.1%, macro F1 0.987.
- Shuffle baseline: accuracy 77.1% (majority class), macro F1 0.436.

### Step 4: LDA projection
**Action:** LDA with 1 component (binary -> max 1 discriminant).
**Result:** LDA accuracy 98.8%. Clean separation on the single LDA axis.

### Step 5: Probe-PCA alignment
**Action:** Cosine similarity between probe weight vector and top-20 PCA directions.
**Result:** Max |cos sim| = 0.257 (PC3), followed by PC0 at 0.247. Substantially higher than LSTM cluster probe (0.11), meaning the AD-cluster boundary is more aligned with principal variance directions.

### Step 6: Cross-representation probe (LSTM -> AD clusters)
**Action:** Joined LSTM cell-state activations with AD cluster labels on (env_id, agent_id, step). 25,920 matched samples. Trained logistic regression with trajectory-based split.
**Result:** LSTM -> AD-cluster accuracy **88.9%**, macro F1 0.844. C0: F1=0.761, C1: F1=0.927. The AD cluster structure is partially readable from LSTM state but less perfectly than from AD activations themselves.
