# Per-Cluster T-SNE Log

## Key Findings

- **Clusters are NOT homogeneous blobs.** Every cluster in both representations shows clear internal structure when embedded in isolation via T-SNE.
- **LSTM cell-state clusters** show smooth gradients organized by time (time_alive/tick) and spatial position (self_row/self_col). The gradients are continuous, not discrete sub-modes. C0 (early-game) is the most structured, with a clear temporal progression axis. C2 (steady-state, n=86k) is the most diffuse -- a large round cloud with weak spatial gradients.
- **Action decoder clusters** show sharper sub-structure than LSTM. C0 (unequipped, n=6568) fragments into spatial sub-regions (self_row/self_col strongly organized). C2 (equipped steady-state, n=28,797) shows the richest internal structure: a branching topology with a clear time_alive gradient and spatial arms.
- **Action decoder C1** is small (n=1,136) and corresponds to food-stressed states. It shows a self_row gradient (agents at different map positions experience food stress differently).
- **Top-varying features within clusters** are consistently: time_alive, tick, self_col, self_row, nearest_player_dist. These temporal and spatial features dominate intra-cluster variation, while the features that *distinguish* clusters (n_equipped_items, n_inventory_items, self_food) are relatively uniform within each cluster.
- **LSTM cell-state has higher PCA variance explained** (98-99%) vs action decoder (83-89%), confirming LSTM cell state is lower-dimensional. Despite this, both show comparable internal structure when embedded.

## Detailed Log

### Step 1: Load data and verify alignment
**Action:** Loaded LSTM cell-state activations (170,775 x 256) with pre-existing cluster labels from exp 23. Loaded action decoder activations (36,501 x 256).
**Result:** LSTM labels matched 100% by index (same extraction run). Action decoder needed fresh clustering.

### Step 2: Generate action decoder clusters
**Action:** Pipeline: simple stride-5 subsample -> PCA(30) -> T-SNE 3D (perp=50) -> HDBSCAN (mcs=200, ms=3) -> KNN propagation to full resolution. Used PCA=30 and simple subsampling to match exp 2-3 config (per-trajectory subsampling with PCA=20 produced only 2 clusters).
**Result:** 3 clusters: C0 (6,568 = 18%), C1 (1,136 = 3%), C2 (28,797 = 79%). C0 = unequipped, C1 = food-stressed, C2 = steady-state equipped. Matches the classic yaofeng 3-cluster structure.

### Step 3: Per-cluster T-SNE embedding
**Action:** For each of 6 clusters (3 LSTM + 3 action decoder): subsample to 3000 points, PCA(30), T-SNE 2D and 3D with perplexity=min(50, N//5). Colored by top-6 highest-std features (2D) and top-1 feature (3D).
**Result:** All clusters show internal structure. See key findings above. Total runtime: 225 seconds.

### Step 4: Build summary and output
**Action:** Created summary.png (2x3 grid: rows=LSTM/action-decoder, cols=C0/C1/C2), individual per-cluster plots (2D and 3D), and run_stats.json.
**Result:** 12 individual plots + 1 summary. All saved under experiments/per_cluster_tsne/.

## Cluster Detail

### LSTM Cell-State
| Cluster | N | PCA Var | Top Varying (intra) | Discriminating (vs rest) |
|---------|------|---------|---------------------|--------------------------|
| C0 (early-game) | 36,146 | 98.4% | time_alive, tick, self_col | n_equipped_items |
| C1 (combat/trade) | 48,095 | 98.6% | time_alive, tick, self_col | n_inventory_items |
| C2 (steady-state) | 86,534 | 99.4% | time_alive, tick, self_row | tick |

### Action Decoder
| Cluster | N | PCA Var | Top Varying (intra) | Discriminating (vs rest) |
|---------|------|---------|---------------------|--------------------------|
| C0 (unequipped) | 6,568 | 83.3% | self_col, self_row, time_alive | n_equipped_items |
| C1 (food-stressed) | 1,136 | 89.4% | time_alive, tick, self_col | self_food |
| C2 (steady-state) | 28,797 | 88.2% | time_alive, tick, self_col | n_equipped_items |
