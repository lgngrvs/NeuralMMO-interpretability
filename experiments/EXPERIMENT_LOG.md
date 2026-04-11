# T-SNE Clustering Experiments Log

Date: 2026-04-10

## Code Changes

### `analyze_activations.py`
- Added `run_tsne()` function wrapping sklearn TSNE
- Added CLI flags: `--tsne`, `--tsne-perplexity`, `--tsne-learning-rate`, `--tsne-early-exaggeration`, `--tsne-max-iter`
- Modified pipeline to use T-SNE instead of UMAP when `--tsne` is passed
- Output directory naming includes `-tsne` suffix
- Visualization titles/filenames adapt to reduction method

### `sweep_cluster_params.py`
- Added Strategy 4: PCA -> T-SNE 2D -> HDBSCAN
- Added `--tsne-only` and `--strategies` CLI flags
- Pre-computes T-SNE embeddings with caching by (pca_dims, perplexity)
- Parallelized T-SNE pre-computation with ThreadPoolExecutor

### `sweep_tsne_3d.py` (new)
- Dedicated 3D T-SNE sweep script
- Parallelized T-SNE fitting with ThreadPoolExecutor
- Generates 2x2 multi-angle 3D scatter plots, Cohen's d heatmaps, summaries

### `extract_activations.py`
- Added `--layer` CLI argument: `action_decoder` (default), `lstm_cell`, `lstm_hidden`, `encoder_output`
- Hooks different modules based on layer selection (LSTM cell/hidden via `recurrent_wrapper.recurrent`, encoder via `inner.proj_fc`)

---

## Experiments

### 1. T-SNE_sweep — baseline_10M 2D T-SNE sweep
- **Dir**: `experiments/T-SNE_sweep/`
- **Data**: baseline_10M, subsample=5, 2041 points
- **Method**: PCA [20,30] -> T-SNE 2D [perp 5,15,30,50] -> HDBSCAN sweep
- **Result**: 96 configs, 14 passing verification. Best score 1.249 (perp=50, mcs=50, ms=10, 7 clusters). Higher perplexity (30-50) dominates. T-SNE reveals spatial specialization clusters (east/north/west) and a harvester specialist cluster not as cleanly separated in UMAP.

### 2. T-SNE-sweep-yaofeng-200M — yaofeng 2D T-SNE sweep
- **Dir**: `experiments/T-SNE-sweep-yaofeng-200M/`
- **Data**: yaofeng_200M, subsample=5, 7314 points
- **Method**: Same as above
- **Result**: 96 configs, 45 passing (47%). Best score 1.022 (perp=30, mcs=400, ms=5). Higher verification pass rate than baseline but lower top scores. Equipment (`n_equipped_items`) is the dominant discriminating feature.

### 3. T-SNE-3d-sweep-yaofeng_200M — yaofeng 3D T-SNE sweep
- **Dir**: `experiments/T-SNE-3d-sweep-yaofeng_200M/`
- **Data**: yaofeng_200M, subsample=5, ~7300 points
- **Method**: PCA [20,30] -> T-SNE 3D -> HDBSCAN sweep
- **Result**: 62 passing (65%). Best score 0.940 (perp=50, PCA=30, 3 clusters, 9.8% noise). 3D produces cleaner, more robust clusters than 2D but fewer of them. Same 3 clusters found regardless of HDBSCAN params.

### 4. T-SNE-3d-sweep-takeru_200M — takeru 3D T-SNE sweep
- **Dir**: `experiments/T-SNE-3d-sweep-takeru_200M/`
- **Data**: takeru_200M, subsample=5, 5272 points
- **Method**: Same as yaofeng 3D
- **Result**: **0 configs pass verification.** Best score 0.474. Takeru activations are much harder to cluster — more uniform representation, higher noise rates.

### 5. lifetime_clusters — Agent lifetime cluster membership analysis
- **Dir**: `experiments/lifetime_clusters/`
- **Data**: yaofeng_200M (config 1) and takeru_200M (config 5) from 3D sweeps
- **Method**: KNN-inferred labels at full resolution (subsample=1), smoothed feature overlay plots
- **Key finding**: Yaofeng clusters are temporally coherent (30% transition rate, agents stay in clusters for long stretches). Takeru clusters are "confetti" (67% transition rate, agents flicker between clusters every tick). This is due to Yaofeng having a 2-layer LSTM that smooths representations over time, while Takeru has no recurrence.
- **Death gap handling**: Identified that agents die and respawn mid-episode. Fixed visualization to show life segments separately with relative tick x-axis.

### 6. cluster_separability — Statistical separability of yaofeng clusters
- **Dir**: `experiments/cluster_separability/`
- **Data**: yaofeng_200M, subsample=5
- **Method**: 4 metrics on 30-dim PCA space: silhouette scores, K-NN purity, Mahalanobis distance from C2, density along inter-centroid axes
- **Result**: Both C0 (unequipped) and C1 (food-stressed) are genuinely distinct from C2. K-NN purity: C0=98%, C1=85%. Mahalanobis: C0 at 50.8 (9x expected), C1 at 18.3. Clear density dips between all cluster pairs.

### 7. c2_subclustering — Sub-clustering yaofeng's dominant C2 cluster
- **Dir**: `experiments/c2_subclustering/`
- **Data**: yaofeng_200M C2 points only (~4737), subsample=5
- **Method**: PCA 30 -> T-SNE 2D (perp=30) -> HDBSCAN
- **Result**: 15 sub-clusters within C2. Primarily separated by resource stress (water/food depletion), spatial position, and inventory state.

### 8. c2_subclustering_perp_sweep — C2 subclustering perplexity sweep (2D)
- **Dir**: `experiments/c2_subclustering_perp_sweep/`
- **Method**: Perplexities [40, 50, 60, 70] on C2 in 2D
- **Result**: Non-monotonic — perp=50 is sweet spot (12 sub-clusters, 10% noise). Perp=40 too coarse (3 clusters), perp=70 over-fragments (27 clusters). Core signals (dehydration, food-stress, combat) robust across all perplexities.

### 9. c2_subclustering_perp_sweep_3d — C2 subclustering perplexity sweep (3D)
- **Dir**: `experiments/c2_subclustering_perp_sweep_3d/`
- **Method**: Same perplexities in 3D
- **Result**: 3D over-fragments (22-28 clusters at perp 40/50/70) with higher noise. Perp=60 is most stable in 3D (3 clusters, 18.9% noise). Same feature signatures preserved.

### 10. T-SNE-3d-takeru-highperp — Takeru with higher perplexity + more data
- **Dir**: `experiments/T-SNE-3d-takeru-highperp/`
- **Data**: takeru_200M combined (original + extra), subsample=2, 13144 points
- **Method**: Perplexities [40, 50, 60, 70, 80], HDBSCAN sweep
- **Result**: Best at perp=40 (score 0.66), declining at higher perplexity. 0 passing verification. Equipment axis still dominates. 93% transition rate — still confetti.

### 11. T-SNE-3d-takeru-perp120 — Takeru at perplexity 120
- **Dir**: `experiments/T-SNE-3d-takeru-perp120/`
- **Data**: Combined takeru, subsample=10, 11898 points
- **Method**: PCA 20, perp=120, mcs=400, ms=25
- **Result**: 5 clusters, 61% noise, 0.66 transition rate. Equipment-searching clusters present (C0/C3 unequipped, C1/C4 equipped). Larger mcs reduces transition rate (0.90 at mcs=50 -> 0.66 at mcs=400) but increases noise.

### 12. T-SNE-3d-yaofeng-lstm-cell — Yaofeng LSTM cell state sweep
- **Dir**: `experiments/T-SNE-3d-yaofeng-lstm-cell/`
- **Data**: yaofeng_200M LSTM cell state, subsample=5, 34202 points
- **Method**: PCA [20,30] -> T-SNE 3D [perp 5,15,30,50] -> HDBSCAN (parallelized)
- **Result**: Best score 0.781, 0 passing. Cell state is much more diffuse than hidden state — 72% of variance in 1 PCA component vs 26% for hidden state. Needs only 20 dims for 99% variance (vs 156 for hidden state).

### 13. T-SNE-3d-yaofeng-lstm-cell-highperp — LSTM cell at high perplexity
- **Dir**: `experiments/T-SNE-3d-yaofeng-lstm-cell-highperp/`
- **Data**: Same cell state data, subsample=5
- **Method**: Perplexities [100, 200, 300, 400]
- **Result**: Score improved to 1.024 at perp=100 (31% better), but still 0 passing due to 43% noise. Higher perplexity (200-400) actually hurt — spread embedding too uniformly. n/100 heuristic did not help.

### 14. T-SNE-3d-yaofeng-lstm-cell-sparse — LSTM cell with sparse subsampling
- **Dir**: `experiments/T-SNE-3d-yaofeng-lstm-cell-sparse/`
- **Data**: LSTM cell state, subsample=50, 3484 points
- **Method**: Perplexities [50, 75, 100], thorough HDBSCAN sweep (mcs 15-300, ms 3-25)
- **Result**: **6 configs pass verification** (vs 0 before). Best: perp=75, mcs=150, ms=3, 3 clusters, 29% noise, score 0.877. Transition rate 0.28 (very coherent). Clusters: "Mature Explorer" (36%), "Early Game" (17%), "Combat/Social" (19%). Sparse subsampling was the key fix — reduced autocorrelation enough for HDBSCAN to find clean density structure.

### 15. T-SNE-3d-takeru-combined — Takeru combined datasets sweep
- **Dir**: `experiments/T-SNE-3d-takeru-combined/`
- **Data**: Original + extra combined, subsample=10, 11898 points
- **Method**: Perplexities [30-80], HDBSCAN sweep
- **Result**: Best score 0.610 (29% improvement over single dataset). Still 0 passing due to noise >30%. Equipment axis dominates, 93% transition rate.

### 16. T-SNE-3d-takeru-sparse — Takeru with sparse subsampling
- **Dir**: `experiments/T-SNE-3d-takeru-sparse/`
- **Data**: Combined takeru, subsample [25, 50, 100]
- **Method**: Perplexities [50, 75, 100, 120], thorough HDBSCAN sweep
- **Result**: sub=100 (1268 points) best: noise 28.8%, transition rate 0.43 (huge improvement from 0.96). Score 0.797. Clusters include spatial (spawn area d=-12.6, map edge d=-4.0), equipment, and resource-stress modes. Fails verification on step entropy.

### 17. T-SNE-3d-takeru-10x — Takeru with 10x more data (in progress)
- **Dir**: `experiments/T-SNE-3d-takeru-10x/`
- **Data**: 10x data collection (32 new episodes, seeds 3-10), subsample=100
- **Status**: Running

---

## Key Findings

### Architecture matters for clustering
- **Yaofeng (2-layer LSTM)**: Clean 3-cluster structure (unequipped / food-stressed / steady-state). Temporally coherent (28-30% transition rate). LSTM smooths representations over time.
- **Takeru (no LSTM, feedforward)**: Activations flicker between clusters tick-by-tick (66-93% transition rate). Equipment axis is the only robust signal. No temporal coherence because each frame is processed independently.

### PCA dimensionality reveals representation structure
- Yaofeng hidden state: 26% in top PC, needs 156 dims for 99% — rich, distributed representation
- Yaofeng LSTM cell state: 72% in top PC, needs 20 dims for 99% — primarily a 1D progression counter
- Both encode the same behavioral modes, but the cell state encodes them more diffusely

### Subsampling is critical
- High autocorrelation in dense time series inflates noise in HDBSCAN
- Sparse subsampling (50-100x) dramatically improves cluster quality
- LSTM cell state: 0 passing configs at sub=5 -> 6 passing at sub=50
- Takeru: transition rate dropped from 0.96 (sub=25) to 0.43 (sub=100)

### T-SNE perplexity sweet spots
- For ~7k points: perplexity 30-50 works well
- For ~34k points: perplexity 100 (not 200-400)
- For ~3.5k points: perplexity 50-75
- n/100 heuristic did not consistently help; lower perplexity often better

### C2 subclustering
- The dominant steady-state cluster contains 15+ sub-clusters at fine resolution
- Core robust sub-modes: dehydration, food-stress, combat engagement, spatial position
- These appear at every perplexity tested, in both 2D and 3D
