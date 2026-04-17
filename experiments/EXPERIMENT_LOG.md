# Neural MMO Interpretability — Experiment Log

This is the single source of truth for all interpretability experiments run on Neural MMO RL agent activations. Experiments span T-SNE clustering, linear/nonlinear probes, causal mediation analysis, and activation patching.

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

### More data confirms but attenuates spatial signal in Takeru
- With 1x data (1268 points), spatial cluster had self_row d=-12.6 — suspiciously large
- With 5-8x data (~7900 points), same cluster survives at d=-5 to -7 — real but inflated by small sample
- The cluster captures early-game agents near spawn (low row, no equipment, full food)
- More data did NOT help Takeru form stable behavioral modes — the ~96% transition rate persists

### Representations are mostly linear
- Linear probes (ridge/logistic regression) capture most of the structure in 256-dim activations
- MLP probes provide marginal gains on some features (combat/skill levels) but never dramatically outperform linear probes
- The activation space encodes behavioral features in largely linearly-accessible directions

### Tick is a real but subtle causal modulator
- The tick probe direction is 14-94x stronger than random directions in activation patching — a real causal signal
- But it accounts for <1% of full-swap behavioral change in movement (0.004 vs 0.70 KL)
- Largest tick-direction effect is on economic heads (gold_quantity 78x random, inventory_price 94x random)
- Most early-vs-late behavioral change comes from other co-varying dimensions (position, resources, combat state)

### n_visible_players encodes through indirect channels
- The model does not receive other-player observations, yet the probe achieves R²=0.57
- After controlling for combat, position, and time features, R²=0.40 survives — 70% of the signal is NOT explained by obvious confounders
- Remaining signal likely comes from entity distances routed through attack masking, resource depletion patterns, or other indirect channels

---

## Data Collection

### Original data (S3)
- Source: `s3://doxascope-tests/activation_data/`
- 6 policies: baseline_10M, learner, takeru_100M, takeru_200M, yaofeng_100M, yaofeng_200M
- ~32k records per policy, ~1k after dead-filtering + subsampling (rate=10)

### 10x data collection
- Collected 40 episodes per policy using `--streaming` mode (JSONL with periodic disk flushing to avoid OOM)
- 3 policies run in parallel on separate GPUs (L4 x3): baseline_10M, takeru_100M, yaofeng_100M
- Results: 540k-610k records per policy, 280k alive after filtering, ~28k after subsampling
- Stored in `activation_data_10x/` as `.jsonl` files with `.cache.npz` binary caches for fast reloading

**Key change**: Added `StreamingRecordWriter` to `extract_activations.py` to flush records to disk every 5000 records instead of accumulating in RAM. Previous large runs crashed the server due to OOM.

---

## Experiments

### T-SNE Clustering (Experiments 1-17)

#### 1. T-SNE_sweep — baseline_10M 2D T-SNE sweep
- **Dir**: `experiments/T-SNE_sweep/`
- **Data**: baseline_10M, subsample=5, 2041 points
- **Method**: PCA [20,30] -> T-SNE 2D [perp 5,15,30,50] -> HDBSCAN sweep
- **Result**: 96 configs, 14 passing verification. Best score 1.249 (perp=50, mcs=50, ms=10, 7 clusters). Higher perplexity (30-50) dominates. T-SNE reveals spatial specialization clusters (east/north/west) and a harvester specialist cluster not as cleanly separated in UMAP.

#### 2. T-SNE-sweep-yaofeng-200M — yaofeng 2D T-SNE sweep
- **Dir**: `experiments/T-SNE-sweep-yaofeng-200M/`
- **Data**: yaofeng_200M, subsample=5, 7314 points
- **Method**: Same as above
- **Result**: 96 configs, 45 passing (47%). Best score 1.022 (perp=30, mcs=400, ms=5). Higher verification pass rate than baseline but lower top scores. Equipment (`n_equipped_items`) is the dominant discriminating feature.

#### 3. T-SNE-3d-sweep-yaofeng_200M — yaofeng 3D T-SNE sweep
- **Dir**: `experiments/T-SNE-3d-sweep-yaofeng_200M/`
- **Data**: yaofeng_200M, subsample=5, ~7300 points
- **Method**: PCA [20,30] -> T-SNE 3D -> HDBSCAN sweep
- **Result**: 62 passing (65%). Best score 0.940 (perp=50, PCA=30, 3 clusters, 9.8% noise). 3D produces cleaner, more robust clusters than 2D but fewer of them. Same 3 clusters found regardless of HDBSCAN params.

#### 4. T-SNE-3d-sweep-takeru_200M — takeru 3D T-SNE sweep
- **Dir**: `experiments/T-SNE-3d-sweep-takeru_200M/`
- **Data**: takeru_200M, subsample=5, 5272 points
- **Method**: Same as yaofeng 3D
- **Result**: **0 configs pass verification.** Best score 0.474. Takeru activations are much harder to cluster — more uniform representation, higher noise rates.

#### 5. lifetime_clusters — Agent lifetime cluster membership analysis
- **Dir**: `experiments/lifetime_clusters/`
- **Data**: yaofeng_200M (config 1) and takeru_200M (config 5) from 3D sweeps
- **Method**: KNN-inferred labels at full resolution (subsample=1), smoothed feature overlay plots
- **Key finding**: Yaofeng clusters are temporally coherent (30% transition rate, agents stay in clusters for long stretches). Takeru clusters are "confetti" (67% transition rate, agents flicker between clusters every tick). This is due to Yaofeng having a 2-layer LSTM that smooths representations over time, while Takeru has no recurrence.
- **Death gap handling**: Identified that agents die and respawn mid-episode. Fixed visualization to show life segments separately with relative tick x-axis.

#### 6. cluster_separability — Statistical separability of yaofeng clusters
- **Dir**: `experiments/cluster_separability/`
- **Data**: yaofeng_200M, subsample=5
- **Method**: 4 metrics on 30-dim PCA space: silhouette scores, K-NN purity, Mahalanobis distance from C2, density along inter-centroid axes
- **Result**: Both C0 (unequipped) and C1 (food-stressed) are genuinely distinct from C2. K-NN purity: C0=98%, C1=85%. Mahalanobis: C0 at 50.8 (9x expected), C1 at 18.3. Clear density dips between all cluster pairs.

#### 7. c2_subclustering — Sub-clustering yaofeng's dominant C2 cluster
- **Dir**: `experiments/c2_subclustering/`
- **Data**: yaofeng_200M C2 points only (~4737), subsample=5
- **Method**: PCA 30 -> T-SNE 2D (perp=30) -> HDBSCAN
- **Result**: 15 sub-clusters within C2. Primarily separated by resource stress (water/food depletion), spatial position, and inventory state.

#### 8. c2_subclustering_perp_sweep — C2 subclustering perplexity sweep (2D)
- **Dir**: `experiments/c2_subclustering_perp_sweep/`
- **Method**: Perplexities [40, 50, 60, 70] on C2 in 2D
- **Result**: Non-monotonic — perp=50 is sweet spot (12 sub-clusters, 10% noise). Perp=40 too coarse (3 clusters), perp=70 over-fragments (27 clusters). Core signals (dehydration, food-stress, combat) robust across all perplexities.

#### 9. c2_subclustering_perp_sweep_3d — C2 subclustering perplexity sweep (3D)
- **Dir**: `experiments/c2_subclustering_perp_sweep_3d/`
- **Method**: Same perplexities in 3D
- **Result**: 3D over-fragments (22-28 clusters at perp 40/50/70) with higher noise. Perp=60 is most stable in 3D (3 clusters, 18.9% noise). Same feature signatures preserved.

#### 10. T-SNE-3d-takeru-highperp — Takeru with higher perplexity + more data
- **Dir**: `experiments/T-SNE-3d-takeru-highperp/`
- **Data**: takeru_200M combined (original + extra), subsample=2, 13144 points
- **Method**: Perplexities [40, 50, 60, 70, 80], HDBSCAN sweep
- **Result**: Best at perp=40 (score 0.66), declining at higher perplexity. 0 passing verification. Equipment axis still dominates. 93% transition rate — still confetti.

#### 11. T-SNE-3d-takeru-perp120 — Takeru at perplexity 120
- **Dir**: `experiments/T-SNE-3d-takeru-perp120/`
- **Data**: Combined takeru, subsample=10, 11898 points
- **Method**: PCA 20, perp=120, mcs=400, ms=25
- **Result**: 5 clusters, 61% noise, 0.66 transition rate. Equipment-searching clusters present (C0/C3 unequipped, C1/C4 equipped). Larger mcs reduces transition rate (0.90 at mcs=50 -> 0.66 at mcs=400) but increases noise.

#### 12. T-SNE-3d-yaofeng-lstm-cell — Yaofeng LSTM cell state sweep
- **Dir**: `experiments/T-SNE-3d-yaofeng-lstm-cell/`
- **Data**: yaofeng_200M LSTM cell state, subsample=5, 34202 points
- **Method**: PCA [20,30] -> T-SNE 3D [perp 5,15,30,50] -> HDBSCAN (parallelized)
- **Result**: Best score 0.781, 0 passing. Cell state is much more diffuse than hidden state — 72% of variance in 1 PCA component vs 26% for hidden state. Needs only 20 dims for 99% variance (vs 156 for hidden state).

#### 13. T-SNE-3d-yaofeng-lstm-cell-highperp — LSTM cell at high perplexity
- **Dir**: `experiments/T-SNE-3d-yaofeng-lstm-cell-highperp/`
- **Data**: Same cell state data, subsample=5
- **Method**: Perplexities [100, 200, 300, 400]
- **Result**: Score improved to 1.024 at perp=100 (31% better), but still 0 passing due to 43% noise. Higher perplexity (200-400) actually hurt — spread embedding too uniformly. n/100 heuristic did not help.

#### 14. T-SNE-3d-yaofeng-lstm-cell-sparse — LSTM cell with sparse subsampling
- **Dir**: `experiments/T-SNE-3d-yaofeng-lstm-cell-sparse/`
- **Data**: LSTM cell state, subsample=50, 3484 points
- **Method**: Perplexities [50, 75, 100], thorough HDBSCAN sweep (mcs 15-300, ms 3-25)
- **Result**: **6 configs pass verification** (vs 0 before). Best: perp=75, mcs=150, ms=3, 3 clusters, 29% noise, score 0.877. Transition rate 0.28 (very coherent). Clusters: "Mature Explorer" (36%), "Early Game" (17%), "Combat/Social" (19%). Sparse subsampling was the key fix — reduced autocorrelation enough for HDBSCAN to find clean density structure.

#### 15. T-SNE-3d-takeru-combined — Takeru combined datasets sweep
- **Dir**: `experiments/T-SNE-3d-takeru-combined/`
- **Data**: Original + extra combined, subsample=10, 11898 points
- **Method**: Perplexities [30-80], HDBSCAN sweep
- **Result**: Best score 0.610 (29% improvement over single dataset). Still 0 passing due to noise >30%. Equipment axis dominates, 93% transition rate.

#### 16. T-SNE-3d-takeru-sparse — Takeru with sparse subsampling
- **Dir**: `experiments/T-SNE-3d-takeru-sparse/`
- **Data**: Combined takeru, subsample [25, 50, 100]
- **Method**: Perplexities [50, 75, 100, 120], thorough HDBSCAN sweep
- **Result**: sub=100 (1268 points) best: noise 28.8%, transition rate 0.43 (huge improvement from 0.96). Score 0.797. Clusters include spatial (spawn area d=-12.6, map edge d=-4.0), equipment, and resource-stress modes. Fails verification on step entropy.

#### 17. T-SNE-3d-takeru-10x — Takeru with 5-8x more data
- **Dir**: `experiments/T-SNE-3d-takeru-10x/`
- **Data**: 7 datasets combined (original + extra + batches 3-7), ~7900 points at subsample=100
- **Method**: PCA 20D, T-SNE 3D [perp 50,75,100,120], HDBSCAN sweep (parallelized)
- **Result**: Best score 1.981 (perp=100, mcs=15, ms=3, 2 clusters). 1 config passes verification (perp=75, mcs=200, 7 clusters, 28% noise). The d=-12.6 spatial cluster from the 1x experiment shrunk to d=-5 to -7 with more data — confirming it was inflated by small sample size but the signal is real. Transition rates remain ~96% — Takeru's feedforward architecture fundamentally prevents temporally coherent clusters regardless of data quantity.

### Linear and Nonlinear Probes (Experiments 18-20)

#### 18. Linear Probes (baseline)
- **Script**: `train_probes.py`
- **Method**: Ridge regression (continuous) / Logistic regression (binary) on 256-dim activations with StandardScaler. Trajectory-based train/test split to avoid temporal leakage. Three conditions: full (256d), PCA (50d), shuffle baseline.

##### 1x data results (1,031 samples)
Many features had negative R² (worse than predicting the mean), indicating insufficient data. Best performers: tick/time_alive (R²=0.87), self_col (0.71), max_combat_level (0.69), self_health (0.68).

##### 10x data results (28,089 samples)
Dramatic improvement across the board:
- tick/time_alive: 0.87 -> **0.89**
- self_col: 0.71 -> **0.85**
- self_row: 0.61 -> **0.82**
- max_combat_level: 0.69 -> **0.79**
- range_level: 0.28 -> **0.76** (biggest jump)
- n_visible_players: 0.03 -> **0.57**
- fishing_level: 0.03 -> **0.54**
- Features that were negative R² at 1x became positive at 10x

**Results**: `experiments/linear_baseline/`, `experiments/linear_10x/`

#### 19. Class Reweighting
- **Script**: `train_probes.py --class-weight balanced`
- **Method**: For binary features, uses sklearn's `class_weight="balanced"`. For continuous features, computes sample weights via quantile binning (inverse frequency weighting).

Reweighting had modest effects. Binary feature recall improved slightly for rare classes (is_trading, is_using_item) but overall metrics were similar. The main benefit is fairer evaluation on imbalanced features.

**Results**: `experiments/linear_balanced/`

#### 20. Nonlinear (MLP) Probes
- **Script**: `train_nonlinear_probes.py`
- **Architecture**: 2-layer ReLU MLP: `Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)`

##### MLP-64 (hidden_factor=4, ~16.5k params)
- Catastrophic overfitting on many features, especially PCA condition
- self_health R² went negative (-2.46 on 1x data, 0.73 on 10x)
- Some features (item_level, n_listed_items) had R² < -0.5 even on 10x
- Best on spatial/temporal features: self_col (0.92), tick (0.94)

##### MLP-8 (hidden_factor=32, ~2k params)
- Fixed catastrophic overfitting on full activations
- PCA condition still bad (hidden=50//32=1 unit bottleneck)
- Beat linear on combat/skill levels: max_combat 0.82 vs 0.79, fishing 0.63 vs 0.54
- Worse on vitals: self_health 0.49 vs 0.72 (underfitting)

##### Conclusion
Representations are **mostly linear**. MLPs provide marginal gains on some features but never dramatically outperform linear probes. The activation space encodes behavioral features in largely linearly-accessible directions.

**Results**: `experiments/nonlinear_baseline/`, `experiments/nonlinear_10x/`, `experiments/nonlinear_10x_small/`

### Causal Analysis (Experiments 21-22)

#### 21. Causal Mediation Analysis
- **Script**: `mediation_analysis.py`
- **Method**: Frisch-Waugh-Lovell residualization. Regress out mediator features from both activations and target, then probe residuals. See `docs/mediation_analysis_explainer.md` for methodology.

##### n_visible_players mediation

The model architecture does not receive other-player observations, yet the probe achieves R²=0.57. We tested whether this is mediated through features the model can observe:

| Mediator Set | Residualized R² | % Explained by Mediators |
|---|---|---|
| Combat (in_combat, damage, combat_level, is_attacking) | 0.413 | 27.3% |
| Position (self_row, self_col) | 0.570 | -0.3% |
| Time (tick, time_alive) | 0.422 | 25.7% |
| All combined | **0.399** | **29.7%** |

**Surprising finding**: R²=0.40 survives after controlling for all obvious confounders. The mediator sets overlap heavily (combat and time are largely redundant). The remaining signal may come from entity distances routed through attack masking, resource depletion patterns, or other indirect channels not included in the mediator set.

##### Tick control for all features

Many features correlate with game progression. After residualizing tick:

**Genuinely encoded (<5% via tick)**:
- self_health (0.72, -0.2%), self_food (0.71), self_water (0.73)
- self_row (0.82), self_col (0.85)
- self_damage, n_inventory_items, n_visible_npcs

**Heavily tick-confounded (>50% via tick)**:
- time_alive (101% -- pure tick proxy)
- self_gold (86%), nearest_entity_dist (65%), max_harvest_level (60%)
- herbalism_level (51%), nearest_player_dist (55%)

**Partially confounded (25-50%)**:
- Combat levels (melee 31%, range 46%, mage 47%)
- n_visible_players (26%)

**Results**: `experiments/mediation_analysis/`

#### 22. Subspace Activation Patching
- **Script**: `activation_patching.py`
- **Method**: Interchange intervention (Geiger et al., 2021) along the learned tick probe direction. Instead of swapping entire activation vectors, we decompose into tick-relevant and orthogonal components and swap only the tick component.

Given the normalized probe direction **d**, for a target activation **a_target** and source **a_source**:
```
a_patched = a_target + ((a_source . d) - (a_target . d)) * d
```

We patch mid-tick (400-600) and late-tick (>800) components into early-tick (0-10) activations (n=35,197 early samples for robust baselines). All 4 embedding-free action heads are evaluated: move (5-way), attack_style (3-way), gold_quantity (99-way), inventory_price (99-way).

##### Controls
- **Random direction baseline**: Same swap along 50 random unit vectors -> null distribution
- **Full activation swap**: Replace entire vector (upper bound on effect)
- **Orthogonal complement swap**: Replace everything EXCEPT the tick direction
- **Dose-response curve**: Interpolate intervention strength t from 0 to 1.5

##### Results

| Condition | move KL | attack_style KL | gold_qty KL | inv_price KL |
|---|---|---|---|---|
| Tick direction only | 0.0038 | 0.0071 | 0.0206 | 0.0255 |
| Full activation swap | 0.6982 | 0.0378 | 0.1345 | 0.1454 |
| Orthogonal complement | 0.6867 | 0.0461 | 0.1089 | 0.1072 |
| Random direction (mean) | 0.0003 | 0.0002 | 0.0003 | 0.0003 |

##### Key findings

1. **The tick direction is 14-94x stronger than random directions** — a real causal signal, not noise.
2. **But accounts for <1% of full-swap behavioral change in move** (0.004 vs 0.70 KL). Almost all early/late behavioral differences live in the orthogonal complement.
3. **Largest tick-direction effect is on economic heads**: gold_quantity (78x random) and inventory_price (94x random), not movement — tick primarily modulates economic behavior.
4. **Move probability shifts are consistent but small**: South +0.024, West -0.023 when patching late->early. Dose-response is clean and monotonic.
5. **Attack style shows larger tick-mediated shift**: Melee +0.058, Range -0.036 (late->early) — the model shifts toward melee as game time increases, causally through the tick direction.

**Interpretation**: The tick representation is real and causal, but it's a subtle modulator rather than the primary driver of behavioral differences. Most early-vs-late behavioral change comes from other co-varying dimensions (position, resources, combat state) that are not captured by the 1-d tick probe direction.

**Results**: `experiments/activation_patching/`

---

## Code Changes

### `analyze_activations.py`
- Added `run_tsne()` function wrapping sklearn TSNE
- Added CLI flags: `--tsne`, `--tsne-perplexity`, `--tsne-learning-rate`, `--tsne-early-exaggeration`, `--tsne-max-iter`
- Modified pipeline to use T-SNE instead of UMAP when `--tsne` is passed
- Output directory naming includes `-tsne` suffix
- Visualization titles/filenames adapt to reduction method
- JSONL support in `load_data()`, parallel cache builder for fast loading

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
- Added `StreamingRecordWriter`, `--streaming`/`--flush-interval` flags

### `train_probes.py`
- Added class reweighting (`--class-weight balanced`)

### `train_nonlinear_probes.py` (new)
- 2-layer MLP probes

### `mediation_analysis.py` (new)
- Causal mediation via Frisch-Waugh-Lovell residualization

### `activation_patching.py` (new)
- Subspace activation patching via interchange interventions

### `docs/mediation_analysis_explainer.md` (new)
- Methodology explainer for causal mediation analysis

---

## Result Paths

| Experiment Category | Results Directory |
|---|---|
| T-SNE clustering parameter sweeps | `experiments/clustering_param_sweep/` |
| Linear probes (1x baseline) | `experiments/linear_baseline/` |
| Linear probes (10x data) | `experiments/linear_10x/` |
| Linear probes (balanced weights) | `experiments/linear_balanced/` |
| Nonlinear probes (MLP baseline) | `experiments/nonlinear_baseline/` |
| Nonlinear probes (MLP 10x) | `experiments/nonlinear_10x/` |
| Nonlinear probes (MLP-8, 10x) | `experiments/nonlinear_10x_small/` |
| Causal mediation analysis | `experiments/mediation_analysis/` |
| Activation patching | `experiments/activation_patching/` |
| Activation controls analysis | `experiments/activation_controls/` |
