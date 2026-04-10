# Interpretability Experiments Log

## Overview

This document describes the experiments run to probe the internal representations of Neural MMO RL agents. We extract 256-dimensional activation vectors from the action decoder of trained policies and train probes to predict behavioral features from these activations.

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

## Experiment 1: Linear Probes (baseline)

**Script**: `train_probes.py`
**Method**: Ridge regression (continuous) / Logistic regression (binary) on 256-dim activations with StandardScaler. Trajectory-based train/test split to avoid temporal leakage. Three conditions: full (256d), PCA (50d), shuffle baseline.

### 1x data results (1,031 samples)
Many features had negative R² (worse than predicting the mean), indicating insufficient data. Best performers: tick/time_alive (R²=0.87), self_col (0.71), max_combat_level (0.69), self_health (0.68).

### 10x data results (28,089 samples)
Dramatic improvement across the board:
- tick/time_alive: 0.87 -> **0.89**
- self_col: 0.71 -> **0.85**
- self_row: 0.61 -> **0.82**
- max_combat_level: 0.69 -> **0.79**
- range_level: 0.28 -> **0.76** (biggest jump)
- n_visible_players: 0.03 -> **0.57**
- fishing_level: 0.03 -> **0.54**
- Features that were negative R² at 1x became positive at 10x

**Results**: `results/linear_baseline/`, `results/linear_10x/`

## Experiment 2: Class Reweighting

**Script**: `train_probes.py --class-weight balanced`
**Method**: For binary features, uses sklearn's `class_weight="balanced"`. For continuous features, computes sample weights via quantile binning (inverse frequency weighting).

Reweighting had modest effects. Binary feature recall improved slightly for rare classes (is_trading, is_using_item) but overall metrics were similar. The main benefit is fairer evaluation on imbalanced features.

**Results**: `results/linear_balanced/`

## Experiment 3: Nonlinear (MLP) Probes

**Script**: `train_nonlinear_probes.py`
**Architecture**: 2-layer ReLU MLP: `Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)`

### MLP-64 (hidden_factor=4, ~16.5k params)
- Catastrophic overfitting on many features, especially PCA condition
- self_health R² went negative (-2.46 on 1x data, 0.73 on 10x)
- Some features (item_level, n_listed_items) had R² < -0.5 even on 10x
- Best on spatial/temporal features: self_col (0.92), tick (0.94)

### MLP-8 (hidden_factor=32, ~2k params)
- Fixed catastrophic overfitting on full activations
- PCA condition still bad (hidden=50//32=1 unit bottleneck)
- Beat linear on combat/skill levels: max_combat 0.82 vs 0.79, fishing 0.63 vs 0.54
- Worse on vitals: self_health 0.49 vs 0.72 (underfitting)

### Conclusion
Representations are **mostly linear**. MLPs provide marginal gains on some features but never dramatically outperform linear probes. The activation space encodes behavioral features in largely linearly-accessible directions.

**Results**: `results/nonlinear_baseline/`, `results/nonlinear_10x/`, `results/nonlinear_10x_small/`

## Experiment 4: Causal Mediation Analysis

**Script**: `mediation_analysis.py`
**Method**: Frisch-Waugh-Lovell residualization. Regress out mediator features from both activations and target, then probe residuals. See `docs/mediation_analysis_explainer.md` for methodology.

### n_visible_players mediation

The model architecture does not receive other-player observations, yet the probe achieves R²=0.57. We tested whether this is mediated through features the model can observe:

| Mediator Set | Residualized R² | % Explained by Mediators |
|---|---|---|
| Combat (in_combat, damage, combat_level, is_attacking) | 0.413 | 27.3% |
| Position (self_row, self_col) | 0.570 | -0.3% |
| Time (tick, time_alive) | 0.422 | 25.7% |
| All combined | **0.399** | **29.7%** |

**Surprising finding**: R²=0.40 survives after controlling for all obvious confounders. The mediator sets overlap heavily (combat and time are largely redundant). The remaining signal may come from entity distances routed through attack masking, resource depletion patterns, or other indirect channels not included in the mediator set.

### Tick control for all features

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

**Results**: `results/mediation_analysis/`

## Experiment 5: Subspace Activation Patching

**Script**: `activation_patching.py`
**Method**: Interchange intervention (Geiger et al., 2021) along the learned tick probe direction. Instead of swapping entire activation vectors, we decompose into tick-relevant and orthogonal components and swap only the tick component.

Given the normalized probe direction **d**, for a target activation **a_target** and source **a_source**:
```
a_patched = a_target + ((a_source . d) - (a_target . d)) * d
```

We patch mid-tick (400-600) and late-tick (>800) components into early-tick (0-10) activations (n=35,197 early samples for robust baselines). All 4 embedding-free action heads are evaluated: move (5-way), attack_style (3-way), gold_quantity (99-way), inventory_price (99-way).

### Controls
- **Random direction baseline**: Same swap along 50 random unit vectors → null distribution
- **Full activation swap**: Replace entire vector (upper bound on effect)
- **Orthogonal complement swap**: Replace everything EXCEPT the tick direction
- **Dose-response curve**: Interpolate intervention strength t from 0 to 1.5

### Results

| Condition | move KL | attack_style KL | gold_qty KL | inv_price KL |
|---|---|---|---|---|
| Tick direction only | 0.0038 | 0.0071 | 0.0206 | 0.0255 |
| Full activation swap | 0.6982 | 0.0378 | 0.1345 | 0.1454 |
| Orthogonal complement | 0.6867 | 0.0461 | 0.1089 | 0.1072 |
| Random direction (mean) | 0.0003 | 0.0002 | 0.0003 | 0.0003 |

### Key findings

1. **The tick direction is 14-94x stronger than random directions** — a real causal signal, not noise.
2. **But accounts for <1% of full-swap behavioral change in move** (0.004 vs 0.70 KL). Almost all early/late behavioral differences live in the orthogonal complement.
3. **Largest tick-direction effect is on economic heads**: gold_quantity (78x random) and inventory_price (94x random), not movement — tick primarily modulates economic behavior.
4. **Move probability shifts are consistent but small**: South +0.024, West -0.023 when patching late→early. Dose-response is clean and monotonic.
5. **Attack style shows larger tick-mediated shift**: Melee +0.058, Range -0.036 (late→early) — the model shifts toward melee as game time increases, causally through the tick direction.

**Interpretation**: The tick representation is real and causal, but it's a subtle modulator rather than the primary driver of behavioral differences. Most early-vs-late behavioral change comes from other co-varying dimensions (position, resources, combat state) that are not captured by the 1-d tick probe direction.

**Results**: `results/activation_patching/`

## File Changes Summary

| File | Change |
|---|---|
| `extract_activations.py` | Added `StreamingRecordWriter`, `--streaming`/`--flush-interval` flags |
| `analyze_activations.py` | JSONL support in `load_data()`, parallel cache builder for fast loading |
| `train_probes.py` | Added class reweighting (`--class-weight balanced`) |
| `train_nonlinear_probes.py` | New file: 2-layer MLP probes |
| `mediation_analysis.py` | New file: causal mediation via residualization |
| `docs/mediation_analysis_explainer.md` | New file: methodology explainer |
| `activation_patching.py` | New file: subspace activation patching via interchange interventions |
