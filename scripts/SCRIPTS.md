# Interpretability Scripts

All scripts are run from the repo root via `uv run python scripts/<script>.py`.

## Core Pipeline (run in order)

### 1. `extract_activations.py` — Extract activations from trained policies

Records per-timestep 256-dim hidden-state activations from the action decoder during policy evaluation. Supports multiple extraction layers (action_decoder, lstm_cell, lstm_hidden, encoder_output).

```bash
uv run python scripts/extract_activations.py pve -p takeru -o activation_data
uv run python scripts/extract_activations.py pve -p yaofeng -o activation_data --layer lstm_cell
uv run python scripts/extract_activations.py --smoke-test  # quick sanity check
```

**When to use:** First step in any interpretability analysis. Run once per policy/layer combination. Use `--streaming` for large-scale extraction (avoids OOM). Output goes to `activation_data/<policy_name>/`.

### 2. `analyze_activations.py` — Cluster and visualize activation space

Reduces dimensionality (PCA, UMAP, or T-SNE), clusters with HDBSCAN or K-means, and generates visualization plots. Computes 35 behavioral features and tests cluster-feature associations via Cohen's d.

```bash
uv run python scripts/analyze_activations.py activation_data/yaofeng_200M --subsample 10
uv run python scripts/analyze_activations.py activation_data/yaofeng_200M --tsne --tsne-perplexity 50
uv run python scripts/analyze_activations.py activation_data/yaofeng_200M --cluster-before-umap --pre-cluster-dims 30
```

**When to use:** Exploratory analysis of activation structure. Start here after extracting activations. Used in all T-SNE clustering experiments (experiments 1-17 in EXPERIMENT_LOG.md).

### 3. `train_probes.py` — Linear probes (ridge + logistic regression)

Fits ridge regression (continuous features) and logistic regression (binary features) on PCA-reduced activations. Trajectory-based train/test split to avoid temporal leakage. Reports R^2 and AUC with shuffle baselines.

```bash
uv run python scripts/train_probes.py activation_data/yaofeng_200M
uv run python scripts/train_probes.py activation_data/yaofeng_200M --class-weight balanced
```

**When to use:** To quantify how much behavioral information is linearly decodable from activations. Used in probe experiments (EXPERIMENT_LOG.md experiments 18-19). Provides the probe directions used by `activation_patching.py`.

### 4. `train_nonlinear_probes.py` — Nonlinear probes (2-layer MLP)

Fits 2-layer ReLU MLP probes to compare against linear probes. Tests whether representations encode features nonlinearly.

```bash
uv run python scripts/train_nonlinear_probes.py activation_data/yaofeng_200M
```

**When to use:** After linear probes, to test whether nonlinear methods reveal additional structure. Key finding: representations are mostly linear — MLPs provide marginal gains (EXPERIMENT_LOG.md experiment 20).

### 5. `mediation_analysis.py` — Causal mediation (FWL residualization)

Uses Frisch-Waugh-Lovell residualization to test whether probe-detected features are confounded. Regresses out mediator features from both activations and target, then re-probes residuals.

```bash
uv run python scripts/mediation_analysis.py activation_data/yaofeng_200M
```

**When to use:** When probe results are suspicious — e.g., n_visible_players has high R^2 but the model can't directly observe other players. Tests whether the signal is mediated through features the model can observe (EXPERIMENT_LOG.md experiment 21).

### 6. `activation_patching.py` — Subspace activation patching

Interchange interventions along probe-identified directions. Patches individual feature directions between activation vectors and measures effect on action logits via KL divergence.

```bash
uv run python scripts/activation_patching.py activation_data/yaofeng_200M
```

**When to use:** To establish causal links between probe directions and behavior. Tests whether the direction identified by probes actually matters for downstream action selection (EXPERIMENT_LOG.md experiment 22).

## Sweep Scripts

### `sweep_cluster_params.py` — Clustering hyperparameter sweep

Sweeps PCA dimensions, HDBSCAN/K-means parameters, and UMAP settings. Scores configurations by a composite metric (weighted Cohen's d, noise ratio, entropy). Used for initial clustering experiments.

```bash
uv run python scripts/sweep_cluster_params.py activation_data/yaofeng_200M --subsample 5
uv run python scripts/sweep_cluster_params.py activation_data/yaofeng_200M --tsne-only --strategies 4
```

**When to use:** To find optimal clustering hyperparameters for a new policy or dataset. Results go to `results/clustering_param_sweep/`.

### `sweep_tsne_3d.py` — 3D T-SNE sweep

Dedicated 3D T-SNE sweep with parallelized fitting, multi-angle 3D scatter plots, and Cohen's d heatmaps. Supersedes the T-SNE mode in `sweep_cluster_params.py` for 3D work.

```bash
uv run python scripts/sweep_tsne_3d.py activation_data/yaofeng_200M --subsample 5
```

**When to use:** When 2D T-SNE doesn't capture enough structure. Used extensively in experiments 3-4, 10-17 in EXPERIMENT_LOG.md. 3D T-SNE produces cleaner clusters for yaofeng but still struggles with takeru.

## Visualization Scripts

### `agent_life_visualization.py` — Per-agent timeline plots

Plots per-agent timelines showing health, food, water, combat level, inventory, and actions over the agent's lifespan. Can overlay cluster assignments.

```bash
uv run python scripts/agent_life_visualization.py activation_data/takeru_100M
uv run python scripts/agent_life_visualization.py activation_data/takeru_200M --num_agents 5
```

**When to use:** To understand individual agent behavior over time. Useful for validating cluster assignments and spotting behavioral patterns.

### `cluster_life_phase.py` — Cluster-by-life-phase analysis

Bins each agent's lifespan into phases and plots cluster distribution as a stacked area chart. Tests whether clusters correspond to life phases (early/mid/late game).

```bash
uv run python scripts/cluster_life_phase.py activation_data/yaofeng_200M --cluster-dir <dir>
```

**When to use:** After clustering, to test temporal coherence of cluster assignments.

## Sanity Check Scripts

### `activation_controls.py` — Activation sensitivity controls

Paired-observation experiments: modifies specific observation features (terrain, health, inventory, etc.) and measures L2/cosine changes in the hidden state. Verifies that the model's hidden state actually responds to observation content.

```bash
uv run python scripts/activation_controls.py
```

**When to use:** Before trusting probe results, to verify the model encodes the features you're probing for. Key finding: health is the dominant self-entity signal; combat awareness features (freeze, combat_tick, attacker_id) are not encoded.

### `activation_controls_v2.py` — Combat stat controls

Extended version testing additional combat-related observation features.

```bash
uv run python scripts/activation_controls_v2.py
```

**When to use:** Follow-up to `activation_controls.py` for deeper combat feature investigation.

## Shell Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_policies.sh` | Batch policy evaluation |
| `train_baseline.sh` | Train baseline policy |
| `slurm_jobs.sh` | Submit SLURM jobs |
| `slurm_run.sh` | SLURM GPU runner |
| `slurm_run_cpu.sh` | SLURM CPU runner |
| `upload_checkpoints.sh` | Upload checkpoints to S3 |
| `upload_latest_checkpoint.sh` | Upload latest checkpoint |
| `pre-git-check.sh` | Pre-commit lint/format check |
