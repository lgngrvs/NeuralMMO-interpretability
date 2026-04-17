# Repository Structure

Last updated: 2026-04-17

## Top-Level Layout

```
NeuralMMO-interpretability/
├── train.py                  # Main training entry point
├── train_helper.py           # Training utilities (PufferLib PPO wrapper)
├── evaluate.py               # Policy evaluation (PvP/PvE)
├── syllabus_wrapper.py       # Curriculum learning wrapper (Syllabus)
├── config.yaml               # All training hyperparameters & env settings
│
├── scripts/                  # Interpretability & experiment scripts (see scripts/SCRIPTS.md)
├── agent_zoo/                # Policy implementations (neurips23_start_kit, yaofeng, takeru, hybrid)
├── reinforcement_learning/   # RL training infrastructure (clean_pufferl.py)
├── analysis/                 # Evaluation result processing scripts
│
├── activation_data/          # Extracted activations (JSON/JSONL + .cache.npz)
├── results/                  # All experiment outputs (plots, models, metrics)
├── experiments/              # Experiment-specific scripts + EXPERIMENT_LOG.md
├── policies/                 # Trained model checkpoints
├── maps/                     # Environment maps (train/, train_takeru/, train_yaofeng/)
│
├── curriculum_generation/    # Curriculum generation utilities
├── neurips23_evaluation/     # NeurIPS 2023 evaluation code
├── tests/                    # Test suite
├── docs/                     # Documentation and reference files
│
├── CLAUDE.md                 # AI agent instructions
├── README.md                 # Project README
└── pyproject.toml            # Package config (uv/pip)
```

## scripts/

Interpretability pipeline scripts. See `scripts/SCRIPTS.md` for full documentation.

```
scripts/
├── SCRIPTS.md                    # Documentation for all scripts
├── extract_activations.py        # Step 1: Extract activations from trained policies
├── analyze_activations.py        # Step 2: Cluster & visualize (UMAP/T-SNE + HDBSCAN)
├── train_probes.py               # Step 3: Linear probes (ridge + logistic regression)
├── train_nonlinear_probes.py     # Step 4: Nonlinear probes (2-layer MLP)
├── mediation_analysis.py         # Step 5: Causal mediation (FWL residualization)
├── activation_patching.py        # Step 6: Subspace activation patching
├── sweep_cluster_params.py       # Clustering hyperparameter sweep
├── sweep_tsne_3d.py              # 3D T-SNE sweep
├── agent_life_visualization.py   # Per-agent timeline visualization
├── cluster_life_phase.py         # Cluster-by-life-phase analysis
├── activation_controls.py        # Sanity-check: activation sensitivity controls
├── activation_controls_v2.py     # Sanity-check: combat stat controls
├── evaluate_policies.sh          # Shell: batch policy evaluation
├── pre-git-check.sh              # Shell: pre-commit checks
├── slurm_jobs.sh                 # Shell: SLURM job submission
├── slurm_run.sh                  # Shell: SLURM GPU runner
├── slurm_run_cpu.sh              # Shell: SLURM CPU runner
├── train_baseline.sh             # Shell: baseline training
├── upload_checkpoints.sh         # Shell: upload checkpoints to S3
└── upload_latest_checkpoint.sh   # Shell: upload latest checkpoint
```

## agent_zoo/

Each agent exports `Policy`, `Recurrent`, and `RewardWrapper`. The policy's action decoder produces 256-dim hidden states that are the target of all interpretability work.

```
agent_zoo/
├── neurips23_start_kit/   # NeurIPS 2023 starter
├── yaofeng/               # 2-layer LSTM, clean 3-cluster structure
├── takeru/                # Feedforward (no recurrence), harder to cluster
└── hybrid/                # Hybrid approach
```

## results/

Every experiment has its own subdirectory. Each should contain a `summary.png` with key results.

```
results/
├── linear_baseline/              # Linear probes on 1x data
├── linear_10x/                   # Linear probes on 10x data
├── linear_balanced/              # Linear probes with class reweighting
├── nonlinear_baseline/           # MLP probes on 1x data
├── nonlinear_10x/                # MLP probes on 10x data (large hidden)
├── nonlinear_10x_small/          # MLP probes on 10x data (small hidden)
├── mediation_analysis/           # Causal mediation (FWL residualization)
├── activation_patching/          # Subspace activation patching
├── clustering_param_sweep/       # Clustering hyperparameter sweep (K-means, HDBSCAN, UMAP)
├── agent_lifetime_visualization/ # Agent lifetime plots
└── activation_controls/          # Activation sensitivity sanity checks
```

## activation_data/

Extracted activations from trained policies. Not git-tracked. ~43GB total.

```
activation_data/
├── baseline_10M/            # Baseline policy, 10M steps
├── learner/                 # Learner policy
├── takeru_100M/             # Takeru at 100M steps
├── takeru_200M/             # Takeru at 200M steps (primary)
├── takeru_200M_batch_3-9/   # Additional Takeru batches (10x data collection)
├── takeru_200M_extra/       # Extra Takeru data
├── yaofeng_100M/            # Yaofeng at 100M steps
├── yaofeng_200M/            # Yaofeng at 200M steps (default for experiments)
└── yaofeng_200M_lstm_cell/  # Yaofeng LSTM cell state activations
```

## experiments/

Experiment-specific scripts and the global experiment log. Each subdirectory contains a script used for one specific experiment run.

```
experiments/
├── EXPERIMENT_LOG.md              # Global experiment log (all experiments indexed)
├── T-SNE_sweep/                   # 2D T-SNE sweep on baseline_10M
├── T-SNE-sweep-yaofeng-200M/      # 2D T-SNE sweep on yaofeng_200M
├── T-SNE-3d-takeru-*/             # Various 3D T-SNE experiments on Takeru
├── T-SNE-3d-yaofeng-lstm-cell-*/  # LSTM cell state clustering experiments
├── c2_subclustering*/             # Sub-clustering of yaofeng's dominant C2 cluster
├── cluster_separability/          # Statistical separability analysis
├── lifetime_*.py                  # Agent lifetime cluster analysis scripts
└── (each subdir has its own run script)
```

## Import Dependency Graph

```
Training pipeline:
  train_helper.py ← train.py ← evaluate.py ← extract_activations.py

Interpretability pipeline (all in scripts/):
  analyze_activations.py    ← train_probes.py ← train_nonlinear_probes.py
                            ← train_probes.py ← mediation_analysis.py
                            ← sweep_cluster_params.py ← sweep_tsne_3d.py

  (standalone: activation_controls, activation_patching, agent_life_visualization, cluster_life_phase)
```
