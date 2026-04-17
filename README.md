![figure](https://neuralmmo.github.io/_static/banner.jpg)

# ![icon](https://neuralmmo.github.io/_build/html/_images/icon.png) Welcome to the Platform!

[![PyPI version](https://badge.fury.io/py/nmmo.svg)](https://badge.fury.io/py/nmmo)
[![](https://dcbadge.vercel.app/api/server/BkMmFUC?style=plastic)](https://discord.gg/BkMmFUC)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341)](https://twitter.com/jsuarez5341)

[Documentation](https://neuralmmo.github.io "Neural MMO Documentation") is hosted by github.io.

## Installation

After cloning this repo, run:

```
pip install -e .[dev]
```

## Training

To test if the installation was successful (with the `--debug` mode), run the following command:

```
python train.py --debug --no-track
```

To log the training process, edit the wandb section in `config.yaml` and remove `--no-track` from the command line. The `config.yaml` file contains various configuration settings for the project.

### Agent zoo and your custom policy

This baseline comes with four different models under the `agent_zoo` directory: `neurips23_start_kit`, `yaofeng`, `takeru`, and `hybrid`. You can use any of these models by specifying the `-a` argument.

```
python train.py -a hybrid
```

You can also create your own policy by creating a new module under the `agent_zoo` directory, which should contain `Policy`, `Recurrent`, and `RewardWrapper` classes.

### Curriculum Learning using Syllabus

The training script supports automatic curriculum learning using the [Syllabus](https://github.com/RyanNavillus/Syllabus) library. To use it, add `--syllabus` to the command line.

```
python train.py --syllabus
```

## Replay generation

The `policies` directory contains a set of trained policies. For your models, create a directory and copy the checkpoint files to it. To generate a replay, run the following command:

```
python train.py -m replay -p policies
```

The replay file ends with `.replay.lzma`. You can view the replay using the [web viewer](https://kywch.github.io/nmmo-client/).

## Evaluation

The evaluation script supports the pvp and pve modes. The pve mode spawns all agents using only one policy. The pvp mode spawns groups of agents, each controlled by a different policy.

To evaluate models in the `policies` directory, run the following command:

```
python evaluate.py policies pvp -r 10
```

This generates 10 results json files in the same directory (by using `-r 10`), each of which contains the results from 200 episodes. Then the task completion metrics can be viewed using:

```
python analysis/proc_eval_result.py policies
```

## Interpretability Tools

A pipeline for extracting, clustering, and analyzing hidden-state activations from trained policies. The typical workflow is:

1. Extract activations from a trained policy
2. Cluster and analyze the activation space
3. Visualize agent behavior and cluster structure

### 1. Extract Activations (`scripts/extract_activations.py`)

Records per-timestep hidden-state activations from the action decoder, along with observations and actions, during policy evaluation.

```bash
# Extract from a single policy
python scripts/extract_activations.py pve -p takeru -o activation_data

# List available policies
python scripts/extract_activations.py --list

# Quick smoke test
python scripts/extract_activations.py --smoke-test
```

Outputs a directory per policy under `activation_data/` containing `activations.json` with per-timestep records of activations, observations, and actions.

### 2. Analyze Activations (`scripts/analyze_activations.py`)

Clusters activation vectors using UMAP + HDBSCAN and tests whether clusters correspond to interpretable behavioral features. Computes Cohen's d effect sizes vs random baselines, checks for temporal/agent confounds, and generates multiple visualization modes.

```bash
# Standard clustering analysis
python scripts/analyze_activations.py activation_data/takeru_200M --subsample 10

# High-D clustering: PCA first, then HDBSCAN, then UMAP for viz
python scripts/analyze_activations.py activation_data/takeru_200M \
    --cluster-before-umap --pre-cluster-dims 30

# PCA-feature correlation analysis only (fast, no UMAP/HDBSCAN)
python scripts/analyze_activations.py activation_data/takeru_200M \
    --metric-only --cluster-before-umap --pre-cluster-dims 30

# All visualization modes
python scripts/analyze_activations.py activation_data/takeru_200M \
    --cluster-before-umap --pre-cluster-dims 30 \
    --feature-scatter --umap-pairs --dendrogram-explorer
```

**Behavioral features tracked** (14 total): `n_visible_entities`, `n_visible_npcs`, `n_visible_players`, `self_health`, `self_food`, `self_water`, `self_gold`, `max_combat_level`, `in_combat`, `n_inventory_items`, `tick`, `is_moving`, `is_attacking`, `is_trading`.

**Visualization modes:**

| Flag | Output | Description |
|------|--------|-------------|
| *(default)* | `umap_scatter.png` | UMAP embedding colored by HDBSCAN cluster |
| *(default)* | `cluster_features.png` | Heatmap of Cohen's d per feature per cluster |
| *(default)* | `umap_metric_scatter.png` | UMAP colored by rolling-average feature values (opacity-scaled) |
| `--cluster-before-umap` | `pca_pairs.png` | PCA direction pair plots (PC1&2, PC3&4, ...) |
| `--cluster-before-umap` | `pca_feature_correlations.png` | Heatmap of Pearson r between PCs and features, with multivariate R² |
| `--cluster-before-umap` | `pca_feature_profiles.png` | Per-feature scatter + binned mean profiles along top 3 correlated PCs |
| `--feature-scatter` | `feature_scatter.png` | All 91 feature-pair scatter matrix |
| `--umap-pairs` | `umap_pairs.png` | N-D UMAP direction pair plots on raw activations |
| `--dendrogram-explorer` | `dendrogram_explorer.html` | Interactive Plotly slider over HDBSCAN condensed tree hierarchy |

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--subsample N` | 10 | Keep every Nth record per trajectory |
| `--hdbscan-min-cluster` | 15 | HDBSCAN min_cluster_size |
| `--umap-neighbors` | 15 | UMAP n_neighbors |
| `--pre-cluster-dims` | 30 | PCA dimensions before clustering |
| `--umap-pairs-dims` | 10 | Number of UMAP dimensions for pairs plot |
| `--metric-only` | off | Skip HDBSCAN and 2D UMAP; still runs PCA if `--cluster-before-umap` |

### 3. Agent Life Visualization (`scripts/agent_life_visualization.py`)

Plots per-agent timelines showing health, food, water, combat level, inventory, and actions over the agent's lifespan. Optionally overlays cluster assignments.

```bash
# Visualize the p75-lifespan agent
python scripts/agent_life_visualization.py activation_data/takeru_200M

# Sample 5 random agents, with cluster overlay
python scripts/agent_life_visualization.py activation_data/takeru_200M \
    --num_agents 5 --cluster_dir analysis_results/run_dir \
    --output agent_plots/
```

### 4. Cluster Life Phase Analysis (`scripts/cluster_life_phase.py`)

Checks whether cluster assignments follow a life-phase pattern (e.g. early-game vs late-game behavior). Bins each agent's lifespan into phases and plots the cluster distribution across phases as a stacked area chart.

```bash
python scripts/cluster_life_phase.py activation_data/takeru_200M \
    --cluster-dir analysis_results/run_dir \
    --n-bins 20 --min-lifespan 50
```
