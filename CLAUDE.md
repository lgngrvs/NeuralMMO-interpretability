# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Workflow

ALWAYS use subagents (the Agent tool) for any non-trivial work. This is mandatory, not optional. Specifically:
- **All feature implementations** must be done by subagents. Break work into parallel subagents wherever possible.
- **Research and exploration** of the codebase should be delegated to Explore subagents.
- **Planning** complex changes should use Plan subagents before implementation.
- Use subagents even when the task only *might* benefit — err on the side of spawning one.
- The top-level Claude instance should act as an orchestrator: plan, delegate to subagents, review their results, and synthesize.

### Live Progress Tracking

**Always maintain a visible task list when working with subagents.** The user monitors progress via **Ctrl+T**.

- Before spawning each subagent, call `TaskCreate` with a clear description of what it will do.
- Set the task to `in_progress` when the subagent starts.
- Set the task to `completed` when the subagent returns, and briefly note the outcome.
- For multi-step experiments, create sub-tasks for each major phase (e.g., "Extract activations", "Fit probes", "Generate plots") and update them as they complete.
- Keep task descriptions short and informative — the user should be able to glance at Ctrl+T and immediately understand what's running, what's done, and what's next.

### Hardware-Aware Parallelization

Before kicking off experiment subagents, inspect the available hardware (CPU count, RAM, GPU count/memory via `nvidia-smi`) and include concrete resource guidance in the subagent prompt. The goal is to **fully utilize but not over-subscribe** the machine. For example:
- Tell the subagent how many parallel workers/jobs to use (e.g., `n_jobs=<cpu_count - 1>`).
- Tell the subagent how much GPU memory is free and how to batch accordingly.
- When spawning *n* subagents in parallel, tell each one to use at most **1/n** of the total resources (CPUs, GPU memory, RAM). Exception: if some subagents are lightweight (e.g., file I/O, plotting, code generation only), exclude them from the fraction and allocate their share to the compute-heavy subagents.

**>>> IF GPUs ARE AVAILABLE AND A SCRIPT CAN BE GPU-PARALLELIZED, YOU MUST ALWAYS USE GPU PARALLELIZATION. NO EXCEPTIONS. <<<** Check for GPUs at the start of every experiment run. If they exist, ensure all torch/sklearn-compatible workloads use them. Never fall back to CPU-only when GPUs are free.

## Self-Maintenance

If you notice important information that this CLAUDE.md file is missing or has wrong (e.g., new conventions, gotchas, corrected commands), ask the user for permission to update CLAUDE.md with that information.

## Default Experiment Settings

When running interpretability experiments (probes, clustering, mediation analysis, activation patching, etc.), use the **Yaofeng 200M** policy and its activation data (`activation_data/yaofeng_200M`) by default, unless the user specifies a different policy or dataset.

## Experiment Results

All experiment outputs must be saved under `results/<experiment_name>/`. Each experiment directory **must** contain a `results/<experiment_name>/summary.png` — a single image (possibly multi-panel) showing the key results of the experiment. This is what the user looks at first.

**Never overwrite prior results.** Every experimental run gets its own directory. If re-running an experiment (same or modified), create a new directory — e.g., `results/linear_probes_2_larger_pca_dim/`. The suffix should briefly describe what changed from the prior run.

**Displaying plots:** After an experiment finishes, print `imgcat results/<experiment_name>/summary.png` so the user can immediately see the results in-terminal. For additional detail plots:
```
imgcat results/<experiment_name>/summary.png
imgcat results/<experiment_name>/detail_plot_1.png
```

### Experiment Log

Maintain a running log at `results/<experiment_name>/LOG.md` for each experiment. Structure:

```markdown
# <Experiment Name> Log

## Key Findings
- (top-level insights, winning hyperparameters, surprising results — update as the experiment progresses)

## Detailed Log

### Step N: <short title>
**Action:** 1-2 lines on what was done / what subagent was spawned.
**Result:** 2-3 lines on what was found, key numbers, whether it succeeded/failed, and why.
```

Keep the detailed log entries concise — the goal is retrace-ability, not a transcript. Update "Key Findings" whenever a step produces a notable result.

Additionally, maintain a **global** experiment log at `results/EXPERIMENT_LOG.md` that indexes all experiments. Each entry should link to the experiment's directory and LOG.md, with a one-line summary of what was run and what was found. This is the single place to see all experiments at a glance.

## Project Overview

Interpretability research on Neural MMO RL agents. The repo contains both the training infrastructure (PufferLib-based PPO on Neural MMO) and a full interpretability pipeline: activation extraction, clustering (UMAP + HDBSCAN), linear/nonlinear probes, causal mediation analysis, and activation patching.

## Common Commands

This project uses `uv` for package management and running commands. Always prefer `uv` over raw `pip` or `python`.

### Install
```bash
uv sync --extra dev
```

### Add / Remove Dependencies
```bash
uv add <package>           # add to [project.dependencies]
uv add --group dev <pkg>   # add to [project.optional-dependencies] dev group
uv remove <package>
```

### Lint & Format (Ruff)
```bash
uv run ruff format .          # auto-format
uv run ruff check .           # lint
uv run ruff check . --fix     # lint with auto-fix
```

### Tests
```bash
uv run pytest tests/
uv run pytest tests/test_task_encoder.py            # single file
uv run pytest tests/test_task_encoder.py::TestName  # single test
```

### Training
```bash
uv run python train.py --debug --no-track    # quick smoke test
uv run python train.py -a takeru             # train a specific agent
uv run python train.py --syllabus            # with curriculum learning
```

### Interpretability Pipeline
```bash
# 1. Extract activations
uv run python scripts/extract_activations.py pve -p takeru -o activation_data

# 2. Cluster & visualize
uv run python scripts/analyze_activations.py activation_data/takeru_200M --subsample 10

# 3. Linear probes (ridge regression + logistic)
uv run python scripts/train_probes.py activation_data/takeru_200M

# 4. Nonlinear probes (2-layer MLP)
uv run python scripts/train_nonlinear_probes.py activation_data/takeru_200M

# 5. Causal mediation analysis
uv run python scripts/mediation_analysis.py activation_data/takeru_200M

# 6. Activation patching (interchange interventions)
uv run python scripts/activation_patching.py activation_data/takeru_200M
```

### Evaluation
```bash
uv run python evaluate.py policies pvp -r 10
uv run python analysis/proc_eval_result.py policies
```

## Architecture

### Training Stack
`train.py` loads an agent from `agent_zoo/` and config from `config.yaml`, then delegates to `train_helper.py` which uses `reinforcement_learning/clean_pufferl.py` (PufferLib-based PPO). Each agent in `agent_zoo/` (neurips23_start_kit, yaofeng, takeru, hybrid) exports `Policy`, `Recurrent`, and `RewardWrapper`. The policy's action decoder produces 256-dim hidden states that are the target of all interpretability analysis.

### Activation Extraction
`scripts/extract_activations.py` registers forward hooks on the action decoder layers to capture per-timestep 256-dim activations. Output is JSON/JSONL under `activation_data/<policy_name>/` with per-record: agent_id, env_id, activations, observations, actions. Uses `StreamingRecordWriter` for large-scale extraction.

### Behavioral Features
35 features derived from observations, grouped into: vitals (health, food, water, gold), position, entity counts, combat stats, inventory, skill levels, temporal (tick, time_alive), and action flags (is_moving, is_attacking, is_trading, is_using_item). These features are the dependent variables for probes and the labels for cluster analysis.

### Probes
`scripts/train_probes.py` fits ridge regression (continuous features) and logistic regression (binary features) on PCA-reduced (50d) activations. Uses trajectory-based train/test split to avoid temporal leakage. Reports R^2 and AUC per feature, with shuffle baselines. `scripts/train_nonlinear_probes.py` adds 2-layer MLP probes for comparison.

### Causal Analysis
`scripts/mediation_analysis.py` uses Frisch-Waugh-Lovell residualization to test for confounds between features. `scripts/activation_patching.py` performs interchange interventions along probe-identified directions, measuring effect on action logits via KL divergence.

## Key Configuration

- `config.yaml`: All training hyperparameters, environment settings, and per-agent overrides
- Python 3.10-3.11 required (`>=3.10,<3.12`)
- Ruff: line-length 100, Black-compatible formatting, rules E4/E7/E9/F
- Pre-commit hooks run ruff lint + format
- CI (GitHub Actions): runs `ruff format .` and `ruff check .` on push/PR

## Data Directories

- `activation_data/`: Extracted activations (JSON/JSONL + binary caches)
- `results/`: All experiment outputs (plots, models, metrics)
- `policies/`: Trained model checkpoints
- `maps/`: Environment maps for different agents (train/, train_takeru/, train_yaofeng/)

## Repo Structure

See `docs/REPO_STRUCTURE.md` for a full directory tree, import dependency graph, and layout of all scripts, results, and data directories. See `scripts/SCRIPTS.md` for documentation of each interpretability script.
