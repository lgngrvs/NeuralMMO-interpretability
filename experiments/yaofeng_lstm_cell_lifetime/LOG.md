# Yaofeng LSTM Cell-State Lifetime Log

Re-clusters the Yaofeng 200M LSTM cell-state activations at the user-specified
config and overlays cluster membership on per-agent lifetime timelines.

## Key Findings

- **3 clusters recovered** (user expected ~2) at perp=50, mcs=200, ms=3.
  Matches the 3-cluster structure from the prior `T-SNE-3d-yaofeng-lstm-cell-sparse`
  experiment (same data, same subsampling, same PCA; different HDBSCAN params).
- **Subsample noise: 25.3%** (880/3484 points unassigned in the T-SNE space).
  KNN (k=5, on PCA-20 space, trained only on non-noise subsampled points) then
  assigns a label to every one of the 170,775 full-resolution alive records.
- **Full-resolution tick-to-tick transition rate: 0.0194** (1.94%).
  Cluster membership is *extremely* temporally coherent — an agent typically
  stays in one cluster for ~50 consecutive ticks before switching. This is the
  expected LSTM-cell behavior (it acts as a smoothed behavioral-mode counter).
- **Cluster interpretation (from Cohen's d, backed up by per-agent plots):**
  - **C0 (21% of full-res)** — "Early game / fresh spawn". time_alive = -3.73,
    tick = -3.73, item_level = -3.54, melee_level = -3.31. Agents start every
    life here.
  - **C1 (28% of full-res)** — "Engaged / combat & trade". n_inventory_items = +0.60,
    n_equipped_items = +0.46, in_combat = +0.32, nearest_entity_dist = -0.3.
    The "busy" mode: fights, trades, proximity to others.
  - **C2 (51% of full-res)** — "Steady-state mature". self_health = +0.80,
    n_equipped_items = +0.59, n_visible_players = -0.61, time_alive = +0.6.
    Healthy, well-equipped, alone — agents spend most of their mature lifetime here.
- **Per-agent plots confirm the lifecycle interpretation.** The cluster strip
  at the bottom of each agent timeline shows:
    - Life begins in C0 (early-game) for the first ~100–200 ticks, then the
      LSTM "promotes" the agent to C2 once vitals and equipment are established.
    - C1 appears as short intermittent bursts during combat / trading events,
      interleaved with C2 steady-state stretches.
    - When an agent dies and respawns (new life), the cluster strip resets to C0
      at tick 0 of the new life — the LSTM cell evidently resets with the agent.

## Detailed Log

### Step 1: Setup and driver script
**Action:** Created `/workspace/NeuralMMO-interpretability/experiments/yaofeng_lstm_cell_lifetime/`,
wrote `run_analysis.py` that loads the npz cache directly (no `activations.json`
in the lstm_cell dir), subsamples at rate 50 per trajectory, runs PCA(20) +
T-SNE(3D, perp=50, iters=1000) + HDBSCAN(mcs=200, ms=3), KNN-propagates to
full-res, writes `cluster_labels.npy` in the format `agent_life_visualization.py`
expects, plots, then spawns `scripts/agent_life_visualization.py` with
`--num_agents 5 --cluster_dir ... --output per_agent/`.
**Result:** N/A (setup).

### Step 2: Pipeline run
**Action:** Ran `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
uv run python experiments/yaofeng_lstm_cell_lifetime/run_analysis.py` in the
background. The other takeru-bigsample subagent (192 disk workers) slowed the
initial npz load considerably — ~2 min just to import sklearn + load 170MB npz.
**Result:** Pipeline finished in 153 s after disk contention cleared. PCA 20D
explains 99.1% variance. T-SNE KL divergence ended at 1.15. HDBSCAN found
3 clusters (sizes 628 / 709 / 1267, noise 880) at the subsample level;
KNN-propagated to full-res cluster sizes 36146 / 48095 / 86534.

### Step 3: Per-agent plots
**Action:** `agent_life_visualization.py --num_agents 5 --seed 42` completed
but only produced 2 plots because 3 of the 5 sampled agents had agent_id=0 /
0 alive ticks (the JSON contains dead-from-spawn records for these). Re-ran
with `--num_agents 10 --seed 1` to gather 6 more distinct-agent plots (total
7 unique agent PNGs in `per_agent/`).
**Result:** 7 per-agent PNGs, all showing 3 lives each (multiple deaths +
respawns in the 200M env). Cluster-match rates (fraction of full-res ticks
whose label came from a non-noise-subsampled-neighbor-majority) range 44%–95%.
High-match agents show an extremely clean C0 → C2 transition in each life.

### Step 4: summary.png
**Action:** Rebuilt `summary.png` via `_rebuild_summary.py` picking 3 per-agent
plots spanning short/medium/long lifespans (agent-42, agent-52, agent-65 of 69).
Top row: T-SNE 3D (4 angles) + Cohen's d heatmap. Middle row: summary text
including per-cluster size + top features. Bottom row: 3 per-agent lifetime
timelines with cluster strip at the bottom of each.
**Result:** `summary.png` written at 150 dpi. Used 22-inch-wide figure for
readability. Also preserved all intermediate PNGs.

## Numbers at a glance

| Metric                        | Value     |
|-------------------------------|-----------|
| Full-res records              | 170,775   |
| Subsampled points             | 3,484     |
| PCA(20) explained var         | 99.1%     |
| Clusters                      | 3         |
| Subsample noise               | 25.3%     |
| Full-res tick-to-tick trans.  | 0.0194    |
| C0 size (full-res)            | 21.2%     |
| C1 size (full-res)            | 28.2%     |
| C2 size (full-res)            | 50.7%     |
| Wall time                     | 153 s     |
