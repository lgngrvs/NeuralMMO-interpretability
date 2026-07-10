# T-SNE 3D + HDBSCAN on ALL Takeru 200M data — Log

## Key Findings

- **N = 10,488 points** across **1,066 unique trajectories**, combining:
  `takeru_200M` + `takeru_200M_extra` + `takeru_200M_batch_3..9`. Only a modest
  bump over prior 7-batch 10x run (~7,900 points) because batches 8 & 9 had the
  largest .jsonl files but not proportionally more alive-on-every-step records.
- **Best composite score: 1.422** (perp=50, mcs=50, ms=3, 5 clusters, 34.9% noise,
  weighted |d|=2.64). That's **-28% vs Exp 17's best** of 1.981 on 7,900 points
  (perp=100, mcs=15, ms=3). More data did **not** move the needle.
- **0 of 100 configs pass verification** (vs 1/96 in Exp 17). The reason is the
  same as in every prior takeru run: at least one cluster has very low **step
  entropy** (≈0.21) — i.e., a cluster dominated by early-game spawn frames.
- **Transition rate 0.9666** for the #1 config — **no improvement** over the
  ≈0.96 observed at the same subsample rate in Exps 15/17. Takeru's feed-forward
  architecture still produces confetti labels tick-to-tick, independent of data.
- **The d = -21.8 "spawn-area" cluster returns.** In the 1x run (Exp 16) this
  cluster had `self_row` d ≈ -12.6; with 7 batches (Exp 17) it shrank to d = -5
  to -7; with all 9 batches it blows back up to d = -21.8. That oscillation
  means its exact magnitude is sample-size-sensitive, but the signal (early-game
  agents near the top/left of the map with no equipment and full food) is real.
- **Dominant discriminating features are stable across ranks 1-3:** `self_row`,
  `n_equipped_items`, `item_level`, `self_food`, `max_combat_level`,
  `n_visible_players`. Same axes as every prior takeru clustering experiment.
- **Conclusion:** the "takeru clusters are unstable" phenomenon is **not a
  sample-size issue.** It is an architectural one. Doubling data from 7,900 to
  10,488 points sharpens the spawn cluster but does not produce any new stable
  behavioral modes and does not improve temporal coherence.

## Detailed Log

### Step 1: Cache rebuild for batch_8 and batch_9
**Action:** Ran `_build_caches.py` to parse the 20 GB and 8.6 GB raw `activations.jsonl`
files into binary `.cache.npz`. Previous caches existed for batch_8 but were
stale (older than the jsonl), and batch_9 had no cache at all.
**Result:** batch_8 → 175,134 alive records (198 MB cache, 2.2 min with 64 workers).
batch_9 → 72,564 alive records (82 MB cache, 1.4 min with 64 workers). Both ran
in parallel via ProcessPoolExecutor on 128 CPUs. Total build: ~2.2 min.

### Step 2: Load all 9 datasets with subsample=100
**Action:** Iterated through all takeru_200M subdirs, calling `load_data` on
directories with source jsonl+cache and using direct cache load for cache-only
directories (batches 3-7 and extra). Concatenated the per-dataset arrays.
**Result:**
| dataset | subsampled points |
|---|---|
| takeru_200M (base) | 282 |
| takeru_200M_extra | 986 |
| takeru_200M_batch_3 | 1093 |
| takeru_200M_batch_4 | 1165 |
| takeru_200M_batch_5 | 955 |
| takeru_200M_batch_6 | 2037 |
| takeru_200M_batch_7 | 1367 |
| takeru_200M_batch_8 | 1816 |
| takeru_200M_batch_9 | 787 |
| **Total** | **10,488** |
Unique (env_id, agent_id) trajectories: 1,066. Env ids were offset by dataset
index × 100,000 to prevent cross-dataset trajectory collisions during the
transition-rate computation.

### Step 3: PCA to 20D
**Action:** sklearn PCA with `n_components=20`, `random_state=42`.
**Result:** 90.1 % cumulative variance explained. Matches prior runs.

### Step 4: Parallel T-SNE 3D at 5 perplexities
**Action:** 5 × sklearn TSNE at perp ∈ {50, 75, 100, 125, 150}, `max_iter=1000`,
`random_state=42`. Launched via `ThreadPoolExecutor(max_workers=5)` with
OMP/OpenBLAS/MKL threads capped at 4 each (≈20 threads total).
**Result:** All 5 embeddings produced in 1.4-1.8 min each; total phase ~1.9 min
(bottlenecked by the slowest). Good parallel scaling.

### Step 5: HDBSCAN sweep (100 combos)
**Action:** grid = {50,75,100,125,150} × {50,100,200,400,800} × {3,5,10,25}
filtered by `ms ≤ mcs` → 100 combos. Each scored via `score_combo` (size-weighted
top-3 |Cohen's d| × cluster-count factor × noise factor × entropy factor), with
a min-trustworthy-cluster size of 209 (2 % of N).
**Result:** 0 / 100 pass verification. Top 5 all at mcs=50:
```
perp mcs ms  cl  noise%  |d|    ent   score
 50   50  3   5  34.9%  2.639  0.828  1.422
 50   50  5   6  37.0%  2.528  0.834  1.328
150   50  3   6  45.1%  2.862  0.792  1.244
125   50  3   5  43.2%  2.664  0.807  1.221
150   50  5   5  51.9%  2.858  0.779  1.072
```
Higher mcs (200/400/800) consistently collapsed to 2 clusters and low weighted d.
All 5 perplexities show the same qualitative pattern — a single very-strong
cluster (the spawn cluster with |d| > 20 on self_row) plus several weaker ones.

### Step 6: Detailed outputs for top-3 configs
**Action:** For each of ranks 1-3, wrote 2×2 3D scatter, Cohen's-d heatmap,
text summary, `cluster_labels.npy` (rows `[env_id, agent_id, step, label]`),
and transition matrix.
**Result:** Transition rates:
- Rank 1 (perp=50, mcs=50, ms=3, 5 clusters): **0.9666**
- Rank 2 (perp=50, mcs=50, ms=5, 6 clusters): **0.9622**
- Rank 3 (perp=150, mcs=50, ms=3, 6 clusters): **0.9533**
All three are essentially the 0.96 ceiling we've seen in every prior takeru
run at subsample=100. See `config_*/transitions.txt` for the full matrices.

### Step 7: Comparison + summary plots
**Action:** Plotted best-per-perplexity 1×5 comparison (`comparison.png`) and
2×2 `summary.png` (best scatter, best Cohen's d heatmap, score-vs-perplexity,
text block).
**Result:** Score vs perplexity is non-monotonic and U-shaped:
perp=50 → 1.42, perp=75 → 0.96, perp=100 → 1.00, perp=125 → 1.22, perp=150 → 1.24.
Both extremes of the perplexity sweep win; the middle sags.

### Top cluster features in best config (perp=50, mcs=50, ms=3)

| C | n | top features (d) |
|---|---|---|
| 0 | 237 | self_row **-21.83**, n_equipped_items -5.44, self_food +5.05, item_level -4.93, n_inventory_items -4.44 |
| 1 | 444 | n_equipped_items +1.17, self_food -0.75, max_combat_level +0.74, range_level +0.72, mage_level +0.69 |
| 2 | 335 | item_level -3.32, n_equipped_items -2.85, self_col -2.61, n_inventory_items -2.22, time_alive -2.05 |
| 3 | 318 | self_col +0.44, self_food +0.38, self_health +0.37, n_visible_entities -0.34, n_visible_players -0.33 |
| 4 | 266 | n_equipped_items +0.66, n_visible_players -0.64, max_combat_level +0.60, melee_level +0.60, range_level +0.59 |

**Interpretation:**
- **C0** — spawn / early-game at low row, no equipment, full food. The
  oversized d=-21.8 reflects a small tight mode; this is the same cluster
  identified in Exps 10/16/17.
- **C1 / C4** — equipped fighters (positive d on combat/skill features).
- **C2** — "unequipped + no-gear explorer, negative column" (left side of map).
- **C3** — mid-game balanced (weak discriminators, mostly positional).

These are the same 4-5 behavioral axes we've found in every takeru run
(spawn, equipped-fighter, unequipped-explorer, resource-stress). Adding data
sharpens them slightly but doesn't reveal any qualitatively new modes.
