# Activation Control Experiments

Sanity-check experiments verifying that Takeru (100M) hidden state activations
encode basic observation content. Motivated by difficulty finding linear
representations via standard probes.

## Setup

- **Model**: Takeru 100M (`policies/takeru_100M.pt`) — feedforward encoder + 1-layer LSTM
- **Data**: 41,406 activation records from `takeru_100M/activations.json`
  - 52% are **dead agents** with zeroed observations (filtered out, leaving ~19,900 live records)
  - 500 live-agent samples used per control
- **Method**: For each control, create paired observations (original vs modified), encode
  both through the model, and measure L2 distance and cosine similarity of the 256-d hidden state.
- **Scripts**: `activation_controls.py` (main controls), `activation_controls_v2.py` (combat stat controls)

## Architecture Finding

Both Takeru and Yaofeng build the hidden state from **self-state only**:

```
hidden = proj_fc(cat([tile_enc, self_entity_enc, inventory_enc, market_enc, task_enc]))
```

**Other entities do not enter the hidden state.** They are only used in the action
decoder as embedding lookup tables for target selection (attack_target, give_player, etc.).
This means:

- `attack_style` and `move` logits are computed as `Linear(hidden)` with **no access to
  entity information** — the model decides whether/how to attack blind to whether targets exist
- `attack_target` logits use `hidden @ entity_embeddings.T` — target selection does see entities
- This is identical in both Takeru and Yaofeng (same architectural pattern)

### Additional model notes

- Takeru checkpoint has `num_layers=1` LSTM despite code default of `num_layers=0`
- Yaofeng has `num_layers=2` LSTM
- Pre-LSTM hidden norms ~18, post-LSTM ~1.6 (significant compression)

## Results

### Hidden State Controls

| Control | L2 mean | Cosine sim | % of baseline |
|---|---|---|---|
| Tile terrain (original vs all-grass) | 1.43 | 0.42 | 69% |
| Self health = 65 (5th percentile) | 0.71 | 0.95 | 34% |
| Self health = 0 (dead) | 2.43 | 0.68 | 118% |
| Inventory (original vs empty) | 0.39 | 0.92 | 19% |
| Damage = 10 (95th percentile) | 0.29 | 0.99 | 14% |
| Damage = 0 (reset) | 0.04 | 1.00 | 2% |
| Freeze = 3 (max) | 0.02 | 1.00 | 1% |
| Latest combat tick = current | 0.002 | 1.00 | 0.1% |
| Attacker ID = 0 | 0.000 | 1.00 | 0% |
| **Random pair baseline** | **2.06** | **0.22** | **100%** |

### Action Logit Controls (Entity Presence)

| Metric | Value |
|---|---|
| Attack style L2 (self-only vs self + adjacent) | **0.0000** |
| Move logit L2 (self-only vs self + adjacent) | **0.0000** |
| Attack target max logit diff | **4.58** |

## Key Findings

1. **Activations do encode observation content.** Tile terrain, health, and inventory all
   produce measurable hidden state changes. The model is not collapsed or unresponsive.

2. **Health is the dominant self-entity signal.** Even the in-distribution perturbation
   (health 100 → 65) produces 34% of baseline variance. Most other combat features
   (freeze, combat tick, attacker ID) are essentially invisible to the model.

3. **Combat awareness features are not encoded.** `latest_combat_tick`, `attacker_id`, and
   `freeze` produce near-zero hidden state changes. The model cannot determine from its
   hidden state whether it was recently in combat or who attacked it.

4. **Entity presence is architecturally invisible to the hidden state.** Other agents only
   affect target-selection logits via embedding dot-products. This is confirmed by the
   attack_style/move logit controls showing exactly 0.0 difference.

5. **Data quality issue: 52% dead-agent contamination.** Over half the activation records are
   from dead agents with zeroed observations. Any probe analysis must filter these out.

## Implications for Linear Probes

- Probes finding R^2 = 0.6 for `n_visible_players` on hidden states are likely driven by
  **confounds**: position (row/col), time_alive, and health correlate with local player
  density without the model directly representing entity presence.
- Combat stats (freeze, combat_tick, attacker_id) are unlikely confound drivers since the
  model barely encodes them.
- The dead-agent contamination (52% of data) may inflate or distort probe results if not
  filtered.
