# Causal Mediation Analysis for Linear Probe Interpretability

## The Problem: Probes Can Lie

Linear probes are a standard tool for asking "does this network represent feature Z?"
Train a linear model from activations to Z; if R² is high, the network "knows" Z. But
this reasoning has a gap: **a probe can achieve high R² for a feature the model does not
represent, if that feature correlates with something the model does represent.**

In our case, a probe predicting `n_visible_players` from a 256-d hidden vector achieves
R² = 0.57 -- but the agent architecture never receives other-player observations. The
model *cannot* directly encode this feature. So what is the probe picking up on?

## Mediation Analysis: The Causal DAG View

Mediation analysis comes from causal inference. The idea is to distinguish **direct**
effects from **indirect** (mediated) effects by reasoning about causal structure.

Consider three variables: the activations (A), a potential mediator (M), and the target
feature (Y). Two competing causal stories:

```
  Story 1: Mediated               Story 2: Direct
  (probe is a shortcut)           (genuine representation)

  Real feature M                  Real feature M
       |                               |
       v                               v
  Activations A ----> Y          Activations A ----> Y
       ^                               ^
       |                               |
  (A encodes M,                  (A encodes Y
   Y correlates                   independently
   with M)                        of M)
```

In Story 1, the model encodes M (say, game tick), and Y (say, `n_visible_players`)
happens to correlate with M in the training data. The probe exploits the
A -> M -> Y pathway without A ever encoding Y directly.

In Story 2, A encodes information about Y that is not reducible to its encoding of M.

Mediation analysis asks: **if we surgically remove the information about M, does the
A -> Y relationship survive?**

## The Residualization Method

We use a classical technique backed by the **Frisch-Waugh-Lovell (FWL) theorem** from
econometrics. The procedure has three steps:

1. **Residualize activations.** Regress each activation dimension on the mediator
   features M. Keep the residuals: A_resid = A - A_hat(M). These residuals contain
   only the variation in A that is linearly independent of M.

2. **Residualize the target.** Regress Y on the same mediator features M. Keep the
   residuals: Y_resid = Y - Y_hat(M). This removes the component of Y that is
   linearly predictable from M.

3. **Probe residuals on residuals.** Fit a linear probe from A_resid to Y_resid. The
   R² of this probe measures the *direct* linear relationship between activations and
   the target, with the mediators partialled out.

The FWL theorem guarantees that this three-step procedure yields identical coefficients
to including M as control variables in a single joint regression. The advantage of the
residualization framing is interpretability: we get a clean R² for the "direct" pathway
that is directly comparable to the original probe's R².

## Interpretation Guide

| Outcome after controlling for M | Interpretation |
|---------------------------------|----------------|
| R² drops to ~0 | The original probe was **entirely mediated** through M. The model does not represent Y independently; the probe was exploiting correlations with M. |
| R² stays high (close to original) | **Genuine representation.** The model encodes information about Y that is linearly independent of M. |
| R² drops partially | **Mixed signal.** Part of the probe's accuracy came from mediation through M, but some independent signal remains. Further investigation needed. |

## Caveats

**Linear mediation only.** Residualization removes *linear* relationships between A and
M. If the model encodes M nonlinearly and Y correlates with that nonlinear encoding, the
mediated signal can survive residualization. This means a post-control R² above zero is a
necessary but not sufficient condition for genuine representation.

**Omitted mediator bias.** If the true mediator is a feature we did not include in M, we
will fail to explain away the probe's performance. The analysis is only as good as our
set of candidate mediators. Domain knowledge about what the model can and cannot observe
is essential for choosing the right controls.

**Not a causal experiment.** Unlike activation patching (which intervenes on activations
directly), residualization is an observational statistical technique. It identifies
linear statistical independence, not causal independence in the strictest sense.

## Application: The n_visible_players Example

Our RL agent plays Neural MMO, a multi-agent game. The agent's policy network produces
a 256-d hidden activation vector at each timestep. We train linear probes to predict 35
behavioral features from these activations.

The probe for `n_visible_players` achieves R² = 0.57. But the agent's observation space
does not include other players' information -- so this signal must be indirect.

Hypothesis: `n_visible_players` correlates with game tick (more players die as the game
progresses) and with map-position features (players cluster in certain areas). The model
*does* encode tick and position because those are in its observation space.

Test: we residualize activations and `n_visible_players` on a mediator set including
`tick`, spatial features, and other observation-derived quantities. If R² drops to near
zero, we confirm the probe was a correlational shortcut. If it remains elevated, we need
to revisit our assumptions about the model's observation space.

This pattern generalizes: for any probe with surprisingly high R², residualization
against plausible mediators is the first diagnostic to run.
