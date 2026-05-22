# Closing the Online MBRL Loop at Video-World-Model Scale

_The empty cell: a video-scale world model acting as both inference-time planner and training simulator for a VLA-scale policy, with both co-improving from real robot rollouts._

---

## Table of Contents

- [Thesis](#thesis)
- [What's already published, mapped to (a), (b), (c)](#whats-already-published-mapped-to-a-b-c)
- [Why the gap exists](#why-the-gap-exists)
- [Why now](#why-now)
- [A concrete paper-shaped build](#a-concrete-paper-shaped-build)
- [Expected contributions](#expected-contributions)
- [Failure modes to confront honestly](#failure-modes-to-confront-honestly)
- [Sources](#sources)

---

## Thesis

The full Dyna/Dreamer loop combines three things: (a) inference-time planning over a world model, (b) policy training from imagined rollouts, and (c) online co-improvement of both from real-robot experience. All three have been demonstrated together at compact-latent-WM scale by [FOWM](https://arxiv.org/abs/2310.16029) and [TD-MPC2](https://arxiv.org/abs/2310.16828). The video-WM and VLA-scale variants ([VLAW](https://arxiv.org/abs/2602.12063), [RehearseVLA](https://arxiv.org/html/2605.00080v1), [V-JEPA 2](https://arxiv.org/abs/2506.09985), [GEN-1](https://generalistai.com/blog/apr-02-2026-GEN-1), [π0](https://www.physicalintelligence.company/blog/pi0)) each cover two of the three ingredients, never all three. The empty cell, video-scale WM acting as both inference-time planner and training simulator for a VLA-scale policy, with both co-improving from real rollouts, is the research direction worth writing.

## What's already published, mapped to (a), (b), (c)

[DayDreamer](https://arxiv.org/abs/2206.14176) (Wu/Escontrela/Hafner, 2022) hits (b) and (c) but not (a). Actor-critic trained inside latent imagined rollouts, runs feedforward at deploy. WM is a training-time simulator only.

[FOWM](https://arxiv.org/abs/2310.16029) (Feng/Hansen, CoRL 2023) hits all three. TD-MPC base with MPPI planning at inference, learned policy prior trained from WM imagination, WM fine-tuned online from real rollouts. Compact RSSM-style latent WM, single-robot, short-horizon manipulation. Right loop, wrong scale.

[TD-MPC2](https://arxiv.org/abs/2310.16828) (Hansen et al., 2024) is the simulation-scale refinement, 317M-parameter single agent across 80 tasks, same three-ingredient recipe.

[VLAW](https://arxiv.org/abs/2602.12063) and [RehearseVLA](https://arxiv.org/html/2605.00080v1) (2026) hit (b) and (c) at video-WM/VLA scale via iterative co-improvement. Inference-time planning appears absent: the VLA runs feedforward at deploy.

[V-JEPA 2](https://arxiv.org/abs/2506.09985) (Meta, 2025) does (a) explicitly via latent-space MPC over a self-supervised video model, with V-JEPA 2-AC post-trained on 62 hours of Droid robot video. Online co-improvement on real robots at scale not yet demonstrated.

[GEN-1](https://generalistai.com/blog/apr-02-2026-GEN-1), [π0](https://www.physicalintelligence.company/blog/pi0), GR-2, and Cosmos Policy bet on implicit planning via flow-matching action heads on transformer backbones with video co-training. No explicit MPC at deploy.

### Coverage by work

Ingredients: **(a)** inference-time planning · **(b)** policy training from imagined rollouts · **(c)** online co-improvement from real rollouts.

| Work                       | (a)      | (b)     | (c) | Scale                               |
| -------------------------- | -------- | ------- | --- | ----------------------------------- |
| DayDreamer                 | ✗        | ✓       | ✓   | compact latent WM                   |
| FOWM                       | ✓        | ✓       | ✓   | compact latent WM, single robot     |
| TD-MPC2                    | ✓        | ✓       | ✓   | 317M params, sim, 80 tasks          |
| VLAW / RehearseVLA         | ✗        | ✓       | ✓   | video WM + VLA                      |
| V-JEPA 2                   | ✓        | partial | ✗   | video WM                            |
| GEN-1 / π0 / GR-2 / Cosmos | implicit | ✓       | ✓   | video co-train + VLA                |
| **This direction**         | ✓        | ✓       | ✓   | **video WM + VLA — the empty cell** |

The grid:

|                       | (a) + (b) + (c) loop closed      |
| --------------------- | -------------------------------- |
| Compact latent WM     | yes (FOWM, TD-MPC2), small scale |
| Video WM + VLA policy | no, empty cell                   |

## Why the gap exists

Inference compute is the dominant blocker. MPPI over a video diffusion WM needs hundreds of rollouts per control step, each costing seconds to minutes on a frontier model. The arithmetic does not close at real-robot control rates. RSSM-based FOWM rollouts are milliseconds; video-WM rollouts are not.

The architectural counter-bet is winning. The 2025 to 2026 recipe (flow-matching action head on a transformer trained with a video co-objective) folds planning implicitly into a single forward pass. π0, GEN-1, GR-2, and Cosmos Policy all sit here. Most research compute has gone into making this implicit-planning policy stronger, not into making explicit WM planning faster.

Reward sourcing degrades with horizon. FOWM's small learned reward networks in latent space work for short-horizon manipulation. At VLA scale with sparse long-horizon tasks, reward must come from a VLM judge, preference data, or a self-supervised proxy. All three are noisy, and the noise compounds across the imagined-rollout horizon.

Real-robot data collection rate-limits the loop. Resets, monitoring, and hardware wear cap how fast (c) can drive WM updates. [VLAW](https://arxiv.org/abs/2602.12063) explicitly frames this as the motivating constraint.

## Why now

Three primitives matured in the past year.

[V-JEPA 2](https://arxiv.org/abs/2506.09985)-style latent-space planning over video-trained dynamics models cuts MPC inference cost by orders of magnitude versus pixel-space rollouts.

VLA-grade policies with flow-matching heads ([π0](https://www.physicalintelligence.company/blog/pi0), [GEN-1](https://generalistai.com/blog/apr-02-2026-GEN-1)-class) give a strong proposal distribution to MPC. The planner only needs to refine, not search from scratch, so per-step sample budget shrinks.

VLM judges and self-supervised reward proxies are now usable as cheap reward signals over imagined rollouts, and conservative MBRL methods (ensemble-disagreement penalization, KL-constrained policy updates) have enough noise tolerance to absorb the judge noise.

## A concrete paper-shaped build

Pretrain a video-scale multimodal world model on heterogeneous data (wearable human video, robot teleop, simulation). Co-train a flow-matching VLA policy head against the same backbone.

Distill a fast latent dynamics model from the video WM, targeting under 10 ms per latent step to support MPC at control rate. This is the planning surrogate.

At deploy, run MPPI in the latent dynamics model using the VLA policy as proposal. Score rollouts with a learned value head plus a VLM judge.

Every real execution updates three things: the video WM (supervised loss against ground-truth observations), the VLA policy (imagined rollouts in the updated WM), and the value head (observed task outcomes).

Ensemble the latent dynamics model. Penalize planning trajectories the ensemble disagrees on (conservatism, avoids WM exploitation). KL-constrain the policy update toward the data-collecting policy.

Use ensemble disagreement as an intrinsic exploration bonus (Plan2Explore-style) to seek WM-uncertain regions for data collection.

## Expected contributions

First demonstration of the full Dyna loop at video-WM and VLA scale on real robots. Quantification of when explicit MPC beats implicit planning, ideally on long-horizon or novel tasks where the feedforward policy's implicit planning horizon is exceeded. Calibration of WM-based policy evaluation against real-robot success rates (reusing the inference-time WM as a faithful proxy environment). A recipe for handling WM exploitation at video scale, replicating the small-scale conservatism story from FOWM at much larger model and task scale.

## Failure modes to confront honestly

Latent-dynamics distillation may lose fidelity on the long tail, making ensemble disagreement spike everywhere and planning useless. VLM-judge noise may dominate the reward signal, causing the policy to satisfy the judge rather than the task. The implicit-planning baseline (a strong feedforward VLA) may win outright on the realistic-horizon tasks the paper can actually evaluate, undercutting the central claim. Real-robot data collection rate may be too low to drive meaningful WM updates, collapsing the loop into "fine-tune the policy on offline data with extra compute."

The bet is that long-horizon, novel, contact-rich tasks are where implicit planning runs out of headroom. That is where explicit WM planning plus online co-improvement should differentiate, or fail to.

## Sources

- [DayDreamer (Wu et al., 2022)](https://arxiv.org/abs/2206.14176)
- [FOWM (Feng/Hansen et al., 2023)](https://arxiv.org/abs/2310.16029)
- [TD-MPC2 (Hansen et al., 2024)](https://arxiv.org/abs/2310.16828)
- [V-JEPA 2 (Meta, 2025)](https://arxiv.org/abs/2506.09985)
- [VLAW (2026)](https://arxiv.org/abs/2602.12063)
- [GEN-1 (Generalist, 2026)](https://generalistai.com/blog/apr-02-2026-GEN-1)
- [π0 (Black et al., 2024)](https://www.physicalintelligence.company/blog/pi0)
- [MOTO (Rafailov et al., 2023)](https://arxiv.org/abs/2401.03306)
- [World Model for Robot Learning Survey (2026)](https://arxiv.org/html/2605.00080v1)
