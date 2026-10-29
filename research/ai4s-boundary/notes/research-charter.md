# P006 LABS: conditional GO research charter (2026-10-29)

## Research decision
**GO for a falsifiable methodological investigation.** Primary project: P006 low-autocorrelation binary sequences (LABS). Secondary monitor: P007 Thomson 960. The primary deliverable is a reproducible *generalizable* search improvement; a previously unknown certificate is a high-risk stretch goal, not the sole success criterion.

## Why this is plausible
- There is exploitable sequential structure: LABS has long-range interactions and many local minima. The same family is defined across many sequence lengths, allowing train/test splits.
- Classical search is extremely competitive: 2025 GPU memetic-tabu improved 16 lengths N=92..120, but it *matched* the previously known value for N=100 rather than improving it. Thus beating E=578 is still open, not easy.
- Learning itself is not novel: 2022 DNN+LSTM-adjusted Tabu tenure already exists, 2026 MixBinNet generates low-correlation binary sequences, and 2026 Thompson-sampling-guided walks improved many larger instances. A generic DQN/PPO wrapper or energy predictor is insufficient novelty.

## Exact scientific hypothesis
**H1:** Given structurally informative features of an existing LABS local minimum (aperiodic autocorrelations C_k, per-bit exact delta energies, recent accepted trajectory), a learned policy can choose *coordinated multi-bit escape moves* whose **post-relaxation outcome** is better than strong non-learned perturbation and tenure policies at equal total compute.

Why different from prior work: optimize the *expected basin reached after a fixed local-search rollout* rather than one-step energy reduction or tuning only Tabu tenure. Structured composite moves are learned using delayed evaluation, not just randomly selecting flip counts. This is a working research hypothesis, not a claim of unexplored novelty; literature comparison and ablations may falsify it.

## Sequential work packages
1. **Validity and classical baselines.** Correct exact energy and O(N^2) all-bit delta evaluation; restore good single-flip Tabu; progress to memetic population/recombination, efficient evaluator and self-avoiding-walk comparisons. Verify source algorithm details rather than claiming a partial port is the published method.
2. **Measure failure modes.** Generate local minima with seeded solvers; measure escape-rate vs. basin score and autocorrelation signatures. Compare multi-bit random moves, tabu tenure, segment inversions, learned/no-learned policies. Control restarts and training budget.
3. **Counterfactual basin data.** From a local minimum, sample actions, run an *identical* local-relaxation budget, label outcome by basin energy and novelty. Keep train/evaluation seeds disjoint; compress duplicate symmetric solutions.
4. **Learned search.** Start with a compact CNN/attention-based state encoder and behavior-cloned action ranking; only pursue RL/actor-critic after comparing to contextual bandits and non-neural feature models. Measure model inference overhead.
5. **Generalization, then challenge.** Train on length subsets; hold out at least two lengths and random seeds; if it generalizes under strong fair comparisons, use best policy for N=100 and exact Few Shots verifier.

## Metrics and required controls
- Report minimum/median energy, *time-to-target* distributions, probability of reaching predeclared target, wall-clock cost, count of objective and per-neighbor evaluations, full cost of data generation and training.
- >=20 seeds when affordable, fixed hardware, common initial starts, multiple compute budgets, uncertainty intervals; avoid benchmarking one algorithm for 6000 random proposals vs another evaluating all 100 neighbors per step.
- Compare four ablations: full learned strategy, randomly shuffled learned logits, same action family with handcrafted ranking, and structurally blind features.
- For N=100, official baseline E=578; next admissible score is E=574. Never report a breakthrough without exact certificate verified by Few Shots and independent score reproduction.

## Gate (our prospective success criteria, not achieved outcomes)
- **Gate A:** classical baselines validated on official known best for several solved/tractable lengths; correctness tests and experimental logging pass.
- **Gate B:** in held-out lengths, learned method improves median time-to-target by >=15% *after accounting for inference*, or improves hit rate at equal wall-time and equal evaluation budget, consistently across two test lengths with 20+ seeds. The threshold is our predeclared practical criterion, not a statistically established minimum.
- **Gate C:** if Gate B fails with strong ablations and suitable training data, stop adding model complexity and reassess mechanism or switch to P007.
- **Stretch:** find E<=574 for N=100 and submit only a valid exact certificate; no guarantee of reaching it.

## Resources and references
Target consumer GPU: RTX 4070, small neural model; optimized classical search is CPU/C++ first and GPU parallel later if measured necessary. No expensive LLM or large transformer training required.

References:
- Z. Zhang et al. (2025), *New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search*: https://arxiv.org/abs/2504.00987 (N=100 ties old best, N=92..120 includes 16 improvements).
- D. M. et al. (2022), *A Deep Neural Network as a TABU Support in Solving LABS Problem*: https://doi.org/10.1007/978-3-031-08754-7_32 (existing adaptive tenure).
- B. Pšeničnik et al. (2026), *Prioritizing Search Space Regions in the Low Autocorrelation Binary Sequences Problem*: https://arxiv.org/abs/2607.09688 (Thompson sampling + GPU self-avoiding walks).
- B. Yoon et al. (2026), *Mixer-Based Neural Network for Binary Sequence Generation*: https://doi.org/10.1049/ell2.70699 (related multi-objective neural sequence generation, different exact objective).
- P007 backup: P. Amore & D. J. Wales (2026), *Efficient Searches for Low-Energy Structures in Clusters with Thousands of Particles*: https://doi.org/10.1063/5.0355867

## Engineering boundaries
All prototype code, raw logs, reports, train/test splits and checkpoints stay under research/ai4s-boundary. Do not change official verifiers, workflows, scores, problem metadata or participant shots. No valid record has yet been found. Work directly on this fork per repository owner's authorization; no PR.
