# Research notebook — 2026-10-29

## First hypotheses and observed failures

### P006

**Hypothesis:** an online neural contextual policy for choosing perturbation sizes {1,2,4,8} learns better local minima escape tactics. **Observed:** no consistent gain across N=40/60/100; the single-flip baseline remained competitive or better, and the N=100 official score E=578 was not improved after warm-start trials.

Why might this fail? The inputs (normalized energy, stagnation, elapsed fraction, acceptance rate) contain almost no information about **which** autocorrelation lags or bits make the current basin difficult. Immediate one-step improvement reward tends to prefer small greedy changes, not delayed escape trajectories. This is a concrete mechanism hypothesis, not a proven explanation.

**Next falsifiable step:** encode the autocorrelation spectrum and per-bit energy deltas; compare neural multi-step perturbations with strong greedy descent, tabu, self-avoiding walk, and multi-spin flips using equal energy calls and wall time. Test transfer across unseen sequence lengths and >=20 random seeds. Report confidence intervals and ablations.

### P007

**Hypothesis:** perturbing non-six-coordinated vertices in the convex-hull triangulation gives easier exits from energy basins. **Observed:** at N=32 and 60, both the simple defect strategy and random-patch baseline returned to essentially identical low-energy basins.

Why might this fail? Degree alone may not identify unstable defect topology; local Gaussian motion plus relaxation can simply restore the same scar structure. To test this, identify defect-pair geometries and coordinated topological moves, label actual basin transitions, and reproduce N=960 published and D2 record certificates with the exact verifier before any full-scale optimization.

## Scientific boundary

Separate search-algorithm improvement from a new verified instance record and from a general constructive mathematical theorem. The current results are negative pilots. P006 score is exact integer energy; P007 small-N energies are floating-point research measurements, not valid Few Shots certificates.

Limitations: P006 only six seeds and 6000 proposals per run; P007 only three seeds, 12 patch perturbations, possibly incomplete L-BFGS convergence, unequal objective evaluations; CPU runtime constraints. No GPU training. Larger P006 run timed out and yielded no validated outcome.
