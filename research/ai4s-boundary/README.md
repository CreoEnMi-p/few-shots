# AI4S boundary experiments: P006 / P007

Independent research in the fork, **not** an official Few Shots submission. This folder does not change any scoring code, official certificate, baseline, or problem metadata.

## Research questions

**P006 (LABS):** Can an online-trained two-layer neural proposal policy select multi-bit perturbations more effectively than same-budget single-bit and uniformly mixed annealing?

**P007 (Thomson):** After identical L-BFGS relaxation, does preferentially perturbing non-six-coordinated spherical defects reach better energy basins than random patch perturbation?

## Reproduce (CPU)

Requires Python 3.11+, NumPy, SciPy. The initial experiments were run on a CPU, **not** an RTX 4070.

From this directory:

```bash
python -m pip install numpy scipy
python -m unittest discover -s src -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 python src/p006_labs.py --lengths 40 60 100 --seeds 6 --budget 6000 --output results/p006.json
OPENBLAS_NUM_THREADS=1 python src/p007_thomson.py --lengths 32 60 --seeds 3 --steps 12 --maxiter 100 --output results/p007.json
```

The separate P006 record-seeded challenge used 4 seeds, 25,000 proposals per strategy and the 100-sign baseline sequence from the official `problems/p006-labs-100/baseline/certificate.json` (supply its string using `--warmstart`).

## Formal research phase

**Decision:** GO on P006 as a conditional algorithm-discovery project, not a commitment to break the N=100 world record. Read `notes/research-charter.md` for falsifiable hypotheses, literature, benchmarks and stop/go criteria. The earlier neural-vs-single-flip comparison is a small pilot with weak baselines and is **not** decisive.

A locally verified classical reference is `src/p006_tabu_reference.py`, with exact single-flip evaluation and `src/test_p006_tabu_reference.py` correctness tests. It is **not** the complete GPU memetic-tabu state of the art.

## Results

See `results/summary.json` for all seed-level final best scores and `notes/interpretation.md` for interpretation, limitations and next experiments.

- P006: the online neural policy was **not** consistently better; single-flip was better on average at N=40 and N=60. At N=100 the neural and single-flip means were close with only six seeds. No E<578 found from the best-known baseline in four 25,000-proposal warm-start trials per method.
- P007: at N=32 and N=60 the simple degree-defect-guided perturbation produced essentially the same relaxed minima as random patch perturbation. **No N=960 experiment or valid 960-point certificate was produced.**
- A larger 12-seed / 12,000-proposal P006 run exceeded the available execution limit. It produced no complete output and is **not** included in the results.

## Fairness and caveats

P006 matches proposal count and initial random seed, not CPU time (online neural updates cost extra). P007 matches perturbation/relaxation counts, not total optimizer evaluations. These are small pilot experiments, not world records and not evidence that neural methods cannot help. Research smoke tests are runnable manually. An extra research workflow was removed to preserve the upstream repository's strict workflow whitelist; the existing repository test workflow remains unchanged.

**Repository organization:** keep experiment code and reports under `research/`; put only validated record-breaking certificates into the official `problems/*/shots/` layout.
