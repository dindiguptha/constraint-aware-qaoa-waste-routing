# constraint-aware-qaoa-waste-routing
Constraint-Aware QAOA for Smart Waste Collection Routing: A Comparative Study of Standard and XY-Mixer QAOAnsatz

Companion code for the NQComp 2027 paper. Run files in
order; each stage imports from earlier ones, so keep them all in the same
folder.

## Setup
```
pip install pennylane pyqubo dwave-neal simulated-bifurcation matplotlib numpy networkx torch scikit-learn
```
Requires Python 3.10 or 3.11 recommended (pyqubo/dwave-neal wheel
availability can be inconsistent on very new Python versions like 3.13).

## File-by-file guide
- `stage1_setup.py` — problem instance (depot + 4 bins), distance matrix
- `stage2_baselines.py` — nearest-neighbor heuristic + exact brute-force solver
- `stage3_qubo.py` — one-hot QUBO formulation (PyQUBO), validated via simulated annealing
- `stage3b_sim_bifurcation.py` — quantum-inspired baseline (Simulated Bifurcation) on the same QUBO
- `stage4a_qubo_to_ising.py` — QUBO -> Ising conversion, validated against known energy
- `stage4b_qaoa.py` — standard QAOA in PennyLane (penalty-based constraints)
- `stage4c_feasibility.py` — feasibility-rate analysis for standard QAOA
- `stage5_qaoansatz.py` — QAOAnsatz with XY-mixer (structural constraint enforcement)
- `stage6_log_encoding.py` — binary/log qubit encoding. IMPORTANT: uses
  `compile(strength=500.0)` — PyQUBO's default `strength=5` is too weak
  relative to our own penalty weights (up to 50), which lets the solver
  violate auxiliary-variable constraints and report spuriously low
  energies for infeasible routes. This was a real bug we found via
  validation (see paper Section 4.4) and fixed by raising `strength`.
  Do not remove this parameter.
- `stage7_final_comparison.py` — single-run comparison across 5 methods (table + plot)
- `stage8_multi_trial.py` — multi-trial (10 shot-batches) on ONE instance:
  reproduces paper Table 1 (2.78%/14.06%, 5.06x)
- `stage9_clustering.py` — 8-bin divide-and-conquer via k-means clustering
  + our validated 4-bin QAOAnsatz pipeline per cluster. Reproduces the
  19.35% optimality gap discussed in paper Section 4.2, and confirms the
  gap comes entirely from the classical stitching step (each cluster's
  quantum sub-route matches its own true local optimum).
- `stage10_multi_instance.py` — 10 different random instances, 1 sample
  each: reproduces the first generalization result (2.44%/13.08%, 5.36x)
- `stage11_full_study.py` — the paper's CURRENT headline result: 10
  instances x 10 shot-batches = 100 measurements/method, decomposing
  variance into shot-noise (within-instance) vs. geometry
  (between-instance) components. Reproduces Table 2 in the paper
  (2.52%/12.84%, 5.10x, with the variance decomposition). Saves raw
  per-measurement results to `stage11_raw_results.json`.
- `stage12_significance.py` — loads `stage11_raw_results.json` and runs
  the significance tests reported in paper Section 4.1: Welch's t-test,
  Mann-Whitney U test, and a bootstrap 95% CI on the mean difference.
  Reproduces the paper's exact reported statistics (t=16.43, p<1e-29;
  Mann-Whitney p<1e-33; bootstrap CI [9.14%, 11.60%]). Requires
  `stage11_full_study.py` to be run first.

## Recommended run order
```
python stage1_setup.py       # sanity check the instance
python stage2_baselines.py   # classical baselines
python stage3_qubo.py        # validate QUBO formulation
python stage4a_qubo_to_ising.py   # validate Ising conversion
python stage8_multi_trial.py      # Table 1 result
python stage11_full_study.py      # Table 2 result (takes ~8-10 min)
python stage12_significance.py    # significance tests (needs stage11's output)
```

## Known hardware limits (see paper Section 4.3)
This code is validated at NUM_BINS=4 (16 qubits). Scaling to 6 bins
(36 qubits) requires >1TB RAM for full statevector simulation and will
fail with an out-of-memory error on standard hardware -- this is a
genuine simulation limit, not a bug. Stage 9's clustering approach is
the validated workaround for handling more bins without hitting this wall.

## All numbers below were verified by actually re-running each script
- stage3_qubo.py: route [0,4,3,1,2,0], cost 25.063, matches brute-force
- stage8_multi_trial.py: 2.78%+/-0.57% vs 14.06%+/-1.14%, 5.06x
- stage9_clustering.py: 19.35% gap, both clusters match local optima exactly
- stage10_multi_instance.py: 2.44%+/-0.83% vs 13.08%+/-6.29%, 5.36x
- stage11_full_study.py: 2.52%+/-0.80% vs 12.84%+/-6.20%, 5.10x
- stage12_significance.py: t=16.432, p=1.861e-30; Mann-Whitney p=1.187e-34;
  bootstrap 95% CI on mean difference [9.14%, 11.60%]
