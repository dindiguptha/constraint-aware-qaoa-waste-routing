"""
Stage 12: Statistical Significance Testing

Loads the raw 100-measurement-per-method results saved by
stage11_full_study.py and computes the significance tests reported in
the paper's Results and Discussion section (4.1):

  - Welch's two-sample t-test (unequal variances, appropriate given the
    markedly different spreads observed between standard QAOA and
    QAOAnsatz)
  - Mann-Whitney U test (non-parametric robustness check, appropriate
    since feasibility-rate percentages are bounded and not necessarily
    normally distributed)
  - Bootstrap 95% confidence interval on the mean difference

Run stage11_full_study.py FIRST -- it produces stage11_raw_results.json,
which this script reads. This keeps every reported number in the paper
traceable to a script, rather than a one-off calculation done outside
the staged pipeline.
"""

import json
import numpy as np
from scipy import stats

RAW_RESULTS_FILE = "stage11_raw_results.json"
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42


def load_raw_results(path=RAW_RESULTS_FILE):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["std_all_rates"]), np.array(data["ansatz_all_rates"])


def run_significance_tests(std_rates, ansatz_rates):
    results = {}

    results["std_n"] = len(std_rates)
    results["std_mean"] = float(std_rates.mean())
    results["std_sd"] = float(std_rates.std())
    results["ansatz_n"] = len(ansatz_rates)
    results["ansatz_mean"] = float(ansatz_rates.mean())
    results["ansatz_sd"] = float(ansatz_rates.std())

    # Welch's t-test: does NOT assume equal variances, appropriate here
    # since QAOAnsatz's SD (6.20) is roughly 8x standard QAOA's SD (0.80)
    t_stat, p_ttest = stats.ttest_ind(ansatz_rates, std_rates, equal_var=False)
    results["welch_t"] = float(t_stat)
    results["welch_p"] = float(p_ttest)

    # Mann-Whitney U: non-parametric, robust to non-normality / bounded data
    u_stat, p_mw = stats.mannwhitneyu(ansatz_rates, std_rates, alternative="greater")
    results["mannwhitney_u"] = float(u_stat)
    results["mannwhitney_p"] = float(p_mw)

    # Bootstrap 95% CI on the mean difference (QAOAnsatz - Standard QAOA)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        s = rng.choice(std_rates, size=len(std_rates), replace=True)
        a = rng.choice(ansatz_rates, size=len(ansatz_rates), replace=True)
        diffs[i] = a.mean() - s.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    results["mean_difference"] = float(ansatz_rates.mean() - std_rates.mean())
    results["bootstrap_ci_low"] = float(ci_low)
    results["bootstrap_ci_high"] = float(ci_high)
    results["n_bootstrap"] = N_BOOTSTRAP

    return results


def print_report(results):
    print(f"Standard QAOA: n={results['std_n']}, "
          f"mean={results['std_mean']:.3f}%, SD={results['std_sd']:.3f}%")
    print(f"QAOAnsatz:     n={results['ansatz_n']}, "
          f"mean={results['ansatz_mean']:.3f}%, SD={results['ansatz_sd']:.3f}%")

    print(f"\nWelch's two-sample t-test (unequal variances):")
    print(f"  t = {results['welch_t']:.3f}, p = {results['welch_p']:.3e}")

    print(f"\nMann-Whitney U test (QAOAnsatz > Standard QAOA):")
    print(f"  U = {results['mannwhitney_u']:.1f}, p = {results['mannwhitney_p']:.3e}")

    print(f"\nBootstrap 95% CI on mean difference "
          f"({results['n_bootstrap']} resamples):")
    print(f"  Mean difference (QAOAnsatz - Standard QAOA): "
          f"{results['mean_difference']:.2f}%")
    print(f"  95% CI: [{results['bootstrap_ci_low']:.2f}%, "
          f"{results['bootstrap_ci_high']:.2f}%]")
    excludes_zero = results['bootstrap_ci_low'] > 0
    print(f"  Excludes zero: {'YES' if excludes_zero else 'NO'}")


if __name__ == "__main__":
    try:
        std_rates, ansatz_rates = load_raw_results()
    except FileNotFoundError:
        raise SystemExit(
            f"\n{RAW_RESULTS_FILE} not found. Run stage11_full_study.py "
            f"first -- it produces this file as its final step.\n"
        )

    results = run_significance_tests(std_rates, ansatz_rates)
    print_report(results)

    with open("stage12_significance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to stage12_significance_results.json")
