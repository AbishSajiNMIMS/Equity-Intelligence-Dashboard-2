"""
Statistical helper functions used by the dashboard.
This module keeps the math separate from the UI so the app code
can focus on inputs, charts, and presentation.
"""

import numpy as np
from scipy import stats
from scipy.stats import f_oneway, norm, t as t_dist, ttest_1samp, ttest_ind, ttest_rel


# -----------------------------------------------------------------------------
# Estimation helpers:
# these return compact dictionaries that the dashboard can display directly.
# -----------------------------------------------------------------------------
def confidence_interval_mean(data: np.ndarray, confidence: float = 0.95) -> dict:
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    critical = norm.ppf((1 + confidence) / 2) if n > 30 else t_dist.ppf((1 + confidence) / 2, n - 1)
    margin = critical * se
    return {
        "mean": round(mean, 6),
        "lower": round(mean - margin, 6),
        "upper": round(mean + margin, 6),
        "std_error": round(se, 6),
        "n": n,
        "confidence": confidence,
        "method": "Z-interval (large sample)" if n > 30 else "t-interval (small sample)",
    }


def mle_normal(data: np.ndarray) -> dict:
    mu = np.mean(data)
    sigma = np.std(data, ddof=0)
    ll = np.sum(norm.logpdf(data, mu, sigma))
    return {
        "mu_mle": round(mu, 6),
        "sigma_mle": round(sigma, 6),
        "log_likelihood": round(ll, 4),
        "aic": round(4 - 2 * ll, 4),
        "interpretation": f"The MLE estimates the stock's daily return as μ = {mu * 100:.3f}% with σ = {sigma * 100:.3f}%",
    }


def plot_likelihood_surface(data: np.ndarray, mu_range: tuple = None) -> dict:
    sigma = np.std(data, ddof=0)
    mu = np.mean(data)
    low, high = mu_range or (mu - 4 * sigma, mu + 4 * sigma)
    mu_grid = np.linspace(low, high, 300)
    log_likelihoods = np.array([np.sum(norm.logpdf(data, point, sigma)) for point in mu_grid])
    return {"mu_grid": mu_grid, "log_likelihoods": log_likelihoods, "mle_mu": mu, "peak_ll": log_likelihoods.max()}


# -----------------------------------------------------------------------------
# Hypothesis test helpers:
# each function returns the statistic, p-value, and a plain-English conclusion.
# -----------------------------------------------------------------------------
def z_test_mean(data: np.ndarray, pop_mean: float, pop_std: float, alpha: float = 0.05, tail: str = "two") -> dict:
    sample_mean = np.mean(data)
    z_stat = (sample_mean - pop_mean) / (pop_std / np.sqrt(len(data)))
    if tail == "two":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        z_critical = norm.ppf(1 - alpha / 2)
    elif tail == "right":
        p_value = 1 - norm.cdf(z_stat)
        z_critical = norm.ppf(1 - alpha)
    else:
        p_value = norm.cdf(z_stat)
        z_critical = -norm.ppf(1 - alpha)
    reject = p_value < alpha
    return {
        "test": "Z-Test (One Sample)",
        "z_statistic": round(z_stat, 4),
        "z_critical": round(z_critical, 4),
        "p_value": round(p_value, 6),
        "alpha": alpha,
        "reject_H0": reject,
        "conclusion": (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"The mean return {'IS' if reject else 'IS NOT'} significantly different from {pop_mean:.4%}."
        ),
    }


def t_test_one_sample(data: np.ndarray, pop_mean: float, alpha: float = 0.05) -> dict:
    t_stat, p_value = ttest_1samp(data, pop_mean)
    df = len(data) - 1
    reject = p_value < alpha
    return {
        "test": "One-Sample t-Test",
        "t_statistic": round(t_stat, 4),
        "t_critical": round(t_dist.ppf(1 - alpha / 2, df), 4),
        "p_value": round(p_value, 6),
        "df": df,
        "alpha": alpha,
        "reject_H0": reject,
        "conclusion": (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"Mean return {'IS' if reject else 'IS NOT'} significantly different from {pop_mean:.4%}."
        ),
    }


def t_test_two_sample(data1: np.ndarray, data2: np.ndarray, label1: str = "Stock A", label2: str = "Stock B", alpha: float = 0.05) -> dict:
    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
    reject = p_value < alpha
    return {
        "test": "Two-Sample Welch's t-Test",
        "label1": label1,
        "label2": label2,
        "mean1": round(np.mean(data1), 6),
        "mean2": round(np.mean(data2), 6),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "alpha": alpha,
        "reject_H0": reject,
        "conclusion": (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"Returns of {label1} and {label2} are {'SIGNIFICANTLY DIFFERENT' if reject else 'NOT significantly different'}."
        ),
    }


def t_test_paired(data1: np.ndarray, data2: np.ndarray, label1: str = "Before", label2: str = "After", alpha: float = 0.05) -> dict:
    n = min(len(data1), len(data2))
    t_stat, p_value = ttest_rel(data1[:n], data2[:n])
    reject = p_value < alpha
    return {
        "test": "Paired t-Test",
        "label1": label1,
        "label2": label2,
        "mean_diff": round(np.mean(data1[:n] - data2[:n]), 6),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "alpha": alpha,
        "reject_H0": reject,
        "conclusion": (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"There {'IS' if reject else 'IS NOT'} a significant difference between {label1} and {label2} periods."
        ),
    }


# -----------------------------------------------------------------------------
# Group comparison helper:
# ANOVA is used in the sector-analysis tab to compare several means at once.
# -----------------------------------------------------------------------------
def one_way_anova(*groups, labels: list = None, alpha: float = 0.05) -> dict:
    f_stat, p_value = f_oneway(*groups)
    reject = p_value < alpha
    return {
        "test": "One-Way ANOVA",
        "f_statistic": round(f_stat, 4),
        "p_value": round(p_value, 6),
        "alpha": alpha,
        "reject_H0": reject,
        "conclusion": (
            f"{'REJECT' if reject else 'FAIL TO REJECT'} H₀. "
            f"Sector returns {'ARE' if reject else 'ARE NOT'} significantly different."
        ),
    }
