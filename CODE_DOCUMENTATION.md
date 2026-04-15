# Equity Intelligence Dashboard — Complete Code Documentation

> This document explains every part of the dashboard, how each code file connects to the others, what each function returns (as a dictionary/JSON), which tab displays it, and the theory behind every statistical and ML technique used.

---

## Table of Contents

1. [Project Architecture](#1-project-architecture)
2. [File-by-File Breakdown](#2-file-by-file-breakdown)
   - 2.1 [data_loader.py — Data Access & Feature Engineering](#21-data_loaderpy--data-access--feature-engineering)
   - 2.2 [abish_stats.py — Statistical Tests & Estimation](#22-abish_statspy--statistical-tests--estimation)
   - 2.3 [laksh_ml.py — Machine Learning Models](#23-laksh_mlpy--machine-learning-models)
   - 2.4 [streamlit_app.py — Dashboard UI](#24-streamlit_apppy--dashboard-ui)
3. [Tab-by-Tab Walkthrough](#3-tab-by-tab-walkthrough)
   - 3.1 [Tab 1 — Overview](#31-tab-1--overview)
   - 3.2 [Tab 2 — Risk Profile](#32-tab-2--risk-profile)
   - 3.3 [Tab 3 — Hypothesis Tests](#33-tab-3--hypothesis-tests)
   - 3.4 [Tab 4 — Sector Analysis](#34-tab-4--sector-analysis)
   - 3.5 [Tab 5 — ML Predictions](#35-tab-5--ml-predictions)
   - 3.6 [Tab 6 — Ridge & Lasso](#36-tab-6--ridge--lasso)
4. [Data Flow Summary](#4-data-flow-summary)
5. [Complete Theory Reference](#5-complete-theory-reference)

---

## 1. Project Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     streamlit_app.py (UI Layer)                     │
│  Sidebar inputs → load_data() → 6 Tabs with charts & metrics       │
├──────────┬──────────────────────┬───────────────────────────────────┤
│          │                      │                                   │
│  data_loader.py          abish_stats.py                laksh_ml.py  │
│  (Data Layer)          (Statistics Layer)            (ML Layer)      │
│                                                                     │
│  • Download prices       • Confidence intervals    • Logistic Reg    │
│  • Calculate returns     • MLE estimation          • Ridge Reg       │
│  • Sector mapping        • Z-test, t-tests         • Lasso Reg       │
│  • Feature engineering   • ANOVA                   • Predictions     │
└─────────────────────────────────────────────────────────────────────┘
```

There are **4 Python files** and **6 dashboard tabs**:

| File | Role | Used by Tabs |
|------|------|-------------|
| `data_loader.py` | Downloads stock prices from Yahoo Finance, calculates returns, engineers ML features | All 6 tabs |
| `abish_stats.py` | Statistical estimation (CI, MLE) and hypothesis testing (Z, t, ANOVA) | Tab 2, 3, 4 |
| `laksh_ml.py` | Trains logistic, Ridge, and Lasso models; generates predictions | Tab 5, 6 |
| `streamlit_app.py` | Main dashboard: sidebar inputs, 6 tabs, charts, metrics, theory boxes | — |

---

## 2. File-by-File Breakdown

### 2.1 `data_loader.py` — Data Access & Feature Engineering

**Purpose:** This file is the data backbone. It downloads stock prices from Yahoo Finance, computes daily returns, and engineers the features that the ML models need. No statistics or ML happen here — just data preparation.

---

#### `SECTORS` dictionary (Lines 15–21)

```python
SECTORS = {
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "IT":      ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "FMCG":    ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS"],
    "Pharma":  ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS"],
    "Energy":  ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS"],
}
```

**What it does:** Maps 5 Indian market sectors to their respective NSE ticker symbols (5 stocks per sector = 25 stocks total). The `.NS` suffix is the Yahoo Finance code for the National Stock Exchange of India.

**Where it's used in the dashboard:** Imported in `streamlit_app.py` line 23 and used in the sidebar (lines 97–100) to populate the stock selection dropdowns. When you pick "Banking" from the sector dropdown, this dictionary tells the app to offer HDFCBANK.NS, ICICIBANK.NS, etc.

---

#### `_download()` function (Lines 29–30)

```python
def _download(tickers, period="2y"):
    return yf.download(tickers, period=period, auto_adjust=True, progress=False)
```

**What it does:** Internal helper that calls the `yfinance` library to download stock price data. `auto_adjust=True` means prices are adjusted for splits and dividends. `progress=False` hides the download progress bar (cleaner for a web app).

**Not called directly by the dashboard** — used internally by `get_price_data()` and `engineer_features()`.

---

#### `_extract_field()` function (Lines 33–54)

```python
def _extract_field(raw: pd.DataFrame, field: str, labels=None) -> pd.DataFrame:
```

**What it does:** Yahoo Finance sometimes returns data with simple column names (`Close`, `Volume`) and sometimes with multi-level column names (a `MultiIndex`). This helper normalises both cases into a simple, flat DataFrame. It searches both levels of a MultiIndex for the requested field name.

**Why it's needed:** Without this, the app would crash whenever Yahoo Finance changes its column format between single-ticker and multi-ticker downloads.

---

#### `get_price_data()` function (Lines 62–63)

```python
def get_price_data(tickers: list, period: str = "2y") -> pd.DataFrame:
    return _extract_field(_download(tickers, period), "Close", tickers).dropna(how="all")
```

**What it does:** Downloads stock data and extracts only the closing prices. Returns a DataFrame where each column is a ticker and each row is a trading day.

**Output example:**

| Date | HDFCBANK.NS | ICICIBANK.NS |
|------|-------------|--------------|
| 2024-01-02 | 1650.30 | 1020.45 |
| 2024-01-03 | 1655.10 | 1018.20 |

**Where it's used:** Called in `streamlit_app.py` inside `load_data()` (lines 77–78) to get prices for both selected stocks. These prices feed the Overview tab's price charts.

---

#### `get_daily_returns()` function (Lines 66–67)

```python
def get_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
```

**What it does:** Converts a price series into daily percentage returns using the formula:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

`.pct_change()` computes this automatically. `.dropna()` removes the first row (which has no previous price to compare against).

**Output example:** A value of 0.0032 means the stock went up 0.32% that day.

**Where it's used:** Called in `load_data()` (lines 82–83) in `streamlit_app.py`. The resulting return series (`ret1`, `ret2`) are used in almost every tab — histograms, statistical tests, ML features.

---

#### `get_sector_returns()` function (Lines 70–74)

```python
def get_sector_returns(period: str = "2y") -> dict:
    return {
        sector: get_daily_returns(get_price_data(tickers, period)).mean(axis=1)
        for sector, tickers in SECTORS.items()
    }
```

**What it does:** For each of the 5 sectors, downloads all 5 stock prices, converts to returns, and then averages across the 5 stocks to get one "sector return" per day. Returns a dictionary where each key is a sector name and each value is a pandas Series of daily returns.

**Output (dict):**
```python
{
    "Banking": pd.Series([0.001, -0.003, ...]),   # daily avg return of 5 banking stocks
    "IT": pd.Series([0.002, -0.001, ...]),
    "FMCG": pd.Series([...]),
    "Pharma": pd.Series([...]),
    "Energy": pd.Series([...]),
}
```

**Where it's used:** Called in `load_data()` (line 84). The result feeds the **Sector Analysis tab (Tab 4)**, where ANOVA compares whether sector means are statistically different.

---

#### `get_stock_info()` function (Lines 77–81)

```python
def get_stock_info(ticker: str) -> dict:
    try:
        return {"name": yf.Ticker(ticker).info.get("longName", ticker)}
    except Exception:
        return {"name": ticker}
```

**What it does:** Looks up the full company name for a ticker symbol (e.g., "HDFCBANK.NS" → "HDFC Bank Limited"). Falls back to the ticker code if the lookup fails.

**Output (dict):**
```json
{"name": "HDFC Bank Limited"}
```

**Where it's used:** Called in `load_data()` (line 85). Displayed in the **Overview tab (Tab 1)** as a metric card showing the company name (line 142).

---

#### `engineer_features()` function (Lines 89–119)

```python
def engineer_features(ticker: str, period: str = "2y") -> pd.DataFrame:
```

**What it does:** This is the most important data function. It builds the full feature matrix that the ML models need. Starting from raw close prices and volume, it creates:

| Feature | Lines | Formula / Logic | Theory |
|---------|-------|----------------|--------|
| `return` | 99 | `close.pct_change()` | Basic daily percentage return |
| `lag_1`, `lag_2`, `lag_3`, `lag_5` | 100–101 | `return.shift(lag)` | **Autoregressive features**: past returns may predict future returns (momentum/reversal effects) |
| `rolling_mean_5`, `rolling_mean_20` | 103–104 | `return.rolling(window).mean()` | **Moving average of returns**: smooths out noise, captures short-term (5-day) and medium-term (20-day) trend direction |
| `rolling_std_5`, `rolling_std_20` | 103–105 | `return.rolling(window).std()` | **Rolling volatility**: measures how "jumpy" the stock has been recently |
| `rsi` | 107–111 | 14-period RSI calculation | **Relative Strength Index**: momentum oscillator (0–100). Above 70 = overbought, below 30 = oversold |
| `volume_ratio` | 113 | `volume / volume.rolling(20).mean()` | **Volume spike detector**: ratio > 1 means today's volume is above the 20-day average (unusual activity) |
| `dist_ma20`, `dist_ma50` | 114–116 | `(close - MA) / MA` | **Distance to moving average**: positive = price is above its trend, negative = below. Measures how stretched the price is |
| `target` | 118 | `(return.shift(-1) > 0).astype(int)` | **Binary label**: 1 if tomorrow's return is positive (UP), 0 if negative (DOWN). This is what the logistic model predicts |

**RSI calculation detail (Lines 107–111):**
```python
delta = df["close"].diff()                    # price change each day
gain = delta.clip(lower=0).rolling(14).mean() # average of only positive changes
loss = (-delta.clip(upper=0)).rolling(14).mean()  # average of only negative changes
rs = gain / loss                              # relative strength ratio
df["rsi"] = 100 - (100 / (1 + rs))           # scale to 0–100
```

**Output:** A DataFrame with all the columns above, plus `close` and `volume`. All rows with NaN values (from rolling windows) are dropped via `.dropna()`.

**Where it's used:** Called in `load_data()` (line 86) as `df_ml`. This DataFrame is passed to all three ML training functions in Tabs 5 and 6.

---

### 2.2 `abish_stats.py` — Statistical Tests & Estimation

**Purpose:** Contains all the statistical math. Every function takes numerical data (numpy arrays of daily returns) and returns a dictionary with results, making it easy for the dashboard to display them.

---

#### `confidence_interval_mean()` function (Lines 16–30)

```python
def confidence_interval_mean(data: np.ndarray, confidence: float = 0.95) -> dict:
```

**Theory — Confidence Intervals:**

A confidence interval gives a range of plausible values for the true population mean. If you repeatedly sampled from the market and computed a 95% CI each time, about 95% of those intervals would contain the true mean return.

**Key concepts:**
- **Standard Error (SE)** = s / √n — how much the sample mean varies from sample to sample. Larger samples → smaller SE → tighter interval.
- **Critical value**: comes from the Z-distribution (if n > 30, "large sample") or the t-distribution (if n ≤ 30, "small sample"). The t-distribution has heavier tails, so the interval is wider for small samples.

**Formula:**

$$CI = \bar{x} \pm c \cdot SE$$

where c is the critical value (Z or t depending on sample size).

**Code logic:**
- Line 20: If sample size > 30, uses `norm.ppf()` (Z critical value). Otherwise uses `t_dist.ppf()` (t critical value with n-1 degrees of freedom).
- Line 21: Margin of error = critical value × standard error.

**Output (dict):**
```json
{
    "mean": 0.000234,
    "lower": -0.000412,
    "upper": 0.000880,
    "std_error": 0.000329,
    "n": 499,
    "confidence": 0.95,
    "method": "Z-interval (large sample)"
}
```

**Dashboard location:** **Tab 2 — Risk Profile**, left column (lines 192–197 of `streamlit_app.py`). Shows the mean, CI range, standard error, and method as text and a metric card.

---

#### `mle_normal()` function (Lines 33–43)

```python
def mle_normal(data: np.ndarray) -> dict:
```

**Theory — Maximum Likelihood Estimation (MLE):**

MLE asks: "If stock returns follow a Normal distribution, what μ (mean) and σ (standard deviation) make the observed data most probable?"

The answer is:
- **μ̂ = sample mean** (the average of all daily returns)
- **σ̂ = population standard deviation** (using `ddof=0`, dividing by n, not n-1)

**Log-Likelihood** measures how well the Normal(μ̂, σ̂) model fits the data. Higher = better fit.

$$\ell(\mu, \sigma) = \sum_{i=1}^{n} \log f(x_i \mid \mu, \sigma)$$

**AIC (Akaike Information Criterion)** penalises model complexity. For a 2-parameter Normal model: AIC = 4 − 2ℓ. Lower AIC = better balance between fit and simplicity.

**Code logic:**
- Line 34–35: Compute MLE estimates (just the mean and std dev).
- Line 36: Sum of log-probabilities under Normal(μ̂, σ̂).
- Line 38: AIC = 2k − 2ℓ where k = 2 (two parameters: μ and σ).

**Output (dict):**
```json
{
    "mu_mle": 0.000234,
    "sigma_mle": 0.015678,
    "log_likelihood": 1523.4567,
    "aic": -3042.9134,
    "interpretation": "The MLE estimates the stock's daily return as μ = 0.023% with σ = 1.568%"
}
```

**Dashboard location:** **Tab 2 — Risk Profile**, right column (lines 200–206). Shows MLE μ, MLE σ, log-likelihood, AIC, and interpretation text.

---

#### `plot_likelihood_surface()` function (Lines 46–52)

```python
def plot_likelihood_surface(data: np.ndarray, mu_range: tuple = None) -> dict:
```

**Theory — Likelihood Surface:**

This function visualises how the log-likelihood changes as you vary μ while holding σ fixed at its MLE value. The peak of the curve is at the MLE estimate of μ — that's the value that maximises the likelihood.

**Code logic:**
- Line 50: Creates a grid of 300 candidate μ values spanning ±4 standard deviations around the MLE mean.
- Line 51: For each candidate μ, computes the total log-likelihood.

**Output (dict):**
```json
{
    "mu_grid": [array of 300 μ values],
    "log_likelihoods": [array of 300 log-likelihood values],
    "mle_mu": 0.000234,
    "peak_ll": 1523.4567
}
```

**Dashboard location:** **Tab 2 — Risk Profile** (lines 208–212). Rendered as a Plotly line chart with a dashed vertical line at the MLE μ peak.

---

#### `z_test_mean()` function (Lines 59–83)

```python
def z_test_mean(data, pop_mean, pop_std, alpha=0.05, tail="two") -> dict:
```

**Theory — Z-Test:**

The Z-test checks whether the sample mean is significantly different from a hypothesised population mean, **when the population standard deviation (σ) is known**.

$$H_0: \mu = \mu_0 \quad \text{(null hypothesis: mean equals the hypothesised value)}$$
$$z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

- If |z| > z_critical → reject H₀ → the mean IS significantly different.
- If |z| ≤ z_critical → fail to reject H₀ → no strong evidence of a difference.

**The p-value** is the probability of getting a test statistic as extreme as the one observed, if H₀ were true. Small p-value (< α) = strong evidence against H₀.

**Code logic:**
- Line 61: Computes z-statistic.
- Lines 62–70: Adjusts p-value and critical value based on whether the test is two-tailed, right-tailed, or left-tailed.
- Line 71: Decision: reject if p < α.

**Output (dict):**
```json
{
    "test": "Z-Test (One Sample)",
    "z_statistic": 1.8234,
    "z_critical": 1.96,
    "p_value": 0.068234,
    "alpha": 0.05,
    "reject_H0": false,
    "conclusion": "FAIL TO REJECT H₀. The mean return IS NOT significantly different from 0.0000%."
}
```

**Dashboard location:** **Tab 3 — Hypothesis Tests**, when "Z-Test" is selected (lines 272–278). Shows z-statistic, z-critical, p-value as metric cards, then a success/error banner with the conclusion.

---

#### `t_test_one_sample()` function (Lines 86–102)

```python
def t_test_one_sample(data, pop_mean, alpha=0.05) -> dict:
```

**Theory — One-Sample t-Test:**

Same idea as the Z-test, but used when σ is **unknown** (estimated from the sample). The test statistic follows a t-distribution with n−1 degrees of freedom, which has heavier tails than the Normal distribution (more uncertainty).

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

**When to use Z vs t:**
- **Z-test**: σ is known (rare in practice, but used here when you input a specific σ)
- **t-test**: σ is unknown, estimated from the sample (more common and more conservative)

**Code logic:**
- Line 87: Uses scipy's `ttest_1samp()` which handles the calculation.
- Line 88: Degrees of freedom = n − 1.

**Output (dict):**
```json
{
    "test": "One-Sample t-Test",
    "t_statistic": 1.7456,
    "t_critical": 1.9650,
    "p_value": 0.081345,
    "df": 498,
    "alpha": 0.05,
    "reject_H0": false,
    "conclusion": "FAIL TO REJECT H₀. Mean return IS NOT significantly different from 0.0000%."
}
```

**Dashboard location:** **Tab 3 — Hypothesis Tests**, when "One-Sample t-Test" is selected (lines 280–287).

---

#### `t_test_two_sample()` function (Lines 105–122)

```python
def t_test_two_sample(data1, data2, label1="Stock A", label2="Stock B", alpha=0.05) -> dict:
```

**Theory — Two-Sample Welch's t-Test:**

Compares the **means of two independent groups** (Stock 1 vs Stock 2). The Welch version does not assume the two groups have the same variance — it adjusts the degrees of freedom accordingly.

$$H_0: \mu_1 = \mu_2 \quad \text{(both stocks have the same average return)}$$
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$

**Application:** Are HDFC Bank's returns really different from TCS's returns, or is the difference just due to random day-to-day noise?

**Code logic:**
- Line 106: Uses scipy's `ttest_ind()` with `equal_var=False` (Welch's version).

**Output (dict):**
```json
{
    "test": "Two-Sample Welch's t-Test",
    "label1": "HDFCBANK.NS",
    "label2": "TCS.NS",
    "mean1": 0.000456,
    "mean2": 0.000312,
    "t_statistic": 0.8234,
    "p_value": 0.410567,
    "alpha": 0.05,
    "reject_H0": false,
    "conclusion": "FAIL TO REJECT H₀. Returns of HDFCBANK.NS and TCS.NS are NOT significantly different."
}
```

**Dashboard location:** **Tab 3 — Hypothesis Tests**, when "Two-Sample t-Test" is selected (lines 289–301). Also shows a box plot comparing the two stocks' return distributions.

---

#### `t_test_paired()` function (Lines 125–142)

```python
def t_test_paired(data1, data2, label1="Before", label2="After", alpha=0.05) -> dict:
```

**Theory — Paired t-Test:**

Used when the two samples are **matched** (not independent). The test works on the **differences** d_i = x_{i,1} − x_{i,2} rather than the raw values. Here, the dashboard splits a single stock's return series into a first half and second half and tests whether the stock's behavior changed over time.

$$d_i = x_{i,\text{first}} - x_{i,\text{second}}$$
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

**Application:** Did HDFC Bank perform differently in the first year vs the second year of the analysis window?

**Code logic:**
- Line 126: Takes the minimum length of both arrays to ensure equal pairing.
- Line 127: Uses scipy's `ttest_rel()` for related (paired) samples.

**Output (dict):**
```json
{
    "test": "Paired t-Test",
    "label1": "First Half",
    "label2": "Second Half",
    "mean_diff": 0.000123,
    "t_statistic": 0.5678,
    "p_value": 0.570234,
    "alpha": 0.05,
    "reject_H0": false,
    "conclusion": "FAIL TO REJECT H₀. There IS NOT a significant difference between First Half and Second Half periods."
}
```

**Dashboard location:** **Tab 3 — Hypothesis Tests**, when "Paired t-Test" is selected (lines 303–309).

---

#### `one_way_anova()` function (Lines 149–162)

```python
def one_way_anova(*groups, labels=None, alpha=0.05) -> dict:
```

**Theory — One-Way ANOVA (Analysis of Variance):**

ANOVA extends the two-sample t-test to **k groups** (here, 5 sectors). Instead of asking "are these two means equal?", it asks "are ALL these means equal?"

$$H_0: \mu_1 = \mu_2 = \cdots = \mu_k$$
$$F = \frac{MS_{\text{between}}}{MS_{\text{within}}}$$

- **MS_between**: How much the group means vary from the overall mean (between-group variance).
- **MS_within**: How much individual returns vary within each group (within-group variance).
- A large F means the groups are more different from each other than you'd expect from random noise.

**Application:** Do Banking, IT, FMCG, Pharma, and Energy sectors have genuinely different average returns, or are the differences just noise?

**Code logic:**
- Line 150: Uses scipy's `f_oneway()` which takes any number of arrays.

**Output (dict):**
```json
{
    "test": "One-Way ANOVA",
    "f_statistic": 2.3456,
    "p_value": 0.052345,
    "alpha": 0.05,
    "reject_H0": false,
    "conclusion": "FAIL TO REJECT H₀. Sector returns ARE NOT significantly different."
}
```

**Dashboard location:** **Tab 4 — Sector Analysis** (lines 330–336). Shows F-statistic and p-value as metrics, plus a conclusion banner.

---

### 2.3 `laksh_ml.py` — Machine Learning Models

**Purpose:** Trains three ML models (Logistic Regression, Ridge Regression, Lasso Regression) and returns results as dictionaries. All models use the 12 engineered features from `data_loader.py`.

---

#### `FEATURE_COLS` list (Lines 23–36)

```python
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_5",
    "rolling_mean_5", "rolling_std_5",
    "rolling_mean_20", "rolling_std_20",
    "rsi", "volume_ratio",
    "dist_ma20", "dist_ma50",
]
```

**What it does:** Defines the 12 features every model uses. These match exactly what `engineer_features()` creates in `data_loader.py`. This list is the single source of truth for feature selection.

---

#### `_split_series()` function (Lines 44–46)

```python
def _split_series(X, y, test_size=0.2):
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]
```

**What it does:** Splits data into training (80%) and testing (20%) sets **chronologically** — not randomly. This is critical for time-series data because you must not train on future data and test on past data (that would be "data leakage").

**Why not sklearn's train_test_split?** Because `train_test_split` shuffles randomly, which would let the model "peek" at future returns during training.

---

#### `_coef_frame()` function (Lines 49–57)

```python
def _coef_frame(coefs, selected=None):
```

**What it does:** Converts a model's coefficient array into a sorted DataFrame showing which features have the largest influence. Used by all three models for the coefficient bar charts.

**Output:** A DataFrame with columns: `feature`, `coefficient`, `abs_coef`, and optionally `selected` (for Lasso).

---

#### `_coef_path()` function (Lines 60–61)

```python
def _coef_path(model_cls, alphas, X, y, **kwargs):
    return np.array([model_cls(alpha=alpha, **kwargs).fit(X, y).coef_ for alpha in alphas])
```

**What it does:** Trains the model at many different alpha values and records the coefficients each time. This creates the "regularization path" — a chart showing how each feature's coefficient shrinks as the penalty increases.

---

#### `train_logistic_model()` function (Lines 68–106)

```python
def train_logistic_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
```

**Theory — Logistic Regression for Stock Direction:**

Logistic regression predicts a **binary outcome** (UP or DOWN). It models the probability of the stock going up tomorrow as:

$$P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$$

This is the **sigmoid function** — it squashes any linear combination of features into a probability between 0 and 1.

The **log-odds** (logit) form is linear:

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta^T X$$

**How to interpret coefficients:**
- A positive coefficient (e.g., `lag_1 = +0.15`) means: when yesterday's return is higher, the probability of an UP day tomorrow increases.
- A negative coefficient means the feature pushes toward DOWN.
- The magnitude tells you how strong the influence is.

**Code walkthrough:**
- Lines 69–71: Extract feature matrix X (12 features) and target y (0 or 1) from the engineered DataFrame. Split chronologically.
- Lines 73–86: Build a **Pipeline** with two steps:
  1. `StandardScaler()` — normalises each feature to mean=0, std=1 (important because features have different scales: RSI is 0–100, returns are tiny decimals).
  2. `LogisticRegression()` — `class_weight="balanced"` adjusts for any imbalance in up vs down days; `solver="lbfgs"` is the optimisation algorithm.
- Line 87: Trains the model on 80% of the data.
- Lines 89–92: Evaluates on the test set using accuracy and 5-fold time-series cross-validation.
- Line 92: `TimeSeriesSplit(n_splits=5)` ensures that in each fold, training data comes before test data.

**Output (dict):**
```json
{
    "model": <trained sklearn Pipeline>,
    "accuracy": 0.5421,
    "cv_accuracy_mean": 0.5234,
    "cv_accuracy_std": 0.0312,
    "confusion_matrix": [[45, 52], [38, 65]],
    "coef_df": "<DataFrame of sorted coefficients>",
    "interpretation": "The model achieves 54.2% accuracy. Top signal: rolling_std_5 (coef = -0.3421)"
}
```

**Dashboard location:** **Tab 5 — ML Predictions** (lines 376–406). Shows:
- Metric cards: Signal (UP/DOWN), P(Up), P(Down), Confidence
- Model performance: Test accuracy, CV accuracy
- Confusion matrix heatmap
- Feature coefficient bar chart

---

#### `predict_tomorrow()` function (Lines 109–116)

```python
def predict_tomorrow(model_result: dict, latest_row: pd.Series) -> dict:
```

**What it does:** Takes the trained model and the most recent row of data (today's features) and predicts whether tomorrow will be UP or DOWN.

**Code logic:**
- Line 110: `predict_proba()` returns [P(DOWN), P(UP)] for the latest data point.
- Lines 112–116: Picks the direction with higher probability and reports the confidence.

**Output (dict):**
```json
{
    "direction": "UP",
    "prob_up": 0.5823,
    "prob_down": 0.4177,
    "confidence": 58.2
}
```

**Dashboard location:** **Tab 5 — ML Predictions** (lines 379–383). Displayed as four metric cards at the top: Signal, P(Up), P(Down), Confidence.

---

#### `train_ridge_model()` function (Lines 123–152)

```python
def train_ridge_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
```

**Theory — Ridge Regression (L2 Regularization):**

Ridge regression predicts the **actual return value** (not just direction). It adds a penalty to the standard least-squares loss to prevent overfitting:

$$\min_\beta \sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j \beta_j^2$$

- The first term is the regular prediction error (MSE).
- The second term is the **L2 penalty** — it penalises large coefficients.
- **λ (alpha)** controls the trade-off: higher α = more shrinkage = simpler model.

**Key property of Ridge:** It shrinks all coefficients **toward zero** but never exactly to zero. Every feature stays in the model, just with reduced influence.

**Code walkthrough:**
- Lines 124–127: Target is tomorrow's return (`return.shift(-1)`). X is trimmed to match.
- Lines 129–131: Features are standardised with `StandardScaler()`.
- Lines 133–134: `RidgeCV` tests 100 alpha values (from 0.001 to 10000) using 5-fold time-series cross-validation to find the best alpha.
- Line 135: Final model trained with the best alpha.
- Lines 136–138: Evaluate with RMSE (Root Mean Squared Error) and R² (coefficient of determination).

**R² interpretation:** R² = 0.05 means the model explains 5% of the variance in returns. Stock returns are extremely hard to predict, so even small R² values are meaningful.

**Output (dict):**
```json
{
    "best_alpha": 10.0,
    "r2_score": 0.0234,
    "rmse": 0.015423,
    "coef_df": "<DataFrame of sorted coefficients>",
    "coef_path": "<2D array: 100 alphas × 12 features>",
    "alpha_path": "<array of 100 alpha values>",
    "interpretation": "Ridge (α=10.000) explains 2.3% of return variance. RMSE = 1.542%. Top feature: lag_1."
}
```

**Dashboard location:** **Tab 6 — Ridge & Lasso**, left column (lines 436–444). Shows coefficient bar chart, R², RMSE, and interpretation.

---

#### `train_lasso_model()` function (Lines 155–190)

```python
def train_lasso_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
```

**Theory — Lasso Regression (L1 Regularization):**

Lasso also adds a penalty but uses the **absolute values** of coefficients instead of squares:

$$\min_\beta \sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j |\beta_j|$$

**Key property of Lasso:** It can shrink coefficients **exactly to zero**, effectively performing **automatic feature selection**. Weak or redundant features get eliminated entirely.

**Ridge vs Lasso comparison:**
| Property | Ridge (L2) | Lasso (L1) |
|----------|-----------|-----------|
| Penalty shape | β² (smooth) | |β| (sharp corners) |
| Zeros out features? | No — shrinks all, keeps all | Yes — can eliminate features |
| Best when... | All features contribute a little | Only a few features matter |

**Code walkthrough:**
- Lines 156–161: Same setup as Ridge (target = tomorrow's return, StandardScaler).
- Lines 165–166: `LassoCV` searches 80 alpha values with time-series CV.
- Lines 167: Final Lasso model with best alpha.
- Line 171: Identifies which features have non-zero coefficients (`selected`).
- Lines 173: Extracts the names of selected features.

**Output (dict):**
```json
{
    "best_alpha": 0.00012,
    "r2_score": 0.0189,
    "rmse": 0.015567,
    "coef_df": "<DataFrame with 'selected' column>",
    "coef_path": "<2D array: 80 alphas × 12 features>",
    "alpha_path": "<array of 80 alpha values>",
    "n_selected": 5,
    "n_zeroed": 7,
    "selected_features": ["lag_1", "rolling_std_5", "rsi", "dist_ma20", "volume_ratio"],
    "interpretation": "Lasso (α=0.00012) selected 5 out of 12 features. R² = 1.9%. Key features: lag_1, rolling_std_5, rsi"
}
```

**Dashboard location:** **Tab 6 — Ridge & Lasso**, right column (lines 446–455). Shows coefficient bar chart (selected features in blue, zeroed in grey), count of selected/zeroed features, and feature names.

---

#### `compare_ridge_lasso()` function (Lines 197–214)

```python
def compare_ridge_lasso(ridge_result: dict, lasso_result: dict) -> pd.DataFrame:
```

**What it does:** Creates a side-by-side comparison table of the two models.

**Output (DataFrame displayed as table):**

| Metric | Ridge | Lasso |
|--------|-------|-------|
| R² Score | 0.0234 | 0.0189 |
| RMSE | 0.015423 | 0.015567 |
| Best Alpha | 10.0 | 0.00012 |
| Features Used | 12 | 5 |

**Dashboard location:** **Tab 6 — Ridge & Lasso** (line 433). Displayed as an interactive Streamlit dataframe at the top of the tab.

---

### 2.4 `streamlit_app.py` — Dashboard UI

**Purpose:** This is the main application file. It imports everything from the other three files, creates the sidebar, loads data, and builds the 6-tab interface. It does NOT contain any math or ML logic — only UI code.

---

#### Imports (Lines 1–24)

```python
from abish_stats import (confidence_interval_mean, mle_normal, one_way_anova,
    plot_likelihood_surface, t_test_one_sample, t_test_paired, t_test_two_sample, z_test_mean)
from data_loader import (engineer_features, get_daily_returns, get_price_data,
    get_sector_returns, get_stock_info, SECTORS)
from laksh_ml import (FEATURE_COLS, compare_ridge_lasso, predict_tomorrow,
    train_lasso_model, train_logistic_model, train_ridge_model)
```

This is where all four files connect. `streamlit_app.py` is a consumer of the other three files.

---

#### Display Helpers (Lines 36–68)

| Helper | Lines | Purpose |
|--------|-------|---------|
| `pct(value, digits)` | 36–37 | Formats a decimal as a percentage string: `0.0015` → `"0.15%"` |
| `show_result(result, details)` | 40–44 | Shows a green "Fail to reject" or red "Reject" banner + conclusion text. Used after every hypothesis test. |
| `line_chart(series, title, color)` | 47–51 | Creates a Plotly line chart from a pandas Series. Used for price history charts. |
| `coef_chart(df, title, x_title, selected)` | 54–61 | Creates a horizontal bar chart of model coefficients. Blue for positive (or selected), orange for negative (or grey for zeroed). |
| `show_theory(title, notes, formulas)` | 64–68 | Creates a collapsible "Theory and Formula" expander box with bullet points and LaTeX formulas. Used in every tab. |

---

#### Cached Data Loading (Lines 75–87)

```python
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker1, ticker2, period):
```

**What `@st.cache_data` does:** Caches the result for 1 hour (`ttl=3600` seconds). If you switch tabs or interact with sliders, Streamlit re-runs the entire script — but this decorator ensures the expensive Yahoo Finance downloads only happen once per hour per unique (ticker1, ticker2, period) combination.

**Returns a dictionary with 7 items:**

```python
{
    "prices1": pd.DataFrame,    # closing prices for Stock 1
    "prices2": pd.DataFrame,    # closing prices for Stock 2
    "ret1": pd.Series,          # daily returns for Stock 1
    "ret2": pd.Series,          # daily returns for Stock 2
    "sector_rets": dict,        # {sector_name: pd.Series} for all 5 sectors
    "info1": dict,              # {"name": "HDFC Bank Limited"}
    "df_ml": pd.DataFrame,      # full feature-engineered dataset for Stock 1
}
```

---

#### Sidebar (Lines 94–102)

```python
with st.sidebar:
    primary_sector = st.selectbox("Primary Sector", list(SECTORS))
    ticker1 = st.selectbox("Stock 1", SECTORS[primary_sector])
    compare_sector = st.selectbox("Compare Sector", [s for s in SECTORS if s != primary_sector])
    ticker2 = st.selectbox("Stock 2", SECTORS[compare_sector])
    period = st.select_slider("Data Period", PERIODS, value="2y")
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)
```

The sidebar has 6 inputs:
1. **Primary Sector** — dropdown from SECTORS keys
2. **Stock 1** — dropdown from the selected sector's tickers
3. **Compare Sector** — dropdown excluding the primary sector
4. **Stock 2** — dropdown from the comparison sector's tickers
5. **Data Period** — slider: 6mo, 1y, 2y, 3y, 5y
6. **Significance Level (α)** — slider: 0.01 to 0.10 (used by hypothesis tests and confidence intervals)

---

#### Tab Creation (Line 122)

```python
tabs = st.tabs(["Overview", "Risk Profile", "Hypothesis Tests", "Sector Analysis", "ML Predictions", "Ridge & Lasso"])
```

This single line creates all 6 tabs. Each tab is accessed via `tabs[0]` through `tabs[5]`.

---

## 3. Tab-by-Tab Walkthrough

### 3.1 Tab 1 — Overview
**Name:** "Overview"  
**Code Location:** `streamlit_app.py` lines 128–167  
**Functions Called:**
- `get_stock_info()` → company name (from `data_loader.py`)
- `get_price_data()` → price chart data (from `data_loader.py`)
- `get_daily_returns()` → return histogram data (from `data_loader.py`)

**What You See:**
1. **Theory expander** (lines 129–139): Explains daily returns and annualised return formulas
2. **Three metric cards** (lines 141–144): Company name, average daily return, annualised return
3. **Two line charts** (lines 146–148): Price history for Stock 1 (blue) and Stock 2 (orange)
4. **Overlaid histogram** (lines 150–167): Return distributions for both stocks

**Theory Applied:**
- Daily return: r_t = (P_t − P_{t−1}) / P_{t−1}
- Annualised return: approximately 252 × daily mean (252 trading days/year)

---

### 3.2 Tab 2 — Risk Profile
**Name:** "Risk Profile"  
**Code Location:** `streamlit_app.py` lines 173–212  
**Functions Called:**
- `confidence_interval_mean()` → CI dict (from `abish_stats.py`)
- `mle_normal()` → MLE dict (from `abish_stats.py`)
- `plot_likelihood_surface()` → likelihood curve data (from `abish_stats.py`)

**What You See:**
1. **Theory expander** (lines 174–187): SE, CI, and MLE formulas
2. **Left column — Confidence Interval** (lines 192–197): Mean, CI range, std error, method
3. **Right column — Maximum Likelihood** (lines 200–206): MLE μ, MLE σ, log-likelihood, AIC
4. **Likelihood surface chart** (lines 208–212): Curve with dashed line at MLE peak

---

### 3.3 Tab 3 — Hypothesis Tests
**Name:** "Hypothesis Tests"  
**Code Location:** `streamlit_app.py` lines 218–309  
**Functions Called (depending on radio selection):**
- `z_test_mean()` (from `abish_stats.py`) — lines 273
- `t_test_one_sample()` (from `abish_stats.py`) — line 282
- `t_test_two_sample()` (from `abish_stats.py`) — line 290
- `t_test_paired()` (from `abish_stats.py`) — line 305

**What You See:**
1. **Radio buttons** (line 220): Choose between 4 test types
2. **Theory expander**: Changes based on selected test (lines 222–269)
3. **Input widgets**: Z-Test has a "Known σ" input; One-Sample has a "Null Mean" input
4. **Metric cards**: Test statistic, critical value, p-value
5. **Result banner**: Green (fail to reject) or red (reject) with conclusion text
6. **Box plot** (Two-Sample only, lines 297–301): Side-by-side comparison of return distributions

---

### 3.4 Tab 4 — Sector Analysis
**Name:** "Sector Analysis"  
**Code Location:** `streamlit_app.py` lines 315–352  
**Functions Called:**
- `get_sector_returns()` → sector return dict (from `data_loader.py`, via `load_data`)
- `one_way_anova()` → ANOVA result dict (from `abish_stats.py`)

**What You See:**
1. **Theory expander** (lines 316–327): ANOVA hypothesis and F-statistic formula
2. **Metric cards** (lines 333–335): F-statistic and p-value
3. **Result banner** (line 336): Whether sector means differ significantly
4. **Descriptive statistics table** (lines 338–352): Mean, Std Dev, Skewness, Kurtosis, and Observations for each sector

---

### 3.5 Tab 5 — ML Predictions
**Name:** "ML Predictions"  
**Code Location:** `streamlit_app.py` lines 358–406  
**Functions Called:**
- `train_logistic_model()` → logistic result dict (from `laksh_ml.py`)
- `predict_tomorrow()` → prediction dict (from `laksh_ml.py`)

**What You See:**
1. **Theory expander** (lines 359–369): Sigmoid function, log-odds, coefficient interpretation
2. **Four metric cards** (lines 379–383): Signal (UP/DOWN), P(Up), P(Down), Confidence
3. **Left column — Performance** (lines 386–390): Test accuracy, CV accuracy ± std, interpretation
4. **Right column — Confusion Matrix** (lines 392–403): 2×2 heatmap (Actual vs Predicted, Down/Up)
5. **Coefficient bar chart** (lines 405–406): Horizontal bars showing feature importance

---

### 3.6 Tab 6 — Ridge & Lasso
**Name:** "Ridge & Lasso"  
**Code Location:** `streamlit_app.py` lines 412–478  
**Functions Called:**
- `train_ridge_model()` → Ridge result dict (from `laksh_ml.py`)
- `train_lasso_model()` → Lasso result dict (from `laksh_ml.py`)
- `compare_ridge_lasso()` → comparison DataFrame (from `laksh_ml.py`)

**What You See:**
1. **Theory expander** (lines 413–424): L2 and L1 penalty formulas, regularization explanation
2. **Comparison table** (line 433): Side-by-side R², RMSE, Alpha, Features Used
3. **Left column — Ridge** (lines 436–444): Coefficient chart, R², RMSE, interpretation
4. **Right column — Lasso** (lines 446–455): Coefficient chart (blue=selected, grey=zeroed), selected/zeroed count, feature names
5. **Regularization Path chart** (lines 457–478): Multi-line chart showing all 12 feature coefficients vs log10(alpha), with a dashed line at the chosen alpha

---

## 4. Data Flow Summary

```
User selects: Sector → Stock → Period → α
                  │
                  ▼
          load_data() [cached 1hr]
         ┌────────┼────────────────┐
         │        │                │
    data_loader   │           data_loader
   get_price_data │       engineer_features
   get_daily_returns       get_sector_returns
   get_stock_info │
         │        │                │
         ▼        ▼                ▼
    prices1/2   ret1/ret2       df_ml, sector_rets
    info1
         │        │          ┌─────┴──────┐
         │        │          │            │
         ▼        ▼          ▼            ▼
      Tab 1     Tab 2,3    Tab 5        Tab 6
    Overview   Risk/Tests  Logistic   Ridge/Lasso
                  │
                  ▼
              Tab 4
          Sector Analysis
```

**Which data goes where:**

| Data Variable | Created by | Used in Tabs |
|--------------|-----------|-------------|
| `prices1`, `prices2` | `data_loader.get_price_data()` | Tab 1 (price charts) |
| `ret1`, `ret2` | `data_loader.get_daily_returns()` | Tab 1 (histogram), Tab 2 (CI, MLE), Tab 3 (all tests) |
| `info1` | `data_loader.get_stock_info()` | Tab 1 (company name) |
| `sector_rets` | `data_loader.get_sector_returns()` | Tab 4 (ANOVA, descriptive stats) |
| `df_ml` | `data_loader.engineer_features()` | Tab 5 (logistic), Tab 6 (Ridge, Lasso) |

---

## 5. Complete Theory Reference

### 5.1 Daily Return
$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
Measures the one-day percentage change in stock price. A return of 0.01 means a 1% gain.

### 5.2 Standard Error
$$SE = \frac{s}{\sqrt{n}}$$
Measures uncertainty of the sample mean. Larger sample → smaller SE → more precise estimate.

### 5.3 Confidence Interval
$$CI = \bar{x} \pm c \cdot SE$$
Where c is Z_{α/2} (large sample) or t_{α/2, n-1} (small sample). A 95% CI means "if we repeated this analysis 100 times, about 95 intervals would contain the true mean."

### 5.4 Maximum Likelihood Estimation
$$\hat{\mu} = \bar{x}, \quad \hat{\sigma} = \sqrt{\frac{1}{n}\sum(x_i - \bar{x})^2}$$
$$\ell(\mu, \sigma) = \sum_{i=1}^{n} \log f(x_i \mid \mu, \sigma)$$
MLE finds the parameter values that make the observed data most probable under a Normal distribution model.

### 5.5 AIC (Akaike Information Criterion)
$$AIC = 2k - 2\ell$$
Where k = number of parameters (2 for Normal: μ and σ), ℓ = log-likelihood. Balances model fit against complexity. Lower = better.

### 5.6 Z-Test
$$z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$
Tests if a sample mean differs from a hypothesised value when σ is known. The Z-statistic follows a standard Normal distribution under H₀.

### 5.7 One-Sample t-Test
$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$
Same as Z-test but σ is unknown (estimated by s). Uses t-distribution with n−1 degrees of freedom. More conservative (wider intervals).

### 5.8 Two-Sample Welch's t-Test
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$
Compares two independent group means without assuming equal variances. Used here to compare Stock 1 vs Stock 2 returns.

### 5.9 Paired t-Test
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}, \quad d_i = x_{i,1} - x_{i,2}$$
Tests matched pairs (same stock, different time periods). Works on differences rather than raw values.

### 5.10 One-Way ANOVA
$$F = \frac{MS_{\text{between}}}{MS_{\text{within}}}$$
Compares k group means simultaneously. Large F → groups differ. Used to compare all 5 sector average returns at once.

### 5.11 RSI (Relative Strength Index)
$$RSI = 100 - \frac{100}{1 + RS}, \quad RS = \frac{\text{Avg Gain}_{14}}{\text{Avg Loss}_{14}}$$
Momentum oscillator (0–100). Above 70 = overbought (may fall). Below 30 = oversold (may rise).

### 5.12 Logistic Regression
$$P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$$
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta^T X$$
Predicts binary UP/DOWN direction. Positive coefficients push probability toward UP. Uses sigmoid function to constrain output to [0, 1].

### 5.13 Ridge Regression (L2)
$$\min_\beta \sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j \beta_j^2$$
Shrinks coefficients but keeps all features. Good when many features contribute small amounts. Higher λ = more shrinkage.

### 5.14 Lasso Regression (L1)
$$\min_\beta \sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j |\beta_j|$$
Can zero out coefficients entirely → automatic feature selection. Good when only a few features truly matter. The "sharp corners" of the L1 penalty geometry are what cause exact zeros.

### 5.15 R² (Coefficient of Determination)
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
Proportion of variance explained. R² = 0 means the model is no better than predicting the mean. R² = 1 means perfect prediction. For stock returns, even R² = 0.02–0.05 can be practically useful.

### 5.16 RMSE (Root Mean Squared Error)
$$RMSE = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$
Average prediction error in the same units as the target variable (daily return percentage). Lower = better.

### 5.17 Cross-Validation (Time Series)
`TimeSeriesSplit(n_splits=5)` creates 5 train/test splits where each split uses more training data and the test set always follows the training set chronologically. This prevents data leakage from the future.

### 5.18 StandardScaler
$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$
Transforms each feature to have mean=0 and std=1. Necessary because features like RSI (0–100) and daily returns (±0.05) have very different scales, and regularization penalises all coefficients equally.

---

*Document generated for the Equity Intelligence Dashboard project. All line numbers reference the current codebase.*
