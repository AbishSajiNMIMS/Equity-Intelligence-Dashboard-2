"""
Main Streamlit dashboard for the equity project.
Each tab shows one part of the workflow:
descriptive analysis, inference, sector comparison, and ML.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from abish_stats import (
    confidence_interval_mean,
    mle_normal,
    one_way_anova,
    plot_likelihood_surface,
    t_test_one_sample,
    t_test_paired,
    t_test_two_sample,
    z_test_mean,
)
from data_loader import engineer_features, get_daily_returns, get_price_data, get_sector_returns, get_stock_info, SECTORS
from laksh_ml import FEATURE_COLS, compare_ridge_lasso, predict_tomorrow, train_lasso_model, train_logistic_model, train_ridge_model

st.set_page_config(page_title="Equity Intelligence Dashboard", page_icon="📈", layout="wide")

COLORS = {"primary": "#1f77b4", "secondary": "#ff7f0e"}
PERIODS = ["6mo", "1y", "2y", "3y", "5y"]


# -----------------------------------------------------------------------------
# Small display helpers keep repeated formatting in one place.
# The goal is to keep the app plain and readable without custom HTML blocks.
# -----------------------------------------------------------------------------
def pct(value, digits=2):
    return f"{value * 100:.{digits}f}%"


def show_result(result, details=None):
    (st.error if result["reject_H0"] else st.success)("Reject H0" if result["reject_H0"] else "Fail to reject H0")
    for line in details or []:
        st.write(line)
    st.caption(result["conclusion"])


def line_chart(series, title, color):
    fig = px.line(series.rename_axis("Date").reset_index(), x="Date", y=series.name, title=title)
    fig.update_traces(line_color=color)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def coef_chart(df, title, x_title, selected=False):
    if selected:
        colors = ["#1f77b4" if keep else "#bdbdbd" for keep in df["selected"]]
    else:
        colors = ["#1f77b4" if coef > 0 else "#ff7f0e" for coef in df["coefficient"]]
    fig = go.Figure(go.Bar(x=df["coefficient"], y=df["feature"], orientation="h", marker_color=colors))
    fig.update_layout(title=title, xaxis_title=x_title, height=340, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def show_theory(title, notes, formulas=()):
    with st.expander(f"Theory and Formula — {title}", expanded=False):
        st.markdown("\n".join(f"- {note}" for note in notes))
        for formula in formulas:
            st.latex(formula)


# -----------------------------------------------------------------------------
# Cached loading keeps the dashboard responsive.
# The returned dictionary is the shared data source for every tab below.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker1, ticker2, period):
    prices1 = get_price_data([ticker1], period)
    prices2 = get_price_data([ticker2], period)
    return {
        "prices1": prices1,
        "prices2": prices2,
        "ret1": get_daily_returns(prices1)[ticker1].dropna(),
        "ret2": get_daily_returns(prices2)[ticker2].dropna(),
        "sector_rets": get_sector_returns(period),
        "info1": get_stock_info(ticker1),
        "df_ml": engineer_features(ticker1, period),
    }


# -----------------------------------------------------------------------------
# Sidebar inputs define the whole dashboard state:
# selected stocks, analysis window, and significance level.
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Equity Intelligence")
    st.caption("Simple statistical and ML dashboard")
    primary_sector = st.selectbox("Primary Sector", list(SECTORS))
    ticker1 = st.selectbox("Stock 1", SECTORS[primary_sector])
    compare_sector = st.selectbox("Compare Sector", [sector for sector in SECTORS if sector != primary_sector])
    ticker2 = st.selectbox("Stock 2", SECTORS[compare_sector])
    period = st.select_slider("Data Period", PERIODS, value="2y")
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)

with st.spinner("Fetching market data..."):
    try:
        data = load_data(ticker1, ticker2, period)
    except Exception as exc:
        st.error(f"Data fetch error: {exc}")
        st.stop()

prices1 = data["prices1"]
prices2 = data["prices2"]
ret1 = data["ret1"]
ret2 = data["ret2"]
sector_rets = data["sector_rets"]
info1 = data["info1"]
df_ml = data["df_ml"]

st.title("Equity Intelligence Dashboard")
st.caption(f"{ticker1} vs {ticker2} | Period: {period}")

tabs = st.tabs(["Overview", "Risk Profile", "Hypothesis Tests", "Sector Analysis", "ML Predictions", "Ridge & Lasso"])

# -----------------------------------------------------------------------------
# Tab 1 gives a quick descriptive view of the two stocks.
# It uses prices, simple returns, and a histogram for raw distribution shape.
# -----------------------------------------------------------------------------
with tabs[0]:
    show_theory(
        "Overview",
        [
            "Price history shows the raw time path of each stock over the chosen period.",
            "Daily return measures one-day percentage change and the histogram shows spread, skew, and outliers.",
            "Annualised return is a simple scaling of average daily return by roughly 252 trading days.",
        ],
        [
            r"r_t = \frac{P_t - P_{t-1}}{P_{t-1}}",
            r"\bar r_{\mathrm{annual}} \approx 252 \times \bar r_{\mathrm{daily}}",
        ],
    )
    company, avg_return, annualised = st.columns(3)
    company.metric("Company", info1.get("name", ticker1)[:24])
    avg_return.metric("Avg Daily Return", pct(ret1.mean(), 3))
    annualised.metric("Annualised Return", pct(ret1.mean() * 252, 1))

    left, right = st.columns(2)
    left.plotly_chart(line_chart(prices1[ticker1], f"Price History — {ticker1}", COLORS["primary"]), use_container_width=True)
    right.plotly_chart(line_chart(prices2[ticker2], f"Price History — {ticker2}", COLORS["secondary"]), use_container_width=True)

    st.subheader("Return Distribution Comparison")
    dist = pd.concat(
        [
            pd.DataFrame({"Return": ret1, "Stock": ticker1}),
            pd.DataFrame({"Return": ret2, "Stock": ticker2}),
        ]
    )
    fig = px.histogram(
        dist,
        x="Return",
        color="Stock",
        nbins=60,
        barmode="overlay",
        opacity=0.65,
        color_discrete_map={ticker1: COLORS["primary"], ticker2: COLORS["secondary"]},
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 2 focuses on estimation.
# It shows a confidence interval for the mean and normal MLE estimates.
# -----------------------------------------------------------------------------
with tabs[1]:
    show_theory(
        "Risk Profile",
        [
            "The confidence interval gives a plausible range for the true mean daily return.",
            "Standard error shrinks as the sample gets larger, so the interval becomes tighter.",
            "Under a normal model, MLE estimates the mean and standard deviation that best fit the observed returns.",
        ],
        [
            r"SE = \frac{s}{\sqrt{n}}",
            r"CI = \bar{x} \pm c \cdot SE",
            r"\hat{\mu} = \bar{x}, \qquad \hat{\sigma} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2}",
            r"\ell(\mu,\sigma) = \sum_{i=1}^{n}\log f(x_i \mid \mu,\sigma)",
        ],
    )
    ci = confidence_interval_mean(ret1.values, confidence=1 - alpha)
    mle = mle_normal(ret1.values)

    left, right = st.columns(2)
    with left:
        st.subheader("Confidence Interval")
        st.metric("Mean Daily Return", pct(ci["mean"], 4))
        st.write(f"{int((1 - alpha) * 100)}% CI: {pct(ci['lower'], 4)} to {pct(ci['upper'], 4)}")
        st.write(f"Std Error: {pct(ci['std_error'], 5)}")
        st.caption(f"{ci['method']} | n = {ci['n']}")

    with right:
        st.subheader("Maximum Likelihood")
        c1, c2 = st.columns(2)
        c1.metric("MLE μ", pct(mle["mu_mle"], 4))
        c2.metric("MLE σ", pct(mle["sigma_mle"], 4))
        st.write(f"Log-Likelihood: {mle['log_likelihood']}")
        st.write(f"AIC: {mle['aic']}")
        st.caption(mle["interpretation"])

    ll = plot_likelihood_surface(ret1.values)
    ll_fig = px.line(x=ll["mu_grid"] * 100, y=ll["log_likelihoods"], labels={"x": "μ (%)", "y": "Log-Likelihood"}, title="Likelihood Surface")
    ll_fig.add_vline(x=ll["mle_mu"] * 100, line_dash="dash", line_color=COLORS["secondary"])
    ll_fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(ll_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 3 holds the hypothesis tests.
# The displayed formulas change slightly with the selected test.
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Hypothesis Tests")
    test = st.radio("Choose a test", ["Z-Test", "One-Sample t-Test", "Two-Sample t-Test", "Paired t-Test"], horizontal=True)

    if test == "Z-Test":
        show_theory(
            "Z-Test",
            [
                "Use this when the population standard deviation is treated as known.",
                "The test checks whether the sample mean is statistically different from the null value.",
            ],
            [
                r"H_0:\mu=\mu_0",
                r"z = \frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}",
            ],
        )
    elif test == "One-Sample t-Test":
        show_theory(
            "One-Sample t-Test",
            [
                "Use this when the population standard deviation is unknown and estimated from the sample.",
                "The t-distribution adds extra uncertainty through degrees of freedom.",
            ],
            [
                r"H_0:\mu=\mu_0",
                r"t = \frac{\bar{x}-\mu_0}{s/\sqrt{n}}",
            ],
        )
    elif test == "Two-Sample t-Test":
        show_theory(
            "Two-Sample t-Test",
            [
                "This compares the mean returns of two stocks.",
                "The Welch version does not assume equal variance in the two samples.",
            ],
            [
                r"H_0:\mu_1=\mu_2",
                r"t = \frac{\bar{x}_1-\bar{x}_2}{\sqrt{s_1^2/n_1+s_2^2/n_2}}",
            ],
        )
    else:
        show_theory(
            "Paired t-Test",
            [
                "A paired test works on matched differences instead of two independent samples.",
                "Here the app compares the first half and second half of the same stock's return series.",
            ],
            [
                r"d_i = x_{i,\mathrm{first}} - x_{i,\mathrm{second}}",
                r"t = \frac{\bar d}{s_d/\sqrt{n}}",
            ],
        )

    if test == "Z-Test":
        pop_std = st.number_input("Known Population Std Dev (%)", value=1.0, step=0.1) / 100
        result = z_test_mean(ret1.values, 0.0, pop_std, alpha=alpha)
        a, b, c = st.columns(3)
        a.metric("Z-Statistic", result["z_statistic"])
        b.metric("Z-Critical", f"±{result['z_critical']}")
        c.metric("p-value", result["p_value"])
        show_result(result, ["H0: μ = 0.000%", "H1: μ ≠ 0.000%"])

    elif test == "One-Sample t-Test":
        null_mean = st.number_input("Null Hypothesis Mean (%)", value=0.0, step=0.01) / 100
        result = t_test_one_sample(ret1.values, null_mean, alpha=alpha)
        a, b, c = st.columns(3)
        a.metric("t-Statistic", result["t_statistic"])
        b.metric("t-Critical", f"±{result['t_critical']}")
        c.metric("p-value", result["p_value"])
        show_result(result)

    elif test == "Two-Sample t-Test":
        result = t_test_two_sample(ret1.values, ret2.values, ticker1, ticker2, alpha)
        a, b, c = st.columns(3)
        a.metric(f"Mean {ticker1}", pct(result["mean1"], 4))
        b.metric(f"Mean {ticker2}", pct(result["mean2"], 4))
        c.metric("p-value", result["p_value"])
        show_result(result, [f"Test: {result['test']}", f"t-Statistic: {result['t_statistic']}"])

        box = go.Figure()
        box.add_trace(go.Box(y=ret1 * 100, name=ticker1, marker_color=COLORS["primary"]))
        box.add_trace(go.Box(y=ret2 * 100, name=ticker2, marker_color=COLORS["secondary"]))
        box.update_layout(yaxis_title="Daily Return (%)", height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(box, use_container_width=True)

    else:
        mid = min(len(ret1), len(ret2)) // 2
        result = t_test_paired(ret1.values[:mid], ret1.values[mid : 2 * mid], "First Half", "Second Half", alpha)
        a, b = st.columns(2)
        a.metric("t-Statistic", result["t_statistic"])
        b.metric("p-value", result["p_value"])
        show_result(result, [f"Comparing first half vs second half of {ticker1} returns"])

# -----------------------------------------------------------------------------
# Tab 4 compares sectors at a higher level.
# ANOVA asks whether all sector means can reasonably be treated as equal.
# -----------------------------------------------------------------------------
with tabs[3]:
    show_theory(
        "Sector Analysis",
        [
            "One-way ANOVA compares average returns across multiple sectors at once.",
            "If the p-value is small, at least one sector mean likely differs from the others.",
            "The descriptive table gives supporting shape information through standard deviation, skewness, and kurtosis.",
        ],
        [
            r"H_0:\mu_1=\mu_2=\cdots=\mu_k",
            r"F = \frac{MS_{\mathrm{between}}}{MS_{\mathrm{within}}}",
        ],
    )
    labels = list(sector_rets)
    groups = [sector_rets[label].dropna().to_numpy() for label in labels]
    result = one_way_anova(*groups, labels=labels, alpha=alpha)

    st.subheader("One-Way ANOVA")
    a, b = st.columns(2)
    a.metric("F-Statistic", result["f_statistic"])
    b.metric("p-value", result["p_value"])
    show_result(result)

    st.subheader("Descriptive Statistics")
    stats_df = pd.DataFrame(
        [
            {
                "Sector": label,
                "Mean (%)": round(group.mean() * 100, 4),
                "Std Dev (%)": round(group.std(ddof=1) * 100, 4),
                "Skewness": round(pd.Series(group).skew(), 4),
                "Kurtosis": round(pd.Series(group).kurtosis(), 4),
                "Observations": len(group),
            }
            for label, group in zip(labels, groups)
        ]
    )
    st.dataframe(stats_df, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 5 uses a logistic classifier for next-day direction.
# The theory box links probabilities, coefficients, and confusion matrix output.
# -----------------------------------------------------------------------------
with tabs[4]:
    show_theory(
        "ML Predictions",
        [
            "Logistic regression models the probability of an up move on the next day.",
            "Coefficients are read in log-odds terms: positive values push the probability of UP higher, negative values push it lower.",
            "The confusion matrix shows how many up and down cases were classified correctly or incorrectly.",
        ],
        [
            r"P(Y=1\mid X)=\frac{1}{1+e^{-(\beta_0+\beta^T X)}}",
            r"\log\left(\frac{p}{1-p}\right)=\beta_0+\beta^T X",
        ],
    )
    st.subheader("Logistic Regression")
    if len(df_ml) < 100:
        st.warning("Not enough data for ML training. Try a longer period.")
    else:
        with st.spinner("Training logistic model..."):
            log_res = train_logistic_model(df_ml)
        pred = predict_tomorrow(log_res, df_ml.iloc[-1])

        a, b, c, d = st.columns(4)
        a.metric("Signal", pred["direction"])
        b.metric("P(Up)", f"{pred['prob_up'] * 100:.1f}%")
        c.metric("P(Down)", f"{pred['prob_down'] * 100:.1f}%")
        d.metric("Confidence", f"{pred['confidence']}%")

        left, right = st.columns(2)
        with left:
            st.subheader("Model Performance")
            st.write(f"Test Accuracy: {log_res['accuracy'] * 100:.1f}%")
            st.write(f"CV Accuracy: {log_res['cv_accuracy_mean'] * 100:.1f}% ± {log_res['cv_accuracy_std'] * 100:.1f}%")
            st.caption(log_res["interpretation"])

        with right:
            st.subheader("Confusion Matrix")
            cm = px.imshow(
                log_res["confusion_matrix"],
                x=["Down", "Up"],
                y=["Down", "Up"],
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                text_auto=True,
                color_continuous_scale="Blues",
            )
            cm.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(cm, use_container_width=True)

        st.subheader("Feature Coefficients")
        st.plotly_chart(coef_chart(log_res["coef_df"], "Logistic Coefficients", "Coefficient"), use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 6 shows two regularized regression models.
# Ridge shrinks all coefficients, while Lasso can shrink some exactly to zero.
# -----------------------------------------------------------------------------
with tabs[5]:
    show_theory(
        "Ridge and Lasso",
        [
            "Ridge uses an L2 penalty, so it shrinks coefficients smoothly but usually keeps every feature in the model.",
            "Lasso uses an L1 penalty, so it can force weak coefficients exactly to zero and act like feature selection.",
            "The regularization path shows how coefficients change as alpha grows stronger.",
        ],
        [
            r"\min_\beta \sum_i (y_i-\hat y_i)^2 + \lambda \sum_j \beta_j^2",
            r"\min_\beta \sum_i (y_i-\hat y_i)^2 + \lambda \sum_j |\beta_j|",
        ],
    )
    st.subheader("Ridge and Lasso")
    if len(df_ml) < 100:
        st.warning("Not enough data for regularized models. Try a longer period.")
    else:
        with st.spinner("Training regularized models..."):
            ridge_res = train_ridge_model(df_ml)
            lasso_res = train_lasso_model(df_ml)

        st.dataframe(compare_ridge_lasso(ridge_res, lasso_res), use_container_width=True)
        left, right = st.columns(2)

        with left:
            st.subheader("Ridge")
            st.plotly_chart(
                coef_chart(ridge_res["coef_df"], "Ridge Coefficients", f"Coefficient (α={ridge_res['best_alpha']})"),
                use_container_width=True,
            )
            st.write(f"R²: {ridge_res['r2_score']}")
            st.write(f"RMSE: {pct(ridge_res['rmse'], 4)}")
            st.caption(ridge_res["interpretation"])

        with right:
            st.subheader("Lasso")
            st.plotly_chart(
                coef_chart(lasso_res["coef_df"], "Lasso Coefficients", f"Coefficient (α={lasso_res['best_alpha']:.5f})", selected=True),
                use_container_width=True,
            )
            st.write(f"Selected: {lasso_res['n_selected']}")
            st.write(f"Zeroed Out: {lasso_res['n_zeroed']}")
            st.write(f"Kept: {', '.join(lasso_res['selected_features']) or 'None'}")
            st.caption(lasso_res["interpretation"])

        st.subheader("Lasso Regularization Path")
        path = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, feature in enumerate(FEATURE_COLS):
            path.add_trace(
                go.Scatter(
                    x=np.log10(lasso_res["alpha_path"]),
                    y=lasso_res["coef_path"][:, i],
                    mode="lines",
                    name=feature,
                    line=dict(color=colors[i % len(colors)]),
                )
            )
        path.add_vline(x=np.log10(lasso_res["best_alpha"]), line_dash="dash", line_color=COLORS["secondary"])
        path.update_layout(
            title="Regularization Path",
            xaxis_title="log10(alpha)",
            yaxis_title="Coefficient Value",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(path, use_container_width=True)
