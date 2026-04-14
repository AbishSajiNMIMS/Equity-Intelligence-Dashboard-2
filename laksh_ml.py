"""
Machine-learning helpers for the dashboard.
The functions here train compact models and return only the
pieces that the Streamlit app actually displays.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Shared feature list:
# these engineered variables feed both the classifier and the regressors.
# -----------------------------------------------------------------------------
FEATURE_COLS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_5",
    "rolling_mean_5",
    "rolling_std_5",
    "rolling_mean_20",
    "rolling_std_20",
    "rsi",
    "volume_ratio",
    "dist_ma20",
    "dist_ma50",
]


# -----------------------------------------------------------------------------
# Small reusable helpers:
# one splits time-series data, one formats coefficient tables,
# and one builds coefficient paths for the regularization chart.
# -----------------------------------------------------------------------------
def _split_series(X, y, test_size=0.2):
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]


def _coef_frame(coefs, selected=None):
    data = {
        "feature": FEATURE_COLS,
        "coefficient": coefs,
        "abs_coef": np.abs(coefs),
    }
    if selected is not None:
        data["selected"] = selected
    return pd.DataFrame(data).sort_values("abs_coef", ascending=False)


def _coef_path(model_cls, alphas, X, y, **kwargs):
    return np.array([model_cls(alpha=alpha, **kwargs).fit(X, y).coef_ for alpha in alphas])


# -----------------------------------------------------------------------------
# Logistic regression section:
# this powers the direction-prediction tab and the next-day signal widget.
# -----------------------------------------------------------------------------
def train_logistic_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    X = df[FEATURE_COLS].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = _split_series(X, y, test_size)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5), scoring="accuracy")
    coef_df = _coef_frame(model.named_steps["model"].coef_[0])

    return {
        "model": model,
        "accuracy": round(accuracy, 4),
        "cv_accuracy_mean": round(cv_acc.mean(), 4),
        "cv_accuracy_std": round(cv_acc.std(), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "coef_df": coef_df,
        "interpretation": (
            f"The model achieves {accuracy * 100:.1f}% accuracy. "
            f"Top signal: {coef_df.iloc[0]['feature']} "
            f"(coef = {coef_df.iloc[0]['coefficient']:.4f})"
        ),
    }


def predict_tomorrow(model_result: dict, latest_row: pd.Series) -> dict:
    probs = model_result["model"].predict_proba(latest_row[FEATURE_COLS].to_numpy().reshape(1, -1))[0]
    return {
        "direction": "UP" if probs[1] > 0.5 else "DOWN",
        "prob_up": round(probs[1], 4),
        "prob_down": round(probs[0], 4),
        "confidence": round(max(probs) * 100, 1),
    }


# -----------------------------------------------------------------------------
# Regularized regression section:
# Ridge keeps all variables with shrinkage, while Lasso can zero some out.
# -----------------------------------------------------------------------------
def train_ridge_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    X = df[FEATURE_COLS].values
    y = df["return"].shift(-1).dropna().to_numpy()
    X = X[: len(y)]
    X_train, X_test, y_train, y_test = _split_series(X, y, test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    alphas = np.logspace(-3, 4, 100)
    best_alpha = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5)).fit(X_train, y_train).alpha_
    model = Ridge(alpha=best_alpha).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    coef_df = _coef_frame(model.coef_)

    return {
        "best_alpha": round(best_alpha, 4),
        "r2_score": round(r2, 4),
        "rmse": round(rmse, 6),
        "coef_df": coef_df,
        "coef_path": _coef_path(Ridge, alphas, X_train, y_train),
        "alpha_path": alphas,
        "interpretation": (
            f"Ridge (α={best_alpha:.3f}) explains {r2 * 100:.1f}% of return variance. "
            f"RMSE = {rmse * 100:.3f}%. Top feature: {coef_df.iloc[0]['feature']}."
        ),
    }


def train_lasso_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    X = df[FEATURE_COLS].values
    y = df["return"].shift(-1).dropna().to_numpy()
    X = X[: len(y)]
    X_train, X_test, y_train, y_test = _split_series(X, y, test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    alphas = np.logspace(-6, 2, 80)
    best_alpha = LassoCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5), max_iter=5000).fit(X_train, y_train).alpha_
    model = Lasso(alpha=best_alpha, max_iter=5000).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    selected = model.coef_ != 0
    coef_df = _coef_frame(model.coef_, selected)
    chosen = coef_df.loc[coef_df["selected"], "feature"].tolist()

    return {
        "best_alpha": round(best_alpha, 6),
        "r2_score": round(r2, 4),
        "rmse": round(rmse, 6),
        "coef_df": coef_df,
        "coef_path": _coef_path(Lasso, alphas, X_train, y_train, max_iter=5000),
        "alpha_path": alphas,
        "n_selected": int(selected.sum()),
        "n_zeroed": int((~selected).sum()),
        "selected_features": chosen,
        "interpretation": (
            f"Lasso (α={best_alpha:.5f}) selected {int(selected.sum())} out of "
            f"{len(FEATURE_COLS)} features. R² = {r2 * 100:.1f}%. "
            f"Key features: {', '.join(chosen[:3]) or 'None'}"
        ),
    }


# -----------------------------------------------------------------------------
# Comparison table:
# this turns the two model outputs into one simple dataframe for the app.
# -----------------------------------------------------------------------------
def compare_ridge_lasso(ridge_result: dict, lasso_result: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": ["R² Score", "RMSE", "Best Alpha", "Features Used"],
            "Ridge": [
                ridge_result["r2_score"],
                ridge_result["rmse"],
                ridge_result["best_alpha"],
                len(FEATURE_COLS),
            ],
            "Lasso": [
                lasso_result["r2_score"],
                lasso_result["rmse"],
                lasso_result["best_alpha"],
                lasso_result["n_selected"],
            ],
        }
    )
