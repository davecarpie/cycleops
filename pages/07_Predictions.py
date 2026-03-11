"""
Predictions - Streamlit App

Build predictive models for bike flows between two NTAs.
Choose between Linear Regression and Random Forest, then compare
a simple train/val/test split with TimeSeriesSplit cross-validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import backend as be
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

st.header("🔮 Bike Flow Predictions")

st.markdown(
    """
Predict daily bike flows between two neighborhoods using either
**Linear Regression** or **Random Forest**.  Each algorithm is evaluated with
two training strategies — a simple chronological split and time-series
cross-validation — so you can compare how each approach generalises.
"""
)


# ---------------------------------------------------------------------------
# Helper: build a model instance from the selected algorithm
# ---------------------------------------------------------------------------
def _make_model(model_type, n_estimators=100, max_depth=10):
    if model_type == "Random Forest":
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
        )
    return LinearRegression()


# ---------------------------------------------------------------------------
# Core training function (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def train_models(start_nta, end_nta, train_pct, val_pct, model_type, n_estimators=100, max_depth=10):
    """Train both split strategies for the chosen algorithm and return all results."""
    df_all = be.load_all_data()

    df_filtered = df_all[
        (df_all["start_NTA"] == start_nta)
        & (df_all["end_NTA"] == end_nta)
    ].copy()

    if len(df_filtered) == 0:
        return None

    df_filtered = df_filtered.sort_values("started_date").reset_index(drop=True)

    df_filtered["day_of_week"] = df_filtered["started_date"].dt.dayofweek
    df_filtered["month"] = df_filtered["started_date"].dt.month

    for i in range(1, 6):
        df_filtered[f"lag_{i}"] = df_filtered["ride_count"].shift(i)

    df_filtered = df_filtered.dropna().reset_index(drop=True)

    day_dummies = pd.get_dummies(
        df_filtered["day_of_week"], prefix="dow", drop_first=True
    )
    month_dummies = pd.get_dummies(
        df_filtered["month"], prefix="month", drop_first=True
    )
    df_filtered = pd.concat([df_filtered, day_dummies, month_dummies], axis=1)

    df_filtered["time_trend"] = np.arange(len(df_filtered))
    df_filtered["time_trend_poly"] = df_filtered["time_trend"] ** 2
    df_filtered["rolling_avg_30"] = df_filtered["ride_count"].rolling(
        window=30, min_periods=1
    ).mean()

    n_total = len(df_filtered)
    n_train = int(n_total * train_pct / 100)
    n_val = int(n_total * val_pct / 100)

    train_data = df_filtered.iloc[:n_train]
    val_data = df_filtered.iloc[n_train : n_train + n_val]
    pred_data = df_filtered.iloc[n_train + n_val :]

    day_cols = [col for col in df_filtered.columns if col.startswith("dow_")]
    month_cols = [col for col in df_filtered.columns if col.startswith("month_")]
    lag_cols = ["lag_1", "lag_2", "lag_3", "lag_4", "lag_5"]
    trend_cols = ["time_trend", "time_trend_poly", "rolling_avg_30"]
    feature_cols = day_cols + month_cols + lag_cols + trend_cols

    X_train = train_data[feature_cols]
    y_train = train_data["ride_count"]
    X_val = val_data[feature_cols]
    y_val = val_data["ride_count"]
    X_pred = pred_data[feature_cols]
    y_pred_actual = pred_data["ride_count"]

    # --- Strategy 1: Simple Split (train only → predict val & test) ---
    model1 = _make_model(model_type, n_estimators, max_depth)
    model1.fit(X_train, y_train)
    y_train_pred_m1 = model1.predict(X_train)
    y_val_pred_m1 = model1.predict(X_val)
    y_test_pred_m1 = model1.predict(X_pred)

    # --- Strategy 2: Time Series CV (on train+val) then final fit ---
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_trainval)):
        X_tr, X_te = X_trainval.iloc[train_idx], X_trainval.iloc[test_idx]
        y_tr, y_te = y_trainval.iloc[train_idx], y_trainval.iloc[test_idx]
        model_cv = _make_model(model_type, n_estimators, max_depth)
        model_cv.fit(X_tr, y_tr)
        pred = model_cv.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        r2 = r2_score(y_te, pred)
        cv_rows.append({"Fold": fold + 1, "MAE": mae, "RMSE": rmse, "R²": r2})

    model2 = _make_model(model_type, n_estimators, max_depth)
    model2.fit(X_trainval, y_trainval)
    y_test_pred_m2 = model2.predict(X_pred)

    return {
        "train_data": train_data,
        "val_data": val_data,
        "pred_data": pred_data,
        "df_filtered": df_filtered,
        "y_train": y_train,
        "y_val": y_val,
        "y_pred_actual": y_pred_actual,
        "y_train_pred_m1": y_train_pred_m1,
        "y_val_pred_m1": y_val_pred_m1,
        "y_test_pred_m1": y_test_pred_m1,
        "y_test_pred_m2": y_test_pred_m2,
        "model1": model1,
        "model2": model2,
        "cv_rows": cv_rows,
        "feature_cols": feature_cols,
        "model_type": model_type,
    }


def calculate_metrics(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Data Set": label, "MAE": f"{mae:.2f}", "RMSE": f"{rmse:.2f}", "R²": f"{r2:.3f}"}


# ---------------------------------------------------------------------------
# Sidebar-style config widgets
# ---------------------------------------------------------------------------
all_ntas = be.get_all_ntas()

st.subheader("Select Two Neighborhoods")
col1, col2 = st.columns(2)

with col1:
    start_nta = st.selectbox(
        "Origin NTA:",
        all_ntas,
        index=0,
        help="Select the starting neighborhood",
    )

with col2:
    end_nta = st.selectbox(
        "Destination NTA:",
        all_ntas,
        index=min(1, len(all_ntas) - 1),
        help="Select the destination neighborhood",
    )

st.subheader("Model Configuration")

model_type = st.radio(
    "Algorithm:",
    options=["Linear Regression", "Random Forest"],
    horizontal=True,
    help="Linear Regression fits a weighted sum of features. Random Forest builds an ensemble of decision trees.",
)

rf_n_estimators = 100
rf_max_depth = 10
if model_type == "Random Forest":
    with st.expander("Random Forest Settings"):
        rf_n_estimators = st.slider(
            "Number of trees", 10, 500, 100, 10,
            help="More trees generally improve accuracy but take longer to train.",
        )
        rf_max_depth = st.slider(
            "Max tree depth", 2, 30, 10, 1,
            help="Deeper trees capture more complex patterns but may overfit.",
        )

st.subheader("Data Split Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    train_pct = st.slider(
        "Training %",
        10,
        90,
        70,
        5,
        help="Percentage of data used to train the model",
    )
with col2:
    val_pct = st.slider(
        "Validation %",
        5,
        50,
        15,
        5,
        help="Percentage of data used for validation / cross-validation",
    )
with col3:
    test_pct = 100 - train_pct - val_pct
    st.metric("Test %", f"{test_pct}%")

if test_pct <= 0:
    st.error("Training + Validation percentages must be less than 100% to leave room for a test set.")
    st.stop()

if st.button("Train Models", type="primary"):
    with st.spinner("Training models..."):
        results = train_models(
            start_nta, end_nta, train_pct, val_pct,
            model_type, rf_n_estimators, rf_max_depth,
        )
        if results is None:
            st.error(f"No data found for flows from {start_nta} to {end_nta}")
        else:
            st.session_state.results = results
            st.success("Models trained successfully!")

if "results" in st.session_state:
    results = st.session_state.results

    train_data = results["train_data"]
    val_data = results["val_data"]
    pred_data = results["pred_data"]
    y_train = results["y_train"]
    y_val = results["y_val"]
    y_pred_actual = results["y_pred_actual"]
    y_train_pred_m1 = results["y_train_pred_m1"]
    y_val_pred_m1 = results["y_val_pred_m1"]
    y_test_pred_m1 = results["y_test_pred_m1"]
    y_test_pred_m2 = results["y_test_pred_m2"]
    model1 = results["model1"]
    model2 = results["model2"]
    cv_rows = results["cv_rows"]
    feature_cols = results["feature_cols"]
    trained_model_type = results["model_type"]

    st.subheader("📊 Data Summary")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.metric("Algorithm", trained_model_type)
    with col2:
        st.metric("Training Set", f"{len(train_data)} days")
    with col3:
        st.metric("Validation Set", f"{len(val_data)} days")
    with col4:
        st.metric("Test Set", f"{len(pred_data)} days")

    # Readable feature names for display
    day_names = {
        "dow_1": "Tuesday",
        "dow_2": "Wednesday",
        "dow_3": "Thursday",
        "dow_4": "Friday",
        "dow_5": "Saturday",
        "dow_6": "Sunday",
    }
    month_names = {
        f"month_{m}": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ][m - 1]
        for m in range(2, 13)
    }

    readable_names = []
    for col in feature_cols:
        if col in day_names:
            readable_names.append(f"Day: {day_names[col]}")
        elif col in month_names:
            readable_names.append(f"Month: {month_names[col]}")
        elif col.startswith("lag_"):
            days = col.split("_")[1]
            readable_names.append(f"Lag {days} days")
        elif col == "time_trend":
            readable_names.append("Linear Trend")
        elif col == "time_trend_poly":
            readable_names.append("Polynomial Trend")
        elif col == "rolling_avg_30":
            readable_names.append("30-Day Moving Avg")
        else:
            readable_names.append(col)

    # --- Single-date prediction ---
    st.subheader("🔎 Predict for a Specific Date")
    df_filtered = results["df_filtered"]
    all_dates = df_filtered["started_date"].dt.date
    min_date, max_date = all_dates.min(), all_dates.max()

    pred_date = st.date_input(
        "Select a date:",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help="Pick a date within the dataset range to see the model's prediction.",
    )

    pred_date_ts = pd.Timestamp(pred_date)
    match = df_filtered[df_filtered["started_date"].dt.date == pred_date]

    if len(match) == 0:
        st.warning(f"No data for {pred_date}. Choose a date with recorded rides.")
    else:
        row = match.iloc[0]
        X_single = row[feature_cols].values.reshape(1, -1)
        pred_m1 = model1.predict(X_single)[0]
        pred_m2 = model2.predict(X_single)[0]
        actual = row["ride_count"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Actual Rides", f"{actual:,.0f}")
        with col2:
            st.metric("Simple Split Prediction", f"{pred_m1:,.0f}",
                      delta=f"{pred_m1 - actual:+,.0f} error")
        with col3:
            st.metric("Time Series CV Prediction", f"{pred_m2:,.0f}",
                      delta=f"{pred_m2 - actual:+,.0f} error")

        with st.expander("Feature values used for this prediction"):
            feat_vals = pd.DataFrame({
                "Feature": readable_names,
                "Value": [f"{v:.4f}" for v in row[feature_cols].values],
            })
            st.dataframe(feat_vals, use_container_width=True, hide_index=True)

    st.subheader("� Model Comparison")
    def _get_feature_df(model, readable_names, trained_model_type, prefix):
        is_rf = trained_model_type == "Random Forest"
        value_col = "Importance" if is_rf else "Coefficient"
        values = model.feature_importances_ if is_rf else model.coef_
        full_df = pd.DataFrame({"Feature": readable_names, value_col: values})
        filtered_df = full_df[full_df["Feature"].str.startswith(prefix)].sort_values(
            value_col, ascending=False
        ).reset_index(drop=True)
        return filtered_df, value_col

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Simple Split** — trained on training set only")
        metrics_m1 = [
            calculate_metrics(y_train, y_train_pred_m1, "Train"),
            calculate_metrics(y_val, y_val_pred_m1, "Validation"),
            calculate_metrics(y_pred_actual, y_test_pred_m1, "Test"),
        ]
        st.dataframe(pd.DataFrame(metrics_m1), use_container_width=True, hide_index=True)

    with col2:
        st.write("**Time Series CV** — 5-fold cross-validation on train+val")
        cv_metrics_df = pd.DataFrame(cv_rows)
        for col in ["MAE", "RMSE", "R²"]:
            cv_metrics_df[col] = pd.to_numeric(cv_metrics_df[col], errors="coerce")
        avg_cv_metrics = {
            "Average MAE": cv_metrics_df["MAE"].mean(),
            "Average RMSE": cv_metrics_df["RMSE"].mean(),
            "Average R²": cv_metrics_df["R²"].mean(),
        }
        st.dataframe(cv_metrics_df, use_container_width=True, hide_index=True)
        metrics_m2 = [
            {
                "Data Set": "CV Average",
                "MAE": f"{avg_cv_metrics['Average MAE']:.2f}",
                "RMSE": f"{avg_cv_metrics['Average RMSE']:.2f}",
                "R²": f"{avg_cv_metrics['Average R²']:.3f}",
            },
            calculate_metrics(y_pred_actual, y_test_pred_m2, "Final Test"),
        ]
        st.dataframe(pd.DataFrame(metrics_m2), use_container_width=True, hide_index=True)

    # --- Lag coefficients / importances (shared header, side by side) ---
    is_rf = trained_model_type == "Random Forest"
    if is_rf:
        st.subheader("📅 Lag Feature Importances")
        st.caption(
            "Each score (0–1) shows how much a lag feature helps reduce prediction "
            "error across all trees. Higher = more influential."
        )
    else:
        st.subheader("📅 Lag Coefficients")
        st.caption(
            "Each coefficient shows how much yesterday's (or earlier days') ride count "
            "contributes to today's prediction. For example, a coefficient of 0.5 on "
            "Lag 1 means that for every additional ride yesterday, today's prediction "
            "increases by 0.5 rides."
        )

    lag_df_m1, value_col = _get_feature_df(model1, readable_names, trained_model_type, "Lag ")
    lag_df_m2, _ = _get_feature_df(model2, readable_names, trained_model_type, "Lag ")
    lag_height = (max(len(lag_df_m1), len(lag_df_m2)) + 1) * 35 + 3

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Simple Split**")
        st.dataframe(lag_df_m1, use_container_width=True, height=lag_height)
    with col2:
        st.write("**Time Series CV**")
        st.dataframe(lag_df_m2, use_container_width=True, height=lag_height)

    # --- Day of Week coefficients / importances ---
    if is_rf:
        st.subheader("📆 Day of Week Feature Importances")
        st.caption(
            "Each score (0–1) shows how much a day-of-week feature helps reduce prediction "
            "error across all trees. Higher = more influential."
        )
    else:
        st.subheader("📆 Day of Week Coefficients")
        st.caption(
            "Each coefficient shows the change in predicted rides relative to "
            "**Monday** (the reference day). Positive = more rides than Monday, "
            "negative = fewer."
        )

    dow_df_m1, _ = _get_feature_df(model1, readable_names, trained_model_type, "Day: ")
    dow_df_m2, _ = _get_feature_df(model2, readable_names, trained_model_type, "Day: ")
    dow_height = (max(len(dow_df_m1), len(dow_df_m2)) + 1) * 35 + 3

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Simple Split**")
        st.dataframe(dow_df_m1, use_container_width=True, height=dow_height)
        if not is_rf:
            st.caption(
                f"**Intercept: {model1.intercept_:.2f}** — baseline prediction "
                f"when all features are zero."
            )
    with col2:
        st.write("**Time Series CV**")
        st.dataframe(dow_df_m2, use_container_width=True, height=dow_height)
        if not is_rf:
            st.caption(
                f"**Intercept: {model2.intercept_:.2f}** — baseline prediction "
                f"when all features are zero."
            )

else:
    st.info("👆 Select two NTAs, choose an algorithm, and click **Train Models** to begin.")