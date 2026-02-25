"""
Predictions - Streamlit App

Build linear regression predictions for bike flows between two NTAs.
Compares simple train/val/test split with TimeSeriesSplit cross-validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import backend as be
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

st.header("🔮 Bike Flow Predictions")

st.markdown(
    """
Build a simple linear regression model to predict bike flows between two neighborhoods.
The model uses day of week, month, and the previous 5 days of data as features.
"""
)


@st.cache_data
def train_models(start_nta, end_nta, train_pct, val_pct):
    """Train both models and return all results."""
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

    model1 = LinearRegression()
    model1.fit(X_train, y_train)
    y_train_pred_m1 = model1.predict(X_train)
    y_val_pred_m1 = model1.predict(X_val)
    y_test_pred_m1 = model1.predict(X_pred)

    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_trainval)):
        X_tr, X_te = X_trainval.iloc[train_idx], X_trainval.iloc[test_idx]
        y_tr, y_te = y_trainval.iloc[train_idx], y_trainval.iloc[test_idx]
        model_cv = LinearRegression()
        model_cv.fit(X_tr, y_tr)
        pred = model_cv.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        r2 = r2_score(y_te, pred)
        cv_rows.append({"Fold": fold + 1, "MAE": mae, "RMSE": rmse, "R²": r2})

    model2 = LinearRegression()
    model2.fit(X_trainval, y_trainval)
    y_test_pred_m2 = model2.predict(X_pred)

    return {
        "train_data": train_data,
        "val_data": val_data,
        "pred_data": pred_data,
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
    }


def calculate_metrics(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Data Set": label, "MAE": f"{mae:.2f}", "RMSE": f"{rmse:.2f}", "R²": f"{r2:.3f}"}


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

st.subheader("Data Split Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    train_pct = st.slider(
        "Training %",
        10,
        90,
        70,
        5,
        help="Percentage of data for training",
    )
with col2:
    val_pct = st.slider(
        "Validation %",
        5,
        50,
        15,
        5,
        help="Percentage of data for validation",
    )
with col3:
    pred_pct = 100 - train_pct - val_pct
    st.metric("Prediction %", f"{pred_pct}%", help="Remaining data for prediction testing")

if pred_pct < 0:
    st.error("Training and Validation percentages sum to more than 100%!")
    st.stop()

if st.button("Load Data & Train Model", type="primary"):
    with st.spinner("Training models..."):
        results = train_models(start_nta, end_nta, train_pct, val_pct)
        if results is None:
            st.error(f"No data found for flows from {start_nta} to {end_nta}")
        else:
            st.session_state.results = results
            st.success("✅ Models trained! Scroll down to see results.")

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

    st.subheader("📊 Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Set", f"{len(train_data)} days")
    with col2:
        st.metric("Validation Set", f"{len(val_data)} days")
    with col3:
        st.metric("Prediction Set", f"{len(pred_data)} days")

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

    st.subheader("📈 Model Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model 1: Train/Val/Test Split**")
        metrics_m1 = [
            calculate_metrics(y_train, y_train_pred_m1, "Train"),
            calculate_metrics(y_val, y_val_pred_m1, "Validation"),
            calculate_metrics(y_pred_actual, y_test_pred_m1, "Test"),
        ]
        st.dataframe(pd.DataFrame(metrics_m1), use_container_width=True)
        st.write("**Top Coefficients:**")
        coef_m1_df = pd.DataFrame(
            {"Feature": readable_names, "Coefficient": model1.coef_}
        ).sort_values("Coefficient", ascending=False)
        st.dataframe(coef_m1_df.head(10), use_container_width=True)
        st.caption(f"Intercept: {model1.intercept_:.2f}")

    with col2:
        st.write("**Model 2: Time Series CV**")
        cv_metrics_df = pd.DataFrame(cv_rows)
        for col in ["MAE", "RMSE", "R²"]:
            cv_metrics_df[col] = pd.to_numeric(cv_metrics_df[col], errors="coerce")
        avg_cv_metrics = {
            "Average MAE": cv_metrics_df["MAE"].mean(),
            "Average RMSE": cv_metrics_df["RMSE"].mean(),
            "Average R²": cv_metrics_df["R²"].mean(),
        }
        st.dataframe(cv_metrics_df, use_container_width=True)
        metrics_m2 = [
            {
                "Data Set": "CV Average",
                "MAE": f"{avg_cv_metrics['Average MAE']:.2f}",
                "RMSE": f"{avg_cv_metrics['Average RMSE']:.2f}",
                "R²": f"{avg_cv_metrics['Average R²']:.3f}",
            },
            calculate_metrics(y_pred_actual, y_test_pred_m2, "Final Test"),
        ]
        st.dataframe(pd.DataFrame(metrics_m2), use_container_width=True)
        st.write("**Top Coefficients:**")
        coef_m2_df = pd.DataFrame(
            {"Feature": readable_names, "Coefficient": model2.coef_}
        ).sort_values("Coefficient", ascending=False)
        st.dataframe(coef_m2_df.head(10), use_container_width=True)
        st.caption(f"Intercept: {model2.intercept_:.2f}")

    st.subheader("📊 Predictions Visualization")
    model_choice = st.radio(
        "**Select which model's predictions to display:**",
        options=["Model 1: Train/Val/Test Split", "Model 2: Time Series CV"],
        horizontal=True,
    )

    if model_choice == "Model 1: Train/Val/Test Split":
        y_pred_pred = y_test_pred_m1
        model_name = "Model 1 (Train/Val/Test Split)"
    else:
        y_pred_pred = y_test_pred_m2
        model_name = "Model 2 (Time Series CV)"

    max_pred_diff = float(np.max(np.abs(y_test_pred_m1 - y_test_pred_m2)))
    selected_mae = float(mean_absolute_error(y_pred_actual, y_pred_pred))
    st.caption(
        f"Selected model test MAE: {selected_mae:.2f}. "
        f"Max abs difference between model predictions: {max_pred_diff:.2f}."
    )

    st.write(f"**Test Set: Predicted vs Actual Over Time ({model_name})**")
    fig_test = go.Figure()
    fig_test.add_trace(
        go.Scatter(
            x=pred_data["started_date"],
            y=y_pred_actual,
            mode="lines",
            name="Actual",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig_test.add_trace(
        go.Scatter(
            x=pred_data["started_date"],
            y=y_pred_pred,
            mode="lines",
            name="Predicted",
            line=dict(color="#EF553B", width=2, dash="dash"),
        )
    )
    fig_test.update_layout(
        title=f"Test Set Predictions ({model_name})",
        xaxis_title="Date",
        yaxis_title="Bike Flows",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig_test, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Full Time Series", "Actual vs Predicted Scatter", "Residuals", "Error Distribution"]
    )

    with tab1:
        st.write(f"**Full Time Series: Train, Validation, and Test Sets ({model_name})**")
        fig_full = go.Figure()
        fig_full.add_trace(
            go.Scatter(
                x=train_data["started_date"],
                y=y_train,
                mode="lines",
                name="Train Actual",
                line=dict(color="blue", width=2),
            )
        )
        fig_full.add_trace(
            go.Scatter(
                x=val_data["started_date"],
                y=y_val,
                mode="lines",
                name="Validation Actual",
                line=dict(color="green", width=2),
            )
        )
        fig_full.add_trace(
            go.Scatter(
                x=pred_data["started_date"],
                y=y_pred_actual,
                mode="lines",
                name="Test Actual",
                line=dict(color="#636EFA", width=2),
            )
        )
        fig_full.add_trace(
            go.Scatter(
                x=pred_data["started_date"],
                y=y_pred_pred,
                mode="lines",
                name="Test Predicted",
                line=dict(color="#EF553B", width=2, dash="dash"),
            )
        )
        fig_full.update_layout(
            title=f"Full Time Series ({model_name})",
            xaxis_title="Date",
            yaxis_title="Bike Flows",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig_full, use_container_width=True)

    with tab2:
        st.write(f"**Actual vs Predicted Scatter Plot ({model_name})**")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=y_pred_actual,
                y=y_pred_pred,
                mode="markers",
                marker=dict(size=5, color="#636EFA"),
                name="Predictions",
                text=[
                    f"Actual: {a:.0f}<br>Predicted: {p:.0f}"
                    for a, p in zip(y_pred_actual, y_pred_pred)
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        min_val = min(y_pred_actual.min(), y_pred_pred.min())
        max_val = max(y_pred_actual.max(), y_pred_pred.max())
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Prediction",
            )
        )
        fig_scatter.update_layout(
            title=f"Actual vs Predicted ({model_name})",
            xaxis_title="Actual Bike Flows",
            yaxis_title="Predicted Bike Flows",
            height=500,
            hovermode="closest",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.write(f"**Residuals Over Time ({model_name})**")
        test_residuals = y_pred_actual - y_pred_pred
        fig_resid = go.Figure()
        fig_resid.add_trace(
            go.Scatter(
                x=pred_data["started_date"],
                y=test_residuals,
                mode="markers",
                marker=dict(size=6, color=test_residuals, colorscale="RdBu", showscale=True),
                name="Residual",
            )
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Error")
        fig_resid.update_layout(
            title=f"Residuals ({model_name})",
            xaxis_title="Date",
            yaxis_title="Residual (Actual - Predicted)",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig_resid, use_container_width=True)

    with tab4:
        st.write(f"**Error Distribution ({model_name})**")
        test_residuals = y_pred_actual - y_pred_pred
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=test_residuals,
                nbinsx=30,
                name="Error Distribution",
                marker=dict(color="#636EFA"),
            )
        )
        fig_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
        fig_dist.update_layout(
            title=f"Error Distribution ({model_name})",
            xaxis_title="Error (Actual - Predicted)",
            yaxis_title="Frequency",
            height=400,
            hovermode="x",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.write("**Error Statistics:**")
        error_stats = pd.DataFrame(
            {
                "Metric": [
                    "Mean Error",
                    "Std Dev",
                    "Min Error",
                    "Max Error",
                    "Median Error",
                ],
                "Value": [
                    f"{test_residuals.mean():.2f}",
                    f"{test_residuals.std():.2f}",
                    f"{test_residuals.min():.2f}",
                    f"{test_residuals.max():.2f}",
                    f"{test_residuals.median():.2f}",
                ],
            }
        )
        st.dataframe(error_stats, use_container_width=True)
else:
    st.info("👆 Select two NTAs and click 'Load Data & Train Model' to begin")