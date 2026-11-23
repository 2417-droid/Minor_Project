"""
Streamlit application for Household Electricity Load Forecasting.

Run with::

    streamlit run app.py

Features
--------
- Dataset information and consumption trend visualizations
- Select forecast horizon (1 h, 6 h, 24 h)
- View recent historical consumption
- Generate forecasts using the actual trained model
- Compare models on interactive charts
- View evaluation metrics tables
- Clear statement of model limitations
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import joblib

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    FIGURES_DIR,
    HORIZONS,
    METRICS_PATH,
    MODELS_DIR,
    MODEL_PATH,
    MODELS_PATH,
    REPORTS_DIR,
    TARGET,
)
from src.data_preprocessing import load_hourly_data
from src.feature_engineering import build_features, get_feature_columns
from src.predict import forecast, forecast_all_models, load_models, load_all_models


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Household Load Forecasting",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data(ttl=3600)
def get_hourly_data() -> pd.DataFrame:
    """Load (or trigger preprocessing to create) the hourly dataset."""
    try:
        return load_hourly_data()
    except FileNotFoundError:
        st.error("Processed data not found. Please run the pipeline first:\n"
                 "```\npython -m src.data_download\npython -m src.data_preprocessing\n```")
        return None


@st.cache_resource
def get_models():
    """Load saved models. Returns (best_models, all_models, feature_cols)."""
    try:
        best_models, feature_cols = load_models()
        all_models = joblib.load(MODELS_PATH)
        return best_models, all_models, feature_cols
    except FileNotFoundError:
        return None, None, None


@st.cache_data(ttl=3600)
def get_metrics() -> pd.DataFrame | None:
    """Load the evaluation metrics CSV if it exists."""
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.title("⚡ Household Short-Term Electricity Load Forecasting")
    st.markdown(
        "*A machine learning prototype that forecasts an individual household's "
        "electricity consumption using historical smart-meter data.*"
    )
    st.divider()

    # Load data
    hourly = get_hourly_data()
    if hourly is None:
        st.stop()

    best_models, all_models, feature_cols = get_models()
    metrics_df = get_metrics()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "📊 Dataset Overview",
        "🔮 Forecast",
        "📈 Model Comparison",
        "📋 Evaluation Metrics",
        "⚠️ Limitations",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Disclaimer:** This is a historical-data-based forecasting "
        "prototype, not a production grid-control system."
    )

    if page == "📊 Dataset Overview":
        show_dataset_overview(hourly)
    elif page == "🔮 Forecast":
        show_forecast_page(hourly, best_models, all_models, feature_cols)
    elif page == "📈 Model Comparison":
        show_model_comparison(hourly, all_models, feature_cols, metrics_df)
    elif page == "📋 Evaluation Metrics":
        show_metrics_page(metrics_df)
    elif page == "⚠️ Limitations":
        show_limitations_page()


# ── Dataset Overview page ─────────────────────────────────────────────────────
def show_dataset_overview(hourly: pd.DataFrame):
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hours", f"{len(hourly):,}")
    with col2:
        st.metric("Date Range",
                  f"{hourly.index.min().strftime('%Y-%m-%d')} to "
                  f"{hourly.index.max().strftime('%Y-%m-%d')}")
    with col3:
        st.metric("Avg Power", f"{hourly[TARGET].mean():.3f} kW")
    with col4:
        st.metric("Max Power", f"{hourly[TARGET].max():.3f} kW")

    st.markdown("### Dataset Description")
    st.markdown("""
    **Source:** UCI Machine Learning Repository — Individual Household Electric
    Power Consumption dataset.

    - **Location:** A house in Sceaux, France (7 km south of Paris)
    - **Period:** December 2006 – November 2010 (≈ 4 years)
    - **Original sampling:** 1-minute intervals (2,075,259 readings)
    - **Resampled to:** Hourly frequency for this project
    - **License:** CC BY 4.0

    **Variables:**

    | Column | Description | Unit |
    |--------|-------------|------|
    | Global_active_power | Household active power (forecasting target) | kW |
    | Global_reactive_power | Reactive power | kW |
    | Voltage | Voltage | V |
    | Global_intensity | Current intensity | A |
    | Sub_metering_1 | Kitchen (dishwasher, oven, microwave) | Wh |
    | Sub_metering_2 | Laundry (washer, dryer, fridge, light) | Wh |
    | Sub_metering_3 | Water heater & AC | Wh |
    """)

    # Consumption trend
    st.markdown("### Consumption Over Time")
    n_days = st.slider("Show last N days", 7, 365, 30, key="overview_days")

    fig = go.Figure()
    recent = hourly.tail(n_days * 24)
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent[TARGET],
        mode="lines", name="Active Power",
        line=dict(color="steelblue", width=1),
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Global Active Power (kW)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Daily pattern
    st.markdown("### Average Daily Pattern")
    hourly_copy = hourly.copy()
    hourly_copy["hour"] = hourly_copy.index.hour
    daily_pattern = hourly_copy.groupby("hour")[TARGET].mean()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=daily_pattern.index, y=daily_pattern.values,
        mode="lines+markers", name="Avg Power",
        line=dict(color="darkorange", width=2),
    ))
    fig2.update_layout(
        xaxis_title="Hour of Day", yaxis_title="Avg Active Power (kW)",
        xaxis=dict(dtick=1), height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Weekly pattern
    st.markdown("### Average Weekly Pattern")
    hourly_copy["dayofweek"] = hourly_copy.index.dayofweek
    weekly_pattern = hourly_copy.groupby("dayofweek")[TARGET].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=day_names, y=weekly_pattern.values,
        name="Avg Power", marker_color="seagreen",
    ))
    fig3.update_layout(
        xaxis_title="Day of Week", yaxis_title="Avg Active Power (kW)",
        height=350,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Forecast page ─────────────────────────────────────────────────────────────
def show_forecast_page(hourly, best_models, all_models, feature_cols):
    st.header("🔮 Forecast — Actual vs Predicted")

    if best_models is None:
        st.warning("No trained models found. Please run the training pipeline first:\n"
                   "```\npython -m src.train\n```")
        st.stop()

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 3])
    with col1:
        horizon = st.selectbox(
            "Forecast horizon",
            HORIZONS,
            format_func=lambda h: f"t+{h}h  ({h} hour{'s' if h > 1 else ''} ahead)",
        )
    with col2:
        n_history = st.slider(
            "Hours of history to show", 24, 720, 168, key="forecast_history"
        )

    # ── How to read this page (collapsible) ──────────────────────────────────
    with st.expander("📖  How to read the charts on this page", expanded=True):
        st.markdown(f"""
**What "t+{horizon}h" means:**
At every point in time, the model looks at the last few hours of data and
predicts what the power consumption will be **{horizon} hour(s) later**.

---

**Chart 1 — Actual vs Predicted (time-series)**
- 🔵 **Blue solid line = Actual** — what the electricity meter really recorded
- 🔴 **Red dashed line = Predicted** — what the model said it would be
- 🟠 **Orange shading = Error band** — the gap between actual and predicted.
  - *Wider shading = bigger mistake by the model*
  - *No shading = perfect prediction*
- **Hover** over any point to see the exact values

---

**Chart 2 — Difference (Error) over Time**
- This shows `Predicted − Actual` for every hour
- 🔴 **Bar above zero** → model over-predicted (said more power than was actually used)
- 🔵 **Bar below zero** → model under-predicted (said less power than was actually used)
- **Ideal model** would have all bars at zero
- The **MAE line** (yellow dashed) shows the average error magnitude

---

**The 4 stat cards at the top:**
- **Latest Actual** — the most recent real meter reading
- **Forecast** — what the model predicts for {horizon}h from that point
- **Error** — difference between forecast and actual (positive = over, negative = under)
- **MAE** — Mean Absolute Error over the entire displayed window (lower = better)
        """)

    # ── Generate forecast ─────────────────────────────────────────────────────
    try:
        result = forecast(best_models, feature_cols, hourly,
                          horizon=horizon, n_recent=n_history)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.stop()

    diff = result["predicted"] - result["actual"]
    mae  = diff.abs().mean()

    # ── Stat cards ────────────────────────────────────────────────────────────
    latest       = result.iloc[-1]
    latest_error = latest["predicted"] - latest["actual"]
    forecast_time = latest["datetime"] + pd.Timedelta(hours=horizon)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Latest Actual",
        f"{latest['actual']:.3f} kW",
        delta=latest["datetime"].strftime("%Y-%m-%d %H:%M"),
        delta_color="off",
    )
    c2.metric(
        f"Forecast (t+{horizon}h)",
        f"{latest['predicted']:.3f} kW",
        delta=f"for {forecast_time.strftime('%Y-%m-%d %H:%M')}",
        delta_color="off",
    )
    c3.metric(
        "Latest Error",
        f"{latest_error:+.3f} kW",
        delta="over-predicted" if latest_error > 0 else "under-predicted",
        delta_color="inverse",
    )
    c4.metric(
        "Window MAE",
        f"{mae:.3f} kW",
        delta=f"avg error over {n_history}h window",
        delta_color="off",
    )

    st.markdown("---")

    # Build hour index for x-axis (0, 1, 2 ... n_history)
    hours      = list(range(len(result)))
    hours_rev  = hours[::-1]
    act_vals   = result["actual"].tolist()
    pred_vals  = result["predicted"].tolist()
    diff_vals  = diff.tolist()
    date_strs  = result["datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    # ── Chart 1: Actual vs Predicted ──────────────────────────────────────────
    st.markdown("### 📈 Chart 1 — Actual vs Predicted Power Consumption")
    st.caption(
        "🔵 Thick blue = Actual (real meter)  |  "
        "🔴 Red dotted = Predicted by model  |  "
        "🟠 Orange fill = model over-predicted  |  "
        "🟣 Purple fill = model under-predicted"
    )

    fig1 = go.Figure()

    # Orange fill: where predicted > actual (over-estimate)
    fig1.add_trace(go.Scatter(
        x=hours + hours_rev,
        y=pred_vals + act_vals[::-1],
        fill="toself",
        fillcolor="rgba(255,100,0,0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Over-predicted",
        hoverinfo="skip",
        showlegend=True,
    ))

    # Purple fill: where actual > predicted (under-estimate)
    fig1.add_trace(go.Scatter(
        x=hours + hours_rev,
        y=act_vals + pred_vals[::-1],
        fill="toself",
        fillcolor="rgba(130,0,200,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Under-predicted",
        hoverinfo="skip",
        showlegend=True,
    ))

    # Actual line — thick solid blue
    fig1.add_trace(go.Scatter(
        x=hours,
        y=act_vals,
        mode="lines",
        name="Actual",
        line=dict(color="#1155cc", width=3),
        text=[f"Hour {h} | {d}<br>Actual={a:.3f} kW | Predicted={p:.3f} kW | Diff={p-a:+.3f} kW"
              for h, d, a, p in zip(hours, date_strs, act_vals, pred_vals)],
        hoverinfo="text",
    ))

    # Predicted line — thick red dotted
    fig1.add_trace(go.Scatter(
        x=hours,
        y=pred_vals,
        mode="lines",
        name=f"Predicted (t+{horizon}h)",
        line=dict(color="#cc0000", width=2.5, dash="dot"),
        text=[f"Hour {h} | {d}<br>Predicted={p:.3f} kW | Actual={a:.3f} kW | Diff={p-a:+.3f} kW"
              for h, d, a, p in zip(hours, date_strs, act_vals, pred_vals)],
        hoverinfo="text",
    ))

    fig1.update_layout(
        xaxis_title=f"Hour Index  (Hour 0 = earliest data,  Hour {len(result)-1} = most recent)",
        yaxis_title="Global Active Power (kW)",
        height=460,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60),
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#dddddd",
                   tickmode="linear", tick0=0, dtick=max(1, len(result)//20)),
        yaxis=dict(showgrid=True, gridcolor="#dddddd"),
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.info(
        f"**How to read this chart:**  "
        f"Blue solid line = what the electricity meter actually recorded.  "
        f"Red dotted line = what the model predicted.  "
        f"🟠 Orange fill = model said MORE than reality.  "
        f"🟣 Purple fill = model said LESS than reality.  "
        f"Hover on either line to see exact values and error.  "
        f"Average error over this window = **{mae:.3f} kW**."
    )

    # ── Chart 2: Difference / Error ───────────────────────────────────────────
    st.markdown("### 📊 Chart 2 — Error Per Hour  (Predicted − Actual)")
    st.caption(
        "🔴 Bar above zero = model predicted MORE than reality.  "
        "🔵 Bar below zero = model predicted LESS than reality.  "
        "All bars at zero = perfect model."
    )

    bar_colors = ["#d62728" if v >= 0 else "#1f77b4" for v in diff_vals]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=hours,
        y=diff_vals,
        marker_color=bar_colors,
        opacity=0.82,
        name="Error",
        text=[f"Hour {h} | {d}<br>Error = {e:+.3f} kW<br>({'over' if e>=0 else 'under'}-predicted)"
              for h, d, e in zip(hours, date_strs, diff_vals)],
        hoverinfo="text",
    ))
    fig2.add_hline(y=0,    line_width=2,   line_color="black")
    fig2.add_hline(y=mae,  line_width=1.5, line_dash="dot", line_color="goldenrod",
                   annotation_text=f"+MAE = +{mae:.3f} kW", annotation_position="top right")
    fig2.add_hline(y=-mae, line_width=1.5, line_dash="dot", line_color="goldenrod",
                   annotation_text=f"-MAE = -{mae:.3f} kW", annotation_position="bottom right")

    fig2.update_layout(
        xaxis_title=f"Hour Index  (Hour 0 = earliest,  Hour {len(result)-1} = most recent)",
        yaxis_title="Error = Predicted − Actual (kW)",
        height=320,
        hovermode="closest",
        showlegend=False,
        margin=dict(t=20),
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#dddddd",
                   tickmode="linear", tick0=0, dtick=max(1, len(result)//20)),
        yaxis=dict(showgrid=True, gridcolor="#dddddd"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Error summary stats ───────────────────────────────────────────────────
    st.markdown("### 📋 Error Summary")
    over  = (diff > 0).sum()
    under = (diff < 0).sum()
    exact = (diff == 0).sum()
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    col_s1.metric("MAE",  f"{mae:.3f} kW",        delta="mean absolute error")
    col_s2.metric("Max Over-predict",  f"+{diff.max():.3f} kW", delta=f"{over} hours")
    col_s3.metric("Max Under-predict", f"{diff.min():.3f} kW",  delta=f"{under} hours")
    col_s4.metric("% Over-predicted",  f"{100*over/len(diff):.1f}%")
    col_s5.metric("% Under-predicted", f"{100*under/len(diff):.1f}%")

    # ── All-model comparison ──────────────────────────────────────────────────
    st.markdown("---")
    if st.checkbox(f"Compare ALL 5 models side-by-side for t+{horizon}h"):
        try:
            all_results = forecast_all_models(
                all_models, feature_cols, hourly,
                horizon=horizon, n_recent=n_history,
            )
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=hours, y=result["actual"],
                mode="lines", name="Actual",
                line=dict(color="black", width=2.5),
                customdata=hover_dates,
                hovertemplate="Hour %{x}<br><b>Actual: %{y:.3f} kW</b><br>📅 %{customdata}<extra></extra>",
            ))
            palette = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
            for i, (name, df) in enumerate(all_results.items()):
                fig3.add_trace(go.Scatter(
                    x=hours, y=df["predicted"],
                    mode="lines", name=name,
                    line=dict(color=palette[i % len(palette)], width=1.5, dash="dash"),
                    hovertemplate=f"<b>{name}</b><br>Hour %{{x}}<br>%{{y:.3f}} kW<extra></extra>",
                ))
            fig3.update_layout(
                title=f"All 5 Models vs Actual — Horizon t+{horizon}h",
                xaxis_title=f"Hour Index  (0 = earliest,  {len(result)-1} = most recent)",
                yaxis_title="Global Active Power (kW)",
                height=450, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.info(
                "Notice: Persistence (copies last value) and Seasonal Naive "
                "(copies last week) drift further from actual than Ridge/Random Forest. "
                "This confirms ML models add real value."
            )
        except Exception as e:
            st.error(f"Comparison failed: {e}")


# ── Model Comparison page ─────────────────────────────────────────────────────
def show_model_comparison(hourly, all_models, feature_cols, metrics_df):
    st.header("📈 Model Comparison")

    if metrics_df is not None:
        st.markdown("### Validation Metrics")
        val_df = metrics_df[metrics_df["split"] == "validation"].copy()
        st.dataframe(val_df, use_container_width=True)

        st.markdown("### Test Metrics")
        test_df = metrics_df[metrics_df["split"] == "test"].copy()
        st.dataframe(test_df, use_container_width=True)

        # Bar chart comparison
        st.markdown("### MAE Comparison (Test Set)")
        horizons_list = sorted(test_df["horizon"].unique())
        models_list = sorted(test_df["model"].unique())

        fig = go.Figure()
        for model in models_list:
            vals = []
            for h in horizons_list:
                row = test_df[(test_df["horizon"] == h) & (test_df["model"] == model)]
                vals.append(row["MAE"].values[0] if len(row) > 0 else 0)
            fig.add_trace(go.Bar(name=model, x=horizons_list, y=vals))
        fig.update_layout(
            barmode="group", xaxis_title="Horizon", yaxis_title="MAE (kW)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # RMSE comparison
        st.markdown("### RMSE Comparison (Test Set)")
        fig2 = go.Figure()
        for model in models_list:
            vals = []
            for h in horizons_list:
                row = test_df[(test_df["horizon"] == h) & (test_df["model"] == model)]
                vals.append(row["RMSE"].values[0] if len(row) > 0 else 0)
            fig2.add_trace(go.Bar(name=model, x=horizons_list, y=vals))
        fig2.update_layout(
            barmode="group", xaxis_title="Horizon", yaxis_title="RMSE (kW)",
            height=450,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No metrics available. Run `python -m src.evaluate` to generate metrics.")

    # Show saved figures if available
    st.markdown("---")
    st.markdown("### Saved Figures")
    if FIGURES_DIR.exists():
        figures = sorted(FIGURES_DIR.glob("*.png"))
        if figures:
            selected_fig = st.selectbox(
                "Select a figure to view",
                [f.name for f in figures],
            )
            fig_path = FIGURES_DIR / selected_fig
            st.image(str(fig_path), use_container_width=True)
        else:
            st.info("No figures generated yet. Run `python -m src.evaluate`.")
    else:
        st.info("Figures directory not found.")


# ── Evaluation Metrics page ───────────────────────────────────────────────────
def show_metrics_page(metrics_df):
    st.header("📋 Evaluation Metrics")

    if metrics_df is None:
        st.warning("No metrics found. Run `python -m src.evaluate` first.")
        return

    st.markdown("""
    ### Metric Definitions

    - **MAE (Mean Absolute Error):** The average absolute difference between
      predicted and actual values, in kW. Lower is better.
    - **RMSE (Root Mean Square Error):** The square root of the average squared
      errors. Penalizes large errors more heavily than MAE. Lower is better.
    - **R² (Coefficient of Determination):** Proportion of variance in the
      target explained by the model. 1.0 = perfect, 0.0 = predicting the mean,
      negative = worse than the mean.
    """)

    split = st.radio("Select split", ["validation", "test"], horizontal=True)
    subset = metrics_df[metrics_df["split"] == split]
    st.dataframe(subset, use_container_width=True)


# ── Limitations page ──────────────────────────────────────────────────────────
def show_limitations_page():
    st.header("⚠️ Limitations & Disclaimers")

    st.markdown("""
    ### Important Notice

    This application is a **historical-data-based forecasting prototype**,
    not a production grid-control system. It should not be used for real-time
    grid operation, dispatch decisions, or any safety-critical application.

    ### Key Limitations

    1. **Single Household:** The model is trained on data from one household
       in Sceaux, France (2006-2010). Results may not generalise to other
       households, climates, or time periods.

    2. **No External Factors:** The model uses only historical consumption
       patterns. It does not account for:
       - Weather / temperature changes
       - Occupancy changes (holidays, guests)
       - New appliance installations
       - Tariff changes or demand-response events
       - Calendar anomalies (e.g., Christmas on a different weekday)

    3. **Classical ML Only:** Deep learning models (LSTMs, Transformers) are
       not used to keep the project lightweight and interpretable. More
       complex models may achieve better accuracy.

    4. **Historical Patterns:** The model assumes that future consumption
       follows historical patterns. Sudden behavioural changes will cause
       forecast errors.

    5. **No Real-Time Data:** The app uses pre-trained models and historical
       data. In a production system, models would be retrained periodically
       with new data.

    6. **Resampling:** Original 1-minute data is aggregated to hourly.
       Short-term intra-hour variability is lost.

    7. **Direct Multi-Horizon:** Each horizon has a separate model. This
       means predictions for different horizons are independent and may
       not be temporally consistent (e.g., t+1 and t+6 could diverge).

    ### Ethical Considerations

    - Household consumption data is privacy-sensitive. In a deployment, data
      should be anonymised and handled according to applicable privacy laws.
    - Forecasts should not be used for discriminatory pricing or targeting.

    ### Future Improvements

    - Add weather data as an exogenous feature
    - Implement recursive or hybrid forecasting strategies
    - Add confidence intervals / quantile forecasts
    - Explore gradient-boosted trees (XGBoost, LightGBM)
    - Implement online learning for model updates
    - Add anomaly detection for unusual consumption
    """)


if __name__ == "__main__":
    main()
