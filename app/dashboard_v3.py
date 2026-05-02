from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Private 5G + AI Dashboard V3",
    layout="wide"
)

PREDICTIONS_PATH = Path("results/predictions/final_predictions.csv")
BEST_MODELS_PATH = Path("results/metrics/best_models_summary.csv")
ALL_MODELS_PATH = Path("results/metrics/all_models_summary.csv")

# Pozitii gNB (aceleasi ca in MATLAB)
GNB_POSITIONS = pd.DataFrame({
    "gnb_id": [1, 2, 3],
    "x_m": [0, 500, 250],
    "y_m": [0, 0, 433]
})

METRIC_PRED_MAP = {
    "throughput_bps": "throughput_bps_pred",
    "SINR_dB": "SINR_dB_pred",
    "RSRP_dBm": "RSRP_dBm_pred",
    "latency_ms": "latency_ms_pred",
    "PER": "PER_pred"
}

ERROR_COL_MAP = {
    "throughput_bps": "throughput_abs_error",
    "SINR_dB": "sinr_abs_error",
    "RSRP_dBm": "rsrp_abs_error",
    "latency_ms": "latency_abs_error",
    "PER": "per_abs_error"
}

DISPLAY_NAME = {
    "throughput_bps": "Throughput [bps]",
    "SINR_dB": "SINR [dB]",
    "RSRP_dBm": "RSRP [dBm]",
    "latency_ms": "Latency [ms]",
    "PER": "PER"
}

# =========================
# LOADERS
# =========================
@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Fisier lipsa: {PREDICTIONS_PATH}")
    return pd.read_csv(PREDICTIONS_PATH)

@st.cache_data
def load_best_models():
    if BEST_MODELS_PATH.exists():
        return pd.read_csv(BEST_MODELS_PATH)
    return pd.DataFrame()

@st.cache_data
def load_all_models():
    if ALL_MODELS_PATH.exists():
        return pd.read_csv(ALL_MODELS_PATH)
    return pd.DataFrame()

# =========================
# FILTERS
# =========================
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("## Filtre")

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        selected_cells = st.multiselect(
            "Celule",
            options=sorted(df["serving_cell"].unique().tolist()),
            default=sorted(df["serving_cell"].unique().tolist())
        )

    with f2:
        selected_traffic = st.multiselect(
            "Clase trafic",
            options=sorted(df["traffic_class"].unique().tolist()),
            default=sorted(df["traffic_class"].unique().tolist())
        )

    with f3:
        activity_option = st.selectbox(
            "Stare dispozitiv",
            options=["Toate", "Doar active", "Doar inactive"]
        )

    with f4:
        selected_device = st.selectbox(
            "Device ID",
            options=["Toate"] + sorted(df["device_id"].unique().tolist())
        )

    t1, t2 = st.columns([3, 1])

    with t1:
        time_min = int(df["time_step"].min())
        time_max = int(df["time_step"].max())

        selected_time = st.slider(
            "Interval timp",
            min_value=time_min,
            max_value=time_max,
            value=(time_min, time_max)
        )

    with t2:
        only_critical_errors = st.checkbox("Doar erori mari")

    filtered = df[
        (df["serving_cell"].isin(selected_cells)) &
        (df["traffic_class"].isin(selected_traffic)) &
        (df["time_step"] >= selected_time[0]) &
        (df["time_step"] <= selected_time[1])
    ].copy()

    if activity_option == "Doar active":
        filtered = filtered[filtered["is_active"] == 1].copy()
    elif activity_option == "Doar inactive":
        filtered = filtered[filtered["is_active"] == 0].copy()

    if selected_device != "Toate":
        filtered = filtered[filtered["device_id"] == selected_device].copy()

    if only_critical_errors:
        filtered = filtered[
            (filtered["throughput_abs_error"] > filtered["throughput_abs_error"].quantile(0.95)) |
            (filtered["sinr_abs_error"] > filtered["sinr_abs_error"].quantile(0.95)) |
            (filtered["latency_abs_error"] > filtered["latency_abs_error"].quantile(0.95)) |
            (filtered["per_abs_error"] > filtered["per_abs_error"].quantile(0.95))
        ].copy()

    return filtered

# =========================
# TOPOLOGY
# =========================
def plot_topology(df: pd.DataFrame, metric_for_color: str):
    plot_df = df.copy()

    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42)

    fig = px.scatter(
        plot_df,
        x="x_m",
        y="y_m",
        color=metric_for_color,
        symbol="serving_cell",
        hover_data=["device_id", "serving_cell", "traffic_class", "time_step"],
        title=f"Topologia retelei - colorare dupa {DISPLAY_NAME.get(metric_for_color, metric_for_color)}"
    )

    fig.add_trace(
        go.Scatter(
            x=GNB_POSITIONS["x_m"],
            y=GNB_POSITIONS["y_m"],
            mode="markers+text",
            marker=dict(size=18, symbol="x", color="black"),
            text=[f"gNB {i}" for i in GNB_POSITIONS["gnb_id"]],
            textposition="top center",
            name="gNB-uri"
        )
    )

    fig.update_layout(
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        height=650
    )

    return fig

# =========================
# MAIN
# =========================
def main():
    st.title("Integrarea AI si IoT in retelele celulare private")
    st.caption("Dashboard V3 - filtre imbunatatite si topologie retea")

    df = load_predictions()
    best_models_df = load_best_models()
    all_models_df = load_all_models()

    filtered_df = apply_filters(df)

    if filtered_df.empty:
        st.warning("Nu exista date pentru filtrele selectate.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Real vs Predicted",
        "Error Analysis",
        "Cells & Traffic",
        "Topology",
        "Model Summary"
    ])

    with tab1:
        st.subheader("Rezumat general")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Randuri filtrate", f"{len(filtered_df):,}")
        c2.metric("Devices unice", f"{filtered_df['device_id'].nunique():,}")
        c3.metric("Throughput mediu [Mbps]", f"{filtered_df['throughput_bps'].mean() / 1e6:.2f}")
        c4.metric("SINR mediu [dB]", f"{filtered_df['SINR_dB'].mean():.2f}")
        c5.metric("Latency medie [ms]", f"{filtered_df['latency_ms'].mean():.2f}")
        c6.metric("PER mediu", f"{filtered_df['PER'].mean():.4f}")

        agg_df = filtered_df.groupby("time_step", as_index=False).agg({
            "throughput_bps": "mean",
            "SINR_dB": "mean",
            "RSRP_dBm": "mean",
            "latency_ms": "mean",
            "PER": "mean"
        })

        kpi_overview = st.selectbox(
            "Alege KPI pentru evolutie",
            ["throughput_bps", "SINR_dB", "RSRP_dBm", "latency_ms", "PER"],
            key="overview_kpi_v3"
        )

        fig_line = px.line(
            agg_df,
            x="time_step",
            y=kpi_overview,
            title=f"Evolutia medie in timp pentru {DISPLAY_NAME[kpi_overview]}"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.subheader("Comparatie valori reale vs prezise")

        metric_option = st.selectbox(
            "Alege KPI",
            ["throughput_bps", "SINR_dB", "RSRP_dBm", "latency_ms", "PER"],
            key="rvp_metric_v3"
        )

        pred_col = METRIC_PRED_MAP[metric_option]
        sample_df = filtered_df[[metric_option, pred_col]].dropna()

        if len(sample_df) > 5000:
            sample_df = sample_df.sample(5000, random_state=42)

        col_left, col_right = st.columns(2)

        with col_left:
            fig_scatter = px.scatter(
                sample_df,
                x=metric_option,
                y=pred_col,
                opacity=0.5,
                title=f"Real vs Predicted - {DISPLAY_NAME[metric_option]}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_right:
            time_compare = filtered_df.groupby("time_step", as_index=False).agg({
                metric_option: "mean",
                pred_col: "mean"
            })

            fig_compare = px.line(
                time_compare,
                x="time_step",
                y=[metric_option, pred_col],
                title=f"Evolutie medie in timp - {DISPLAY_NAME[metric_option]}"
            )
            st.plotly_chart(fig_compare, use_container_width=True)

    with tab3:
        st.subheader("Analiza erorilor de predictie")

        error_metric = st.selectbox(
            "Alege KPI pentru analiza erorii",
            ["throughput_bps", "SINR_dB", "RSRP_dBm", "latency_ms", "PER"],
            key="error_metric_v3"
        )

        error_col = ERROR_COL_MAP[error_metric]
        pred_col = METRIC_PRED_MAP[error_metric]

        e1, e2, e3 = st.columns(3)
        e1.metric("Eroare medie absoluta", f"{filtered_df[error_col].mean():.6f}")
        e2.metric("Eroare maxima", f"{filtered_df[error_col].max():.6f}")
        e3.metric("Eroare mediana", f"{filtered_df[error_col].median():.6f}")

        col_left, col_right = st.columns(2)

        with col_left:
            fig_hist = px.histogram(
                filtered_df,
                x=error_col,
                nbins=50,
                title=f"Distributia erorii pentru {DISPLAY_NAME[error_metric]}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            top_errors = filtered_df.sort_values(error_col, ascending=False).head(20)[[
                "time_step", "device_id", "serving_cell", error_metric, pred_col, error_col
            ]]
            st.dataframe(top_errors, use_container_width=True)

    with tab4:
        st.subheader("Analiza pe celule si clase de trafic")

        kpi_cells = st.selectbox(
            "Alege KPI pentru comparatie",
            ["throughput_bps", "SINR_dB", "RSRP_dBm", "latency_ms", "PER"],
            key="cells_metric_v3"
        )

        col_left, col_right = st.columns(2)

        with col_left:
            cell_stats = filtered_df.groupby("serving_cell", as_index=False)[kpi_cells].mean()
            fig_bar_cell = px.bar(
                cell_stats,
                x="serving_cell",
                y=kpi_cells,
                title=f"Media {DISPLAY_NAME[kpi_cells]} pe celule"
            )
            st.plotly_chart(fig_bar_cell, use_container_width=True)

        with col_right:
            traffic_stats = filtered_df.groupby("traffic_class", as_index=False)[kpi_cells].mean()
            fig_bar_traffic = px.bar(
                traffic_stats,
                x="traffic_class",
                y=kpi_cells,
                title=f"Media {DISPLAY_NAME[kpi_cells]} pe clase de trafic"
            )
            st.plotly_chart(fig_bar_traffic, use_container_width=True)

        st.dataframe(filtered_df.head(500), use_container_width=True)

    with tab5:
        st.subheader("Topologia retelei")

        topo_metric = st.selectbox(
            "Colorare topologie dupa",
            ["throughput_bps", "SINR_dB", "RSRP_dBm", "latency_ms", "PER"],
            key="topology_metric"
        )

        fig_topology = plot_topology(filtered_df, topo_metric)
        st.plotly_chart(fig_topology, use_container_width=True)

    with tab6:
        st.subheader("Rezumat modele finale")

        if not best_models_df.empty:
            st.markdown("### Cele mai bune modele selectate")
            st.dataframe(best_models_df, use_container_width=True)

            fig_best = px.bar(
                best_models_df,
                x="target",
                y="R2",
                color="model",
                title="Performanta modelelor finale (R2)"
            )
            st.plotly_chart(fig_best, use_container_width=True)

        if not all_models_df.empty:
            st.markdown("### Toate modelele testate")
            st.dataframe(all_models_df, use_container_width=True)

if __name__ == "__main__":
    main()