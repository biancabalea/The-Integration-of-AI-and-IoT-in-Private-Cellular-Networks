from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Private 5G + AI Dashboard V2",
    layout="wide"
)

PREDICTIONS_PATH = Path("results/predictions/final_predictions.csv")
BEST_MODELS_PATH = Path("results/metrics/best_models_summary.csv")
ALL_MODELS_PATH = Path("results/metrics/all_models_summary.csv")

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
# HELPERS
# =========================
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

def filter_data(df):
    st.sidebar.header("Filtre")

    selected_cells = st.sidebar.multiselect(
        "Celula",
        options=sorted(df["serving_cell"].unique().tolist()),
        default=sorted(df["serving_cell"].unique().tolist())
    )

    selected_traffic = st.sidebar.multiselect(
        "Clasa trafic",
        options=sorted(df["traffic_class"].unique().tolist()),
        default=sorted(df["traffic_class"].unique().tolist())
    )

    selected_activity = st.sidebar.multiselect(
        "Stare dispozitiv",
        options=sorted(df["is_active"].unique().tolist()),
        default=sorted(df["is_active"].unique().tolist())
    )

    selected_devices = st.sidebar.multiselect(
        "Device ID",
        options=sorted(df["device_id"].unique().tolist()),
        default=[]
    )

    time_min = int(df["time_step"].min())
    time_max = int(df["time_step"].max())

    selected_time = st.sidebar.slider(
        "Interval timp",
        min_value=time_min,
        max_value=time_max,
        value=(time_min, time_max)
    )

    filtered = df[
        (df["serving_cell"].isin(selected_cells)) &
        (df["traffic_class"].isin(selected_traffic)) &
        (df["is_active"].isin(selected_activity)) &
        (df["time_step"] >= selected_time[0]) &
        (df["time_step"] <= selected_time[1])
    ].copy()

    if selected_devices:
        filtered = filtered[filtered["device_id"].isin(selected_devices)].copy()

    return filtered

def build_alerts(df, throughput_min_mbps, sinr_min_db, latency_max_ms, per_max):
    alert_df = df.copy()

    alert_df["alert_throughput"] = (alert_df["throughput_bps"] / 1e6) < throughput_min_mbps
    alert_df["alert_sinr"] = alert_df["SINR_dB"] < sinr_min_db
    alert_df["alert_latency"] = alert_df["latency_ms"] > latency_max_ms
    alert_df["alert_per"] = alert_df["PER"] > per_max

    alert_df["is_critical"] = (
        alert_df["alert_throughput"] |
        alert_df["alert_sinr"] |
        alert_df["alert_latency"] |
        alert_df["alert_per"]
    )

    return alert_df

# =========================
# MAIN
# =========================
def main():
    st.title("Integrarea AI si IoT in retelele celulare private")
    st.caption("Dashboard V2 - monitorizare, predictie si analiza a performantei retelei 5G private")

    df = load_predictions()
    best_models_df = load_best_models()
    all_models_df = load_all_models()

    filtered_df = filter_data(df)

    st.sidebar.header("Praguri alerte")
    throughput_min_mbps = st.sidebar.number_input("Throughput minim [Mbps]", value=1.0, step=0.5)
    sinr_min_db = st.sidebar.number_input("SINR minim [dB]", value=5.0, step=1.0)
    latency_max_ms = st.sidebar.number_input("Latency maxima [ms]", value=30.0, step=1.0)
    per_max = st.sidebar.number_input("PER maxim", value=0.10, step=0.01, format="%.2f")

    filtered_df = build_alerts(
        filtered_df,
        throughput_min_mbps=throughput_min_mbps,
        sinr_min_db=sinr_min_db,
        latency_max_ms=latency_max_ms,
        per_max=per_max
    )

    if filtered_df.empty:
        st.warning("Nu exista date pentru filtrele selectate.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Real vs Predicted",
        "Error Analysis",
        "Cells & Traffic",
        "Model Summary"
    ])

    # =========================
    # TAB 1 - OVERVIEW
    # =========================
    with tab1:
        st.subheader("Rezumat general")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Randuri filtrate", f"{len(filtered_df):,}")
        c2.metric("Devices unice", f"{filtered_df['device_id'].nunique():,}")
        c3.metric("Throughput mediu [Mbps]", f"{filtered_df['throughput_bps'].mean() / 1e6:.2f}")
        c4.metric("SINR mediu [dB]", f"{filtered_df['SINR_dB'].mean():.2f}")
        c5.metric("Latency medie [ms]", f"{filtered_df['latency_ms'].mean():.2f}")
        c6.metric("PER mediu", f"{filtered_df['PER'].mean():.4f}")

        st.markdown("### Alerte")
        a1, a2, a3, a4, a5 = st.columns(5)

        critical_count = int(filtered_df["is_critical"].sum())
        critical_pct = 100 * critical_count / len(filtered_df)

        a1.metric("Observatii critice", f"{critical_count:,}")
        a2.metric("Procent critic [%]", f"{critical_pct:.2f}")
        a3.metric("Throughput sub prag", f"{int(filtered_df['alert_throughput'].sum()):,}")
        a4.metric("SINR sub prag", f"{int(filtered_df['alert_sinr'].sum()):,}")
        a5.metric("Latency/PER peste prag", f"{int((filtered_df['alert_latency'] | filtered_df['alert_per']).sum()):,}")

        st.markdown("### Evolutie KPI-uri medii in timp")
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
            key="overview_kpi"
        )

        fig_line = px.line(
            agg_df,
            x="time_step",
            y=kpi_overview,
            title=f"Evolutia medie in timp pentru {DISPLAY_NAME[kpi_overview]}"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # =========================
    # TAB 2 - REAL VS PREDICTED
    # =========================
    with tab2:
        st.subheader("Comparatie valori reale vs prezise")

        metric_option = st.selectbox(
            "Alege KPI",
            ["throughput_bps", "SINR_dB", "RSRP_dBm", "latency_ms", "PER"],
            key="rvp_metric"
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

    # =========================
    # TAB 3 - ERROR ANALYSIS
    # =========================
    with tab3:
        st.subheader("Analiza erorilor de predictie")

        error_metric = st.selectbox(
            "Alege KPI pentru analiza erorii",
            ["throughput_bps", "SINR_dB", "RSRP_dBm", "latency_ms", "PER"],
            key="error_metric"
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

    # =========================
    # TAB 4 - CELLS & TRAFFIC
    # =========================
    with tab4:
        st.subheader("Analiza pe celule si clase de trafic")

        kpi_cells = st.selectbox(
            "Alege KPI pentru comparatie",
            ["throughput_bps", "SINR_dB", "RSRP_dBm", "latency_ms", "PER"],
            key="cells_metric"
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

        st.markdown("### Date filtrate")
        st.dataframe(filtered_df.head(500), use_container_width=True)

    # =========================
    # TAB 5 - MODEL SUMMARY
    # =========================
    with tab5:
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
        else:
            st.info("Fisierul best_models_summary.csv nu a fost gasit.")

        if not all_models_df.empty:
            st.markdown("### Toate modelele testate")
            st.dataframe(all_models_df, use_container_width=True)

if __name__ == "__main__":
    main()