from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Private 5G + AI Dashboard",
    layout="wide"
)

DATA_PATH = Path("results/predictions/final_predictions.csv")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fisierul nu exista: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df

# =========================
# MAIN
# =========================
def main():
    st.title("Integrarea AI si IoT in retelele celulare private")
    st.subheader("Dashboard de monitorizare si predictie pentru reteaua 5G privata")

    df = load_data()

    st.sidebar.header("Filtre")

    selected_cells = st.sidebar.multiselect(
        "Selecteaza celula",
        options=sorted(df["serving_cell"].unique().tolist()),
        default=sorted(df["serving_cell"].unique().tolist())
    )

    selected_traffic = st.sidebar.multiselect(
        "Selecteaza clasa de trafic",
        options=sorted(df["traffic_class"].unique().tolist()),
        default=sorted(df["traffic_class"].unique().tolist())
    )

    selected_activity = st.sidebar.multiselect(
        "Selecteaza starea dispozitivului",
        options=sorted(df["is_active"].unique().tolist()),
        default=sorted(df["is_active"].unique().tolist())
    )

    time_min = int(df["time_step"].min())
    time_max = int(df["time_step"].max())

    selected_time = st.sidebar.slider(
        "Selecteaza intervalul de timp",
        min_value=time_min,
        max_value=time_max,
        value=(time_min, time_max)
    )

    # Filtrare
    filtered_df = df[
        (df["serving_cell"].isin(selected_cells)) &
        (df["traffic_class"].isin(selected_traffic)) &
        (df["is_active"].isin(selected_activity)) &
        (df["time_step"] >= selected_time[0]) &
        (df["time_step"] <= selected_time[1])
    ].copy()

    st.markdown("## Rezumat general")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Nr. randuri", f"{len(filtered_df):,}")
    col2.metric("Throughput mediu [Mbps]", f"{filtered_df['throughput_bps'].mean() / 1e6:.2f}")
    col3.metric("SINR mediu [dB]", f"{filtered_df['SINR_dB'].mean():.2f}")
    col4.metric("Latenta medie [ms]", f"{filtered_df['latency_ms'].mean():.2f}")
    col5.metric("PER mediu", f"{filtered_df['PER'].mean():.4f}")

    st.markdown("## Comparatie valori reale vs prezise")

    metric_option = st.selectbox(
        "Alege KPI-ul pentru comparatie",
        options=[
            "throughput_bps",
            "SINR_dB",
            "RSRP_dBm",
            "latency_ms",
            "PER"
        ]
    )

    pred_col_map = {
        "throughput_bps": "throughput_bps_pred",
        "SINR_dB": "SINR_dB_pred",
        "RSRP_dBm": "RSRP_dBm_pred",
        "latency_ms": "latency_ms_pred",
        "PER": "PER_pred"
    }

    pred_col = pred_col_map[metric_option]

    sample_df = filtered_df[[metric_option, pred_col]].dropna()
    if len(sample_df) > 5000:
        sample_df = sample_df.sample(5000, random_state=42)

    fig_scatter = px.scatter(
        sample_df,
        x=metric_option,
        y=pred_col,
        opacity=0.5,
        title=f"Valori reale vs valori prezise pentru {metric_option}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("## Evolutia KPI-urilor in timp")

    agg_df = filtered_df.groupby("time_step", as_index=False).agg({
        "throughput_bps": "mean",
        "throughput_bps_pred": "mean",
        "SINR_dB": "mean",
        "SINR_dB_pred": "mean",
        "latency_ms": "mean",
        "latency_ms_pred": "mean",
        "PER": "mean",
        "PER_pred": "mean"
    })

    time_metric_map = {
        "throughput_bps": ["throughput_bps", "throughput_bps_pred"],
        "SINR_dB": ["SINR_dB", "SINR_dB_pred"],
        "latency_ms": ["latency_ms", "latency_ms_pred"],
        "PER": ["PER", "PER_pred"]
    }

    if metric_option in time_metric_map:
        y_real, y_pred = time_metric_map[metric_option]

        fig_line = px.line(
            agg_df,
            x="time_step",
            y=[y_real, y_pred],
            title=f"Evolutie in timp pentru {metric_option}"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("## Distributia pe celule")

    cell_stats = filtered_df.groupby("serving_cell", as_index=False).agg({
        "throughput_bps": "mean",
        "SINR_dB": "mean",
        "RSRP_dBm": "mean",
        "latency_ms": "mean",
        "PER": "mean"
    })

    kpi_for_bar = st.selectbox(
        "Alege KPI-ul pentru comparatia intre celule",
        options=["throughput_bps", "SINR_dB", "RSRP_dBm", "latency_ms", "PER"],
        index=0,
        key="bar_metric"
    )

    fig_bar = px.bar(
        cell_stats,
        x="serving_cell",
        y=kpi_for_bar,
        title=f"Media {kpi_for_bar} pe fiecare celula"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("## Date filtrate")

    st.dataframe(filtered_df.head(500), use_container_width=True)

    st.markdown("## Informatii despre modelele finale")
    st.write("- Throughput: Random Forest")
    st.write("- SINR: Random Forest")
    st.write("- RSRP: Linear Regression")
    st.write("- Latency: Random Forest")
    st.write("- PER: Random Forest")


if __name__ == "__main__":
    main()