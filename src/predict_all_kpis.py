from pathlib import Path
import pandas as pd
import joblib

# =========================
# PATHS
# =========================
INPUT_PATH = Path("data/processed/private_5g_iot_dataset_cleaned.csv")
FINAL_MODELS_DIR = Path("models/final")
OUTPUT_DIR = Path("results/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "final_predictions.csv"

# =========================
# FEATURES
# =========================
FEATURES = [
    "traffic_class",
    "serving_cell",
    "is_active",
    "active_devices_serving_cell",
    "active_devices_total",
    "cell_load",
    "distance_serving_m",
    "distance_neighbor1_m",
    "distance_neighbor2_m",
    "pathloss_serving_dB",
    "pathloss_neighbor1_dB",
    "pathloss_neighbor2_dB",
    "packet_size_bytes",
    "packet_rate_pps",
    "generated_traffic_bps",
    "allocated_bandwidth_Hz"
]

# =========================
# TARGETS
# =========================
TARGETS = {
    "throughput_bps": "throughput_model.pkl",
    "SINR_dB": "sinr_model.pkl",
    "RSRP_dBm": "rsrp_model.pkl",
    "latency_ms": "latency_model.pkl",
    "PER": "per_model.pkl"
}

# =========================
# LOAD MODELS
# =========================
def load_models():
    models = {}
    for target, model_file in TARGETS.items():
        model_path = FINAL_MODELS_DIR / model_file

        if not model_path.exists():
            raise FileNotFoundError(f"Model lipsa pentru {target}: {model_path}")

        models[target] = joblib.load(model_path)

    return models

# =========================
# MAIN
# =========================
def main():
    print("=== PREDICTIE FINALA PENTRU TOATE KPI-URILE ===")

    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset incarcat: {df.shape}")

    # pastram coloane utile pentru dashboard
    base_columns = [
        "time_step",
        "device_id",
        "traffic_class",
        "serving_cell",
        "is_active",
        "x_m",
        "y_m",
        "active_devices_serving_cell",
        "active_devices_total",
        "cell_load",
        "distance_serving_m",
        "packet_size_bytes",
        "packet_rate_pps",
        "generated_traffic_bps",
        "allocated_bandwidth_Hz",
        "RSRP_dBm",
        "SINR_dB",
        "throughput_bps",
        "latency_ms",
        "PER"
    ]

    result_df = df[base_columns].copy()
    X = df[FEATURES].copy()

    models = load_models()

    # Predictii pentru fiecare target
    for target, model in models.items():
        y_pred = model.predict(X)

        pred_col = f"{target}_pred"

        if target == "PER":
            y_pred = pd.Series(y_pred).clip(lower=0, upper=1)
        elif target in ["throughput_bps", "latency_ms"]:
            y_pred = pd.Series(y_pred).clip(lower=0)
        else:
            y_pred = pd.Series(y_pred)

        result_df[pred_col] = y_pred

        print(f"Predictii generate pentru: {target}")

    # erori absolute utile pentru dashboard/analiza
    result_df["throughput_abs_error"] = (result_df["throughput_bps"] - result_df["throughput_bps_pred"]).abs()
    result_df["sinr_abs_error"] = (result_df["SINR_dB"] - result_df["SINR_dB_pred"]).abs()
    result_df["rsrp_abs_error"] = (result_df["RSRP_dBm"] - result_df["RSRP_dBm_pred"]).abs()
    result_df["latency_abs_error"] = (result_df["latency_ms"] - result_df["latency_ms_pred"]).abs()
    result_df["per_abs_error"] = (result_df["PER"] - result_df["PER_pred"]).abs()

    result_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFisierul final cu predictii a fost salvat in: {OUTPUT_FILE}")
    print(f"Dimensiune fisier final: {result_df.shape}")


if __name__ == "__main__":
    main()