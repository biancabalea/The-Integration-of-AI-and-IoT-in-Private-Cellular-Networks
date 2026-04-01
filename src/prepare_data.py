from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
INPUT_PATH = Path("data/processed/private_5g_iot_dataset_cleaned.csv")
OUTPUT_DIR = Path("data/processed/ml_ready")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# FEATURE SELECTION
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

TARGET = "throughput_bps"

# =========================
# MAIN
# =========================
def main():
    print("=== PREPARARE DATE PENTRU ML ===")

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset initial:", df.shape)

    # selectam doar coloanele necesare
    df_model = df[FEATURES + [TARGET]].copy()

    print("Dataset dupa selectie:", df_model.shape)

    # separare X si y
    X = df_model[FEATURES]
    y = df_model[TARGET]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print("\nTrain set:", X_train.shape)
    print("Test set:", X_test.shape)

    # salvare
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print("\nDate salvate in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()