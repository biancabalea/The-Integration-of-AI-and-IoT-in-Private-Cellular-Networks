from pathlib import Path
import shutil
import pandas as pd

# =========================
# PATHS
# =========================
MODELS_DIR = Path("models")
FINAL_MODELS_DIR = MODELS_DIR / "final"
METRICS_DIR = Path("results/metrics")

FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAPPING = {
    "throughput_bps": {
        "source": MODELS_DIR / "random_forest_throughput.pkl",
        "destination": FINAL_MODELS_DIR / "throughput_model.pkl",
        "model_name": "RandomForest"
    },
    "SINR_dB": {
        "source": MODELS_DIR / "random_forest_sinr.pkl",
        "destination": FINAL_MODELS_DIR / "sinr_model.pkl",
        "model_name": "RandomForest"
    },
    "RSRP_dBm": {
        "source": MODELS_DIR / "linear_regression_rsrp.pkl",
        "destination": FINAL_MODELS_DIR / "rsrp_model.pkl",
        "model_name": "LinearRegression"
    },
    "latency_ms": {
        "source": MODELS_DIR / "random_forest_latency.pkl",
        "destination": FINAL_MODELS_DIR / "latency_model.pkl",
        "model_name": "RandomForest"
    },
    "PER": {
        "source": MODELS_DIR / "random_forest_per.pkl",
        "destination": FINAL_MODELS_DIR / "per_model.pkl",
        "model_name": "RandomForest"
    }
}

OUTPUT_SUMMARY = METRICS_DIR / "selected_final_models.csv"


def main():
    print("=== ORGANIZARE MODELE FINALE ===")

    selected_models = []

    for target, info in MODEL_MAPPING.items():
        source = info["source"]
        destination = info["destination"]
        model_name = info["model_name"]

        if not source.exists():
            print(f"Lipseste fisierul model pentru {target}: {source}")
            continue

        shutil.copy2(source, destination)

        selected_models.append({
            "target": target,
            "selected_model": model_name,
            "source_file": str(source),
            "final_file": str(destination)
        })

        print(f"Model selectat pentru {target}: {model_name}")
        print(f"Copiat in: {destination}")

    summary_df = pd.DataFrame(selected_models)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    print(f"\nRezumatul modelelor finale a fost salvat in: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()