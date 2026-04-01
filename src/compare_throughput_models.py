from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LINEAR_PATH = METRICS_DIR / "linear_regression_throughput_metrics.csv"
RF_PATH = METRICS_DIR / "random_forest_throughput_metrics.csv"

OUTPUT_COMPARISON_CSV = METRICS_DIR / "throughput_models_comparison.csv"
OUTPUT_COMPARISON_PNG = FIGURES_DIR / "throughput_models_comparison.png"

def main():
    print("=== COMPARATIE MODELE THROUGHPUT ===")

    linear_df = pd.read_csv(LINEAR_PATH)
    rf_df = pd.read_csv(RF_PATH)

    comparison_df = pd.concat([linear_df, rf_df], ignore_index=True)
    comparison_df.to_csv(OUTPUT_COMPARISON_CSV, index=False)

    print("\nTabel comparativ:")
    print(comparison_df)

    # Grafic R2
    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["model"], comparison_df["R2"])
    plt.xlabel("Model")
    plt.ylabel("R2")
    plt.title("Comparatie modele pentru predictia throughput")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_COMPARISON_PNG, dpi=300)
    plt.close()

    print(f"\nComparatia a fost salvata in: {OUTPUT_COMPARISON_CSV}")
    print(f"Graficul a fost salvat in: {OUTPUT_COMPARISON_PNG}")

if __name__ == "__main__":
    main()