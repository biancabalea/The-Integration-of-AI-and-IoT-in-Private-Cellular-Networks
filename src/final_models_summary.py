from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    METRICS_DIR / "throughput_models_comparison.csv",
    METRICS_DIR / "sinr_models_comparison.csv",
    METRICS_DIR / "rsrp_models_comparison.csv",
    METRICS_DIR / "latency_models_comparison.csv",
    METRICS_DIR / "per_models_comparison.csv",
]

OUTPUT_SUMMARY_CSV = METRICS_DIR / "all_models_summary.csv"
OUTPUT_BEST_CSV = METRICS_DIR / "best_models_summary.csv"
OUTPUT_FIGURE = FIGURES_DIR / "best_models_r2_comparison.png"

def main():
    print("=== CENTRALIZARE REZULTATE FINALE ===")

    all_dfs = []
    for file_path in FILES:
        if file_path.exists():
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        else:
            print(f"Fisier lipsa: {file_path}")

    if not all_dfs:
        print("Nu exista fisiere de comparatie.")
        return

    summary_df = pd.concat(all_dfs, ignore_index=True)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\nTabel complet:")
    print(summary_df)

    # Selectare cel mai bun model per target pe baza R2 maxim
    best_models_df = summary_df.loc[summary_df.groupby("target")["R2"].idxmax()].copy()
    best_models_df = best_models_df.sort_values("target")
    best_models_df.to_csv(OUTPUT_BEST_CSV, index=False)

    print("\nCele mai bune modele per target:")
    print(best_models_df)

    # Grafic pentru cele mai bune modele
    plt.figure(figsize=(10, 6))
    plt.bar(best_models_df["target"], best_models_df["R2"])
    plt.xlabel("Target")
    plt.ylabel("R2")
    plt.title("Cele mai bune modele pentru fiecare target")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300)
    plt.close()

    print(f"\nTabel complet salvat in: {OUTPUT_SUMMARY_CSV}")
    print(f"Tabel best models salvat in: {OUTPUT_BEST_CSV}")
    print(f"Grafic salvat in: {OUTPUT_FIGURE}")

if __name__ == "__main__":
    main()