from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_PATH = Path("data/raw/private_5g_iot_dataset_final.csv")
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATA
# =========================
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fisierul nu exista: {path}")
    df = pd.read_csv(path)
    return df

# =========================
# BASIC INSPECTION
# =========================
def inspect_dataset(df: pd.DataFrame) -> None:
    print("\n=== PRIMELE 5 RANDURI ===")
    print(df.head())

    print("\n=== DIMENSIUNE DATASET ===")
    print(df.shape)

    print("\n=== COLOANE ===")
    print(df.columns.tolist())

    print("\n=== TIPURI DE DATE ===")
    print(df.dtypes)

    print("\n=== VALORI LIPSA ===")
    print(df.isnull().sum())

    print("\n=== DUPLICATE ===")
    print(df.duplicated().sum())

# =========================
# DESCRIPTIVE STATS
# =========================
def descriptive_statistics(df: pd.DataFrame) -> None:
    print("\n=== STATISTICI DESCRIPTIVE ===")
    print(df.describe().T)

# =========================
# DATA CLEANING
# =========================
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eliminare duplicate, daca exista
    df = df.drop_duplicates()

    # Eliminare valori imposibile / neplauzibile
    # Poti ajusta dupa nevoie
    df = df[df["PER"].between(0, 1)]
    df = df[df["cell_load"].between(0, 1)]
    df = df[df["allocated_bandwidth_Hz"] >= 0]
    df = df[df["throughput_bps"] >= 0]
    df = df[df["latency_ms"] >= 0]

    return df

# =========================
# PLOTS
# =========================
def plot_histogram(series: pd.Series, title: str, xlabel: str, filename: str, bins: int = 40) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frecventa")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, filename: str, sample_size: int = 5000) -> None:
    plot_df = df[[x_col, y_col]].dropna()

    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df[x_col], plot_df[y_col], alpha=0.5, s=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame, filename: str = "correlation_matrix.png") -> None:
    numeric_df = df.select_dtypes(include=[np.number])

    corr = numeric_df.corr()

    plt.figure(figsize=(16, 12))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Matrice de corelatie")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

# =========================
# TARGET ANALYSIS
# =========================
def analyze_targets(df: pd.DataFrame) -> None:
    target_cols = ["RSRP_dBm", "SINR_dB", "throughput_bps", "latency_ms", "PER"]

    print("\n=== ANALIZA TARGET-URILOR ===")
    for col in target_cols:
        print(f"\n--- {col} ---")
        print(df[col].describe())

# =========================
# SAVE CLEANED DATA
# =========================
def save_cleaned_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDataset curatat salvat in: {output_path}")

# =========================
# MAIN
# =========================
def main() -> None:
    print("=== INCEPE ANALIZA DATASETULUI ===")

    df = load_dataset(DATA_PATH)

    inspect_dataset(df)
    descriptive_statistics(df)

    df_clean = clean_dataset(df)

    print("\n=== DUPA CURATARE ===")
    print(df_clean.shape)

    analyze_targets(df_clean)

    # Histograme KPI
    plot_histogram(df_clean["RSRP_dBm"], "Distributia RSRP", "RSRP [dBm]", "hist_rsrp.png")
    plot_histogram(df_clean["SINR_dB"], "Distributia SINR", "SINR [dB]", "hist_sinr.png")
    plot_histogram(df_clean["throughput_bps"] / 1e6, "Distributia Throughput", "Throughput [Mbps]", "hist_throughput.png")
    plot_histogram(df_clean["latency_ms"], "Distributia Latentei", "Latenta [ms]", "hist_latency.png")
    plot_histogram(df_clean["PER"], "Distributia PER", "PER", "hist_per.png")

    # Relatii importante
    plot_scatter(
        df_clean,
        "SINR_dB",
        "throughput_bps",
        "Throughput in functie de SINR",
        "SINR [dB]",
        "Throughput [bps]",
        "scatter_sinr_throughput.png"
    )

    plot_scatter(
        df_clean,
        "cell_load",
        "latency_ms",
        "Latenta in functie de incarcarea celulei",
        "Cell load",
        "Latenta [ms]",
        "scatter_load_latency.png"
    )

    plot_scatter(
        df_clean,
        "SINR_dB",
        "PER",
        "PER in functie de SINR",
        "SINR [dB]",
        "PER",
        "scatter_sinr_per.png"
    )

    # Corelatii
    plot_correlation_matrix(df_clean)

    # Salvare dataset curatat
    save_cleaned_dataset(df_clean, Path("data/processed/private_5g_iot_dataset_cleaned.csv"))

    print("\n=== ANALIZA FINALIZATA ===")
    print(f"Grafice salvate in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()