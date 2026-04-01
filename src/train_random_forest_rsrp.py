from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_DIR = Path("data/processed/ml_ready_rsrp")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)

def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test

def main():
    print("=== RANDOM FOREST: RSRP PREDICTION ===")

    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R2   = {r2:.4f}")

    joblib.dump(model, MODEL_DIR / "random_forest_rsrp.pkl")

    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    predictions_df.to_csv(
        RESULTS_DIR / "predictions" / "random_forest_rsrp_predictions.csv",
        index=False
    )

    metrics_df = pd.DataFrame([{
        "model": "RandomForest",
        "target": "RSRP_dBm",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }])
    metrics_df.to_csv(
        RESULTS_DIR / "metrics" / "random_forest_rsrp_metrics.csv",
        index=False
    )

    feature_importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    feature_importance_df.to_csv(
        RESULTS_DIR / "metrics" / "random_forest_rsrp_feature_importance.csv",
        index=False
    )

    sample_df = predictions_df.sample(n=min(3000, len(predictions_df)), random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(sample_df["y_true"], sample_df["y_pred"], alpha=0.5, s=10)
    plt.xlabel("Valori reale RSRP [dBm]")
    plt.ylabel("Valori prezise RSRP [dBm]")
    plt.title("Random Forest - RSRP: real vs prezis")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "figures" / "random_forest_rsrp_real_vs_pred.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    top_features = feature_importance_df.head(10)
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importanta")
    plt.ylabel("Feature")
    plt.title("Top 10 feature importance - Random Forest RSRP")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "figures" / "random_forest_rsrp_feature_importance.png", dpi=300)
    plt.close()

    print("Fisierele au fost salvate cu succes.")

if __name__ == "__main__":
    main()