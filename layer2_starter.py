
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ================================================================
# Layer 2 starter script
# ================================================================
# This script assumes Layer 1 has already exported:
# - layer2_residual_design.parquet|csv
# - layer2_feature_manifest.csv
# - layer2_data_contract.json
#
# Date semantics:
# - vintage_period is monthly, first day of month semantics
# - target_quarter is quarterly, first day of quarter semantics
# Do NOT coerce them to month-end timestamps.
# ================================================================

LAYER1_OUTPUT_DIR = Path("outputs/layer1_dfm")
LAYER2_OUTPUT_DIR = Path("outputs/layer2")
PRIMARY_TARGET = "dfm_residual_third_release"
BASE_NOWCAST_COL = "dfm_nowcast"
ORIGIN_COL = "within_quarter_origin"
TIME_ORDER_COL = "vintage_period"
MIN_TRAIN_SIZE = 40
TEST_BLOCK_SIZE = 8


@dataclass
class BacktestResult:
    model_name: str
    tau: int
    predictions: pd.DataFrame
    metrics: Dict[str, float]


def load_layer2_inputs(layer1_output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    design_parquet = layer1_output_dir / "layer2_residual_design.parquet"
    design_csv = layer1_output_dir / "layer2_residual_design.csv"
    manifest_csv = layer1_output_dir / "layer2_feature_manifest.csv"
    contract_json = layer1_output_dir / "layer2_data_contract.json"

    if design_parquet.exists():
        design = pd.read_parquet(design_parquet)
    elif design_csv.exists():
        design = pd.read_csv(design_csv)
    else:
        raise FileNotFoundError("Could not find layer2_residual_design.parquet or layer2_residual_design.csv")

    manifest = pd.read_csv(manifest_csv)
    with open(contract_json, "r", encoding="utf-8") as f:
        contract = json.load(f)

    if "vintage_period" in design.columns:
        design["vintage_period"] = design["vintage_period"].astype(str)
    if "target_quarter" in design.columns:
        design["target_quarter"] = design["target_quarter"].astype(str)

    return design, manifest, contract


def included_feature_columns(manifest: pd.DataFrame) -> List[str]:
    keep = manifest.loc[
        (manifest["role"] == "feature") & (manifest["included_in_training_matrix"] == True),
        "column",
    ].astype(str)
    return keep.tolist()


def sort_for_real_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([TIME_ORDER_COL, "target_quarter"]).reset_index(drop=True)
    return out


def trainable_sample(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.loc[df[target_col].notna()].copy()
    out = sort_for_real_time(out)
    return out


def rmsfe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_models() -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "elastic_net": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000, random_state=42)),
            ]
        ),
        "gbr": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=2,
                    random_state=42,
                )),
            ]
        ),
        "rf": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=500,
                    max_depth=4,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
    }
    return models


def expanding_splits(n_obs: int, min_train_size: int = MIN_TRAIN_SIZE, test_block_size: int = TEST_BLOCK_SIZE):
    start = min_train_size
    while start < n_obs:
        train_end = start
        test_end = min(start + test_block_size, n_obs)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        if len(test_idx) == 0:
            break
        yield train_idx, test_idx
        start = test_end


def backtest_one_tau(
    df_tau: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_name: str,
    model: Pipeline,
) -> BacktestResult:
    df_tau = trainable_sample(df_tau, target_col)
    X = df_tau[feature_cols].copy()
    y = df_tau[target_col].astype(float).to_numpy()

    all_pred_rows = []

    for train_idx, test_idx in expanding_splits(len(df_tau)):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        fitted = clone(model)
        fitted.fit(X_train, y_train)
        y_hat = fitted.predict(X_test)

        fold = df_tau.iloc[test_idx][[
            "vintage_period",
            "target_quarter",
            "within_quarter_origin",
            BASE_NOWCAST_COL,
            target_col,
        ]].copy()
        fold["residual_hat"] = y_hat
        fold["hybrid_nowcast_hat"] = fold[BASE_NOWCAST_COL] + fold["residual_hat"]
        fold["hybrid_error"] = fold[target_col] - fold["residual_hat"]
        fold["model_name"] = model_name
        all_pred_rows.append(fold)

    if not all_pred_rows:
        pred_df = pd.DataFrame(columns=[
            "vintage_period", "target_quarter", "within_quarter_origin", BASE_NOWCAST_COL,
            target_col, "residual_hat", "hybrid_nowcast_hat", "hybrid_error", "model_name"
        ])
        metrics = {
            "n_oos": 0,
            "residual_rmsfe": np.nan,
            "residual_mae": np.nan,
            "hybrid_rmsfe": np.nan,
            "hybrid_mae": np.nan,
            "dfm_only_rmsfe": np.nan,
            "dfm_only_mae": np.nan,
        }
        return BacktestResult(model_name=model_name, tau=int(df_tau[ORIGIN_COL].iloc[0]), predictions=pred_df, metrics=metrics)

    pred_df = pd.concat(all_pred_rows, ignore_index=True)

    # residual metrics
    residual_true = pred_df[target_col].to_numpy(dtype=float)
    residual_pred = pred_df["residual_hat"].to_numpy(dtype=float)

    # DFM-only GDP error equals the residual target itself:
    # g_truth - g_dfm = residual
    dfm_only_gdp_error = residual_true
    hybrid_gdp_error = residual_true - residual_pred

    metrics = {
        "n_oos": int(len(pred_df)),
        "residual_rmsfe": rmsfe(residual_true, residual_pred),
        "residual_mae": float(mean_absolute_error(residual_true, residual_pred)),
        "hybrid_rmsfe": rmsfe(np.zeros_like(hybrid_gdp_error), hybrid_gdp_error),
        "hybrid_mae": float(mean_absolute_error(np.zeros_like(hybrid_gdp_error), hybrid_gdp_error)),
        "dfm_only_rmsfe": rmsfe(np.zeros_like(dfm_only_gdp_error), dfm_only_gdp_error),
        "dfm_only_mae": float(mean_absolute_error(np.zeros_like(dfm_only_gdp_error), dfm_only_gdp_error)),
    }
    return BacktestResult(model_name=model_name, tau=int(df_tau[ORIGIN_COL].iloc[0]), predictions=pred_df, metrics=metrics)


def run_backtests() -> None:
    LAYER2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    design, manifest, contract = load_layer2_inputs(LAYER1_OUTPUT_DIR)
    feature_cols = included_feature_columns(manifest)
    target_col = contract.get("primary_target", PRIMARY_TARGET)

    if target_col not in design.columns:
        raise KeyError(f"Primary target {target_col!r} not found in design table")
    if BASE_NOWCAST_COL not in design.columns:
        raise KeyError(f"Base nowcast column {BASE_NOWCAST_COL!r} not found in design table")
    if ORIGIN_COL not in design.columns:
        raise KeyError(f"Origin column {ORIGIN_COL!r} not found in design table")

    models = make_models()
    all_predictions = []
    metric_rows = []

    for tau in sorted(design[ORIGIN_COL].dropna().astype(int).unique().tolist()):
        df_tau = design.loc[design[ORIGIN_COL].astype(int) == int(tau)].copy()
        df_tau = sort_for_real_time(df_tau)

        print(f"\n=== Backtesting tau={tau} | rows={len(df_tau)} | trainable={df_tau[target_col].notna().sum()} ===")

        for model_name, model in models.items():
            result = backtest_one_tau(df_tau, feature_cols, target_col, model_name, model)
            preds = result.predictions.copy()
            preds["tau"] = tau
            all_predictions.append(preds)

            row = {"tau": tau, "model_name": model_name, **result.metrics}
            if np.isfinite(row["dfm_only_rmsfe"]) and np.isfinite(row["hybrid_rmsfe"]):
                row["rmsfe_gain_vs_dfm_pct"] = 100.0 * (row["dfm_only_rmsfe"] - row["hybrid_rmsfe"]) / row["dfm_only_rmsfe"]
            else:
                row["rmsfe_gain_vs_dfm_pct"] = np.nan
            metric_rows.append(row)
            print(row)

    pred_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)

    pred_df.to_csv(LAYER2_OUTPUT_DIR / "layer2_predictions_oos.csv", index=False)
    metrics_df.to_csv(LAYER2_OUTPUT_DIR / "layer2_metrics_summary.csv", index=False)

    run_config = {
        "layer1_output_dir": str(LAYER1_OUTPUT_DIR),
        "layer2_output_dir": str(LAYER2_OUTPUT_DIR),
        "primary_target": target_col,
        "base_nowcast_col": BASE_NOWCAST_COL,
        "origin_col": ORIGIN_COL,
        "time_order_col": TIME_ORDER_COL,
        "min_train_size": MIN_TRAIN_SIZE,
        "test_block_size": TEST_BLOCK_SIZE,
        "models": list(models.keys()),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
    }
    with open(LAYER2_OUTPUT_DIR / "layer2_run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\nSaved:")
    print("- outputs/layer2/layer2_predictions_oos.csv")
    print("- outputs/layer2/layer2_metrics_summary.csv")
    print("- outputs/layer2/layer2_run_config.json")


if __name__ == "__main__":
    run_backtests()
