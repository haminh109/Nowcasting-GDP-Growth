from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


LAYER1 = Path("outputs/layer1_dfm")
OUT = Path("outputs/layer2_step4_first_pass")

PRIMARY_TARGET = "dfm_residual_third_release"
TRUTH_COL = "third_release"
BASE_COL = "dfm_nowcast"
ORIGIN_COL = "within_quarter_origin"
FEATURE_SET_NAME = "baseline_v1"

MIN_TRAIN_SIZE = 40
TEST_BLOCK_SIZE = 1

EXCLUDED_FIRST_PASS = {
    "news_signed__quarterly_target_history",
    "news_abs__quarterly_target_history",
}


def rmsfe(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .map({"true": True, "false": False, "1": True, "0": False})
    )


def baseline_feature_columns(manifest: pd.DataFrame, feature_sets: dict, contract: dict) -> list[str]:
    eligible = set(
        manifest.loc[
            (manifest["role"] == "feature") &
            (manifest["included_in_training_matrix"] == True),
            "column",
        ].astype(str)
    )

    baseline_v1 = list(feature_sets[FEATURE_SET_NAME])

    forbidden = set(contract["forbidden_feature_columns"]) | set(contract["audit_only_fields"])

    assert set(baseline_v1).issubset(eligible)
    assert EXCLUDED_FIRST_PASS.isdisjoint(baseline_v1)
    assert not (set(baseline_v1) & forbidden)

    return baseline_v1


def make_models() -> dict[str, Pipeline]:
    return {
        "elastic_net": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000, random_state=42)),
        ]),
        "gbr": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                random_state=42
            )),
        ]),
    }


def expanding_walk_forward_splits(n_obs: int, min_train_size: int = 40):
    start = min_train_size
    fold_id = 1
    while start < n_obs:
        yield fold_id, np.arange(0, start), np.array([start])
        start += 1
        fold_id += 1


def summarize_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    scopes = [("overall", pred)]
    for tau in [1, 2, 3]:
        scopes.append((f"tau_{tau}", pred.loc[pred[ORIGIN_COL] == tau].copy()))

    for scope_name, df_scope in scopes:
        if df_scope.empty:
            continue

        for model_name, g in df_scope.groupby("model_name"):
            y_true = g[TRUTH_COL].to_numpy()
            y_dfm = g[BASE_COL].to_numpy()
            y_hybrid = g["hybrid_nowcast_hat"].to_numpy()

            mae_dfm = mean_absolute_error(y_true, y_dfm)
            rmsfe_dfm = rmsfe(y_true, y_dfm)

            mae_hybrid = mean_absolute_error(y_true, y_hybrid)
            rmsfe_hybrid = rmsfe(y_true, y_hybrid)

            rows.append({
                "scope": scope_name,
                "model_name": model_name,
                "n_obs": int(len(g)),
                "mae_dfm": float(mae_dfm),
                "rmsfe_dfm": float(rmsfe_dfm),
                "mae_hybrid": float(mae_hybrid),
                "rmsfe_hybrid": float(rmsfe_hybrid),
                "mae_improvement": float(mae_dfm - mae_hybrid),
                "rmsfe_improvement": float(rmsfe_dfm - rmsfe_hybrid),
                "mae_improvement_pct": float((mae_dfm - mae_hybrid) / mae_dfm * 100.0) if mae_dfm != 0 else np.nan,
                "rmsfe_improvement_pct": float((rmsfe_dfm - rmsfe_hybrid) / rmsfe_dfm * 100.0) if rmsfe_dfm != 0 else np.nan,
            })

    return pd.DataFrame(rows)


design = pd.read_csv(LAYER1 / "layer2_residual_design.csv")
manifest = pd.read_csv(LAYER1 / "layer2_feature_manifest.csv")
feature_sets = json.loads((LAYER1 / "layer2_feature_sets.json").read_text())
contract = json.loads((LAYER1 / "layer2_data_contract.json").read_text())

design["vintage_period"] = pd.PeriodIndex(design["vintage_period"], freq="M")
design["target_quarter"] = pd.PeriodIndex(design["target_quarter"], freq="Q-DEC")
design[ORIGIN_COL] = design[ORIGIN_COL].astype(int)
design["primary_target_available"] = to_bool_series(design["primary_target_available"])

assert contract["primary_target"] == PRIMARY_TARGET

feature_cols = baseline_feature_columns(manifest, feature_sets, contract)
models = make_models()

trainable = (
    design.loc[design["primary_target_available"] == True]
          .loc[design[PRIMARY_TARGET].notna()]
          .sort_values(["target_quarter", "vintage_period"])
          .reset_index(drop=True)
)

pred_rows = []

for tau in [1, 2, 3]:
    df_tau = (
        trainable.loc[trainable[ORIGIN_COL] == tau]
                 .sort_values(["target_quarter", "vintage_period"])
                 .reset_index(drop=True)
    )

    if len(df_tau) <= MIN_TRAIN_SIZE:
        continue

    X = df_tau[feature_cols]
    y = df_tau[PRIMARY_TARGET].to_numpy()

    for fold_id, train_idx, test_idx in expanding_walk_forward_splits(len(df_tau), MIN_TRAIN_SIZE):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]

        base_fold = df_tau.iloc[test_idx][[
            "vintage_period",
            "target_quarter",
            ORIGIN_COL,
            TRUTH_COL,
            BASE_COL,
            PRIMARY_TARGET,
        ]].copy()

        base_fold["fold_id"] = fold_id
        base_fold["n_train"] = len(train_idx)

        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)

            residual_hat = fitted.predict(X_test)

            fold = base_fold.copy()
            fold["model_name"] = model_name
            fold["residual_hat"] = residual_hat
            fold["hybrid_nowcast_hat"] = fold[BASE_COL] + fold["residual_hat"]
            fold["dfm_error"] = fold[TRUTH_COL] - fold[BASE_COL]
            fold["hybrid_error"] = fold[TRUTH_COL] - fold["hybrid_nowcast_hat"]

            pred_rows.append(fold)

if not pred_rows:
    raise ValueError("No OOS predictions were generated. Check trainable rows and min_train_size.")

pred = pd.concat(pred_rows, ignore_index=True)

OUT.mkdir(parents=True, exist_ok=True)

pred_to_save = pred.assign(
    vintage_period=pred["vintage_period"].astype(str),
    target_quarter=pred["target_quarter"].astype(str),
)

pred_to_save.to_csv(OUT / "step4_predictions_oos_long.csv", index=False)

wide = (
    pred_to_save.pivot_table(
        index=[
            "vintage_period",
            "target_quarter",
            ORIGIN_COL,
            TRUTH_COL,
            BASE_COL,
            PRIMARY_TARGET,
            "fold_id",
            "n_train",
            "dfm_error",
        ],
        columns="model_name",
        values=["residual_hat", "hybrid_nowcast_hat", "hybrid_error"],
        aggfunc="first",
    )
    .sort_index()
)

wide.columns = [f"{a}__{b}" for a, b in wide.columns]
wide = wide.reset_index()
wide.to_csv(OUT / "step4_predictions_oos_wide.csv", index=False)

metrics = summarize_metrics(pred)
metrics.to_csv(OUT / "step4_metrics_summary.csv", index=False)

metadata = {
    "primary_target": PRIMARY_TARGET,
    "truth_col": TRUTH_COL,
    "base_col": BASE_COL,
    "origin_col": ORIGIN_COL,
    "feature_set_name": FEATURE_SET_NAME,
    "n_features": len(feature_cols),
    "feature_columns": feature_cols,
    "excluded_first_pass": sorted(EXCLUDED_FIRST_PASS),
    "min_train_size": MIN_TRAIN_SIZE,
    "test_block_size": TEST_BLOCK_SIZE,
    "models": list(models.keys()),
    "n_oos_rows_long": int(len(pred_to_save)),
    "n_unique_test_points": int(
        pred_to_save[["vintage_period", "target_quarter", ORIGIN_COL]].drop_duplicates().shape[0]
    ),
}

(OUT / "step4_run_metadata.json").write_text(json.dumps(metadata, indent=2))