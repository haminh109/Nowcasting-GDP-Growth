from pathlib import Path
import json
import pandas as pd

LAYER1_DIR = Path("outputs/layer1_dfm")

PRIMARY_TARGET = "dfm_residual_third_release"
TRAIN_FLAG = "primary_target_available"
TAU_COL = "within_quarter_origin"

MIN_TRAIN_SIZE = 40
TEST_BLOCK_SIZE = 1

EXCLUDE_FIRST_PASS = {
    "news_signed__quarterly_target_history",
    "news_abs__quarterly_target_history",
}


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def load_layer2_artifacts(layer1_dir: Path = LAYER1_DIR):
    design_parquet = layer1_dir / "layer2_residual_design.parquet"
    design_csv = layer1_dir / "layer2_residual_design.csv"

    if design_parquet.exists():
        design = pd.read_parquet(design_parquet)
    elif design_csv.exists():
        design = pd.read_csv(design_csv)
    else:
        raise FileNotFoundError("Missing layer2_residual_design.parquet/csv")

    manifest = pd.read_csv(layer1_dir / "layer2_feature_manifest.csv")
    feature_sets = json.loads((layer1_dir / "layer2_feature_sets.json").read_text(encoding="utf-8"))
    contract = json.loads((layer1_dir / "layer2_data_contract.json").read_text(encoding="utf-8"))

    return design, manifest, feature_sets, contract


def as_month_period(series: pd.Series) -> pd.PeriodIndex:
    vals = []
    for raw in series.astype(str).str.strip():
        if len(raw) >= 7 and raw[4] == "-":
            vals.append(pd.Period(raw[:7], freq="M"))
        else:
            vals.append(pd.Timestamp(raw).to_period("M"))
    return pd.PeriodIndex(vals, freq="M")


def as_quarter_period(series: pd.Series) -> pd.PeriodIndex:
    vals = []
    for raw in series.astype(str).str.strip():
        if "Q" in raw:
            vals.append(pd.Period(raw, freq="Q-DEC"))
        else:
            vals.append(pd.Timestamp(raw).to_period("Q-DEC"))
    return pd.PeriodIndex(vals, freq="Q-DEC")


def month_start_string(m: pd.Period) -> str:
    return f"{m.year:04d}-{m.month:02d}-01"


def audit_feature_contract(manifest: pd.DataFrame, feature_sets: dict) -> list[str]:
    include_mask = (
        manifest["role"].astype(str).eq("feature")
        & as_bool(manifest["included_in_training_matrix"])
    )

    allowed_from_manifest = manifest.loc[include_mask, "column"].astype(str).tolist()
    baseline_features = feature_sets["baseline_v1"]

    assert set(baseline_features).issubset(set(allowed_from_manifest)), \
        "baseline_v1 contains columns outside allowed feature manifest"

    assert EXCLUDE_FIRST_PASS.isdisjoint(set(baseline_features)), \
        "baseline_v1 must exclude quarterly_target_history news features in first pass"

    return baseline_features


def prepare_tau_sample(
    design: pd.DataFrame,
    tau: int,
    target_col: str,
    train_flag: str,
) -> tuple[pd.DataFrame, dict]:
    d = design.copy()

    d["target_q"] = as_quarter_period(d["target_quarter"])
    d["vintage_m"] = as_month_period(d["vintage_period"])
    d[TAU_COL] = d[TAU_COL].astype(int)
    d["_train_flag_bool"] = as_bool(d[train_flag])

    raw_tau = d.loc[d[TAU_COL] == int(tau)].copy()
    raw_tau = raw_tau.sort_values(["target_q", "vintage_m"]).reset_index(drop=True)

    assert raw_tau["target_q"].is_monotonic_increasing, f"tau={tau}: raw target_q not monotone"
    assert raw_tau["vintage_m"].is_monotonic_increasing, f"tau={tau}: raw vintage_m not monotone"
    assert not raw_tau.duplicated(["target_q"]).any(), f"tau={tau}: duplicate raw target_quarter"
    assert not raw_tau.duplicated(["vintage_m"]).any(), f"tau={tau}: duplicate raw vintage_period"

    flagged_tau = raw_tau.loc[raw_tau["_train_flag_bool"]].copy().reset_index(drop=True)

    trainable_tau = raw_tau.loc[
        raw_tau["_train_flag_bool"] & raw_tau[target_col].notna()
    ].copy().reset_index(drop=True)

    assert trainable_tau["target_q"].is_monotonic_increasing, f"tau={tau}: filtered target_q not monotone"
    assert trainable_tau["vintage_m"].is_monotonic_increasing, f"tau={tau}: filtered vintage_m not monotone"
    assert not trainable_tau.duplicated(["target_q"]).any(), f"tau={tau}: duplicate filtered target_quarter"
    assert not trainable_tau.duplicated(["vintage_m"]).any(), f"tau={tau}: duplicate filtered vintage_period"

    if len(trainable_tau) >= 2:
        qdiff = pd.Series(trainable_tau["target_q"].astype(int)).diff().dropna()
        mdiff = pd.Series(trainable_tau["vintage_m"].astype(int)).diff().dropna()

        assert qdiff.eq(1).all(), f"tau={tau}: filtered target_quarter is not consecutive by 1 quarter"
        assert mdiff.eq(3).all(), f"tau={tau}: filtered vintage_period is not consecutive by 3 months"

    meta = {
        "tau": tau,
        "raw_rows": len(raw_tau),
        "flagged_rows": len(flagged_tau),
        "trainable_rows": len(trainable_tau),
        "dropped_unavailable_or_na": len(raw_tau) - len(trainable_tau),
    }

    return trainable_tau, meta


def expanding_folds(
    n_obs: int,
    min_train_size: int = MIN_TRAIN_SIZE,
    test_block_size: int = TEST_BLOCK_SIZE,
):
    if n_obs <= min_train_size:
        raise ValueError(f"Need n_obs > {min_train_size}, got {n_obs}")

    fold_id = 1
    split = min_train_size

    while split < n_obs:
        test_end = min(split + test_block_size, n_obs)

        train_idx = range(0, split)
        test_idx = range(split, test_end)

        if len(list(test_idx)) == 0:
            break

        yield fold_id, train_idx, test_idx

        split = test_end
        fold_id += 1


def build_fold_summary(df_tau: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for fold_id, train_idx, test_idx in expanding_folds(len(df_tau)):
        train = df_tau.iloc[list(train_idx)].copy()
        test = df_tau.iloc[list(test_idx)].copy()

        assert train["target_q"].max() < test["target_q"].min(), \
            f"fold {fold_id}: target quarter leakage detected"
        assert train["vintage_m"].max() < test["vintage_m"].min(), \
            f"fold {fold_id}: vintage leakage detected"

        rows.append(
            {
                "fold_id": fold_id,
                "train start": f"{train.iloc[0]['target_q']} / {month_start_string(train.iloc[0]['vintage_m'])}",
                "train end": f"{train.iloc[-1]['target_q']} / {month_start_string(train.iloc[-1]['vintage_m'])}",
                "test start": f"{test.iloc[0]['target_q']} / {month_start_string(test.iloc[0]['vintage_m'])}",
                "test end": f"{test.iloc[-1]['target_q']} / {month_start_string(test.iloc[-1]['vintage_m'])}",
                "n_train": len(train),
                "n_test": len(test),
            }
        )

    return pd.DataFrame(rows)


def iter_fold_local_views(
    df_tau: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
):
    for fold_id, train_idx, test_idx in expanding_folds(len(df_tau)):
        train = df_tau.iloc[list(train_idx)].copy()
        test = df_tau.iloc[list(test_idx)].copy()

        assert train["target_q"].max() < test["target_q"].min()
        assert train["vintage_m"].max() < test["vintage_m"].min()

        X_train = train[feature_cols].copy()
        y_train = train[target_col].copy()
        X_test = test[feature_cols].copy()
        y_test = test[target_col].copy()

        yield fold_id, X_train, y_train, X_test, y_test


design, manifest, feature_sets, contract = load_layer2_artifacts()

assert contract["primary_target"] == PRIMARY_TARGET
assert contract["train_sample_flag"] == TRAIN_FLAG

feature_cols = audit_feature_contract(manifest, feature_sets)

for tau in (1, 2, 3):
    df_tau, meta = prepare_tau_sample(
        design=design,
        tau=tau,
        target_col=contract["primary_target"],
        train_flag=contract["train_sample_flag"],
    )

    summary = build_fold_summary(df_tau)

    print(
        f"\n=== tau={tau} | "
        f"raw_rows={meta['raw_rows']} | "
        f"flagged_rows={meta['flagged_rows']} | "
        f"trainable_rows={meta['trainable_rows']} | "
        f"dropped_unavailable_or_na={meta['dropped_unavailable_or_na']} | "
        f"folds={len(summary)} ==="
    )
    print(summary.to_string(index=False))

# Optional Step 4 hook:
# for tau in (1, 2, 3):
#     df_tau, _ = prepare_tau_sample(
#         design=design,
#         tau=tau,
#         target_col=PRIMARY_TARGET,
#         train_flag=TRAIN_FLAG,
#     )
#     for fold_id, X_train, y_train, X_test, y_test in iter_fold_local_views(df_tau, feature_cols, PRIMARY_TARGET):
#         pass