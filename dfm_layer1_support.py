from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

import hybrid_nowcast_utils as base


DEFAULT_ARTIFACTS = [
    "dfm_nowcasts.csv",
    "dfm_states.csv",
    "dfm_news_series.csv",
    "dfm_news_blocks.csv",
    "dfm_coverage.csv",
    "dfm_diagnostics.csv",
    "vintage_manifest.csv",
]


@dataclass
class ProjectFiles:
    project_root: Path
    raw_dir: Path
    output_dir: Path
    experiment_config: Optional[Path]
    routput_monthly: Path
    routput_release_values: Path
    gdpplus_vintages: Optional[Path]
    spf_mean_growth: Optional[Path]
    spf_median_growth: Optional[Path]
    fred_md_dirs: List[Path]
    full_panel_map: Optional[Path] = None


@dataclass
class PreparedFitInput:
    vintage: pd.Period
    quarter: pd.Period
    impact_month: pd.Period
    panel_mode: str
    source_path: Path
    metadata: pd.DataFrame
    monthly_raw: pd.DataFrame
    quarterly_raw: pd.DataFrame
    monthly_std: pd.DataFrame
    quarterly_std: pd.DataFrame
    monthly_means: pd.Series
    monthly_stds: pd.Series
    quarterly_means: pd.Series
    quarterly_stds: pd.Series
    coverage: pd.DataFrame
    tcodes: pd.Series


@dataclass
class DFMVintageBundle:
    prepared: PreparedFitInput
    lag_order: int
    result: object
    states: pd.DataFrame
    loadings: pd.DataFrame
    diagnostics: pd.DataFrame
    nowcast_table: pd.DataFrame


# -----------------------------------------------------------------------------
# Project / file discovery
# -----------------------------------------------------------------------------


def _find_first(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None



def discover_project_files(project_root: Path | str = ".") -> ProjectFiles:
    root = Path(project_root).resolve()
    raw_dir = root / "data" / "raw"
    if not raw_dir.exists():
        candidate = _find_first(root, ["raw"])
        if candidate is None:
            raise FileNotFoundError(
                f"Could not locate a 'data/raw' directory below {root}."
            )
        raw_dir = candidate

    experiment_config = root / "experiment_config.json"
    if not experiment_config.exists():
        experiment_config = _find_first(root, ["experiment_config.json"])

    routput_monthly = _find_first(raw_dir, ["routputMvQd.xlsx", "ROUTPUTMvQd.xlsx"])
    routput_release = _find_first(raw_dir, ["routput_first_second_third.xlsx"])
    gdpplus = _find_first(raw_dir, ["GDPplus_Vintages.xlsx"])
    spf_mean = _find_first(raw_dir, ["meanGrowth.xlsx"])
    spf_median = _find_first(raw_dir, ["medianGrowth.xlsx"])

    if routput_monthly is None:
        raise FileNotFoundError("Could not find routputMvQd.xlsx in data/raw.")
    if routput_release is None:
        raise FileNotFoundError(
            "Could not find routput_first_second_third.xlsx in data/raw."
        )

    fred_md_dir_names = [
        "FRED-MD-MONTHLY",
        "Historical FRED-MD Vintages Final",
        "Historical-vintages-of-FRED-MD-2015-01-to-2024-12",
    ]
    fred_md_dirs: List[Path] = []
    for name in fred_md_dir_names:
        path = raw_dir / name
        if path.exists():
            fred_md_dirs.append(path)

    # Fallback: any directory under raw containing 'FRED-MD'
    if not fred_md_dirs:
        for path in sorted(raw_dir.iterdir()):
            if path.is_dir() and "FRED-MD" in path.name:
                fred_md_dirs.append(path)

    if not fred_md_dirs:
        raise FileNotFoundError(
            "Could not find any local FRED-MD directory below data/raw."
        )

    full_panel_map = _find_first(
        root,
        [
            "series_to_block_full_panel.csv",
            "full_panel_block_map.csv",
            "series_to_block.csv",
        ],
    )

    output_dir = root / "outputs" / "dfm_layer1"
    output_dir.mkdir(parents=True, exist_ok=True)

    return ProjectFiles(
        project_root=root,
        raw_dir=raw_dir,
        output_dir=output_dir,
        experiment_config=experiment_config,
        routput_monthly=routput_monthly,
        routput_release_values=routput_release,
        gdpplus_vintages=gdpplus,
        spf_mean_growth=spf_mean,
        spf_median_growth=spf_median,
        fred_md_dirs=fred_md_dirs,
        full_panel_map=full_panel_map,
    )



def load_experiment_config(config_path: Optional[Path]) -> Dict:
    if config_path is None or not config_path.exists():
        return {
            "project_title": "Real-Time GDP Growth Nowcasting using a Hybrid Dynamic Factor Model with Machine Learning Residual Correction",
            "country": "United States",
            "forecast_task": "Real-time nowcasting of current-quarter real GDP growth at each month-end vintage",
            "benchmark_window": {"start_quarter": "2000Q1", "end_quarter": None},
            "forecast_origins_per_quarter": 3,
            "main_truth": "third-release real GDP growth",
            "robustness_truths": ["latest RTDSM value", "GDPplus"],
            "main_predictor_panel": "FRED-MD vintage archive",
            "supplementary_predictor_panel": "FRED-QD vintage archive (robustness only)",
            "external_benchmark": "Survey of Professional Forecasters (RGDP mean growth)",
            "real_time_rule": "Every observation used at vintage v must be recoverable exactly as of that month-end.",
            "dfm_blocks": [
                "real activity & income",
                "labor market",
                "housing & construction",
                "demand / orders / inventories",
                "prices & inflation",
                "financial conditions",
            ],
            "baseline_factor_design": {
                "global_factor": 1,
                "block_factors": 6,
                "factor_lag_grid": [1, 2, 3],
                "idiosyncratic_ar1": True,
            },
            "artifact_files": DEFAULT_ARTIFACTS.copy(),
        }
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Local FRED-MD catalog
# -----------------------------------------------------------------------------


def parse_local_fred_md_filename(filename: str) -> Optional[pd.Period]:
    name = Path(filename).name
    patterns = [
        r"^(?P<year>\d{4})-(?P<month>\d{2})(?:-MD)?\.csv$",
        r"^FRED_MD_(?P<year>\d{4})m(?P<month>\d{2})\.csv$",
        r"^FRED-MD_(?P<year>\d{4})m(?P<month>\d{2})\.csv$",
    ]
    for pattern in patterns:
        match = re.match(pattern, name, flags=re.IGNORECASE)
        if match:
            year = int(match.group("year"))
            month = int(match.group("month"))
            return pd.Period(f"{year}-{month:02d}", freq="M")
    return None



def _source_priority(path: Path) -> int:
    name = str(path)
    if "FRED-MD-MONTHLY" in name:
        return 0
    if "Historical-vintages-of-FRED-MD-2015-01-to-2024-12" in name:
        return 1
    if "Historical FRED-MD Vintages Final" in name:
        return 2
    return 9



def build_fred_md_catalog(fred_md_dirs: Sequence[Path | str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for directory in fred_md_dirs:
        directory = Path(directory)
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            vintage = parse_local_fred_md_filename(path.name)
            if vintage is None:
                continue
            rows.append(
                {
                    "vintage": vintage,
                    "path": path,
                    "source_dir": directory.name,
                    "priority": _source_priority(path),
                }
            )
    if not rows:
        raise FileNotFoundError("No local monthly FRED-MD csv files were found.")

    catalog = pd.DataFrame(rows).sort_values(["vintage", "priority", "path"]).copy()
    catalog = catalog.drop_duplicates(subset=["vintage"], keep="first")
    return catalog.reset_index(drop=True)



def load_local_fred_md_snapshot(
    vintage: pd.Period,
    catalog: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, Path]:
    row = catalog.loc[catalog["vintage"] == vintage]
    if row.empty:
        raise FileNotFoundError(f"No local FRED-MD snapshot found for {vintage}.")
    path = Path(row.iloc[0]["path"])
    snapshot, tcodes = base.parse_fred_md_csv(path)
    return snapshot, tcodes, path


# -----------------------------------------------------------------------------
# Target / truth construction and status summary
# -----------------------------------------------------------------------------


def build_target_and_truth_tables(files: ProjectFiles) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    target_vintage_table = base.build_target_vintage_table(files.routput_monthly)
    truth_tables = base.build_truth_tables(
        files.routput_release_values,
        gdpplus_path=files.gdpplus_vintages,
        spf_mean_growth_path=files.spf_mean_growth,
    )
    return target_vintage_table, truth_tables



def compute_data_status(
    target_vintage_table: pd.DataFrame,
    truth_tables: Mapping[str, pd.DataFrame],
    fred_md_catalog: pd.DataFrame,
) -> pd.DataFrame:
    latest_predictor_vintage = fred_md_catalog["vintage"].max()
    latest_target_history_vintage = target_vintage_table["vintage"].max()

    latest_target_history_quarter = (
        target_vintage_table.loc[target_vintage_table["vintage"] == latest_predictor_vintage, "quarter"].max()
        if (target_vintage_table["vintage"] == latest_predictor_vintage).any()
        else target_vintage_table["quarter"].max()
    )

    third_truth = truth_tables["truth_third_release"].dropna(subset=["truth_third_release"])
    latest_main_truth_quarter = third_truth["quarter"].max() if not third_truth.empty else pd.NaT

    latest_latest_truth_quarter = truth_tables["truth_latest_rtdsm"].dropna(subset=["truth_latest_rtdsm"])["quarter"].max()
    latest_gdpplus_quarter = (
        truth_tables["truth_gdpplus_latest"].dropna(subset=["truth_gdpplus_latest"])["quarter"].max()
        if "truth_gdpplus_latest" in truth_tables
        else pd.NaT
    )

    rows = [
        ("latest_predictor_vintage", str(latest_predictor_vintage)),
        ("latest_target_history_vintage", str(latest_target_history_vintage)),
        ("latest_target_history_quarter_at_latest_predictor", str(latest_target_history_quarter)),
        ("latest_main_truth_quarter", str(latest_main_truth_quarter)),
        ("latest_latest_rtdsm_truth_quarter", str(latest_latest_truth_quarter)),
        ("latest_gdpplus_truth_quarter", str(latest_gdpplus_quarter)),
        ("latest_forecastable_quarter_from_predictors", str(latest_predictor_vintage.asfreq("Q"))),
    ]
    return pd.DataFrame(rows, columns=["object", "value"])


# -----------------------------------------------------------------------------
# Metadata and canonicalization
# -----------------------------------------------------------------------------


def stable_metadata() -> pd.DataFrame:
    return base.stable_subset_metadata().copy()



def canonicalize_snapshot_columns(
    snapshot: pd.DataFrame,
    tcodes: pd.Series,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    snapshot = snapshot.copy()
    tcodes = tcodes.copy()

    for _, row in metadata.iterrows():
        canonical = str(row["mnemonic"]).strip()
        alias = str(row.get("alias_or_crosswalk", "")).strip()
        if not alias or alias.lower() == "nan":
            continue

        if canonical not in snapshot.columns and alias in snapshot.columns:
            snapshot[canonical] = snapshot[alias]
        if canonical not in tcodes.index and alias in tcodes.index:
            tcodes.loc[canonical] = tcodes.loc[alias]

    return snapshot, tcodes



def _normalize_block_name(value: str) -> str:
    value = str(value).strip().lower()
    mapping = {
        "real activity & income": "real_activity_income",
        "real_activity_income": "real_activity_income",
        "labor market": "labor_market",
        "labor_market": "labor_market",
        "housing & construction": "housing_construction",
        "housing_construction": "housing_construction",
        "demand / orders / inventories": "demand_orders_inventories",
        "demand, orders & inventories": "demand_orders_inventories",
        "demand_orders_inventories": "demand_orders_inventories",
        "prices & inflation": "prices_inflation",
        "prices_inflation": "prices_inflation",
        "financial conditions": "financial_conditions",
        "financial_conditions": "financial_conditions",
    }
    return mapping.get(value, value.replace(" ", "_").replace("&", "").replace("/", "_"))



def load_full_panel_map(
    full_panel_map_path: Path | str,
    tcodes: pd.Series,
) -> pd.DataFrame:
    full_panel_map_path = Path(full_panel_map_path)
    df = pd.read_csv(full_panel_map_path)
    required = {"mnemonic", "block"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Full-panel map {full_panel_map_path} is missing required columns: {sorted(missing)}"
        )
    if "anchor" not in df.columns:
        df["anchor"] = False
    if "description" not in df.columns:
        df["description"] = ""
    if "include_stable_subset" not in df.columns:
        df["include_stable_subset"] = False
    if "caution_note" not in df.columns:
        df["caution_note"] = ""
    if "alias_or_crosswalk" not in df.columns:
        df["alias_or_crosswalk"] = ""
    if "concept_id" not in df.columns:
        df["concept_id"] = ""
    if "tcode" not in df.columns:
        df["tcode"] = np.nan

    df["block"] = df["block"].map(_normalize_block_name)
    df["tcode"] = df["tcode"].where(df["tcode"].notna(), df["mnemonic"].map(tcodes))
    if df["tcode"].isna().any():
        unresolved = df.loc[df["tcode"].isna(), "mnemonic"].tolist()
        warnings.warn(
            "Some full-panel mapped series do not have tcodes resolved from the snapshot: "
            + ", ".join(unresolved[:10])
            + (" ..." if len(unresolved) > 10 else "")
        )
    return df[
        [
            "concept_id",
            "mnemonic",
            "block",
            "tcode",
            "alias_or_crosswalk",
            "anchor",
            "description",
            "include_stable_subset",
            "caution_note",
        ]
    ].copy()



def generate_full_panel_template(
    snapshot_columns: Sequence[str],
    stable_meta: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    template = base.generate_full_panel_block_template(snapshot_columns, stable_meta)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    return output_path


# -----------------------------------------------------------------------------
# Vintage preparation
# -----------------------------------------------------------------------------


def standardize_with_reference(
    panel: pd.DataFrame,
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    means = means.reindex(panel.columns)
    stds = stds.reindex(panel.columns).replace(0, np.nan)
    return (panel - means) / stds



def prepare_vintage_fit_input(
    vintage: pd.Period,
    snapshot: pd.DataFrame,
    tcodes: pd.Series,
    source_path: Path,
    target_vintage_table: pd.DataFrame,
    stable_meta: pd.DataFrame,
    panel_mode: str = "stable_subset",
    full_panel_map_path: Optional[Path | str] = None,
) -> PreparedFitInput:
    snapshot, tcodes = canonicalize_snapshot_columns(snapshot, tcodes, stable_meta)

    if panel_mode == "stable_subset":
        metadata_used = stable_meta.loc[stable_meta["mnemonic"].isin(snapshot.columns)].copy()
    elif panel_mode == "full_panel":
        if full_panel_map_path is None:
            raise FileNotFoundError(
                "panel_mode='full_panel' requires a fully specified series_to_block map."
            )
        metadata_used = load_full_panel_map(full_panel_map_path, tcodes)
        metadata_used = metadata_used.loc[metadata_used["mnemonic"].isin(snapshot.columns)].copy()
    else:
        raise ValueError("panel_mode must be either 'stable_subset' or 'full_panel'.")

    if metadata_used.empty:
        raise ValueError(f"No usable monthly variables remained for vintage {vintage}.")

    metadata_used["block"] = metadata_used["block"].map(_normalize_block_name)
    monthly_cols = metadata_used["mnemonic"].tolist()

    transformed_raw = base.transform_snapshot(snapshot[monthly_cols], tcodes.reindex(monthly_cols))
    transformed_raw = transformed_raw.replace([np.inf, -np.inf], np.nan)

    # Remove columns that are completely missing or have zero in-sample variance.
    keep_cols = []
    for col in transformed_raw.columns:
        s = transformed_raw[col]
        if s.notna().sum() == 0:
            continue
        if pd.to_numeric(s, errors="coerce").std(skipna=True, ddof=0) in [0, 0.0]:
            continue
        keep_cols.append(col)

    transformed_raw = transformed_raw[keep_cols].copy()
    metadata_used = metadata_used.loc[metadata_used["mnemonic"].isin(keep_cols)].copy()
    metadata_used = metadata_used.drop_duplicates(subset=["mnemonic"]).reset_index(drop=True)

    monthly_std, monthly_means, monthly_stds = base.standardize_panel(transformed_raw)

    quarterly_raw = base.build_quarterly_history_for_vintage(target_vintage_table, vintage)
    quarterly_raw = quarterly_raw.replace([np.inf, -np.inf], np.nan)
    quarterly_std, quarterly_means, quarterly_stds = base.standardize_panel(quarterly_raw)

    coverage = base.compute_block_coverage(transformed_raw, metadata_used, vintage)

    return PreparedFitInput(
        vintage=vintage,
        quarter=vintage.asfreq("Q"),
        impact_month=vintage.asfreq("Q").asfreq("M", "end"),
        panel_mode=panel_mode,
        source_path=source_path,
        metadata=metadata_used,
        monthly_raw=transformed_raw,
        quarterly_raw=quarterly_raw,
        monthly_std=monthly_std,
        quarterly_std=quarterly_std,
        monthly_means=monthly_means,
        monthly_stds=monthly_stds,
        quarterly_means=quarterly_means,
        quarterly_stds=quarterly_stds,
        coverage=coverage,
        tcodes=tcodes.reindex(keep_cols),
    )


# -----------------------------------------------------------------------------
# DFM estimation, forecasting, news
# -----------------------------------------------------------------------------


def _anchor_sign_flags(loadings: pd.DataFrame, metadata: pd.DataFrame) -> Dict[str, bool]:
    flags: Dict[str, bool] = {}
    if loadings.empty:
        return flags
    for _, row in metadata.loc[metadata["anchor"].astype(bool)].iterrows():
        anchor = row["mnemonic"]
        block = row["block"]
        if anchor in loadings.index and block in loadings.columns:
            flags[f"anchor_ok_{block}"] = bool(loadings.loc[anchor, block] >= 0)
        else:
            flags[f"anchor_ok_{block}"] = False
    # Global factor: use the first anchor series present.
    anchor_candidates = metadata.loc[metadata["anchor"].astype(bool), "mnemonic"].tolist()
    if "global" in loadings.columns and anchor_candidates:
        anchor = next((a for a in anchor_candidates if a in loadings.index), None)
        if anchor is not None:
            flags["anchor_ok_global"] = bool(loadings.loc[anchor, "global"] >= 0)
    return flags



def predict_current_quarter_std(
    result,
    impact_month: pd.Period,
    quarterly_var: str = "GDP_GROWTH",
) -> float:
    pred = result.predict(start=str(impact_month), end=str(impact_month))
    return float(pred.iloc[0][quarterly_var])



def unstandardize_scalar(value: float, mean: float, std: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(value * std + mean)



def fit_dfm_vintage(
    prepared: PreparedFitInput,
    p_grid: Sequence[int] = (1, 2, 3),
    idiosyncratic_ar1: bool = True,
    maxiter: int = 200,
    tolerance: float = 1e-6,
    disp: bool = False,
    quarterly_var: str = "GDP_GROWTH",
) -> DFMVintageBundle:
    monthly = prepared.monthly_std.copy()
    quarterly = prepared.quarterly_std.copy()

    if monthly.shape[1] < 7:
        raise ValueError(
            f"Vintage {prepared.vintage}: need at least 7 monthly variables for 1 global + 6 block factors; found {monthly.shape[1]}."
        )
    if prepared.metadata["block"].nunique() < 6:
        warnings.warn(
            f"Vintage {prepared.vintage} has only {prepared.metadata['block'].nunique()} represented blocks; expected 6."
        )

    factors = base.build_factor_mapping(monthly.columns.tolist(), prepared.metadata, quarterly_var=quarterly_var)
    factor_names = factors[quarterly_var]

    best_res = None
    best_p = None
    best_aic = None
    fit_errors = []
    for p in p_grid:
        try:
            factor_orders = {tuple(factor_names): int(p)}
            mod = DynamicFactorMQ(
                monthly,
                endog_quarterly=quarterly,
                factors=factors,
                factor_orders=factor_orders,
                idiosyncratic_ar1=idiosyncratic_ar1,
                standardize=False,
            )
            res = mod.fit(maxiter=maxiter, tolerance=tolerance, disp=disp)
            if best_res is None or res.aic < best_aic:
                best_res = res
                best_aic = float(res.aic)
                best_p = int(p)
        except Exception as exc:  # pragma: no cover - broad catch for vintage robustness
            fit_errors.append((p, repr(exc)))
            continue

    if best_res is None:
        raise RuntimeError(
            f"All candidate factor lag orders failed for vintage {prepared.vintage}. Errors: {fit_errors}"
        )

    states, loadings = base._sign_orient_states(best_res, prepared.metadata)
    nowcast_std = predict_current_quarter_std(best_res, prepared.impact_month, quarterly_var=quarterly_var)
    nowcast_ann = unstandardize_scalar(
        nowcast_std,
        float(prepared.quarterly_means[quarterly_var]),
        float(prepared.quarterly_stds[quarterly_var]),
    )

    anchor_flags = _anchor_sign_flags(loadings, prepared.metadata)

    nowcast_table = pd.DataFrame(
        [
            {
                "vintage": str(prepared.vintage),
                "quarter": str(prepared.quarter),
                "impact_month": str(prepared.impact_month),
                "tau": base.tau_from_vintage_and_quarter(prepared.vintage, prepared.quarter),
                "panel_mode": prepared.panel_mode,
                "dfm_nowcast_std_scale": nowcast_std,
                "dfm_nowcast_ann_pct": nowcast_ann,
                "lag_order": best_p,
                "n_monthly_vars": monthly.shape[1],
                "n_blocks_present": int(prepared.metadata["block"].nunique()),
                "aic": float(best_res.aic),
                "bic": float(best_res.bic),
                "llf": float(best_res.llf),
            }
        ]
    )

    diagnostics_row = {
        "vintage": str(prepared.vintage),
        "quarter": str(prepared.quarter),
        "impact_month": str(prepared.impact_month),
        "lag_order": best_p,
        "aic": float(best_res.aic),
        "bic": float(best_res.bic),
        "hqic": float(getattr(best_res, "hqic", np.nan)),
        "llf": float(best_res.llf),
        "nobs": int(best_res.nobs),
        "n_monthly_vars": int(monthly.shape[1]),
        "n_blocks_present": int(prepared.metadata["block"].nunique()),
        "maxiter": int(maxiter),
        "em_iterations": float(getattr(best_res, "mle_retvals", {}).get("iter", np.nan))
        if hasattr(best_res, "mle_retvals")
        else np.nan,
        "fit_errors_other_p": " | ".join([f"p={p}:{err}" for p, err in fit_errors]),
    }
    diagnostics_row.update(anchor_flags)
    diagnostics = pd.DataFrame([diagnostics_row])

    return DFMVintageBundle(
        prepared=prepared,
        lag_order=int(best_p),
        result=best_res,
        states=states,
        loadings=loadings,
        diagnostics=diagnostics,
        nowcast_table=nowcast_table,
    )



def build_state_table(bundle: DFMVintageBundle) -> pd.DataFrame:
    df = bundle.states.copy()
    df["month"] = df.index.astype(str)
    df["vintage"] = str(bundle.prepared.vintage)
    df["quarter"] = str(bundle.prepared.quarter)
    return df.reset_index(drop=True)



def apply_previous_model_to_updated_vintage(
    previous_bundle: DFMVintageBundle,
    updated_prepared: PreparedFitInput,
    quarterly_var: str = "GDP_GROWTH",
):
    prev_cols = previous_bundle.prepared.monthly_std.columns
    updated_monthly_raw = updated_prepared.monthly_raw.reindex(columns=prev_cols)
    updated_monthly_std_prev = standardize_with_reference(
        updated_monthly_raw,
        previous_bundle.prepared.monthly_means,
        previous_bundle.prepared.monthly_stds,
    )

    updated_quarterly_raw = updated_prepared.quarterly_raw[[quarterly_var]].copy()
    updated_quarterly_std_prev = standardize_with_reference(
        updated_quarterly_raw,
        previous_bundle.prepared.quarterly_means,
        previous_bundle.prepared.quarterly_stds,
    )

    applied = previous_bundle.result.apply(
        updated_monthly_std_prev,
        endog_quarterly=updated_quarterly_std_prev,
    )
    return applied



def compute_news_tables(
    previous_bundle: DFMVintageBundle,
    updated_prepared: PreparedFitInput,
    quarterly_var: str = "GDP_GROWTH",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    applied = apply_previous_model_to_updated_vintage(previous_bundle, updated_prepared, quarterly_var=quarterly_var)
    news = applied.news(
        previous_bundle.result,
        impact_date=str(updated_prepared.impact_month),
        impacted_variable=quarterly_var,
        comparison_type="previous",
    )

    details = news.details_by_impact.copy().xs(quarterly_var, level="impacted variable").reset_index()
    details = details.rename(
        columns={
            "impact date": "impact_month",
            "update date": "update_month",
            "updated variable": "updated_variable",
            "forecast (prev)": "forecast_prev_std",
            "news": "news_std",
            "weight": "weight_std",
            "impact": "impact_std",
            "observed": "observed_std",
        }
    )
    details["previous_vintage"] = str(previous_bundle.prepared.vintage)
    details["current_vintage"] = str(updated_prepared.vintage)
    details["quarter"] = str(updated_prepared.quarter)
    details["tau"] = base.tau_from_vintage_and_quarter(updated_prepared.vintage, updated_prepared.quarter)

    meta = updated_prepared.metadata[["mnemonic", "block", "description"]].drop_duplicates()
    details = details.merge(meta, left_on="updated_variable", right_on="mnemonic", how="left")
    details["block"] = details["block"].fillna("unmapped")

    # Convert impacts to original GDP-growth units (annualized percentage points).
    gdp_std = float(previous_bundle.prepared.quarterly_stds[quarterly_var])
    details["impact_ann_pct"] = details["impact_std"] * gdp_std

    # Reverse the standardization for updated monthly variables whenever possible.
    monthly_means = previous_bundle.prepared.monthly_means
    monthly_stds = previous_bundle.prepared.monthly_stds
    quarterly_mean = float(previous_bundle.prepared.quarterly_means[quarterly_var])
    quarterly_std = float(previous_bundle.prepared.quarterly_stds[quarterly_var])

    observed_original = []
    forecast_prev_original = []
    news_original = []
    for _, row in details.iterrows():
        var = row["updated_variable"]
        if var == quarterly_var:
            mean_i = quarterly_mean
            std_i = quarterly_std
        else:
            mean_i = float(monthly_means.get(var, np.nan))
            std_i = float(monthly_stds.get(var, np.nan))
        observed_original.append(row["observed_std"] * std_i + mean_i if pd.notna(std_i) else np.nan)
        forecast_prev_original.append(row["forecast_prev_std"] * std_i + mean_i if pd.notna(std_i) else np.nan)
        news_original.append(row["news_std"] * std_i if pd.notna(std_i) else np.nan)
    details["observed_transformed_units"] = observed_original
    details["forecast_prev_transformed_units"] = forecast_prev_original
    details["news_transformed_units"] = news_original

    block_news = (
        details.groupby(["current_vintage", "quarter", "impact_month", "tau", "block"], as_index=False)
        .agg(
            signed_block_news_ann_pct=("impact_ann_pct", "sum"),
            abs_block_news_ann_pct=("impact_ann_pct", lambda s: np.abs(s).sum()),
            n_updates=("updated_variable", "size"),
        )
    )
    block_news["previous_vintage"] = str(previous_bundle.prepared.vintage)

    return details, block_news


# -----------------------------------------------------------------------------
# Forecast window helpers and benchmarks
# -----------------------------------------------------------------------------


def select_vintages_for_run(
    fred_md_catalog: pd.DataFrame,
    start_quarter: str = "2000Q1",
    end_vintage: Optional[str | pd.Period] = None,
    run_mode: str = "debug",
    debug_n_vintages: int = 6,
    debug_recent: bool = True,
) -> List[pd.Period]:
    start_month = pd.Period(start_quarter, freq="Q").asfreq("M", "start")
    if end_vintage is None:
        end_vintage = fred_md_catalog["vintage"].max()
    end_vintage = pd.Period(str(end_vintage), freq="M")

    vintages = fred_md_catalog.loc[
        (fred_md_catalog["vintage"] >= start_month) & (fred_md_catalog["vintage"] <= end_vintage),
        "vintage",
    ].sort_values().tolist()

    if run_mode == "debug":
        if debug_recent:
            vintages = vintages[-debug_n_vintages:]
        else:
            vintages = vintages[:debug_n_vintages]
    return vintages



def ar_benchmark_nowcast(
    quarterly_history_raw: pd.DataFrame,
    p: int = 2,
    value_col: str = "GDP_GROWTH",
) -> float:
    y = quarterly_history_raw[value_col].dropna().astype(float)
    if len(y) <= p + 1:
        return np.nan

    df = pd.DataFrame({"y": y})
    for lag in range(1, p + 1):
        df[f"lag{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    if df.empty:
        return np.nan

    X = np.column_stack([np.ones(len(df))] + [df[f"lag{lag}"].values for lag in range(1, p + 1)])
    beta = np.linalg.lstsq(X, df["y"].values, rcond=None)[0]
    latest_lags = np.array([1.0] + [y.iloc[-lag] for lag in range(1, p + 1)])
    return float(latest_lags @ beta)


# -----------------------------------------------------------------------------
# Export and master loop
# -----------------------------------------------------------------------------


def export_artifacts(
    output_dir: Path | str,
    nowcasts: pd.DataFrame,
    states: pd.DataFrame,
    news_series: pd.DataFrame,
    news_blocks: pd.DataFrame,
    coverage: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "dfm_nowcasts": output_dir / "dfm_nowcasts.csv",
        "dfm_states": output_dir / "dfm_states.csv",
        "dfm_news_series": output_dir / "dfm_news_series.csv",
        "dfm_news_blocks": output_dir / "dfm_news_blocks.csv",
        "dfm_coverage": output_dir / "dfm_coverage.csv",
        "dfm_diagnostics": output_dir / "dfm_diagnostics.csv",
        "vintage_manifest": output_dir / "vintage_manifest.csv",
    }

    nowcasts.to_csv(paths["dfm_nowcasts"], index=False)
    states.to_csv(paths["dfm_states"], index=False)
    news_series.to_csv(paths["dfm_news_series"], index=False)
    news_blocks.to_csv(paths["dfm_news_blocks"], index=False)
    coverage.to_csv(paths["dfm_coverage"], index=False)
    diagnostics.to_csv(paths["dfm_diagnostics"], index=False)
    manifest.to_csv(paths["vintage_manifest"], index=False)
    return paths



def run_dfm_backbone(
    vintages: Sequence[pd.Period],
    target_vintage_table: pd.DataFrame,
    fred_md_catalog: pd.DataFrame,
    stable_meta: pd.DataFrame,
    output_dir: Path | str,
    panel_mode: str = "stable_subset",
    full_panel_map_path: Optional[Path | str] = None,
    p_grid: Sequence[int] = (1, 2, 3),
    maxiter: int = 200,
    tolerance: float = 1e-6,
    disp: bool = False,
    ar_benchmark_order: int = 2,
) -> Dict[str, pd.DataFrame]:
    nowcasts_parts: List[pd.DataFrame] = []
    states_parts: List[pd.DataFrame] = []
    news_series_parts: List[pd.DataFrame] = []
    news_block_parts: List[pd.DataFrame] = []
    coverage_parts: List[pd.DataFrame] = []
    diagnostics_parts: List[pd.DataFrame] = []
    manifest_rows: List[Dict] = []

    previous_bundle: Optional[DFMVintageBundle] = None
    previous_nowcast_ann = np.nan

    for vintage in vintages:
        snapshot, tcodes, source_path = load_local_fred_md_snapshot(vintage, fred_md_catalog)
        prepared = prepare_vintage_fit_input(
            vintage=vintage,
            snapshot=snapshot,
            tcodes=tcodes,
            source_path=source_path,
            target_vintage_table=target_vintage_table,
            stable_meta=stable_meta,
            panel_mode=panel_mode,
            full_panel_map_path=full_panel_map_path,
        )
        bundle = fit_dfm_vintage(
            prepared,
            p_grid=p_grid,
            maxiter=maxiter,
            tolerance=tolerance,
            disp=disp,
        )
        nowcast = bundle.nowcast_table.copy()
        nowcast["revision_from_previous_vintage_ann_pct"] = (
            nowcast["dfm_nowcast_ann_pct"] - previous_nowcast_ann
            if pd.notna(previous_nowcast_ann)
            else np.nan
        )
        nowcast["ar2_benchmark_ann_pct"] = ar_benchmark_nowcast(
            prepared.quarterly_raw,
            p=ar_benchmark_order,
        )
        nowcasts_parts.append(nowcast)
        states_parts.append(build_state_table(bundle))
        coverage_parts.append(prepared.coverage.copy())
        diagnostics_parts.append(bundle.diagnostics.copy())

        manifest_rows.append(
            {
                "vintage": str(vintage),
                "quarter": str(prepared.quarter),
                "impact_month": str(prepared.impact_month),
                "source_path": str(source_path),
                "panel_mode": panel_mode,
                "n_monthly_vars": int(prepared.monthly_std.shape[1]),
                "n_blocks_present": int(prepared.metadata["block"].nunique()),
                "lag_order": int(bundle.lag_order),
                "dfm_nowcast_ann_pct": float(nowcast.iloc[0]["dfm_nowcast_ann_pct"]),
            }
        )

        if previous_bundle is not None:
            try:
                series_news, block_news = compute_news_tables(previous_bundle, prepared)
                total_news = block_news["signed_block_news_ann_pct"].sum() if not block_news.empty else np.nan
                block_news["reestimated_revision_ann_pct"] = float(nowcast.iloc[0]["revision_from_previous_vintage_ann_pct"])
                block_news["total_news_impact_ann_pct"] = total_news
                news_series_parts.append(series_news)
                news_block_parts.append(block_news)
            except Exception as exc:  # pragma: no cover
                diagnostics_parts.append(
                    pd.DataFrame(
                        [
                            {
                                "vintage": str(vintage),
                                "quarter": str(prepared.quarter),
                                "impact_month": str(prepared.impact_month),
                                "news_error": repr(exc),
                            }
                        ]
                    )
                )

        previous_bundle = bundle
        previous_nowcast_ann = float(nowcast.iloc[0]["dfm_nowcast_ann_pct"])

    nowcasts_df = pd.concat(nowcasts_parts, ignore_index=True) if nowcasts_parts else pd.DataFrame()
    states_df = pd.concat(states_parts, ignore_index=True) if states_parts else pd.DataFrame()
    news_series_df = pd.concat(news_series_parts, ignore_index=True) if news_series_parts else pd.DataFrame()
    news_blocks_df = pd.concat(news_block_parts, ignore_index=True) if news_block_parts else pd.DataFrame()
    coverage_df = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    diagnostics_df = pd.concat(diagnostics_parts, ignore_index=True) if diagnostics_parts else pd.DataFrame()
    manifest_df = pd.DataFrame(manifest_rows)

    export_artifacts(
        output_dir,
        nowcasts=nowcasts_df,
        states=states_df,
        news_series=news_series_df,
        news_blocks=news_blocks_df,
        coverage=coverage_df,
        diagnostics=diagnostics_df,
        manifest=manifest_df,
    )

    return {
        "dfm_nowcasts": nowcasts_df,
        "dfm_states": states_df,
        "dfm_news_series": news_series_df,
        "dfm_news_blocks": news_blocks_df,
        "dfm_coverage": coverage_df,
        "dfm_diagnostics": diagnostics_df,
        "vintage_manifest": manifest_df,
    }


# -----------------------------------------------------------------------------
# Scoring helpers for pre-ML validation
# -----------------------------------------------------------------------------


def attach_truth_and_score(
    nowcasts: pd.DataFrame,
    truth_table: pd.DataFrame,
    truth_col: str = "truth_third_release",
) -> pd.DataFrame:
    scored = nowcasts.merge(
        truth_table[["quarter", truth_col]],
        on="quarter",
        how="left",
    )
    scored["dfm_error"] = scored[truth_col] - scored["dfm_nowcast_ann_pct"]
    scored["ar2_error"] = scored[truth_col] - scored["ar2_benchmark_ann_pct"]
    return scored



def rmsfe(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return np.nan
    return float(np.sqrt(np.mean(series**2)))



def pre_ml_validation_summary(scored: pd.DataFrame, truth_col: str = "truth_third_release") -> pd.DataFrame:
    rows = []
    for tau, group in scored.dropna(subset=[truth_col]).groupby("tau"):
        rows.append(
            {
                "tau": tau,
                "n_scored": len(group),
                "dfm_rmsfe": rmsfe(group["dfm_error"]),
                "ar2_rmsfe": rmsfe(group["ar2_error"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["tau", "n_scored", "dfm_rmsfe", "ar2_rmsfe"])
    out = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    out["dfm_beats_ar2"] = out["dfm_rmsfe"] < out["ar2_rmsfe"]
    return out
