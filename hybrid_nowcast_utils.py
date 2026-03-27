
"""
Utilities for the DFM backbone of the research project
"Real-Time GDP Growth Nowcasting using a Hybrid Dynamic Factor Model with
Machine Learning Residual Correction".

This module focuses on the tầng 1 / DFM backbone described in the outline:
- protocol and vintage-aware configuration
- target/truth construction from RTDSM / ROUTPUT
- stable-subset metadata and crosswalks
- FRED-MD snapshot parsing and transformations
- ragged-edge coverage summaries
- block-structured mixed-frequency DFM with DynamicFactorMQ
- nowcast, news decomposition, and artefact assembly

The code is intentionally conservative about leakage:
- every transformation is done on the vintage-specific sample only
- standardization uses only data available up to the current vintage
- quarterly truth tables are separated from GDP history used in estimation
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ


# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

FRED_MD_ARCHIVE_URLS = {
    "1999_2014": (
        "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/"
        "research/fred-md/historical_fred-md.zip"
        "?hash=8A23C5FAF7A0D743A353D77DF4704028&sc_lang=en"
    ),
    "2015_2024": (
        "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/"
        "research/fred-md/historical-vintages-of-fred-md-2015-01-to-2024-12.zip"
        "?hash=831F98A7EC8D3809881DF067965B50FF&sc_lang=en"
    ),
}

# Fallback pattern widely used in practice when pulling a single monthly file.
FRED_MD_MONTHLY_FALLBACK = (
    "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{vintage}.csv"
)

RTDSM_URLS = {
    "routput_monthly": (
        "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
        "real-time-data/data-files/xlsx/routputMvQd.xlsx"
        "?hash=B3AD0AED9E268AA077E8D9B8AAF2AC83&sc_lang=en"
    ),
    "routput_quarterly": (
        "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
        "real-time-data/data-files/xlsx/ROUTPUTQvQd.xlsx"
        "?hash=34FA1C6BF0007996E1885C8C32E3BEF9&sc_lang=en"
    ),
    "routput_release_values": (
        "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
        "real-time-data/data-files/xlsx/routput_first_second_third.xlsx"
        "?hash=AB8BB59BBBF6840DE1448851E90D7A80&sc_lang=en"
    ),
    "gdpplus_vintages": (
        "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
        "gdpplus/GDPplus_Vintages.xlsx"
    ),
    "spf_mean_growth": (
        "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
        "survey-of-professional-forecasters/historical-data/meanGrowth.xlsx"
        "?hash=0F651C29D8FE2E04AB86BC7DD7EDAD2E&sc_lang=en"
    ),
}

TCODE_RULES = {
    1: "level",
    2: "diff",
    3: "diff2",
    4: "log",
    5: "dlog",
    6: "d2log",
    7: "chg_growth_rate",
}


# ---------------------------------------------------------------------------
# Lightweight data classes
# ---------------------------------------------------------------------------

@dataclass
class PreparedVintagePanel:
    vintage: pd.Period
    monthly: pd.DataFrame
    quarterly: pd.DataFrame
    monthly_means: pd.Series
    monthly_stds: pd.Series
    quarterly_means: pd.Series
    quarterly_stds: pd.Series
    coverage: pd.DataFrame


@dataclass
class DFMFitBundle:
    vintage: pd.Period
    lag_order: int
    result: object
    diagnostics: pd.DataFrame
    states: pd.DataFrame
    nowcast_table: pd.DataFrame


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_if_missing(url: str, path: Path | str, timeout: int = 120) -> Path:
    """
    Download a file only if it is not already cached locally.
    """
    path = Path(path)
    if path.exists():
        return path

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def parse_quarter_str(value: str) -> pd.Period | pd.NaT:
    value = str(value).strip()
    match = re.match(r"^(\d{4}):Q([1-4])$", value)
    if not match:
        return pd.NaT
    year, quarter = match.groups()
    return pd.Period(f"{year}Q{quarter}", freq="Q")


def parse_gdpplus_quarter(value: str) -> pd.Period | pd.NaT:
    value = str(value).strip()
    match = re.match(r"^(\d{4}):0?([1-4])$", value)
    if not match:
        return pd.NaT
    year, quarter = match.groups()
    return pd.Period(f"{year}Q{quarter}", freq="Q")


def parse_routput_month_vintage(col: str) -> Optional[pd.Period]:
    match = re.match(r"^ROUTPUT(\d{2})M(\d{1,2})$", str(col))
    if not match:
        return None
    yy, mm = match.groups()
    year = 1900 + int(yy)
    if year < 1965:
        year += 100
    return pd.Period(f"{year}-{int(mm):02d}", freq="M")


def parse_routput_quarter_vintage(col: str) -> Optional[pd.Period]:
    match = re.match(r"^ROUTPUT(\d{2})Q([1-4])$", str(col))
    if not match:
        return None
    yy, qq = match.groups()
    year = 1900 + int(yy)
    if year < 1965:
        year += 100
    return pd.Period(f"{year}Q{qq}", freq="Q")


def parse_gdpplus_vintage_col(col: str) -> Optional[pd.Timestamp]:
    match = re.match(r"^GDPPLUS_(\d{2})(\d{2})(\d{2})$", str(col))
    if not match:
        return None
    mm, dd, yy = match.groups()
    year = 2000 + int(yy)
    return pd.Timestamp(year=year, month=int(mm), day=int(dd))


def tau_from_vintage_and_quarter(vintage: pd.Period, quarter: pd.Period) -> float:
    if pd.isna(vintage) or pd.isna(quarter):
        return np.nan
    if vintage.asfreq("Q") != quarter:
        return np.nan
    first_month = (quarter.quarter - 1) * 3 + 1
    return vintage.month - first_month + 1


# ---------------------------------------------------------------------------
# Target and truth tables
# ---------------------------------------------------------------------------

def build_target_vintage_table(routput_monthly_path: Path | str) -> pd.DataFrame:
    """
    Build vintage-aware GDP level and GDP growth table from monthly ROUTPUT vintages.
    """
    raw = pd.read_excel(routput_monthly_path, sheet_name=0)
    raw["quarter"] = [parse_quarter_str(x) for x in raw["DATE"]]

    vintage_cols = [c for c in raw.columns if c not in ["DATE", "quarter"]]
    vintage_map = {c: parse_routput_month_vintage(c) for c in vintage_cols}
    valid_cols = [c for c, v in vintage_map.items() if v is not None]

    long = raw[["quarter"] + valid_cols].melt(
        id_vars="quarter", var_name="vintage_col", value_name="y_level"
    )
    long["vintage"] = long["vintage_col"].map(vintage_map)
    long = long.drop(columns="vintage_col").sort_values(["vintage", "quarter"]).reset_index(drop=True)

    long["gdp_growth_ann_pct"] = long.groupby("vintage")["y_level"].transform(
        lambda s: 400 * (np.log(s) - np.log(s.shift(1)))
    )
    long["quarter_str"] = long["quarter"].astype(str)
    long["vintage_str"] = long["vintage"].astype(str)
    long["tau"] = [tau_from_vintage_and_quarter(v, q) for v, q in zip(long["vintage"], long["quarter"])]
    long["is_current_quarter_vintage"] = long["tau"].notna()
    return long


def build_truth_tables(
    release_values_path: Path | str,
    gdpplus_path: Optional[Path | str] = None,
    spf_mean_growth_path: Optional[Path | str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build main and robustness truth tables plus optional SPF benchmark table.
    """
    rel = pd.read_excel(release_values_path, sheet_name="DATA", skiprows=3)
    rel.columns = ["Date", "First", "Second", "Third", "Most_Recent"]
    rel = rel.iloc[1:].copy()
    rel["quarter"] = [parse_quarter_str(x) for x in rel["Date"]]
    for col in ["First", "Second", "Third", "Most_Recent"]:
        rel[col] = pd.to_numeric(rel[col], errors="coerce")

    out = {
        "truth_first_release": rel[["quarter", "First"]].rename(columns={"First": "truth_first_release"}),
        "truth_second_release": rel[["quarter", "Second"]].rename(columns={"Second": "truth_second_release"}),
        "truth_third_release": rel[["quarter", "Third"]].rename(columns={"Third": "truth_third_release"}),
        "truth_latest_rtdsm": rel[["quarter", "Most_Recent"]].rename(columns={"Most_Recent": "truth_latest_rtdsm"}),
    }

    if gdpplus_path is not None:
        gdpplus = pd.read_excel(gdpplus_path)
        gdpplus["quarter"] = [parse_gdpplus_quarter(x) for x in gdpplus["Date"]]
        gdpplus_cols = [c for c in gdpplus.columns if c not in ["Date", "quarter"]]
        latest_col = max(gdpplus_cols, key=lambda c: parse_gdpplus_vintage_col(c))
        out["truth_gdpplus_latest"] = gdpplus[["quarter", latest_col]].rename(
            columns={latest_col: "truth_gdpplus_latest"}
        )

        gdpplus_long = gdpplus[["quarter"] + gdpplus_cols].melt(
            id_vars="quarter", var_name="vintage_col", value_name="gdpplus"
        )
        gdpplus_long["vintage_date"] = gdpplus_long["vintage_col"].map(parse_gdpplus_vintage_col)
        out["gdpplus_vintages"] = gdpplus_long

    if spf_mean_growth_path is not None:
        spf = pd.read_excel(spf_mean_growth_path, sheet_name="RGDP")
        spf_long = spf.melt(
            id_vars=["YEAR", "QUARTER"],
            value_vars=[c for c in spf.columns if c.startswith("drgdp")],
            var_name="horizon_col",
            value_name="spf_mean_rgdp_growth",
        )
        spf_long["horizon"] = spf_long["horizon_col"].str.extract(r"(\d+)$").astype(float)
        spf_long["survey_quarter"] = [
            pd.Period(f"{int(y)}Q{int(q)}", freq="Q")
            for y, q in zip(spf_long["YEAR"], spf_long["QUARTER"])
        ]
        spf_long["target_quarter"] = spf_long.apply(
            lambda r: r["survey_quarter"] + int(r["horizon"]) - 2
            if not pd.isna(r["horizon"])
            else pd.NaT,
            axis=1,
        )
        out["spf_mean_rgdp_growth"] = spf_long

    return out


# ---------------------------------------------------------------------------
# Stable subset metadata
# ---------------------------------------------------------------------------

def stable_subset_metadata() -> pd.DataFrame:
    rows = [
        ("RA1", "RPI", "real_activity_income", 5, "dlog", "", False, "Real Personal Income"),
        ("RA2", "W875RX1", "real_activity_income", 5, "dlog", "", False, "Real personal income ex transfer receipts"),
        ("RA3", "INDPRO", "real_activity_income", 5, "dlog", "", True, "Industrial Production Index"),
        ("RA4", "IPFINAL", "real_activity_income", 5, "dlog", "", False, "IP: Final Products"),
        ("RA5", "IPCONGD", "real_activity_income", 5, "dlog", "", False, "IP: Consumer Goods"),
        ("RA6", "IPBUSEQ", "real_activity_income", 5, "dlog", "", False, "IP: Business Equipment"),
        ("RA7", "IPMANSICS", "real_activity_income", 5, "dlog", "", False, "IP: Manufacturing (SIC)"),
        ("RA8", "CUMFNS", "real_activity_income", 2, "diff", "", False, "Capacity Utilization: Manufacturing"),
        ("LM1", "CLF16OV", "labor_market", 5, "dlog", "", False, "Civilian Labor Force"),
        ("LM2", "UNRATE", "labor_market", 2, "diff", "", False, "Civilian Unemployment Rate"),
        ("LM3", "UEMPMEAN", "labor_market", 2, "diff", "", False, "Average Duration of Unemployment"),
        ("LM4", "CLAIMSx", "labor_market", 5, "dlog", "", False, "Initial Claims"),
        ("LM5", "PAYEMS", "labor_market", 5, "dlog", "", True, "All Employees: Total Nonfarm"),
        ("LM6", "MANEMP", "labor_market", 5, "dlog", "", False, "All Employees: Manufacturing"),
        ("LM7", "AWHMAN", "labor_market", 1, "level", "", False, "Average Weekly Hours: Manufacturing"),
        ("LM8", "CES3000000008", "labor_market", 6, "d2log", "", False, "Average Hourly Earnings: Manufacturing"),
        ("HC1", "HOUST", "housing_construction", 4, "log", "", True, "Housing Starts: Total New Privately Owned"),
        ("HC2", "PERMIT", "housing_construction", 4, "log", "", False, "New Private Housing Permits"),
        ("HC3", "HOUSTNE", "housing_construction", 4, "log", "", False, "Housing Starts: Northeast"),
        ("HC4", "HOUSTMW", "housing_construction", 4, "log", "", False, "Housing Starts: Midwest"),
        ("HC5", "HOUSTS", "housing_construction", 4, "log", "", False, "Housing Starts: South"),
        ("HC6", "HOUSTW", "housing_construction", 4, "log", "", False, "Housing Starts: West"),
        ("DO1", "DPCERA3M086SBEA", "demand_orders_inventories", 5, "dlog", "", False, "Real personal consumption expenditures"),
        ("DO2", "CMRMTSPLx", "demand_orders_inventories", 5, "dlog", "", False, "Real Manufacturing and Trade Industries Sales"),
        ("DO3", "RETAILx", "demand_orders_inventories", 5, "dlog", "", True, "Retail and Food Services Sales"),
        ("DO4", "ACOGNO", "demand_orders_inventories", 5, "dlog", "", False, "New Orders for Consumer Goods"),
        ("DO5", "AMDMNOx", "demand_orders_inventories", 5, "dlog", "", False, "New Orders for Durable Goods"),
        ("DO6", "ANDENOx", "demand_orders_inventories", 5, "dlog", "", False, "New Orders for Nondefense Capital Goods"),
        ("DO7", "AMDMUOx", "demand_orders_inventories", 5, "dlog", "", False, "Unfilled Orders for Durable Goods"),
        ("DO8", "BUSINVx", "demand_orders_inventories", 5, "dlog", "", False, "Total Business Inventories"),
        ("DO9", "ISRATIOx", "demand_orders_inventories", 2, "diff", "", False, "Total Business: Inventories to Sales Ratio"),
        ("PI1", "CPIAUCSL", "prices_inflation", 6, "d2log", "", True, "CPI: All Items"),
        ("PI2", "CPIULFSL", "prices_inflation", 6, "d2log", "", False, "CPI: All Items Less Food"),
        ("PI3", "CUSR0000SAS", "prices_inflation", 6, "d2log", "", False, "CPI: Services"),
        ("PI4", "CUSR0000SAC", "prices_inflation", 6, "d2log", "", False, "CPI: Commodities"),
        ("PI5", "PCEPI", "prices_inflation", 6, "d2log", "", False, "PCE Chain Index"),
        ("PI6", "DDURRG3M086SBEA", "prices_inflation", 6, "d2log", "", False, "PCE deflator: Durable goods"),
        ("PI7", "DSERRG3M086SBEA", "prices_inflation", 6, "d2log", "", False, "PCE deflator: Services"),
        ("PI8", "WPSFD49207", "prices_inflation", 6, "d2log", "PPIFGS", False, "PPI: Finished Goods / Finished Demand crosswalk concept"),
        ("PI9", "OILPRICEx", "prices_inflation", 6, "d2log", "", False, "Crude Oil (spliced WTI/Cushing)"),
        ("FC1", "FEDFUNDS", "financial_conditions", 2, "diff", "", False, "Effective Federal Funds Rate"),
        ("FC2", "TB3MS", "financial_conditions", 2, "diff", "", False, "3-Month Treasury Bill Rate"),
        ("FC3", "GS10", "financial_conditions", 2, "diff", "", False, "10-Year Treasury Rate"),
        ("FC4", "BAA", "financial_conditions", 2, "diff", "", False, "Moody's Seasoned Baa Corporate Bond Yield"),
        ("FC5", "T10YFFM", "financial_conditions", 1, "level", "", True, "10-Year Treasury minus Fed Funds spread"),
        ("FC6", "BAAFFM", "financial_conditions", 1, "level", "", False, "Baa minus Fed Funds spread"),
        ("FC7", "BUSLOANS", "financial_conditions", 6, "d2log", "", False, "Commercial and Industrial Loans"),
        ("FC8", "REALLN", "financial_conditions", 6, "d2log", "", False, "Real Estate Loans at All Commercial Banks"),
        ("FC9", "NONREVSL", "financial_conditions", 6, "d2log", "", False, "Total Nonrevolving Credit"),
        ("FC10", "TWEXAFEGSMTHx", "financial_conditions", 5, "dlog", "TWEXMMTH", False, "Trade Weighted U.S. Dollar Index concept"),
    ]
    meta = pd.DataFrame(
        rows,
        columns=[
            "concept_id",
            "mnemonic",
            "block",
            "tcode",
            "transform_rule_name",
            "alias_or_crosswalk",
            "anchor",
            "description",
        ],
    )
    meta["include_stable_subset"] = True
    meta["caution_note"] = ""
    meta.loc[meta["mnemonic"] == "CLAIMSx", "caution_note"] = (
        "Caution: 2015-08 seasonal-adjustment correction and 2017-04 "
        "monthly-aggregation change."
    )
    meta.loc[meta["mnemonic"] == "BAA", "caution_note"] = (
        "Note temporary Moody's publication issue in 2016-11 to 2017-02."
    )
    meta.loc[meta["mnemonic"] == "WPSFD49207", "caution_note"] = (
        "Use concept-level PPI crosswalk from legacy PPIFGS."
    )
    meta.loc[meta["mnemonic"] == "TWEXAFEGSMTHx", "caution_note"] = (
        "Use concept-level crosswalk from legacy TWEXMMTH."
    )
    return meta


def block_label_pretty(block: str) -> str:
    return {
        "real_activity_income": "real activity & income",
        "labor_market": "labor market",
        "housing_construction": "housing & construction",
        "demand_orders_inventories": "demand / orders / inventories",
        "prices_inflation": "prices & inflation",
        "financial_conditions": "financial conditions",
    }[block]


def anchor_map_from_metadata(metadata: pd.DataFrame) -> Dict[str, str]:
    anchors = metadata.loc[metadata["anchor"], ["mnemonic", "block"]].copy()
    out = {}
    for _, row in anchors.iterrows():
        out[row["block"]] = row["mnemonic"]
    return out


def generate_full_panel_block_template(columns: Sequence[str], metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Create an ex-ante template for full-panel block assignment.
    Stable-subset rows are prefilled; all other series are left blank on purpose.
    """
    existing = metadata.set_index("mnemonic")
    rows = []
    for c in columns:
        if c in existing.index:
            row = existing.loc[c].to_dict()
            row["prefilled"] = True
        else:
            row = {
                "concept_id": "",
                "mnemonic": c,
                "block": "",
                "tcode": np.nan,
                "transform_rule_name": "",
                "alias_or_crosswalk": "",
                "anchor": False,
                "description": "",
                "include_stable_subset": False,
                "caution_note": "",
                "prefilled": False,
            }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FRED-MD ingestion
# ---------------------------------------------------------------------------

def _looks_like_tcode_row(row: pd.Series) -> bool:
    vals = row.iloc[1:].astype(str).str.strip()
    score = vals.str.fullmatch(r"[1-7]").mean()
    return bool(score > 0.5)


def parse_fred_md_csv(path_or_url: str | Path, nrows_scan: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Parse a FRED-MD vintage CSV robustly.

    The exact header layout has changed across examples in the wild. This parser
    scans the first few rows, finds the row containing `sasdate` or `date`, and
    then looks for the nearest row dominated by transformation codes 1..7.
    """
    raw = pd.read_csv(path_or_url, header=None)
    header_candidates = []
    for i in range(min(nrows_scan, len(raw))):
        first = str(raw.iloc[i, 0]).strip().lower()
        if first in {"sasdate", "date"}:
            header_candidates.append(i)

    if not header_candidates:
        raise ValueError("Could not detect the FRED-MD header row (sasdate/date).")
    header_row = header_candidates[0]

    tcode_row = None
    for r in [header_row + 1, header_row - 1, header_row + 2]:
        if 0 <= r < len(raw) and _looks_like_tcode_row(raw.iloc[r]):
            tcode_row = r
            break

    colnames = raw.iloc[header_row].astype(str).tolist()
    data_start = header_row + 1
    tcodes = pd.Series(dtype="float64")

    if tcode_row is not None:
        tcodes = pd.to_numeric(raw.iloc[tcode_row, 1:], errors="coerce")
        tcodes.index = colnames[1:]
        data_start = max(header_row, tcode_row) + 1

    data = raw.iloc[data_start:].copy()
    data.columns = colnames
    first_col = data.columns[0]

    # Standard FRED-MD convention is monthly date in first column
    data[first_col] = pd.PeriodIndex(pd.to_datetime(data[first_col]), freq="M")
    data = data.rename(columns={first_col: "date"}).set_index("date")

    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    return data, tcodes


def request_fred_md_monthly_file(vintage: pd.Period) -> pd.DataFrame:
    """
    Pull a single FRED-MD monthly file using the widely used raw endpoint pattern.
    Useful as a fallback when the historical zip archives are not cached locally.
    """
    url = FRED_MD_MONTHLY_FALLBACK.format(vintage=vintage.strftime("%Y-%m"))
    data, _ = parse_fred_md_csv(url)
    return data


def read_fred_md_from_zip(zip_path: Path | str, vintage: pd.Period) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Read a single monthly vintage file from a local zip archive.
    """
    zip_path = Path(zip_path)
    target_name = f"{vintage.strftime('%Y-%m')}.csv"
    with zipfile.ZipFile(zip_path, "r") as zf:
        matches = [name for name in zf.namelist() if name.endswith(target_name)]
        if not matches:
            raise FileNotFoundError(f"{target_name} not found inside {zip_path}.")
        with zf.open(matches[0]) as handle:
            content = io.BytesIO(handle.read())
            return parse_fred_md_csv(content)


def load_fred_md_snapshot(
    vintage: pd.Period,
    archive_1999_2014: Optional[Path | str] = None,
    archive_2015_2024: Optional[Path | str] = None,
    fallback_to_raw_url: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a monthly FRED-MD snapshot either from local archives or from a raw URL.
    """
    if archive_1999_2014 and vintage <= pd.Period("2014-12", freq="M"):
        return read_fred_md_from_zip(archive_1999_2014, vintage)
    if archive_2015_2024 and pd.Period("2015-01", "M") <= vintage <= pd.Period("2024-12", "M"):
        return read_fred_md_from_zip(archive_2015_2024, vintage)
    if fallback_to_raw_url:
        return parse_fred_md_csv(FRED_MD_MONTHLY_FALLBACK.format(vintage=vintage.strftime("%Y-%m")))
    raise FileNotFoundError(
        "No suitable local FRED-MD archive was supplied and fallback_to_raw_url=False."
    )


# ---------------------------------------------------------------------------
# Transformations and standardization
# ---------------------------------------------------------------------------

def apply_fred_tcode(series: pd.Series, tcode: int) -> pd.Series:
    """
    Apply the official FRED-MD transformation code to a single series.
    """
    if pd.isna(tcode):
        return series.copy()

    s = pd.to_numeric(series, errors="coerce").astype(float)

    if tcode == 1:
        return s
    if tcode == 2:
        return s.diff()
    if tcode == 3:
        return s.diff().diff()
    if tcode == 4:
        return np.log(s)
    if tcode == 5:
        return np.log(s).diff()
    if tcode == 6:
        return np.log(s).diff().diff()
    if tcode == 7:
        return (s / s.shift(1) - 1).diff()

    raise ValueError(f"Unsupported tcode: {tcode}")


def transform_snapshot(snapshot: pd.DataFrame, tcodes: Mapping[str, float]) -> pd.DataFrame:
    """
    Apply FRED-MD tcodes to an entire monthly snapshot.
    """
    out = pd.DataFrame(index=snapshot.index)
    for col in snapshot.columns:
        tcode = tcodes.get(col, np.nan)
        out[col] = apply_fred_tcode(snapshot[col], int(tcode) if not pd.isna(tcode) else np.nan)
    return out


def standardize_panel(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Standardize each series using only the sample currently present in `panel`.
    """
    means = panel.mean(skipna=True)
    stds = panel.std(skipna=True, ddof=0).replace(0, np.nan)
    standardized = (panel - means) / stds
    return standardized, means, stds


def select_latest_observation_per_quarter(
    transformed_panel: pd.DataFrame,
    quarter: pd.Period,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Build the four ragged-edge summaries used later by the residual learner:
    x_last, delta_x, partial-quarter average, and months missing.
    """
    months = pd.period_range(quarter.asfreq("M", "start"), quarter.asfreq("M", "end"), freq="M")
    panel_q = transformed_panel.reindex(months)

    x_last = panel_q.apply(lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan)
    x_prev = transformed_panel.apply(lambda s: s.loc[s.index < months[-1]].dropna().iloc[-1] if s.loc[s.index < months[-1]].dropna().size else np.nan)
    delta_x = x_last - x_prev
    x_pqa = panel_q.mean(skipna=True)
    n_obs = panel_q.notna().sum()
    m_miss = 3 - n_obs

    return x_last, delta_x, x_pqa, m_miss


def compute_block_coverage(
    transformed_panel: pd.DataFrame,
    metadata: pd.DataFrame,
    vintage: pd.Period,
    quarter: Optional[pd.Period] = None,
) -> pd.DataFrame:
    """
    Compute block-level coverage c_{b,v,q} = (1 / (3 * N_b)) sum_i |O_{i,v,q}|.
    """
    if quarter is None:
        quarter = vintage.asfreq("Q")

    months = pd.period_range(quarter.asfreq("M", "start"), quarter.asfreq("M", "end"), freq="M")
    panel_q = transformed_panel.reindex(months)

    meta = metadata.set_index("mnemonic")
    rows = []
    for block, block_meta in meta.groupby("block"):
        cols = [c for c in block_meta.index if c in panel_q.columns]
        if not cols:
            continue
        observed_months = panel_q[cols].notna().sum().sum()
        n_b = len(cols)
        coverage = observed_months / (3 * n_b)
        rows.append({
            "vintage": str(vintage),
            "quarter": str(quarter),
            "block": block,
            "coverage": coverage,
            "n_series_in_block": n_b,
            "observed_month_cells": int(observed_months),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DFM preparation and estimation
# ---------------------------------------------------------------------------

def build_quarterly_history_for_vintage(
    target_vintage_table: pd.DataFrame,
    vintage: pd.Period,
    value_col: str = "gdp_growth_ann_pct",
) -> pd.DataFrame:
    """
    Extract the GDP history that was actually available at the month-end vintage.
    """
    subset = target_vintage_table.loc[target_vintage_table["vintage"] == vintage, ["quarter", value_col]].copy()
    subset = subset.dropna().set_index("quarter").sort_index()
    return subset.rename(columns={value_col: "GDP_GROWTH"})


def build_factor_mapping(
    monthly_columns: Sequence[str],
    metadata: pd.DataFrame,
    quarterly_var: str = "GDP_GROWTH",
    include_global: bool = True,
    quarterly_loads_on_all_blocks: bool = True,
) -> Dict[str, List[str]]:
    """
    Create the block-structured factor loading dictionary expected by DynamicFactorMQ.
    """
    meta = metadata.set_index("mnemonic")
    mapping: Dict[str, List[str]] = {}
    blocks_present: List[str] = []

    for col in monthly_columns:
        if col not in meta.index:
            continue
        block = meta.loc[col, "block"]
        blocks_present.append(block)
        factors = [block]
        if include_global:
            factors = ["global"] + factors
        mapping[col] = factors

    unique_blocks = list(dict.fromkeys(blocks_present))
    if quarterly_loads_on_all_blocks:
        q_factors = unique_blocks.copy()
        if include_global:
            q_factors = ["global"] + q_factors
        mapping[quarterly_var] = q_factors

    return mapping


def extract_loading_table(result) -> pd.DataFrame:
    """
    Reconstruct the factor-loading matrix from parameter names.
    """
    rows = []
    for name, value in zip(result.param_names, result.params):
        if not str(name).startswith("loading."):
            continue
        match = re.match(r"^loading\.(.+?)->(.+)$", str(name))
        if match:
            factor, variable = match.groups()
            rows.append({"variable": variable, "factor": factor, "loading": float(value)})

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows).pivot(index="variable", columns="factor", values="loading")
    return table


def _sign_orient_states(
    result,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orient smoothed factors so that anchor loadings are positive.

    Returns
    -------
    states : DataFrame
        Smoothed factor states after sign normalization.
    loadings : DataFrame
        Loading table with the same sign normalization.
    """
    states = result.factors.smoothed.copy()
    loadings = extract_loading_table(result)

    meta = metadata.set_index("mnemonic")
    for block in states.columns:
        if block == "global":
            anchor_candidates = metadata.loc[metadata["anchor"], "mnemonic"].tolist()
            anchor = anchor_candidates[0] if anchor_candidates else None
        else:
            candidates = metadata.loc[(metadata["block"] == block) & (metadata["anchor"]), "mnemonic"].tolist()
            anchor = candidates[0] if candidates else None

        if anchor is None or anchor not in loadings.index or block not in loadings.columns:
            continue

        if pd.notna(loadings.loc[anchor, block]) and loadings.loc[anchor, block] < 0:
            states[block] = -states[block]
            loadings[block] = -loadings[block]

    return states, loadings


def prepare_vintage_panel(
    vintage: pd.Period,
    transformed_monthly_snapshot: pd.DataFrame,
    metadata: pd.DataFrame,
    quarterly_history: pd.DataFrame,
) -> PreparedVintagePanel:
    """
    Prepare standardized monthly + quarterly panels plus coverage summaries.
    """
    # Restrict to stable subset columns that are present in the vintage.
    stable_cols = [c for c in metadata["mnemonic"] if c in transformed_monthly_snapshot.columns]
    monthly = transformed_monthly_snapshot[stable_cols].copy()

    # The monthly panel is already truncated at the vintage by construction.
    monthly_std, monthly_means, monthly_stds = standardize_panel(monthly)
    quarterly_std, quarterly_means, quarterly_stds = standardize_panel(quarterly_history)

    coverage = compute_block_coverage(monthly, metadata, vintage)

    return PreparedVintagePanel(
        vintage=vintage,
        monthly=monthly_std,
        quarterly=quarterly_std,
        monthly_means=monthly_means,
        monthly_stds=monthly_stds,
        quarterly_means=quarterly_means,
        quarterly_stds=quarterly_stds,
        coverage=coverage,
    )


def fit_dfm_vintage(
    prepared: PreparedVintagePanel,
    metadata: pd.DataFrame,
    p_grid: Sequence[int] = (1, 2, 3),
    maxiter: int = 200,
    tolerance: float = 1e-6,
    disp: bool = False,
) -> DFMFitBundle:
    """
    Fit the block-structured mixed-frequency DFM at a single vintage and select p
    from a small admissible grid by AIC.
    """
    monthly = prepared.monthly
    quarterly = prepared.quarterly
    if monthly.shape[1] < 7:
        raise ValueError(
            "Need at least 7 monthly variables to estimate 1 global + 6 block factors."
        )

    factors = build_factor_mapping(monthly.columns.tolist(), metadata)
    factor_names = factors["GDP_GROWTH"]

    best = None
    best_res = None
    best_p = None

    for p in p_grid:
        factor_orders = {tuple(factor_names): int(p)}
        mod = DynamicFactorMQ(
            monthly,
            endog_quarterly=quarterly,
            factors=factors,
            factor_orders=factor_orders,
            idiosyncratic_ar1=True,
            standardize=False,
        )
        res = mod.fit(maxiter=maxiter, tolerance=tolerance, disp=disp)
        score = res.aic
        if best is None or score < best:
            best = score
            best_res = res
            best_p = int(p)

    oriented_states, oriented_loadings = _sign_orient_states(best_res, metadata)

    quarter = prepared.vintage.asfreq("Q")
    current_month = prepared.vintage
    current_factor_state = oriented_states.loc[current_month]
    nowcast = best_res.forecast(1, original_scale=False).iloc[0]["GDP_GROWTH"]

    nowcast_table = pd.DataFrame(
        [{
            "vintage": str(prepared.vintage),
            "quarter": str(quarter),
            "tau": tau_from_vintage_and_quarter(prepared.vintage, quarter),
            "dfm_nowcast_std_scale": nowcast,
            "lag_order": best_p,
            "aic": best_res.aic,
            "bic": best_res.bic,
            "llf": best_res.llf,
        }]
    )

    diagnostics = pd.DataFrame(
        [{
            "vintage": str(prepared.vintage),
            "lag_order": best_p,
            "aic": best_res.aic,
            "bic": best_res.bic,
            "hqic": getattr(best_res, "hqic", np.nan),
            "llf": best_res.llf,
            "nobs": best_res.nobs,
            "iterations": getattr(best_res, "mle_retvals", {}).get("iter", np.nan)
            if hasattr(best_res, "mle_retvals")
            else np.nan,
        }]
    )

    return DFMFitBundle(
        vintage=prepared.vintage,
        lag_order=best_p,
        result=best_res,
        diagnostics=diagnostics,
        states=oriented_states,
        nowcast_table=nowcast_table,
    )


# ---------------------------------------------------------------------------
# News decomposition and artefacts
# ---------------------------------------------------------------------------

def apply_updated_vintage(
    fitted_bundle: DFMFitBundle,
    updated_prepared: PreparedVintagePanel,
):
    """
    Re-apply the fitted parameterization to an updated vintage using the same
    standardization basis retained inside the fitted result.
    """
    return fitted_bundle.result.apply(
        updated_prepared.monthly,
        endog_quarterly=updated_prepared.quarterly,
        retain_standardization=True,
    )


def news_block_aggregation(
    news,
    metadata: pd.DataFrame,
    impacted_variable: str = "GDP_GROWTH",
) -> pd.DataFrame:
    """
    Aggregate series-level news into block-level signed and absolute news.
    """
    details = news.details_by_impact.copy()

    # Keep only rows for the selected impacted variable
    details = details.xs(impacted_variable, level="impacted variable")
    details = details.reset_index()

    meta = metadata[["mnemonic", "block"]].drop_duplicates()
    details = details.merge(meta, left_on="updated variable", right_on="mnemonic", how="left")
    details["block"] = details["block"].fillna("unmapped")

    grouped = (
        details.groupby(["impact date", "block"], as_index=False)
        .agg(
            signed_block_news=("impact", "sum"),
            abs_block_news=("impact", lambda s: np.abs(s).sum()),
            n_updates=("impact", "size"),
        )
    )
    return grouped


def extract_series_news_table(news, impacted_variable: str = "GDP_GROWTH") -> pd.DataFrame:
    """
    Flatten the statsmodels NewsResults details table into a simple dataframe.
    """
    details = news.details_by_impact.copy()
    details = details.xs(impacted_variable, level="impacted variable")
    details = details.reset_index()
    return details


def dfm_state_table(bundle: DFMFitBundle) -> pd.DataFrame:
    """
    Return a tidy state table for export.
    """
    df = bundle.states.copy()
    df["month"] = df.index.astype(str)
    return df.reset_index(drop=True)


def pseudo_real_time_loop_template(
    vintages: Sequence[pd.Period],
    target_vintage_table: pd.DataFrame,
    metadata: pd.DataFrame,
    snapshot_loader,
) -> pd.DataFrame:
    """
    Template loop for the DFM backbone. `snapshot_loader` should be a callable
    that accepts a monthly vintage and returns (snapshot, tcodes).
    """
    outputs = []
    previous_bundle = None

    for vintage in vintages:
        snapshot, tcodes = snapshot_loader(vintage)
        transformed = transform_snapshot(snapshot, tcodes)
        quarterly_history = build_quarterly_history_for_vintage(target_vintage_table, vintage)
        prepared = prepare_vintage_panel(
            vintage=vintage,
            transformed_monthly_snapshot=transformed,
            metadata=metadata,
            quarterly_history=quarterly_history,
        )
        bundle = fit_dfm_vintage(prepared, metadata)
        outputs.append(bundle.nowcast_table)

        if previous_bundle is not None:
            updated_result = apply_updated_vintage(previous_bundle, prepared)
            news = updated_result.news(previous_bundle.result, comparison_type="previous")
            # At this point you would export:
            # - extract_series_news_table(news)
            # - news_block_aggregation(news, metadata)
            # - prepared.coverage
            # - dfm_state_table(bundle)

        previous_bundle = bundle

    return pd.concat(outputs, ignore_index=True)


# ---------------------------------------------------------------------------
# Smoke test data from statsmodels package
# ---------------------------------------------------------------------------

def load_frbny_debug_sample(data_dir: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the two small monthly snapshots used in the statsmodels FRBNY nowcast tests.
    """
    data_dir = Path(data_dir)
    old = pd.read_csv(data_dir / "2016-06-29.csv")
    new = pd.read_csv(data_dir / "2016-07-29.csv")
    return old, new


def _parse_us_debug_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.PeriodIndex(pd.to_datetime(out["Date"], format="%m/%d/%y"), freq="M")
    return out.set_index("Date")


def transform_frbny_debug_sample(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a compact 8-monthly-variable + 1-quarterly-variable sample that is
    quick to estimate and is useful only for code sanity checks.
    """
    df = _parse_us_debug_dates(df)

    monthly_vars = ["INDPRO", "PAYEMS", "HOUST", "RSAFS", "CPIAUCSL", "IR", "UNRATE", "PERMIT"]
    m = df[monthly_vars].copy()

    transformed = pd.DataFrame(index=m.index)
    transformed["INDPRO"] = np.log(m["INDPRO"]).diff() * 100
    transformed["PAYEMS"] = np.log(m["PAYEMS"]).diff() * 100
    transformed["HOUST"] = np.log(m["HOUST"])
    transformed["RSAFS"] = np.log(m["RSAFS"]).diff() * 100
    transformed["CPIAUCSL"] = np.log(m["CPIAUCSL"]).diff().diff() * 100
    transformed["IR"] = m["IR"].diff()
    transformed["UNRATE"] = m["UNRATE"].diff()
    transformed["PERMIT"] = np.log(m["PERMIT"])

    q = df[["GDPC1"]].copy()
    q.index = q.index.to_timestamp()
    q = q.resample("QE").last()
    q.index = pd.PeriodIndex(q.index, freq="Q")
    q["GDP_GROWTH"] = ((q["GDPC1"] / q["GDPC1"].shift(1)) ** 4 - 1) * 100
    q = q[["GDP_GROWTH"]]

    transformed = transformed.loc["1990":]
    q = q.loc["1990":]
    return transformed, q
