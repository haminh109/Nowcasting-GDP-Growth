from pathlib import Path
import pandas as pd, json

ROOT = Path("outputs/layer1_dfm")

def read_design(root):
    for name, reader in [
        ("layer2_residual_design.parquet", pd.read_parquet),
        ("layer2_residual_design.csv", pd.read_csv),
    ]:
        p = root / name
        if p.exists():
            return reader(p), name
    raise FileNotFoundError("Missing layer2_residual_design.parquet and .csv")

def as_period(s, freq):
    if str(s.dtype).startswith("period["):
        return s
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.PeriodIndex(s, freq=freq)
    return pd.PeriodIndex(s.astype(str), freq=freq)

design, design_file = read_design(ROOT)
manifest = pd.read_csv(ROOT / "layer2_feature_manifest.csv")
feature_sets = json.loads((ROOT / "layer2_feature_sets.json").read_text())
contract = json.loads((ROOT / "layer2_data_contract.json").read_text())
handoff_md = (ROOT / "README_layer2_handoff.md").read_text()
layer1_protocol = json.loads((ROOT / "layer1_protocol.json").read_text())

assert "Primary training table" in handoff_md
assert contract["primary_target"] == "dfm_residual_third_release"
assert layer1_protocol["truth_main"] == "third_release"

# keep monthly/quarterly period semantics; never convert to month-end
design["vintage_period"] = as_period(design["vintage_period"], "M")
design["target_quarter"] = as_period(design["target_quarter"], "Q-DEC")
design[contract["train_sample_flag"]] = design[contract["train_sample_flag"]].astype(bool)

included_manifest = set(
    manifest.loc[
        (manifest["role"] == "feature") &
        (manifest["included_in_training_matrix"] == True),
        "column",
    ]
)

baseline_seed = feature_sets.get("baseline_v1", contract["feature_columns"])
forced_first_pass_drop = {
    "news_signed__quarterly_target_history",
    "news_abs__quarterly_target_history",
}
forbidden = set(contract["forbidden_feature_columns"])
audit_only = set(contract["audit_only_fields"])

final_features = [
    c for c in baseline_seed
    if c in included_manifest
    and c not in forbidden
    and c not in audit_only
    and c not in forced_first_pass_drop
    and c != "within_quarter_origin"
]

assert set(final_features).issubset(set(design.columns))
assert not design.duplicated(contract["primary_key"]).any()

design["_complete_case_final_features"] = design[final_features].notna().all(axis=1)
design["_trainable_complete_case_watch"] = (
    design[contract["train_sample_flag"]] & design["_complete_case_final_features"]
)

row_counts = (
    design.groupby("within_quarter_origin")
    .agg(
        total_rows=("within_quarter_origin", "size"),
        target_available_rows=(contract["train_sample_flag"], "sum"),
        final_trainable_rows=(contract["train_sample_flag"], "sum"),  # contract-only
        complete_case_trainable_rows_watch=("_trainable_complete_case_watch", "sum"),
    )
    .reset_index()
)

mi = manifest.set_index("column")
feature_groups = {
    "dfm backbone": [
        c for c in final_features if mi.loc[c, "feature_group"] == "dfm_backbone"
    ],
    "states": [
        c for c in final_features if mi.loc[c, "feature_group"] == "factor_states"
    ],
    "signed news": [
        c for c in final_features
        if mi.loc[c, "feature_group"] == "news_blocks" and c.startswith("news_signed__")
    ],
    "abs news": [
        c for c in final_features
        if mi.loc[c, "feature_group"] == "news_blocks" and c.startswith("news_abs__")
    ],
    "coverage": [
        c for c in final_features
        if mi.loc[c, "feature_group"] in {"coverage", "coverage_counts"}
    ],
    "diagnostics": [
        c for c in final_features if mi.loc[c, "feature_group"] == "diagnostics"
    ],
    "other engineered features": [
        c for c in final_features if mi.loc[c, "feature_group"] == "other_features"
    ],
}

truth_cols = {"third_release", "latest_rtdsm", "gdpplus"}
drop_records = []

for c in design.columns:
    if c in final_features or c.startswith("_"):
        continue
    reasons = []
    if c in contract["metadata_columns"]:
        reasons.append("metadata")
    if c in contract["primary_key"]:
        reasons.append("primary_key")
    if c == "within_quarter_origin":
        reasons.append("metadata_not_feature")
    if c == contract["train_sample_flag"]:
        reasons.append("train_sample_flag")
    if c in truth_cols:
        reasons.append("truth_column")
    if c == contract["primary_target"]:
        reasons.append("primary_target")
    elif c in contract["robustness_targets"]:
        reasons.append("robustness_target")
    if c in forced_first_pass_drop:
        reasons.append("excluded_in_first_pass_baseline")
    if c in forbidden:
        reasons.append("forbidden_feature_column")
    if c in audit_only:
        reasons.append("audit_only_field")
    drop_records.append((c, "|".join(dict.fromkeys(reasons)) or "not_selected"))

manifest_absent = manifest.loc[
    ~manifest["column"].isin(design.columns),
    ["column", "included_in_training_matrix", "exclusion_reason"],
]
for _, r in manifest_absent.iterrows():
    reasons = ["absent_from_design"]
    if r["included_in_training_matrix"] is False:
        reasons.append("included_in_training_matrix=False")
    if pd.notna(r["exclusion_reason"]):
        reasons.append(str(r["exclusion_reason"]))
    drop_records.append((r["column"], "|".join(reasons)))

state_kept = [c for c in final_features if c.startswith("state__")]

print(f"design_file={design_file}")
print(f"total_rows={len(design)}")
print(f"primary_target={contract['primary_target']}")
print(f"primary_target_available_rows={int(design[contract['train_sample_flag']].sum())}")
print(f"final_trainable_rows_contract={int(design[contract['train_sample_flag']].sum())}")
print(f"final_trainable_rows_complete_case_watch={int(design['_trainable_complete_case_watch'].sum())}")
print(f"final_feature_count={len(final_features)}")

print("final_feature_names=")
for c in final_features:
    print(f"  - {c}")

print("dropped_columns_and_reasons=")
for c, reason in drop_records:
    print(f"  - {c}: {reason}")

print("row_counts_by_tau=")
print(row_counts.to_string(index=False))

print("state_features_kept=")
for c in state_kept:
    print(f"  - {c}")

print("feature_groups=")
for g, cols in feature_groups.items():
    print(f"[{g}] ({len(cols)})")
    for c in cols:
        print(f"  - {c}")