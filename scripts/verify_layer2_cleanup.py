from pathlib import Path
import json
import pandas as pd

OUT = Path("outputs/layer1_dfm")

# files
required = [
    OUT / "layer2_data_contract.json",
    OUT / "layer2_feature_manifest.csv",
    OUT / "layer2_feature_sets.json",
    OUT / "README_layer2_handoff.md",
]
for p in required:
    assert p.exists(), f"Missing required cleanup artifact: {p}"

# protocol sanitized
proto_path = OUT / "layer1_protocol_sanitized.json"
assert proto_path.exists(), "Missing sanitized protocol"

with open(proto_path, "r", encoding="utf-8") as f:
    proto = json.load(f)

assert proto["repo_root"] == ".", "repo_root not sanitized"
assert proto["output_dir"] == "outputs/layer1_dfm", "output_dir not sanitized"
assert proto["export_same_tau_residual_lags"] is False, "same-tau lag safety unexpectedly changed"

# contract
with open(OUT / "layer2_data_contract.json", "r", encoding="utf-8") as f:
    contract = json.load(f)

assert contract["primary_target"] == "dfm_residual_third_release"
assert contract["train_sample_flag"] == "primary_target_available"

# manifest
manifest = pd.read_csv(OUT / "layer2_feature_manifest.csv")
included = manifest.loc[manifest["included_in_training_matrix"] == True, "column"].tolist()

# feature sets
with open(OUT / "layer2_feature_sets.json", "r", encoding="utf-8") as f:
    feature_sets = json.load(f)

baseline = feature_sets["baseline_v1"]
for c in baseline:
    assert c in included, f"{c} not in included_in_training_matrix"

assert "news_signed__quarterly_target_history" not in baseline
assert "news_abs__quarterly_target_history" not in baseline

# training table
if (OUT / "layer2_residual_design.parquet").exists():
    df = pd.read_parquet(OUT / "layer2_residual_design.parquet")
else:
    df = pd.read_csv(OUT / "layer2_residual_design.csv")

# parse periods
df["vintage_period"] = pd.PeriodIndex(df["vintage_period"], freq="M")
df["target_quarter"] = pd.PeriodIndex(df["target_quarter"], freq="Q-DEC")

# unique key
dupes = df.duplicated(["vintage_period", "target_quarter"]).sum()
assert dupes == 0, f"Found duplicate keys: {dupes}"

# target-known sample
train_df = df.loc[df["primary_target_available"] == True].copy()
assert train_df["dfm_residual_third_release"].notna().all(), "Target-known rows contain null target"

print("All cleanup checks passed.")
print("Rows total:", len(df))
print("Rows target-known:", len(train_df))
print("Baseline feature count:", len(baseline))