from pathlib import Path
import json
import pandas as pd

ROOT = Path(".").resolve()
OUT = ROOT / "outputs" / "layer1_dfm"

contract_path = OUT / "layer2_data_contract.json"
manifest_path = OUT / "layer2_feature_manifest.csv"
protocol_path = OUT / "layer1_protocol.json"
table_parquet = OUT / "layer2_residual_design.parquet"
table_csv = OUT / "layer2_residual_design.csv"

print("ROOT:", ROOT)
print("OUT :", OUT)
print()

print("Exists:")
for p in [contract_path, manifest_path, protocol_path, table_parquet, table_csv]:
    print(f"  {p.name}: {p.exists()}")

with open(contract_path, "r", encoding="utf-8") as f:
    contract = json.load(f)

print("\nContract summary:")
print("  primary_training_table_name:", contract["primary_training_table_name"])
print("  primary_key:", contract["primary_key"])
print("  primary_target:", contract["primary_target"])
print("  train_sample_flag:", contract["train_sample_flag"])
print("  n_feature_columns:", len(contract["feature_columns"]))

manifest = pd.read_csv(manifest_path)
print("\nManifest summary:")
print("  rows:", len(manifest))
print("  included_in_training_matrix:", manifest["included_in_training_matrix"].sum())
print("  target rows:", (manifest["role"] == "target").sum())

sparse = manifest[
    (manifest["included_in_training_matrix"] == True) &
    (manifest["nonmissing_share"] < 0.5)
][["column", "nonmissing_share"]]
print("\nSparse included features:")
print(sparse if not sparse.empty else "  None")

with open(protocol_path, "r", encoding="utf-8") as f:
    protocol = json.load(f)

print("\nProtocol summary:")
print("  repo_root:", protocol.get("repo_root"))
print("  output_dir:", protocol.get("output_dir"))
print("  export_same_tau_residual_lags:", protocol.get("export_same_tau_residual_lags"))