from pathlib import Path
import json
import pandas as pd

OUT = Path("outputs/layer1_dfm")

def _read_training_table():
    parquet_path = OUT / "layer2_residual_design.parquet"
    csv_path = OUT / "layer2_residual_design.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("No layer2_residual_design parquet/csv found")

def load_layer2_inputs(feature_set_name="baseline_v1", target_name="dfm_residual_third_release"):
    df = _read_training_table()

    with open(OUT / "layer2_data_contract.json", "r", encoding="utf-8") as f:
        contract = json.load(f)

    with open(OUT / "layer2_feature_sets.json", "r", encoding="utf-8") as f:
        feature_sets = json.load(f)

    feature_cols = feature_sets[feature_set_name]

    # preserve period semantics
    df["vintage_period"] = pd.PeriodIndex(df["vintage_period"], freq="M")
    df["target_quarter"] = pd.PeriodIndex(df["target_quarter"], freq="Q-DEC")

    # train sample
    train_flag = contract["train_sample_flag"]
    train_df = df.loc[df[train_flag] == True].copy()

    # metadata
    metadata_cols = contract["metadata_columns"]

    # X, y, metadata
    X = train_df[feature_cols].copy()
    y = train_df[target_name].copy()
    meta = train_df[metadata_cols].copy()

    return {
        "full_table": df,
        "train_table": train_df,
        "X": X,
        "y": y,
        "meta": meta,
        "feature_cols": feature_cols,
        "contract": contract,
    }

if __name__ == "__main__":
    obj = load_layer2_inputs()
    print("train_table rows:", len(obj["train_table"]))
    print("X shape:", obj["X"].shape)
    print("y non-null:", obj["y"].notna().sum())
    print("first 5 features:", obj["feature_cols"][:5])