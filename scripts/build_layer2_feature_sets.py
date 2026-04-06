from pathlib import Path
import json
import pandas as pd

OUT = Path("outputs/layer1_dfm")
manifest = pd.read_csv(OUT / "layer2_feature_manifest.csv")

included = manifest.loc[
    manifest["included_in_training_matrix"] == True, "column"
].tolist()

sparse_optional = [
    "news_signed__quarterly_target_history",
    "news_abs__quarterly_target_history",
]

baseline = [c for c in included if c not in sparse_optional]

feature_sets = {
    "baseline_v1": baseline,
    "full_included_manifest": included,
    "optional_sparse_news": sparse_optional,
}

with open(OUT / "layer2_feature_sets.json", "w", encoding="utf-8") as f:
    json.dump(feature_sets, f, ensure_ascii=False, indent=2)

print("Written:", OUT / "layer2_feature_sets.json")
print("baseline_v1:", len(baseline))
print("full_included_manifest:", len(included))