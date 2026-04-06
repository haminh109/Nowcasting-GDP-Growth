from pathlib import Path
import json
import subprocess
from datetime import datetime, timezone

ROOT = Path(".").resolve()
OUT = ROOT / "outputs" / "layer1_dfm"

src = OUT / "layer1_protocol.json"
dst = OUT / "layer1_protocol_sanitized.json"

with open(src, "r", encoding="utf-8") as f:
    protocol = json.load(f)

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return None

def get_git_branch():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return None

protocol["repo_root"] = "."
protocol["output_dir"] = "outputs/layer1_dfm"
protocol["export_contract_version"] = "1.0.0"
protocol["export_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
protocol["git_commit"] = get_git_commit()
protocol["git_branch"] = get_git_branch()

with open(dst, "w", encoding="utf-8") as f:
    json.dump(protocol, f, ensure_ascii=False, indent=2)

print(f"Written: {dst}")