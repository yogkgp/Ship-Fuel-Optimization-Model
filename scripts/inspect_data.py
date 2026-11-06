"""Inspect original parquet before any modelling; write full schema and quality."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data/raw/CPS_Triton.parquet"
frame = pd.read_parquet(path, engine="pyarrow")
report = {
    "file": path.name,
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "rows": len(frame), "columns": len(frame.columns),
    "schema": {c: str(t) for c, t in frame.dtypes.items()},
    "missing": frame.isna().sum().to_dict(),
    "duplicate_rows": int(frame.duplicated().sum()),
    "index_type": str(type(frame.index)),
    "numeric_summary": frame.describe().to_dict(),
}
(ROOT / "reports/metrics/inspection.json").write_text(
    json.dumps(report, indent=2, default=str), encoding="utf-8")
print(f"Rows={len(frame):,}; columns={len(frame.columns)}")
print(frame.dtypes.to_string())
print(frame.select_dtypes("number").describe().to_string())
