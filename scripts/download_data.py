"""Download the verified revision without overwriting an existing raw file."""
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "eb6a6ec011c1c9a2cbce21459e22be4c77ef84dd"
FILES = {
    "CPS_Triton.parquet": "327581b0d35c756db207e6992a1d78f6f7e4f68dcfd299c77188912fdbd6aab8",
    "CPS_Poseidon.parquet": "fcc4fff3e86de3d02b034376429e4cd98c794befb4e8f610187643d52552571b",
    "OSS_Ceto.parquet": "a81eefff2bb6bedbb85455e4fb408906a1de5356f0633fda450d7e794410b7ef",
}
for filename, expected in FILES.items():
    destination = ROOT / "data/raw" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        url = f"https://huggingface.co/datasets/krohnedigital/FuelCast/resolve/{REVISION}/{filename}"
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        temporary = destination.with_suffix(".download")
        temporary.write_bytes(response.content)
        temporary.replace(destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Source hash differs from verified file: {filename}")
    print(f"Raw data ready: {destination}")
