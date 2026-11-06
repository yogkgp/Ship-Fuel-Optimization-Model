r"""Modal deployment wrapper for the Streamlit dashboard.

Deploy with:
    .\.venv\Scripts\modal.exe deploy modal_deploy.py
"""

import os
import subprocess
from pathlib import Path

import modal


LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root/project")
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements.txt"))
    .add_local_dir(LOCAL_ROOT / "app", REMOTE_ROOT / "app", copy=True)
    .add_local_dir(LOCAL_ROOT / "src", REMOTE_ROOT / "src", copy=True)
    .add_local_dir(LOCAL_ROOT / "models", REMOTE_ROOT / "models", copy=True)
    .add_local_dir(LOCAL_ROOT / "data" / "processed", REMOTE_ROOT / "data" / "processed", copy=True)
    .add_local_dir(LOCAL_ROOT / "reports" / "metrics", REMOTE_ROOT / "reports" / "metrics", copy=True)
)

app = modal.App("ship-fuel-optimization", image=image)


@app.function(cpu=2.0, memory=4096, timeout=3600, scaledown_window=300)
@modal.concurrent(max_inputs=20)
@modal.web_server(PORT, startup_timeout=120)
def dashboard():
    """Start Streamlit inside the Modal container."""
    os.chdir(REMOTE_ROOT)
    subprocess.Popen(
        [
            "streamlit",
            "run",
            "app/streamlit_app.py",
            "--server.address=0.0.0.0",
            f"--server.port={PORT}",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--browser.gatherUsageStats=false",
        ]
    )
