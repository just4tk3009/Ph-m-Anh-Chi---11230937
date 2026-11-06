import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def synthetic_cfg(tmp_path_factory):
    """Build the whole Bronze->Gold chain on SYNTHETIC data in a temp folder."""
    from fdic_ews.storage.gold import build_gold
    from fdic_ews.storage.silver import build_silver
    from fdic_ews.utils.config import load_config

    d = tmp_path_factory.mktemp("synthetic")
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic_bronze.py"),
                    "--data-dir", str(d / "data"), "--banks", "150"], check=True)
    cfg = load_config(None, str(d / "data"), str(d / "reports"))
    build_silver(cfg)
    build_gold(cfg)
    return cfg
