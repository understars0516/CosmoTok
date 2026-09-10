import os
from pathlib import Path

PROJECT_DIR = Path(__file__).absolute().parent.parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_DIR / "data"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", PROJECT_DIR / "cache"))
