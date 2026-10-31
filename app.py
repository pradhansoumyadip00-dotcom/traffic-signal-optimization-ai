import sys
import os
from pathlib import Path

# Bind project root and dashboard directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT_DIR / "dashboard"
for _p in [str(ROOT_DIR), str(DASHBOARD_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load and execute dashboard/app.py
_app_file = DASHBOARD_DIR / "app.py"
with open(_app_file, "r", encoding="utf-8") as _f:
    _code = _f.read()
exec(_code, globals())
