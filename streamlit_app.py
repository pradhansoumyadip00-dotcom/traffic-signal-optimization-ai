import sys
from pathlib import Path
import runpy

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Run the primary dashboard application
dashboard_path = ROOT_DIR / "dashboard" / "app.py"
runpy.run_path(str(dashboard_path), run_name="__main__")

