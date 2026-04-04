import sys
from pathlib import Path

pytest_plugins = ["tests.tetriszero.fixtures"]


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
