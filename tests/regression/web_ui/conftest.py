"""web_ui regression fixtures.

The repo root (which exposes the top-level ``web_ui`` package) is made
importable by ``pythonpath = ["."]`` in pyproject's pytest config, so no
sys.path mutation is needed here.

For unit tests in tests/regression/web_ui/unit/, we also need web_ui/backend
in the path to import services/models/api modules directly.
"""

import sys
from pathlib import Path

# Add web_ui/backend to sys.path for unit test imports
_backend_path = Path(__file__).parent.parent.parent.parent / "web_ui" / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))
