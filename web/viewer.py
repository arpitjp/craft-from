"""Local viewer server — thin wrapper around viewer/server.py.

Usage:
    python3 web/viewer.py output/output.litematic
    python3 web/viewer.py output/output.litematic --port 8432
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from viewer.server import main

if __name__ == "__main__":
    main()
