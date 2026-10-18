"""Double-click launcher. Prefer Finish.bat the first time so libraries can install."""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
os.chdir(root)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.main import main

main()
