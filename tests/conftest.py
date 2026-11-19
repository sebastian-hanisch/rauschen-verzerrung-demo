"""Gemeinsamer Importpfad für die Tests (die Module der Demo liegen im Wurzelordner)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
