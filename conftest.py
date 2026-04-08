"""Pytest configuration — adds project root to sys.path."""

import sys
from pathlib import Path

# Add project root to Python path so 'from models.xxx' imports work
sys.path.insert(0, str(Path(__file__).parent))
