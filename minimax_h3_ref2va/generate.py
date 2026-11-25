"""Generate a video with ARLIFT."""
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND.parent))

from arlift.cli import main

if __name__ == "__main__":
    main(BACKEND)
