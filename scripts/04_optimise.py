#!/usr/bin/env python3
"""Stage 4 - BAYESIAN OPTIMISATION (PyTorch).  Run with --help for options; see README for the 4-stage workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bwbopt.stages.optimise_stage import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
