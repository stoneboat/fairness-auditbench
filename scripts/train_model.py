#!/usr/bin/env python
"""CLI entry-point for training models.

Usage examples:
    # fast smoke on CPU or GPU
    python scripts/train_model.py --dataset acs_public_coverage --model logreg --fast-dev-run --seed 0

    python scripts/train_model.py --dataset acs_public_coverage --model ft_transformer \\
        --fast-dev-run --seed 0 --max-epochs 2

    # full run
    python scripts/train_model.py --dataset acs_public_coverage --model ft_transformer \\
        --states CA --year 2018 --seed 0
"""

import os
import sys

# Ensure the src/ directory is importable even without editable install
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(os.path.dirname(_here), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from fairness_auditbench.cli import main

if __name__ == "__main__":
    main()
