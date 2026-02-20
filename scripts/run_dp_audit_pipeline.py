"""Orchestrator script to optionally train, synthesize, and run audit."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="End-to-End DP Auditing Pipeline")
    parser.add_argument("--train", action="store_true", help="Force train model first")
    parser.add_argument("--dataset", type=str, default="acs_public_coverage")
    parser.add_argument("--states", nargs="+", default=["CA"])
    parser.add_argument("--model", type=str, default="logreg")
    parser.add_argument("--synth", type=str, default="private_pgm")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast-dev-run", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    base_cmd = [sys.executable]
    states_args = ["--states"] + args.states
    fast_dev = ["--fast-dev-run"] if args.fast_dev_run else []
    
    if args.train:
        logger.info("=== 1. Training Model ===")
        cmd = base_cmd + [
            "scripts/train_model.py",
            "--dataset", args.dataset,
            "--model", args.model,
            "--seed", str(args.seed),
        ] + states_args + fast_dev
        subprocess.run(cmd, check=True)
    
    logger.info("=== 2. Synthesizing Audit Table ===")
    cmd = base_cmd + [
        "scripts/synthesize_audit_table.py",
        "--dataset", args.dataset,
        "--synth", args.synth,
        "--epsilon", str(args.epsilon),
        "--seed", str(args.seed),
    ] + states_args + fast_dev
    subprocess.run(cmd, check=True)
    
    logger.info("=== 3. Auditing Fairness ===")
    cmd = base_cmd + [
        "scripts/audit_fairness.py",
        "--dataset", args.dataset,
        "--model", args.model,
        "--model-seed", str(args.seed),
        "--synth", args.synth,
        "--epsilon", str(args.epsilon),
        "--synth-seed", str(args.seed),
    ] + states_args + fast_dev
    subprocess.run(cmd, check=True)
    
    logger.info("=== Pipeline Complete! ===")

if __name__ == "__main__":
    main()
