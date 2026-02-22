"""CLI to run fairness auditing on real vs synthetic data."""

import argparse
import logging
from pathlib import Path

from fairness_auditbench.runners.audit import run_audit

def main():
    parser = argparse.ArgumentParser(description="Audit fairness on real vs synthetic data.")
    
    parser.add_argument("--dataset", type=str, default="acs_public_coverage")
    parser.add_argument("--states", nargs="+", default=["CA"])
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--sensitive-col", type=str, default=None)
    
    parser.add_argument("--model", type=str, choices=["logreg", "ft_transformer"], required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    
    parser.add_argument("--audit-split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--synth", type=str, default="private_pgm")
    parser.add_argument("--epsilon", type=float, default=None, help="Privacy budget (ignored if synth is non-DP)")
    parser.add_argument("--synth-seed", type=int, default=0)
    
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--fast-dev-run", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    run_audit(
        dataset=args.dataset,
        states=args.states,
        year=args.year,
        model_type=args.model,
        model_seed=args.model_seed,
        audit_split=args.audit_split,
        synth_name=args.synth,
        epsilon=args.epsilon,
        synth_seed=args.synth_seed,
        out_dir=args.out_dir,
        fast_dev_run=args.fast_dev_run,
        sensitive_col=args.sensitive_col,
    )

if __name__ == "__main__":
    main()
