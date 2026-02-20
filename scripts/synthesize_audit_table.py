"""CLI to synthesize a DP audit table."""

import argparse
import logging
from pathlib import Path

from fairness_auditbench.runners.synthesize import run_synthesis


def main():
    parser = argparse.ArgumentParser(description="Generate DP synthetic audit table.")
    parser.add_argument("--dataset", type=str, default="acs_public_coverage")
    parser.add_argument("--states", nargs="+", default=["CA"])
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--horizon", type=str, default="1-Year")
    parser.add_argument("--survey", type=str, default="person")
    parser.add_argument("--sensitive-col", type=str, default=None)
    
    parser.add_argument("--audit-split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--synth", type=str, default="private_pgm")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--bins", type=int, default=32, help="Numeric bins (PrivatePGM)")
    parser.add_argument("--degree", type=int, default=2, help="Marginal degree (PrivatePGM)")
    
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--fast-dev-run", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    run_synthesis(
        dataset=args.dataset,
        states=args.states,
        year=args.year,
        audit_split=args.audit_split,
        synth_name=args.synth,
        epsilon=args.epsilon,
        delta=args.delta,
        seed=args.seed,
        bins=args.bins,
        degree=args.degree,
        out_dir=args.out_dir,
        fast_dev_run=args.fast_dev_run,
        sensitive_col=args.sensitive_col,
    )

if __name__ == "__main__":
    main()
