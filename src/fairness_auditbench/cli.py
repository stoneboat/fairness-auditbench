"""Command-line interface for fairness_auditbench training."""

import argparse
import logging
import sys

from fairness_auditbench.config import TrainConfig
from fairness_auditbench.runners.train import run_training


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fairness_auditbench",
        description="Train models for fairness auditing benchmarks.",
    )

    # Dataset
    p.add_argument("--dataset", default="acs_public_coverage", help="Dataset name (default: acs_public_coverage)")
    p.add_argument("--states", nargs="+", default=["CA"], help="US state(s) (default: CA)")
    p.add_argument("--year", type=int, default=2018, help="Survey year (default: 2018)")
    p.add_argument("--horizon", default="1-Year", help="ACS horizon (default: 1-Year)")
    p.add_argument("--survey", default="person", help="ACS survey type (default: person)")
    p.add_argument("--sensitive-col", default=None, help="Override sensitive attribute column")

    # Model
    p.add_argument("--model", default="logreg", choices=["logreg", "ft_transformer"], help="Model to train")

    # Reproducibility
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    # Dev / debug
    p.add_argument("--fast-dev-run", action="store_true", help="Subsample data for fast smoke test")
    p.add_argument("--fast-dev-n", type=int, default=500, help="Rows to keep in fast dev run")

    # Paths
    p.add_argument("--data-dir", default="data", help="Root data directory")
    p.add_argument("--output-dir", default="results", help="Root output directory")

    # Training hyper-params
    p.add_argument("--max-epochs", type=int, default=50, help="Max training epochs (FT-Transformer)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    p.add_argument("--logreg-max-iter", type=int, default=1000, help="Max iterations for LogReg")

    return p


def main(argv=None):
    """Parse CLI args, build config, and run training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    config = TrainConfig(
        dataset=args.dataset,
        states=args.states,
        year=args.year,
        horizon=args.horizon,
        survey=args.survey,
        sensitive_col=args.sensitive_col,
        model=args.model,
        seed=args.seed,
        fast_dev_run=args.fast_dev_run,
        fast_dev_n=args.fast_dev_n,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        logreg_max_iter=args.logreg_max_iter,
    )

    summary = run_training(config)

    print("\n✅ Training complete!")
    print(f"   Model:   {summary['model']}")
    print(f"   Dataset: {summary['dataset']}")
    print(f"   Metrics: {summary['metrics']}")
    print(f"   Saved:   {summary['model_dir']}")


if __name__ == "__main__":
    main()
