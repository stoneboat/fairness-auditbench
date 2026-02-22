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
    parser.add_argument("--epsilon", type=float, default=None, help="Privacy budget (ignored if synth is non-DP)")
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--bins", type=int, default=32, help="Numeric bins (PrivatePGM)")
    parser.add_argument("--degree", type=int, default=2, help="Marginal degree (PrivatePGM)")
    
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--fast-dev-run", action="store_true")
    
    # DPCTGAN specific arguments
    parser.add_argument("--dpctgan-epochs", type=int, default=300)
    parser.add_argument("--dpctgan-batch-size", type=int, default=500)
    parser.add_argument("--dpctgan-pac", type=int, default=1)
    parser.add_argument("--dpctgan-sigma", type=float, default=5.0)
    parser.add_argument("--dpctgan-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dpctgan-preprocessor-eps", type=float, default=None)
    parser.add_argument("--dpctgan-disabled-dp", action="store_true", default=False)
    parser.add_argument("--dpctgan-loss", type=str, default="cross_entropy")
    parser.add_argument("--dpctgan-no-cuda", action="store_true", default=False)
    parser.add_argument("--dpctgan-no-verbose", action="store_true", default=False)

    # CTGAN specific arguments
    parser.add_argument("--ctgan-epochs", type=int, default=300)
    parser.add_argument("--ctgan-batch-size", type=int, default=500)
    parser.add_argument("--ctgan-pac", type=int, default=10)
    parser.add_argument("--ctgan-embedding-dim", type=int, default=128)
    parser.add_argument("--ctgan-generator-dim", type=str, default="256,256")
    parser.add_argument("--ctgan-discriminator-dim", type=str, default="256,256")
    parser.add_argument("--ctgan-generator-lr", type=float, default=2e-4)
    parser.add_argument("--ctgan-discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--ctgan-generator-decay", type=float, default=1e-6)
    parser.add_argument("--ctgan-discriminator-decay", type=float, default=1e-6)
    parser.add_argument("--ctgan-discriminator-steps", type=int, default=1)
    parser.add_argument("--ctgan-disable-gpu", action="store_true", default=False)
    parser.add_argument("--ctgan-verbose", action="store_true", default=False)

    args = parser.parse_args()
    
    # Override epochs for fast dev run if it's the default
    if args.fast_dev_run and args.dpctgan_epochs == 300:
        args.dpctgan_epochs = 20
    if args.fast_dev_run and args.ctgan_epochs == 300:
        args.ctgan_epochs = 20

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    synth_kwargs = {}
    if args.synth in ["dpctgan", "patectgan"]:
        synth_kwargs = dict(
            epochs=args.dpctgan_epochs,
            batch_size=args.dpctgan_batch_size,
            pac=args.dpctgan_pac,
            sigma=args.dpctgan_sigma,
            max_per_sample_grad_norm=args.dpctgan_max_grad_norm,
            preprocessor_eps=args.dpctgan_preprocessor_eps,
            disabled_dp=args.dpctgan_disabled_dp,
            loss=args.dpctgan_loss,
            cuda=not args.dpctgan_no_cuda,
            verbose=not args.dpctgan_no_verbose,
        )
    elif args.synth == "ctgan":
        synth_kwargs = dict(
            epochs=args.ctgan_epochs,
            batch_size=args.ctgan_batch_size,
            pac=args.ctgan_pac,
            embedding_dim=args.ctgan_embedding_dim,
            generator_dim=args.ctgan_generator_dim,
            discriminator_dim=args.ctgan_discriminator_dim,
            generator_lr=args.ctgan_generator_lr,
            discriminator_lr=args.ctgan_discriminator_lr,
            generator_decay=args.ctgan_generator_decay,
            discriminator_decay=args.ctgan_discriminator_decay,
            discriminator_steps=args.ctgan_discriminator_steps,
            enable_gpu=not args.ctgan_disable_gpu,
            verbose=args.ctgan_verbose,
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
        **synth_kwargs
    )

if __name__ == "__main__":
    main()
