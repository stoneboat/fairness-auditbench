# fairness-auditbench

The goal of this repository is to investigate if Differential Privacy (DP) synthetic-data generation algorithms can retain enough high-dimensional correlation to accurately validate fairness metrics in complex ML models.

Currently, we evaluate marginal-based DP generation algorithms (implemented via [dpmm](https://github.com/dpcomp-org/dpmm) for Private-PGM) against two types of models:
- **Logistic Regression**
- **FT-Transformer** (a neural network for tabular data)

## Quick Start

### 1. Environment Setup (Cluster)

```bash
# From the repo root:
bash scripts/local_scripts/cluster_install_gpu.sh
```

This will:
- Load `pytorch/25` and `anaconda3` modules
- Create / activate a conda env at `/tmp/python-venv/fairness-auditbench_venv`
- Install Python dependencies from `requirements.txt`
- Install the package in editable mode (`pip install -e .`)
- Register a Jupyter kernel

### 2. Activate the Environment

```bash
module load pytorch/25 anaconda3
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/fairness-auditbench_venv
```

### 3. Train Models

```bash
# Logistic Regression — fast smoke test
python scripts/train_model.py \
    --dataset acs_public_coverage \
    --model logreg \
    --fast-dev-run --seed 0

# FT-Transformer — fast smoke test
python scripts/train_model.py \
    --dataset acs_public_coverage \
    --model ft_transformer \
    --fast-dev-run --seed 0 --max-epochs 2

# Full training run
python scripts/train_model.py \
    --dataset acs_public_coverage \
    --model ft_transformer \
    --states CA --year 2018 --seed 0
```

### 4. Changing the Sensitive Attribute

By default the sensitive attribute is `RAC1P` (race). Override with `--sensitive-col`:

```bash
python scripts/train_model.py \
    --dataset acs_public_coverage \
    --model logreg \
    --sensitive-col DIS \
    --seed 0
```

### 5. DP Synthesizer Pipeline & Auditing

You can synthesize a DP audit table (containing $X, y, A$) using `Private-PGM` or a baseline:

```bash
python scripts/synthesize_audit_table.py \
    --dataset acs_public_coverage \
    --synth private_pgm \
    --epsilon 1.0 \
    --seed 0 \
    --fast-dev-run
```

Once a model is trained and an audit table is synthesized, run the fairness audit:

```bash
python scripts/audit_fairness.py \
    --dataset acs_public_coverage \
    --model logreg \
    --model-seed 0 \
    --synth private_pgm \
    --epsilon 1.0 \
    --synth-seed 0 \
    --fast-dev-run
```

Results are appended to `results/metrics.jsonl`.

### 6. CLI Help

```bash
python -m fairness_auditbench --help
```

## Output Structure

```
results/
  models/
    acs_public_coverage/
      logreg/seed=0/
        pipeline.joblib
        metrics.json
      ft_transformer/seed=0/
        model.pt
        hparams.json
        preprocessor.joblib
        metrics.json
  runs/
    <timestamp>_<model>_<dataset>.json   # run summaries
  synth/
    <dataset>/<synth_name>/eps=<eps>/seed=<seed>/audit_split=<split>/
      synthetic.parquet
      metadata.json
  metrics.jsonl                          # fairness auditing results
```

## Project Layout

```
fairness-auditbench/
  src/fairness_auditbench/    # importable Python package
    datasets/                 # pluggable dataset loaders
    models/                   # model implementations
    preprocess/               # sklearn / torch preprocessors
    runners/                  # training orchestrator
    cli.py                    # argparse CLI
    config.py                 # TrainConfig dataclass
  scripts/
    train_model.py            # main entry-point script
    local_scripts/            # cluster setup scripts
  Notebooks/                  # step-by-step notebooks
  data/                       # cached downloads (gitignored)
  results/                    # outputs (gitignored)
```
