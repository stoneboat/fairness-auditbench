"""FT-Transformer model (pure PyTorch).

A simplified Feature-Tokenizer Transformer for tabular binary classification:
  - Learned embeddings for each categorical feature
  - Per-feature linear projection for numerical features → tokens
  - Prepend a [CLS] token
  - Standard Transformer encoder
  - MLP head on [CLS] output → single logit
"""

import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from fairness_auditbench.config import TrainConfig
from fairness_auditbench.datasets.base import DatasetSpec
from fairness_auditbench.models.base import BaseModel
from fairness_auditbench.preprocess.tabular_torch import TorchTabularPreprocessor
from fairness_auditbench.utils.device import get_device
from fairness_auditbench.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn


class _FTTransformerNet(nn.Module):
    """Minimal FT-Transformer architecture."""

    def __init__(
        self,
        vocab_sizes: List[int],
        n_numerical: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_cat = len(vocab_sizes)
        self.n_num = n_numerical

        # ---- Categorical embeddings ---
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(vs, d_model, padding_idx=0) for vs in vocab_sizes]
        )

        # ---- Numerical projections ----
        if n_numerical > 0:
            self.num_proj = nn.Linear(1, d_model)  # shared projection per scalar
            self.num_biases = nn.Parameter(torch.zeros(n_numerical, d_model))
        else:
            self.num_proj = None

        # ---- CLS token ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ---- Head ----
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),  # single logit for binary classification
        )

    def forward(self, cat_codes: torch.Tensor, num_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cat_codes : LongTensor [B, C_cat]
        num_values : FloatTensor [B, C_num]

        Returns
        -------
        logits : FloatTensor [B]
        """
        tokens = []
        # Categorical tokens
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(cat_codes[:, i]))  # [B, d]
        # Numerical tokens
        if self.num_proj is not None and num_values.shape[1] > 0:
            # num_values: [B, C_num] → [B, C_num, 1] → [B, C_num, d]
            num_tokens = self.num_proj(num_values.unsqueeze(-1)) + self.num_biases
            for j in range(num_values.shape[1]):
                tokens.append(num_tokens[:, j, :])  # [B, d]

        if not tokens:
            raise ValueError("No features provided to FT-Transformer")

        # Stack tokens: [B, S, d]
        x = torch.stack(tokens, dim=1)

        # Prepend CLS
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, S+1, d]

        x = self.encoder(x)
        cls_out = x[:, 0, :]  # [B, d]
        logits = self.head(cls_out).squeeze(-1)  # [B]
        return logits


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------


class FTTransformerModel(BaseModel):
    """Trains an FT-Transformer for binary classification."""

    def __init__(self):
        self.net: Optional[_FTTransformerNet] = None
        self.preprocessor: Optional[TorchTabularPreprocessor] = None
        self._metrics: Dict = {}
        self._hparams: Dict = {}

    # ------------------------------------------------------------------

    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        spec: DatasetSpec,
        config: TrainConfig,
    ) -> Dict:
        device = get_device()

        # ---- preprocess ----
        self.preprocessor = TorchTabularPreprocessor()
        self.preprocessor.fit(
            train_df,
            categorical_cols=spec.categorical_cols,
            numerical_cols=spec.numerical_cols,
        )

        train_cat, train_num = self.preprocessor.transform(train_df)
        val_cat, val_num = self.preprocessor.transform(val_df)
        y_train = train_df[spec.label_col].values.astype(np.float32)
        y_val = val_df[spec.label_col].values.astype(np.float32)

        # Convert to tensors
        train_cat_t = torch.from_numpy(train_cat).long()
        train_num_t = torch.from_numpy(train_num).float()
        y_train_t = torch.from_numpy(y_train)
        val_cat_t = torch.from_numpy(val_cat).long()
        val_num_t = torch.from_numpy(val_num).float()
        y_val_t = torch.from_numpy(y_val)

        train_ds = torch.utils.data.TensorDataset(train_cat_t, train_num_t, y_train_t)
        val_ds = torch.utils.data.TensorDataset(val_cat_t, val_num_t, y_val_t)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size * 2, shuffle=False
        )

        # ---- build model ----
        vocab_sizes = self.preprocessor.vocab_sizes()
        n_num = self.preprocessor.n_numerical()
        self._hparams = dict(
            vocab_sizes=vocab_sizes,
            n_numerical=n_num,
            d_model=64,
            n_heads=4,
            n_layers=3,
            d_ff=128,
            dropout=0.1,
        )
        self.net = _FTTransformerNet(**self._hparams).to(device)
        logger.info(
            "FT-Transformer: %d params",
            sum(p.numel() for p in self.net.parameters()),
        )

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=config.lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        # Mixed precision
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_auroc = -1.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, config.max_epochs + 1):
            # --- train ---
            self.net.train()
            total_loss = 0.0
            n_batches = 0
            for cat_b, num_b, y_b in train_loader:
                cat_b, num_b, y_b = (
                    cat_b.to(device),
                    num_b.to(device),
                    y_b.to(device),
                )
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = self.net(cat_b, num_b)
                    loss = criterion(logits, y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # --- val ---
            val_acc, val_auroc = self._evaluate(val_loader, device, use_amp)

            logger.info(
                "Epoch %d/%d  loss=%.4f  val_acc=%.4f  val_auroc=%.4f",
                epoch,
                config.max_epochs,
                avg_loss,
                val_acc,
                val_auroc,
            )

            # Early stopping on val AUROC
            if not math.isnan(val_auroc) and val_auroc > best_auroc:
                best_auroc = val_auroc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.to(device)

        # Final eval
        val_acc, val_auroc = self._evaluate(val_loader, device, use_amp)
        self._metrics = {"accuracy": float(val_acc), "auroc": float(val_auroc)}
        logger.info("FT-Transformer → accuracy=%.4f, AUROC=%.4f", val_acc, val_auroc)
        return self._metrics

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, loader, device, use_amp: bool):
        self.net.eval()
        all_probs, all_labels = [], []
        for cat_b, num_b, y_b in loader:
            cat_b, num_b = cat_b.to(device), num_b.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = self.net(cat_b, num_b)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_b.numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        preds = (all_probs >= 0.5).astype(int)
        acc = (preds == all_labels.astype(int)).mean()
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(all_labels.astype(int), all_probs)
        except ValueError:
            auroc = float("nan")
            warnings.warn("AUROC undefined (single class in val set)")
        return float(acc), float(auroc)

    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        out = ensure_dir(Path(output_dir))
        # State dict
        torch.save(self.net.state_dict(), out / "model.pt")
        # Hparams
        save_json(self._hparams, out / "hparams.json")
        # Preprocessor
        joblib.dump(self.preprocessor, out / "preprocessor.joblib")
        # Metrics
        save_json(self._metrics, out / "metrics.json")
        logger.info("Saved FT-Transformer artefacts to %s", out)
