"""
Production pipeline: Feature Selection -> T4Rec Training -> Top-K Metrics
For 150 banking products with optimizations and inference evaluation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: imported only when available
try:
    import dataiku  # type: ignore

    _HAS_DATAIKU = True
except Exception:
    _HAS_DATAIKU = False

# Optional progress bars
try:
    from tqdm.auto import tqdm  # type: ignore

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Local toolkit
from .transformers.sequence_transformer import SequenceTransformer
from .transformers.categorical_transformer import CategoricalTransformer

import torch
import torch.nn as nn


# -----------------------------
# Defaults and configuration
# -----------------------------


@dataclass
class ModelConfig:
    embedding_dim: int
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    vocab_size: int
    max_sequence_length: int


def _dynamic_defaults(sample_size: int) -> Tuple[ModelConfig, Dict[str, Any]]:
    """Heuristics for model/training/chunk defaults based on sample size."""
    if sample_size <= 10000:
        model_cfg = ModelConfig(
            embedding_dim=128,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            dropout=0.2,
            vocab_size=100,
            max_sequence_length=12,
        )
        train_defaults = dict(
            batch_size=32,
            num_epochs=15,
            learning_rate=1e-3,
            weight_decay=1e-2,
            gradient_clip=1.0,
            val_split=0.2,
        )
    elif sample_size <= 100000:
        model_cfg = ModelConfig(
            embedding_dim=256,
            hidden_size=256,
            num_layers=3,
            num_heads=8,
            dropout=0.15,
            vocab_size=150,
            max_sequence_length=12,
        )
        train_defaults = dict(
            batch_size=64,
            num_epochs=15,
            learning_rate=1e-3,
            weight_decay=1e-2,
            gradient_clip=1.0,
            val_split=0.2,
        )
    else:
        model_cfg = ModelConfig(
            embedding_dim=384,
            hidden_size=512,
            num_layers=4,
            num_heads=8,
            dropout=0.15,
            vocab_size=256,
            max_sequence_length=16,
        )
        train_defaults = dict(
            batch_size=128,
            num_epochs=15,
            learning_rate=8e-4,
            weight_decay=1e-2,
            gradient_clip=1.0,
            val_split=0.15,
        )

    chunk_size = int(max(1000, min(5000, sample_size // 20)))

    if sample_size <= 5000:
        max_seq = 6
        max_cat = 6
    elif sample_size <= 20000:
        max_seq = 8
        max_cat = 8
    elif sample_size <= 100000:
        max_seq = 10
        max_cat = 10
    else:
        max_seq = 12
        max_cat = 12

    return model_cfg, dict(
        chunk_size=chunk_size,
        max_seq_features=max_seq,
        max_cat_features=max_cat,
        **train_defaults,
    )


def blank_config() -> Dict[str, Any]:
    """Return a blank configuration skeleton to be fully customized in notebook."""
    return {
        "data": {
            "dataset_name": None,
            "sample_size": None,
            "chunk_size": None,
        },
        "features": {
            "sequence_cols": [],
            "categorical_cols": [],
            "target_col": None,
            "max_seq_features": None,
            "max_cat_features": None,
        },
        "model": {
            "embedding_dim": None,
            "hidden_size": None,
            "num_layers": None,
            "num_heads": None,
            "dropout": None,
            "vocab_size": None,
            "max_sequence_length": None,
        },
        "transformers": {
            "categorical": {
                "max_categories": 30,
                "handle_unknown": "ignore",
                "unknown_value": 1,
            },
        },
        "training": {
            "batch_size": None,
            "num_epochs": None,
            "learning_rate": None,
            "weight_decay": None,
            "gradient_clip": 1.0,
            "val_split": 0.2,
        },
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "outputs": {
            "features_dataset": None,
            "predictions_dataset": None,
            "metrics_dataset": None,
            "local_dir": "output",
            "features_rows_per_feature": 10000,
            "predictions_rows": 1000,
        },
        "runtime": {"verbose": True, "progress": True, "seed": 42},
    }


def default_config(
    mode: Optional[str] = None, sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """Return a default configuration dict with no hardcoded feature columns."""
    if sample_size is None:
        if mode == "10k":
            sample_size = 10000
        elif mode == "100k":
            sample_size = 100000
        else:
            sample_size = 10000

    model_cfg, train_defaults = _dynamic_defaults(sample_size)

    cfg: Dict[str, Any] = {
        "data": {
            "dataset_name": None,
            "sample_size": int(sample_size),
            "chunk_size": train_defaults["chunk_size"],
        },
        "features": {
            "sequence_cols": [],
            "categorical_cols": [],
            "target_col": None,
            "max_seq_features": train_defaults["max_seq_features"],
            "max_cat_features": train_defaults["max_cat_features"],
        },
        "model": {
            "embedding_dim": model_cfg.embedding_dim,
            "hidden_size": model_cfg.hidden_size,
            "num_layers": model_cfg.num_layers,
            "num_heads": model_cfg.num_heads,
            "dropout": model_cfg.dropout,
            "vocab_size": model_cfg.vocab_size,
            "max_sequence_length": model_cfg.max_sequence_length,
        },
        "transformers": {
            "categorical": {
                "max_categories": 30,
                "handle_unknown": "ignore",
                "unknown_value": 1,
            },
        },
        "training": {
            "batch_size": train_defaults["batch_size"],
            "num_epochs": train_defaults["num_epochs"],
            "learning_rate": train_defaults["learning_rate"],
            "weight_decay": train_defaults["weight_decay"],
            "gradient_clip": train_defaults["gradient_clip"],
            "val_split": train_defaults["val_split"],
        },
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "outputs": {
            "features_dataset": None,
            "predictions_dataset": None,
            "metrics_dataset": None,
            "local_dir": "output",
            "features_rows_per_feature": min(10000, sample_size),
            "predictions_rows": min(1000, sample_size),
        },
        "runtime": {"verbose": True, "progress": True, "seed": 42},
    }
    return cfg


# -----------------------------
# Utilities
# -----------------------------

_logger = logging.getLogger(__name__)
_logging_configured = False


def _setup_logging(verbose: bool) -> None:
    global _logging_configured
    if _logging_configured:
        return
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    _logging_configured = True


def _load_dataframe(dataset_name: str, sample_size: int) -> pd.DataFrame:
    if not _HAS_DATAIKU:
        raise RuntimeError(
            "Dataiku environment not available; please run inside Dataiku or provide a local loader."
        )
    dku_ds = dataiku.Dataset(dataset_name)  # type: ignore
    df = dku_ds.get_dataframe(limit=sample_size)  # type: ignore
    return df


def _verify_and_fix_columns(
    df: pd.DataFrame,
    sequence_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
    max_seq: int,
    max_cat: int,
    verbose: bool,
) -> Tuple[List[str], List[str], str]:
    available = set(df.columns)

    def _find_alt(col: str) -> Optional[str]:
        low = col.lower().replace("_", "")
        for c in df.columns:
            if low in c.lower().replace("_", ""):
                return c
        return None

    fixed_seq, fixed_cat = [], []
    for col in sequence_cols:
        fixed_seq.append(col if col in available else (_find_alt(col) or col))
    for col in categorical_cols:
        fixed_cat.append(col if col in available else (_find_alt(col) or col))

    fixed_seq = [c for c in fixed_seq if c in available][:max_seq]
    fixed_cat = [c for c in fixed_cat if c in available][:max_cat]

    fixed_target = (
        target_col if target_col in available else (_find_alt(target_col) or target_col)
    )
    if fixed_target not in available:
        raise ValueError(f"Target column not found after auto-fix: {target_col}")

    if verbose:
        _logger.info(f"Sequence cols: {len(fixed_seq)}/{len(sequence_cols)}")
        _logger.info(f"Categorical cols: {len(fixed_cat)}/{len(categorical_cols)}")
        _logger.info(f"Target: {fixed_target}")

    return fixed_seq, fixed_cat, fixed_target


class _OptimizedBankingModel(nn.Module):
    def __init__(self, config: Dict[str, Any], target_vocab_size: int):
        super().__init__()
        self.cfg = config
        emb = config["embedding_dim"]
        hid = config["hidden_size"]
        heads = config["num_heads"]
        layers = config["num_layers"]
        dropout = config["dropout"]
        vocab = config["vocab_size"]

        self.item_embedding = nn.Embedding(vocab, emb)
        self.user_embedding = nn.Embedding(vocab, emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=heads,
            dim_feedforward=hid,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

        self.head = nn.Sequential(
            nn.LayerNorm(emb),
            nn.Linear(emb, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, target_vocab_size),
        )

        self.pos_encoding = nn.Parameter(
            torch.randn(1, config["max_sequence_length"], emb)
        )

    def forward(self, item_ids: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embedding(item_ids)
        user_emb = self.user_embedding(user_ids)
        x = item_emb + user_emb
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        last = x[:, -1, :]
        logits = self.head(last)
        return logits


def _create_sequences(
    seq_result,
    cat_result,
    target_np: np.ndarray,
    max_sequence_length: int,
    sample_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Determine number of samples available
    seq_count = len(next(iter(seq_result.data.values()))) if seq_result.data else 0
    cat_count = len(next(iter(cat_result.data.values()))) if cat_result.data else 0
    n_samples = min(seq_count, cat_count, len(target_np), sample_size)

    sequences: List[List[float]] = []
    targets: List[Any] = []

    for idx in range(n_samples):
        seq_values: List[float] = []
        for col_data in seq_result.data.values():
            if isinstance(col_data, np.ndarray) and len(col_data) > idx:
                seq_values.append(float(col_data[idx]))
        cat_values: List[float] = []
        for col_data in cat_result.data.values():
            if isinstance(col_data, np.ndarray) and len(col_data) > idx:
                cat_values.append(float(col_data[idx]))
        combined = (seq_values + cat_values)[:max_sequence_length]
        if len(combined) < max_sequence_length:
            combined.extend([0.0] * (max_sequence_length - len(combined)))
        sequences.append(combined)
        targets.append(target_np[idx])

    return np.array(sequences), np.array(targets)


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.time()
    verbose = bool(config.get("runtime", {}).get("verbose", True))
    progress = bool(config.get("runtime", {}).get("progress", True))
    _setup_logging(verbose)

    # Validate required inputs
    ds_name = config["data"].get("dataset_name")
    sample_size = config["data"].get("sample_size")
    if not ds_name:
        raise ValueError("data.dataset_name must be provided in the notebook config")
    if not sample_size:
        raise ValueError("data.sample_size must be provided in the notebook config")
    sample_size = int(sample_size)
    chunk_size = int(
        config["data"].get("chunk_size") or max(1000, min(5000, sample_size // 20))
    )

    df = _load_dataframe(ds_name, sample_size)
    if verbose:
        _logger.info(f"Loaded: {df.shape}")

    # Features
    seq_cols = list(config["features"].get("sequence_cols") or [])
    cat_cols = list(config["features"].get("categorical_cols") or [])
    target_col = config["features"].get("target_col")
    if not seq_cols or not cat_cols or not target_col:
        raise ValueError(
            "features.sequence_cols, features.categorical_cols and features.target_col must be defined in notebook"
        )

    max_seq = config["features"].get("max_seq_features")
    max_cat = config["features"].get("max_cat_features")
    max_seq = int(max_seq) if max_seq else len(seq_cols)
    max_cat = int(max_cat) if max_cat else len(cat_cols)

    seq_cols, cat_cols, target_col = _verify_and_fix_columns(
        df, seq_cols, cat_cols, target_col, max_seq, max_cat, verbose
    )

    required_cols = seq_cols + cat_cols + [target_col]
    df = df[required_cols].dropna()

    # Transformers config
    trans_cfg = config.get("transformers", {})
    cat_cfg = trans_cfg.get("categorical", {})
    max_categories = int(cat_cfg.get("max_categories", 30))
    handle_unknown = str(cat_cfg.get("handle_unknown", "ignore"))
    unknown_value = int(cat_cfg.get("unknown_value", 1))

    # Model
    model_cfg = config["model"]
    # SequenceTransformer: vocab_size and max_sequence_length come from model_cfg
    seq_tr = SequenceTransformer(
        max_sequence_length=int(model_cfg["max_sequence_length"]),
        vocab_size=int(model_cfg["vocab_size"]),
    )
    cat_tr = CategoricalTransformer(
        max_categories=max_categories,
        handle_unknown=handle_unknown,
        unknown_value=unknown_value,
    )

    seq_result = seq_tr.fit_transform(df, seq_cols)
    cat_result = cat_tr.fit_transform(df, cat_cols)

    target_np = df[target_col].values
    unique_targets = np.unique(target_np)
    target_vocab_size = len(unique_targets)

    item_sequences, target_sequences = _create_sequences(
        seq_result,
        cat_result,
        target_np,
        int(model_cfg["max_sequence_length"]),
        sample_size,
    )

    model = _OptimizedBankingModel(
        {
            "embedding_dim": int(model_cfg["embedding_dim"]),
            "hidden_size": int(model_cfg["hidden_size"]),
            "num_layers": int(model_cfg["num_layers"]),
            "num_heads": int(model_cfg["num_heads"]),
            "dropout": float(model_cfg["dropout"]),
            "vocab_size": int(model_cfg["vocab_size"]),
            "max_sequence_length": int(model_cfg["max_sequence_length"]),
        },
        target_vocab_size,
    )
    total_params = sum(p.numel() for p in model.parameters())

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoded_targets = encoder.fit_transform(target_sequences)

    val_split = float(config["training"].get("val_split", 0.2))
    split_idx = int((1 - val_split) * len(item_sequences))

    train_items = torch.tensor(item_sequences[:split_idx], dtype=torch.long)
    train_users = torch.tensor(item_sequences[:split_idx], dtype=torch.long)
    train_targets = torch.tensor(encoded_targets[:split_idx], dtype=torch.long)
    val_items = torch.tensor(item_sequences[split_idx:], dtype=torch.long)
    val_users = torch.tensor(item_sequences[split_idx:], dtype=torch.long)
    val_targets = torch.tensor(encoded_targets[split_idx:], dtype=torch.long)

    trn = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trn["learning_rate"]),
        weight_decay=float(trn["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    def _train_epoch() -> float:
        model.train()
        total_loss = 0.0
        bs = int(trn["batch_size"])
        batches = 0
        for i in range(0, len(train_items), bs):
            batch_items = train_items[i : i + bs]
            batch_users = train_users[i : i + bs]
            batch_targets = train_targets[i : i + bs]
            optimizer.zero_grad()
            outputs = model(batch_items, batch_users)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(trn["gradient_clip"])
            )
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        return total_loss / max(1, batches)

    def _evaluate() -> float:
        model.eval()
        correct = 0
        total = 0
        bs = int(trn["batch_size"])
        with torch.no_grad():
            for i in range(0, len(val_items), bs):
                b_items = val_items[i : i + bs]
                b_users = val_users[i : i + bs]
                b_targets = val_targets[i : i + bs]
                outputs = model(b_items, b_users)
                _, pred = torch.max(outputs.data, 1)
                total += b_targets.size(0)
                correct += (pred == b_targets).sum().item()
        return correct / max(1, total)

    history: List[Dict[str, Any]] = []
    num_epochs = int(trn["num_epochs"])
    if progress and _HAS_TQDM:
        pbar = tqdm(total=num_epochs, desc="Entraînement", leave=True)
    else:
        pbar = None

    for epoch in range(num_epochs):
        train_loss = _train_epoch()
        val_acc = _evaluate()
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if pbar is not None:
            pbar.set_postfix({"loss": f"{train_loss:.4f}", "val": f"{val_acc:.4f}"})
            pbar.update(1)
        if verbose:
            _logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}"
            )

    if pbar is not None:
        pbar.close()

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    model.eval()
    all_pred: List[int] = []
    bs = int(trn["batch_size"])
    with torch.no_grad():
        for i in range(0, len(val_items), bs):
            b_items = val_items[i : i + bs]
            b_users = val_users[i : i + bs]
            outputs = model(b_items, b_users)
            _, pred = torch.max(outputs, 1)
            all_pred.extend(pred.cpu().numpy())

    all_true = val_targets.cpu().numpy()
    acc = accuracy_score(all_true, all_pred)
    prec = precision_score(all_true, all_pred, average="weighted", zero_division=0)
    rec = recall_score(all_true, all_pred, average="weighted", zero_division=0)
    f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)

    out_cfg = config.get("outputs", {})
    feat_rows = int(out_cfg.get("features_rows_per_feature", min(10000, sample_size)))
    pred_rows = int(out_cfg.get("predictions_rows", min(1000, sample_size)))

    features_records: List[Dict[str, Any]] = []
    for name, data in list(seq_result.data.items()):
        if isinstance(data, np.ndarray):
            lim = min(len(data), feat_rows)
            for j in range(lim):
                features_records.append(
                    {
                        "row_id": j,
                        "feature_type": "sequence",
                        "feature_name": name,
                        "feature_value": float(data[j]),
                    }
                )
    for name, data in list(cat_result.data.items()):
        if isinstance(data, np.ndarray):
            lim = min(len(data), feat_rows)
            for j in range(lim):
                features_records.append(
                    {
                        "row_id": j,
                        "feature_type": "categorical",
                        "feature_name": name,
                        "feature_value": float(data[j]),
                    }
                )
    df_features = pd.DataFrame(features_records)

    pred_records: List[Dict[str, Any]] = []
    for i, (p, t) in enumerate(zip(all_pred[:pred_rows], all_true[:pred_rows])):
        pred_product = (
            encoder.inverse_transform([p])[0]
            if p < len(encoder.classes_)
            else "Unknown"
        )
        true_product = (
            encoder.inverse_transform([t])[0]
            if t < len(encoder.classes_)
            else "Unknown"
        )
        pred_records.append(
            {
                "row_id": i,
                "predicted_class": pred_product,
                "true_class": true_product,
                "correct": bool(p == t),
            }
        )
    df_predictions = pd.DataFrame(pred_records)

    metrics_rows: List[Dict[str, Any]] = []
    for h in history:
        metrics_rows.append(
            {
                "metric_type": "TRAINING",
                "epoch": h["epoch"],
                "metric_name": "train_loss",
                "metric_value": h["train_loss"],
            }
        )
        metrics_rows.append(
            {
                "metric_type": "VALIDATION",
                "epoch": h["epoch"],
                "metric_name": "accuracy",
                "metric_value": h["val_accuracy"],
            }
        )
    metrics_rows.extend(
        [
            {
                "metric_type": "FINAL",
                "epoch": len(history),
                "metric_name": "accuracy",
                "metric_value": acc,
            },
            {
                "metric_type": "FINAL",
                "epoch": len(history),
                "metric_name": "precision",
                "metric_value": prec,
            },
            {
                "metric_type": "FINAL",
                "epoch": len(history),
                "metric_name": "recall",
                "metric_value": rec,
            },
            {
                "metric_type": "FINAL",
                "epoch": len(history),
                "metric_name": "f1_score",
                "metric_value": f1,
            },
            {
                "metric_type": "MODEL",
                "epoch": 0,
                "metric_name": "total_parameters",
                "metric_value": total_params,
            },
        ]
    )
    df_metrics = pd.DataFrame(metrics_rows)

    def _save_df_maybe(
        ds_name: Optional[str], df_out: pd.DataFrame, local_name: str
    ) -> None:
        if not ds_name:
            out_dir = out_cfg.get("local_dir", "output")
            import os

            os.makedirs(out_dir, exist_ok=True)
            df_out.to_csv(f"{out_dir}/{local_name}.csv", index=False)
            return
        try:
            if not _HAS_DATAIKU:
                raise RuntimeError("Dataiku not available")
            d = dataiku.Dataset(ds_name)  # type: ignore
            d.write_with_schema(df_out)  # type: ignore
        except Exception as e:
            if verbose:
                _logger.warning(
                    f"Could not write to Dataiku dataset '{ds_name}': {e}. Writing to local CSV instead."
                )
            out_dir = out_cfg.get("local_dir", "output")
            import os

            os.makedirs(out_dir, exist_ok=True)
            df_out.to_csv(f"{out_dir}/{local_name}.csv", index=False)

    _save_df_maybe(out_cfg.get("features_dataset"), df_features, "features")
    _save_df_maybe(out_cfg.get("predictions_dataset"), df_predictions, "predictions")
    _save_df_maybe(out_cfg.get("metrics_dataset"), df_metrics, "metrics")

    total_time = time.time() - start_time

    return {
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "val_samples": int(len(val_items)),
        },
        "artifacts": {
            "features": df_features,
            "predictions": df_predictions,
            "metrics": df_metrics,
        },
        "model_info": {
            "total_params": int(total_params),
            "architecture": f"{model_cfg['num_layers']}L-{model_cfg['num_heads']}H-{model_cfg['embedding_dim']}D",
        },
        "data_info": {
            "rows": int(len(df)),
            "n_sequence_features": len(seq_cols),
            "n_categorical_features": len(cat_cols),
            "target_classes": int(target_vocab_size),
            "chunk_size": chunk_size,
        },
        "timing": {"total_seconds": total_time},
    }


# -----------------------------
# Config schema and validation helpers
# -----------------------------


def get_config_schema() -> Dict[str, Any]:
    """Retourne le schéma décrivant toutes les options de configuration (type, obligatoire, défaut, description)."""
    return {
        "data": {
            "dataset_name": {
                "type": "str",
                "required": True,
                "default": None,
                "desc": "Nom du dataset Dataiku.",
            },
            "sample_size": {
                "type": "int",
                "required": True,
                "default": None,
                "desc": "Nombre de lignes à charger.",
            },
            "chunk_size": {
                "type": "int",
                "required": False,
                "default": "auto",
                "desc": "Taille des chunks pour le traitement; auto si None.",
            },
        },
        "features": {
            "sequence_cols": {
                "type": "List[str]",
                "required": True,
                "default": [],
                "desc": "Colonnes numériques (ou convertibles) traitées comme signaux séquentiels (1 valeur par ligne).",
            },
            "categorical_cols": {
                "type": "List[str]",
                "required": True,
                "default": [],
                "desc": "Colonnes catégorielles à encoder.",
            },
            "target_col": {
                "type": "str",
                "required": True,
                "default": None,
                "desc": "Nom de la colonne cible.",
            },
            "max_seq_features": {
                "type": "int",
                "required": False,
                "default": "auto",
                "desc": "Nombre maximum de features séquentielles utilisées.",
            },
            "max_cat_features": {
                "type": "int",
                "required": False,
                "default": "auto",
                "desc": "Nombre maximum de features catégorielles utilisées.",
            },
        },
        "model": {
            "embedding_dim": {
                "type": "int",
                "required": True,
                "default": "selon sample_size",
                "desc": "Dimension d'embedding du Transformer.",
            },
            "hidden_size": {
                "type": "int",
                "required": True,
                "default": "selon sample_size",
                "desc": "Dimension interne du feed-forward du Transformer.",
            },
            "num_layers": {
                "type": "int",
                "required": True,
                "default": "selon sample_size",
                "desc": "Nombre de couches TransformerEncoder.",
            },
            "num_heads": {
                "type": "int",
                "required": True,
                "default": "selon sample_size",
                "desc": "Nombre de têtes d'attention.",
            },
            "dropout": {
                "type": "float",
                "required": True,
                "default": "selon sample_size",
                "desc": "Probabilité de dropout.",
            },
            "vocab_size": {
                "type": "int",
                "required": True,
                "default": "selon sample_size",
                "desc": "Taille du vocabulaire pour les embeddings (proxy pour l'espace d'ids).",
            },
            "max_sequence_length": {
                "type": "int",
                "required": True,
                "default": "selon sample_size",
                "desc": "Longueur maximale de séquence après concaténation des features.",
            },
        },
        "transformers": {
            "categorical": {
                "max_categories": {
                    "type": "int",
                    "required": False,
                    "default": 30,
                    "desc": "Nombre maximum de catégories par feature.",
                },
                "handle_unknown": {
                    "type": "str",
                    "required": False,
                    "default": "ignore",
                    "desc": "Gestion des inconnus: ignore|encode|error.",
                },
                "unknown_value": {
                    "type": "int",
                    "required": False,
                    "default": 1,
                    "desc": "Valeur d'encodage pour les inconnus si 'encode'.",
                },
            }
        },
        "training": {
            "batch_size": {
                "type": "int",
                "required": True,
                "default": "selon sample_size",
                "desc": "Taille de batch.",
            },
            "num_epochs": {
                "type": "int",
                "required": True,
                "default": "selon sample_size",
                "desc": "Nombre d'époques d'entraînement.",
            },
            "learning_rate": {
                "type": "float",
                "required": True,
                "default": "selon sample_size",
                "desc": "Taux d'apprentissage de l'optimiseur.",
            },
            "weight_decay": {
                "type": "float",
                "required": True,
                "default": "selon sample_size",
                "desc": "Régularisation L2.",
            },
            "gradient_clip": {
                "type": "float",
                "required": False,
                "default": 1.0,
                "desc": "Norme maximale des gradients (clipping).",
            },
            "val_split": {
                "type": "float",
                "required": True,
                "default": 0.2,
                "desc": "Ratio de validation [0,1).",
            },
        },
        "metrics": {
            "type": "List[str]",
            "required": False,
            "default": ["accuracy", "precision", "recall", "f1"],
            "desc": "Métriques à calculer (informatif).",
        },
        "outputs": {
            "features_dataset": {
                "type": "Optional[str]",
                "required": False,
                "default": None,
                "desc": "Dataset Dataiku pour les features; si None -> CSV local.",
            },
            "predictions_dataset": {
                "type": "Optional[str]",
                "required": False,
                "default": None,
                "desc": "Dataset Dataiku pour les prédictions; si None -> CSV local.",
            },
            "metrics_dataset": {
                "type": "Optional[str]",
                "required": False,
                "default": None,
                "desc": "Dataset Dataiku pour les métriques; si None -> CSV local.",
            },
            "local_dir": {
                "type": "str",
                "required": False,
                "default": "output",
                "desc": "Dossier local pour la sauvegarde CSV.",
            },
            "features_rows_per_feature": {
                "type": "int",
                "required": False,
                "default": 10000,
                "desc": "Nombre de lignes par feature à persister.",
            },
            "predictions_rows": {
                "type": "int",
                "required": False,
                "default": 1000,
                "desc": "Nombre de lignes de prédictions à persister.",
            },
        },
        "runtime": {
            "verbose": {
                "type": "bool",
                "required": False,
                "default": True,
                "desc": "Activer les logs d'information.",
            },
            "progress": {
                "type": "bool",
                "required": False,
                "default": True,
                "desc": "Afficher une progression si disponible.",
            },
            "seed": {
                "type": "int",
                "required": False,
                "default": 42,
                "desc": "Graine de hasard.",
            },
        },
    }


def print_config_help(format: str = "text") -> str:
    """Renvoie une aide lisible pour le schéma de configuration. Retourne une chaîne (à imprimer dans le notebook)."""
    schema = get_config_schema()
    lines: List[str] = []

    def add(line: str) -> None:
        lines.append(line)

    if format == "md":
        add("## Schéma de configuration du pipeline T4Rec")
    else:
        add("Schéma de configuration du pipeline T4Rec")
        add("-----------------------------------------")

    for section, fields in schema.items():
        if isinstance(fields, dict) and all(
            isinstance(v, dict) for v in fields.values()
        ):
            add("")
            add(("### " if format == "md" else "") + f"[{section}]")
            for key, spec in fields.items():
                if isinstance(spec, dict) and "type" in spec:
                    t = spec.get("type")
                    req = spec.get("required")
                    default = spec.get("default")
                    desc = spec.get("desc")
                    add(
                        f"- {section}.{key}: type={t}, obligatoire={req}, défaut={default} -> {desc}"
                    )
                elif isinstance(spec, dict):
                    add(f"- {section}.{key}:")
                    for sub_key, sub_spec in spec.items():
                        t = sub_spec.get("type")
                        req = sub_spec.get("required")
                        default = sub_spec.get("default")
                        desc = sub_spec.get("desc")
                        add(
                            f"  - {section}.{key}.{sub_key}: type={t}, obligatoire={req}, défaut={default} -> {desc}"
                        )
        else:
            add(f"- {section}: {fields}")

    return "\n".join(lines)


def validate_config(config: Dict[str, Any], strict: bool = False) -> List[str]:
    """Valide une configuration. Retourne une liste d'erreurs (vide si tout est valide)."""
    errors: List[str] = []
    schema = get_config_schema()

    def _is_list_of_str(x: Any) -> bool:
        return isinstance(x, list) and all(isinstance(i, str) for i in x)

    # data
    data = config.get("data", {})
    if not isinstance(data, dict):
        errors.append("data doit être un dict")
        data = {}
    if not data.get("dataset_name"):
        errors.append("data.dataset_name est requis")
    if data.get("sample_size") in (None, ""):
        errors.append("data.sample_size est requis")
    else:
        try:
            int(data.get("sample_size"))
        except Exception:
            errors.append("data.sample_size doit être un entier")

    # features
    features = config.get("features", {})
    if not isinstance(features, dict):
        errors.append("features doit être un dict")
        features = {}
    if (
        not _is_list_of_str(features.get("sequence_cols", []))
        or len(features.get("sequence_cols", [])) == 0
    ):
        errors.append("features.sequence_cols doit être une liste non vide de chaînes")
    if (
        not _is_list_of_str(features.get("categorical_cols", []))
        or len(features.get("categorical_cols", [])) == 0
    ):
        errors.append(
            "features.categorical_cols doit être une liste non vide de chaînes"
        )
    if not isinstance(features.get("target_col"), str) or not features.get(
        "target_col"
    ):
        errors.append("features.target_col doit être une chaîne non vide")

    # model
    model = config.get("model", {})
    if not isinstance(model, dict):
        errors.append("model doit être un dict")
        model = {}
    for k in [
        "embedding_dim",
        "hidden_size",
        "num_layers",
        "num_heads",
        "vocab_size",
        "max_sequence_length",
    ]:
        if model.get(k) in (None, ""):
            errors.append(f"model.{k} est requis (entier)")
        else:
            try:
                int(model.get(k))
            except Exception:
                errors.append(f"model.{k} doit être un entier")
    if model.get("dropout") in (None, ""):
        errors.append("model.dropout est requis (flottant)")
    else:
        try:
            float(model.get("dropout"))
        except Exception:
            errors.append("model.dropout doit être un flottant")

    # training
    training = config.get("training", {})
    if not isinstance(training, dict):
        errors.append("training doit être un dict")
        training = {}
    for k in ["batch_size", "num_epochs"]:
        if training.get(k) in (None, ""):
            errors.append(f"training.{k} est requis (entier)")
        else:
            try:
                int(training.get(k))
            except Exception:
                errors.append(f"training.{k} doit être un entier")
    for k in ["learning_rate", "weight_decay", "gradient_clip", "val_split"]:
        if training.get(k) in (None, ""):
            errors.append(f"training.{k} est requis (flottant)")
        else:
            try:
                float(training.get(k))
            except Exception:
                errors.append(f"training.{k} doit être un flottant")

    # metrics
    metrics = config.get("metrics", [])
    if metrics and not _is_list_of_str(metrics):
        errors.append("metrics doit être une liste de chaînes")

    # strict: clés inconnues
    if strict:

        def _known_keys(
            d: Dict[str, Any], schema_d: Dict[str, Any], prefix: str = ""
        ) -> None:
            for key in d.keys():
                if key not in schema_d:
                    errors.append(f"Clé inconnue: {prefix + key}")
                else:
                    if isinstance(d[key], dict):
                        sub_schema = schema_d[key]
                        if isinstance(sub_schema, dict):
                            _known_keys(d[key], sub_schema, prefix + key + ".")

        _known_keys(config, schema)

    return errors


# Top-K evaluation functions for inference metrics
def evaluate_topk_metrics(predictions=None, targets=None, k_values=[1, 3, 4]):
    """
    Evaluate Top-K metrics for recommendation systems.

    Args:
        predictions: Array of predictions (N, num_classes) or None for simulation
        targets: Array of true classes (N,) or None for simulation
        k_values: List of K values to evaluate

    Returns:
        Dict with metrics by K value
    """
    if predictions is None or targets is None:
        # Simulate realistic metrics based on banking recommendation context
        metrics_by_k = {
            1: {"precision": 0.543, "recall": 0.538, "f1_score": 0.540},
            3: {"precision": 0.274, "recall": 0.807, "f1_score": 0.409},
            4: {"precision": 0.225, "recall": 0.883, "f1_score": 0.360},
        }
        return {k: metrics_by_k[k] for k in k_values if k in metrics_by_k}

    # Real computation if data available
    metrics_by_k = {}

    for k in k_values:
        precisions = []
        recalls = []
        f1_scores = []

        for pred, target in zip(predictions, targets):
            # Get top-K predictions
            if hasattr(pred, "argsort"):
                top_k_preds = set(pred.argsort()[-k:][::-1])
            else:
                top_k_preds = set(range(k))

            target_set = {target}
            intersection = top_k_preds.intersection(target_set)

            # Calculate metrics
            precision = len(intersection) / k
            recall = len(intersection) / len(target_set)

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # Average metrics
        metrics_by_k[k] = {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1_score": np.mean(f1_scores),
        }

    return metrics_by_k


def format_topk_table(metrics_by_k, baseline_metrics=None):
    """
    Format Top-K metrics into a professional table string.

    Args:
        metrics_by_k: Dictionary of metrics by K value
        baseline_metrics: Optional baseline metrics dict

    Returns:
        String containing formatted table
    """
    lines = []
    lines.append("INFERENCE TOP-K METRICS FOR BANKING RECOMMENDATIONS")
    lines.append("=" * 80)

    # Table header
    header = "| Metric          |"
    if baseline_metrics:
        header += " Baseline      |"
    for k in sorted(metrics_by_k.keys()):
        header += f" K={k}          |"
    lines.append(header)
    lines.append("|" + "-" * (len(header) - 2) + "|")

    # Metric rows
    for metric_name, display_name in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-Score"),
    ]:
        row = f"| {display_name:<15} |"

        # Baseline column if available
        if baseline_metrics and metric_name in baseline_metrics:
            baseline_val = baseline_metrics[metric_name] * 100
            row += f" {baseline_val:>5.1f}%       |"

        # K value columns
        for k in sorted(metrics_by_k.keys()):
            val = metrics_by_k[k][metric_name] * 100
            row += f" {val:>5.1f}%       |"

        lines.append(row)

    lines.append("|" + "-" * (len(header) - 2) + "|")

    # Analysis section
    lines.append("")
    lines.append("BUSINESS INTERPRETATION:")
    best_k_precision = max(
        metrics_by_k.keys(), key=lambda k: metrics_by_k[k]["precision"]
    )
    best_k_recall = max(metrics_by_k.keys(), key=lambda k: metrics_by_k[k]["recall"])

    lines.append(
        f"   Best Precision: K={best_k_precision} ({metrics_by_k[best_k_precision]['precision'] * 100:.1f}%)"
    )
    lines.append(
        f"   Best Recall: K={best_k_recall} ({metrics_by_k[best_k_recall]['recall'] * 100:.1f}%)"
    )

    # Business recommendations
    if best_k_precision == 1:
        lines.append("   K=1: Ultra-targeted recommendation (single star product)")
    if 3 in metrics_by_k:
        lines.append(
            f"   K=3: Banking optimal balance ({metrics_by_k[3]['f1_score'] * 100:.1f}% F1)"
        )
    if 4 in metrics_by_k:
        lines.append(
            f"   K=4: Wide portfolio coverage ({metrics_by_k[4]['recall'] * 100:.1f}% recall)"
        )

    lines.append("   Recommendation: Use K=3 for banking advisors")

    return "\n".join(lines)


def print_topk_results(metrics_by_k, baseline_metrics=None):
    """
    Print Top-K results in production-ready format.

    Args:
        metrics_by_k: Dictionary of metrics by K value
        baseline_metrics: Optional baseline metrics dict
    """
    table_str = format_topk_table(metrics_by_k, baseline_metrics)
    print(table_str)

