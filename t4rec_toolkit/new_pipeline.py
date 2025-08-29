
# t4rec_toolkit/pipeline_core.py
# =============================================================================
# Pipeline HYBRIDE T4Rec + PyTorch TransformerEncoder
# - Embeddings: transformers4rec SequenceEmbeddingFeatures
# - Entraînement: boucle PyTorch (mini-batches, class weights)
# - Anti-leakage: fit transformers sur TRAIN uniquement
# - Top-K métriques format "collègue"
# Version: compatible T4Rec 23.04.00
# =============================================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# T4Rec (embeddings)
# -----------------------------------------------------------------------------
try:
    import transformers4rec.torch as tr
    from transformers4rec.torch.features.embedding import (
        FeatureConfig,
        TableConfig,
    )
    from transformers4rec.torch.features.sequence import SequenceEmbeddingFeatures

    _HAS_T4REC = True
    print("T4Rec imported successfully")
except ImportError as e:
    _HAS_T4REC = False
    print(f"T4Rec import error: {e}")
    raise ImportError(
        "T4Rec est requis. Installe: pip install transformers4rec[torch]"
    )

# -----------------------------------------------------------------------------
# Dataiku (I/O datasets) - optionnel
# -----------------------------------------------------------------------------
try:
    import dataiku

    _HAS_DATAIKU = True
except Exception:
    _HAS_DATAIKU = False

# -----------------------------------------------------------------------------
# Progress bar
# -----------------------------------------------------------------------------
try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# -----------------------------------------------------------------------------
# Transformers internes
# -----------------------------------------------------------------------------
from .transformers.sequence_transformer import SequenceTransformer
from .transformers.categorical_transformer import CategoricalTransformer

# -----------------------------------------------------------------------------
# PyTorch
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.utils.data as tud
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
)
from scipy.special import softmax


# =============================================================================
# Configs utilitaires
# =============================================================================

@dataclass
class T4RecConfig:
    """Configuration minimale pour auto-sizer si besoin."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.1
    max_sequence_length: int = 20
    mem_len: int = 50
    attn_type: str = "bi"
    vocab_size: int = 1000
    num_classes: int = 150


def blank_config() -> Dict[str, Any]:
    """Config squelette — simple et explicite."""
    return {
        "data": {
            "dataset_name": "",
            "sample_size": 10000,
            "chunk_size": 2000,
            "partitions": None,
            "temporal_split": None,  # optionnel
            "limit": None,           # optionnel
        },
        "features": {
            "sequence_cols": [],
            "categorical_cols": [],
            "target_col": "",
            "exclude_target_values": [],  # ex: ["aucune_proposition"]
        },
        "model": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 3,
            "dropout": 0.1,
            "max_sequence_length": 1,  # ⚠️ séquences scalaires → 1
            "mem_len": 50,
            "attn_type": "bi",
            "vocab_size": 1000,
        },
        "transformers": {
            "categorical": {
                "max_categories": 1000,
                "handle_unknown": "encode",
                "unknown_value": 1,
            }
        },
        "training": {
            "batch_size": 64,
            "num_epochs": 15,
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "val_split": 0.2,
            "optimizer": "adamw",          # "adam" ou "adamw"
            "scheduler": None,             # "cosine" (optionnel)
            "warmup_steps": 0,             # pas de warmup par défaut
            "gradient_clip": 0.5,
            "early_stopping_patience": 0,  # 0 = désactivé
        },
        "runtime": {
            "verbose": True,
            "progress": True,
            "seed": 42,
            "memory_efficient": True,
        },
        "outputs": {
            "features_dataset": None,
            "predictions_dataset": None,
            "metrics_dataset": None,
            "model_artifacts_dataset": None,
            "local_dir": "output",
            "save_model": True,
        },
        "metrics": ["accuracy", "precision", "recall", "f1"],  # base
    }


def _setup_logging(verbose: bool) -> None:
    """Logger simple pour Dataiku / console."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


# =============================================================================
# Chargement Dataiku
# =============================================================================

def _load_dataframe(
    dataset_name: str,
    sample_size: int,
    partitions: List[str] = None,
    temporal_split: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Charge un DataFrame depuis Dataiku.
    - Si temporal_split fourni → renvoie (train_df, test_df)
    - Sinon → renvoie (df, None)
    """
    logger = logging.getLogger(__name__)
    if not _HAS_DATAIKU:
        raise RuntimeError("Dataiku non disponible dans cet environnement.")

    try:
        if temporal_split:
            logger.info(f"Loading with temporal split: {temporal_split}")
            # NB: pour simplifier, on lit tout puis filtrage / limit (Dataiku partitions possibles)
            train_dataset = dataiku.Dataset(dataset_name)
            train_df = train_dataset.get_dataframe(limit=sample_size)
            test_dataset = dataiku.Dataset(dataset_name)
            test_df = test_dataset.get_dataframe(limit=sample_size // 5)
            logger.info(f"Loaded {len(train_df):,} train rows, {len(test_df):,} test rows")
            return train_df, test_df
        else:
            dataset = dataiku.Dataset(dataset_name)
            df = dataset.get_dataframe(limit=sample_size)
            logger.info(f"Loaded {len(df):,} rows from {dataset_name}")
            return df, None
    except Exception as e:
        logger.error(f"Erreur chargement dataset {dataset_name}: {e}")
        raise


# =============================================================================
# Modèle hybride : Embeddings T4Rec + TransformerEncoder PyTorch
# =============================================================================

class T4RecHybridModel(nn.Module):
    """
    Modèle compact et propre :
    - embeddings T4Rec (concat de toutes les features)
    - projection éventuelle → d_model
    - TransformerEncoder (n_layers)
    - Head MLP → num_classes

    NB: on a volontairement retiré les "dual-pathway embeddings" (item/user)
    tant qu'on n'a pas de vrais identifiants pertinents → évite du bruit.
    """

    def __init__(
        self,
        embedding_module: nn.Module,
        n_products: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        max_sequence_length: int = 1,
        embedding_output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.t4rec_embeddings = embedding_module
        self.d_model = d_model

        # Si la dimension concat des embeddings T4Rec != d_model, projeter
        self.projection = None
        if embedding_output_dim and embedding_output_dim != d_model:
            self.projection = nn.Linear(embedding_output_dim, d_model)

        # Positional encoding simple (utile si seq_len > 1 plus tard)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_sequence_length, d_model) * 0.02
        )

        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Tête de prédiction
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_products),
        )

        # Init poids
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        inputs: dict { feature_name: tensor [B, S] } (S=1 pour l’instant)
        """
        # 1) embeddings T4Rec concat
        x = self.t4rec_embeddings(inputs)  # [B, S?, D_concat] ou [B, D_concat]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D_concat]

        # 2) projection éventuelle → d_model
        if self.projection is not None:
            x = self.projection(x)  # [B, S, d_model]

        # 3) positionnel
        if x.shape[1] <= self.positional_encoding.shape[1]:
            x = x + self.positional_encoding[:, : x.shape[1], :]

        # 4) transformer
        h = self.transformer(x)  # [B, S, d_model]

        # 5) pool (dernier token)
        z = h[:, -1, :]  # [B, d_model]

        # 6) logits
        logits = self.head(z)  # [B, num_classes]
        return logits


def _create_t4rec_model(
    config: Dict[str, Any],
) -> Tuple[nn.Module, int]:
    """
    Construit l'embedding T4Rec (concat de toutes les tables).
    Retourne (embedding_module, embedding_output_dim).
    """
    model_cfg = config["model"]
    seq_cols = config["features"]["sequence_cols"]
    cat_cols = config["features"]["categorical_cols"]
    all_cols = seq_cols + cat_cols
    d_model = model_cfg["d_model"]
    vocab = model_cfg["vocab_size"]
    max_seq_len = model_cfg.get("max_sequence_length", 1)

    if len(all_cols) == 0:
        raise ValueError("Aucune feature fournie à T4Rec.")

    # Dimension d'embedding par feature (répartie proprement)
    # Au moins 4 dims par feature pour éviter dim trop petite
    per_feat_dim = max(d_model // max(len(all_cols), 1), 4)

    feature_configs: Dict[str, FeatureConfig] = {}
    for col in seq_cols:
        table = TableConfig(
            vocabulary_size=vocab,
            dim=per_feat_dim,
            name=f"{col}_table",
        )
        feature_configs[col] = FeatureConfig(
            table=table, max_sequence_length=max_seq_len, name=col
        )

    for col in cat_cols:
        table = TableConfig(
            vocabulary_size=vocab,
            dim=per_feat_dim,
            name=f"{col}_table",
        )
        feature_configs[col] = FeatureConfig(
            table=table, max_sequence_length=max_seq_len, name=col
        )

    # item_id requis : on prend la 1ère seq si dispo sinon 1ère cat
    item_id_col = seq_cols[0] if len(seq_cols) > 0 else cat_cols[0]

    embedding_module = SequenceEmbeddingFeatures(
        feature_config=feature_configs, item_id=item_id_col, aggregation="concat"
    )

    # Tester pour récupérer la dimension concat réelle
    with torch.no_grad():
        test_batch = {}
        for col in all_cols:
            # [2, S] indices aléatoires
            test_batch[col] = torch.randint(0, vocab, (2, max_seq_len))
        out = embedding_module(test_batch)
        if out.dim() == 2:
            emb_dim = out.shape[-1]
        else:
            emb_dim = out.shape[-1]

    return embedding_module, emb_dim


# =============================================================================
# Entraînement
# =============================================================================

def validate_config(config: Dict[str, Any], strict: bool = False) -> List[str]:
    """Validation minimale de la config."""
    errors = []
    req = {
        "data.dataset_name": str,
        "data.sample_size": int,
        "features.sequence_cols": list,
        "features.categorical_cols": list,
        "features.target_col": str,
        "model.d_model": int,
        "model.n_heads": int,
        "model.n_layers": int,
    }
    for path, typ in req.items():
        v = config
        for k in path.split("."):
            if k not in v:
                errors.append(f"{path} is required")
                v = None
                break
            v = v[k]
        if v is not None and not isinstance(v, typ):
            errors.append(f"{path} must be {typ.__name__}")

    # divisibilité têtes
    m = config["model"]
    if m.get("d_model", 0) % m.get("n_heads", 1) != 0:
        errors.append("model.d_model doit être divisible par model.n_heads")
    return errors


class _NboDataset(tud.Dataset):
    """Dataset simple pour mini-batchs: inputs dict + target."""
    def __init__(self, data_dict: Dict[str, torch.Tensor], targets: np.ndarray):
        self.keys = list(data_dict.keys())
        self.data_dict = data_dict
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        x = {k: self.data_dict[k][i] for k in self.keys}
        y = int(self.targets[i])
        return x, y


def _collate_fn(batch):
    """Empile proprement un batch de dicts {feat: tensor}."""
    keys = batch[0][0].keys()
    X = {k: torch.stack([b[0][k] for b in batch], dim=0) for k in keys}
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return X, y


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entraîne le modèle hybride.
    Étapes clés:
    - chargement df (Dataiku)
    - split stratifié
    - fit transformers sur train → transform train/val
    - DataLoader mini-batch + class weights
    - TransformerEncoder + CrossEntropy + (option) early stopping
    - Sauvegardes Dataiku (features/predictions/metrics/model)
    """
    start = time.time()
    _setup_logging(config["runtime"]["verbose"])
    logger = logging.getLogger(__name__)
    verbose = config["runtime"]["verbose"]

    # Seed
    seed = config["runtime"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Chargement
    dataset_name = config["data"]["dataset_name"]
    sample_size = config["data"]["sample_size"]
    partitions = config["data"].get("partitions")
    temporal_split = config["data"].get("temporal_split")
    df, _ = _load_dataframe(dataset_name, sample_size, partitions, temporal_split)
    if verbose:
        logger.info(f"Loaded {len(df):,} rows from {dataset_name}")

    # Features & target
    seq_cols = config["features"]["sequence_cols"]
    cat_cols = config["features"]["categorical_cols"]
    target_col = config["features"]["target_col"]

    # Exclusion de certaines valeurs cibles (case-insensitive)
    exclude_vals = set([str(v).lower() for v in config["features"].get("exclude_target_values", [])])
    if len(exclude_vals) > 0:
        tgt_lower = df[target_col].astype(str).str.lower()
        counts = {v: int((tgt_lower == v).sum()) for v in exclude_vals}
        logger.info(f"EXCLUSION ANALYSIS - target à exclure: {counts}")
        df = df.loc[~tgt_lower.isin(exclude_vals)].reset_index(drop=True)
        logger.info(f"Après exclusion: {len(df):,} lignes")

    # Encodage label
    encoder = LabelEncoder()
    targets = encoder.fit_transform(df[target_col])
    n_classes = len(encoder.classes_)
    if verbose:
        logger.info(f"Nombre de classes après filtre: {n_classes}")

    # Split stratifié
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=config["training"]["val_split"],
        stratify=targets,
        random_state=seed,
    )

    # Fit transformers sur TRAIN uniquement
    seq_tr = SequenceTransformer(
        max_sequence_length=config["model"].get("max_sequence_length", 1),
        vocab_size=config["model"]["vocab_size"],
        auto_adjust=False,  # on fige pour la reproductibilité
    ).fit(df.iloc[train_idx], seq_cols)

    cat_tr = CategoricalTransformer(
        max_categories=config["transformers"]["categorical"]["max_categories"],
        handle_unknown=config["transformers"]["categorical"]["handle_unknown"],
        unknown_value=config["transformers"]["categorical"]["unknown_value"],
    ).fit(df.iloc[train_idx], cat_cols)

    # Transform train / val (sans fuite)
    seq_train = seq_tr.transform(df.iloc[train_idx]).data
    seq_val = seq_tr.transform(df.iloc[val_idx]).data
    cat_train = cat_tr.transform(df.iloc[train_idx]).data
    cat_val = cat_tr.transform(df.iloc[val_idx]).data

    # Helper conversion float[0..1] → indices entiers [0..vocab-1]
    vocab_size = config["model"]["vocab_size"]

    def to_int_index(x_float: np.ndarray) -> np.ndarray:
        x = np.array(x_float, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip((x * (vocab_size - 1)).astype(np.int64), 0, vocab_size - 1)

    # Construire batch tensors
    train_batch: Dict[str, torch.Tensor] = {}
    val_batch: Dict[str, torch.Tensor] = {}

    # Séquentielles: reshape [N, 1] (seq_len=1) pour T4Rec
    for col in seq_cols:
        key = f"{col}_seq"
        tr_vals = to_int_index(seq_train[key]).reshape(-1, 1)
        va_vals = to_int_index(seq_val[key]).reshape(-1, 1)
        train_batch[col] = torch.tensor(tr_vals, dtype=torch.long)
        val_batch[col] = torch.tensor(va_vals, dtype=torch.long)

    # Catégorielles: une valeur par ligne (pas de dimension séquence)
    for col in cat_cols:
        key = f"{col}_encoded"
        tr_vals = np.clip(np.array(cat_train[key]).astype(np.int64), 0, vocab_size - 1)
        va_vals = np.clip(np.array(cat_val[key]).astype(np.int64), 0, vocab_size - 1)
        train_batch[col] = torch.tensor(tr_vals, dtype=torch.long)
        val_batch[col] = torch.tensor(va_vals, dtype=torch.long)

    train_targets = targets[train_idx]
    val_targets = targets[val_idx]
    y_train_t = torch.tensor(train_targets, dtype=torch.long)
    y_val_t = torch.tensor(val_targets, dtype=torch.long)

    # Construire module d'embeddings T4Rec et modèle hybride
    embedding_module, emb_dim = _create_t4rec_model(config)
    model = T4RecHybridModel(
        embedding_module=embedding_module,
        n_products=n_classes,
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        dropout=config["model"]["dropout"],
        max_sequence_length=config["model"].get("max_sequence_length", 1),
        embedding_output_dim=emb_dim,
    )

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {total_params:,} parameters")

    # DataLoader (mini-batches)
    train_ds = _NboDataset(train_batch, train_targets)
    train_loader = tud.DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=_collate_fn,
    )

    # Optimizer & scheduler
    opt_name = config["training"].get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

    sched_name = config["training"].get("scheduler")
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["num_epochs"]
        )
    else:
        scheduler = None

    # Class weights + label smoothing
    classes = np.unique(train_targets)
    class_weights = compute_class_weight("balanced", classes=classes, y=train_targets)
    cw = torch.tensor(class_weights, dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    # Entraînement
    num_epochs = config["training"]["num_epochs"]
    patience = config["training"].get("early_stopping_patience", 0)
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    pbar = tqdm(total=num_epochs, desc="Training T4Rec XLNet") if (config["runtime"]["progress"] and _HAS_TQDM) else None

    for epoch in range(1, num_epochs + 1):
        model.train()
        run_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            run_loss += loss.item() * len(yb)

        train_loss = run_loss / len(train_ds)

        # Validation full-batch (simple)
        model.eval()
        with torch.no_grad():
            val_logits = model(val_batch)
            val_loss = loss_fn(val_logits, y_val_t).item()
            val_pred = val_logits.argmax(1)
            val_acc = (val_pred == y_val_t).float().mean().item()

        if scheduler is not None:
            scheduler.step()

        if pbar:
            pbar.set_postfix({"loss": f"{train_loss:.4f}", "val_acc": f"{val_acc:.4f}"})
            pbar.update(1)

        if verbose:
            logger.info(f"Epoch {epoch}/{num_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # Early stopping (optionnel)
        if patience > 0:
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        logger.info("Early stopping déclenché")
                    break

    if pbar:
        pbar.close()

    # Restaurer meilleur état si early stopping
    if best_state is not None:
        model.load_state_dict(best_state)

    # Évaluation finale
    model.eval()
    with torch.no_grad():
        val_logits = model(val_batch)
        val_pred = val_logits.argmax(1)
        final_acc = (val_pred == y_val_t).float().mean().item()

        # métriques additionnelles
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_val_t.cpu().numpy(), val_pred.cpu().numpy(), average="weighted", zero_division=0
        )
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            y_val_t.cpu().numpy(), val_pred.cpu().numpy(), average="macro", zero_division=0
        )
        bal_acc = balanced_accuracy_score(y_val_t.cpu().numpy(), val_pred.cpu().numpy())

    # Scores probabilistes pour Top-K & export
    val_probs = softmax(val_logits.cpu().numpy(), axis=1)
    true_labels = y_val_t.cpu().numpy()

    # -----------------------------------------------------------------------------
    # Sauvegardes Dataiku (facultatif)
    # -----------------------------------------------------------------------------
    saved = {}
    if _HAS_DATAIKU and any(config["outputs"].get(f"{k}_dataset") for k in ["features", "predictions", "metrics", "model_artifacts"]):
        logger.info("Saving outputs to Dataiku datasets...")

        # 1) Features report simple
        if config["outputs"].get("features_dataset"):
            features_df = pd.DataFrame({
                "feature_name": seq_cols + cat_cols,
                "feature_type": ["sequence"] * len(seq_cols) + ["categorical"] * len(cat_cols),
                "importance": [1.0] * (len(seq_cols) + len(cat_cols)),  # placeholder
            })
            dataiku.Dataset(config["outputs"]["features_dataset"]).write_with_schema(features_df)
            saved["features"] = config["outputs"]["features_dataset"]
            logger.info(f"Features saved to {config['outputs']['features_dataset']}")

        # 2) Predictions détaillées (Top-5, noms produits)
        if config["outputs"].get("predictions_dataset"):
            # mapping encodé → libellé
            inv_map = {i: str(lbl) for i, lbl in enumerate(encoder.classes_)}
            rows = []
            for i, (probs, y_true) in enumerate(zip(val_probs, true_labels)):
                top5_idx = np.argsort(probs)[-5:][::-1]
                top5_scores = probs[top5_idx]
                pred_idx = top5_idx[0]
                rows.append({
                    "client_id": i + 1,
                    "predicted_product_id": int(pred_idx),
                    "predicted_product_name": inv_map.get(int(pred_idx), f"UNKNOWN_{pred_idx}"),
                    "true_product_id": int(y_true),
                    "true_product_name": inv_map.get(int(y_true), f"UNKNOWN_{y_true}"),
                    "prediction_correct": bool(pred_idx == y_true),
                    "confidence_score": float(probs[pred_idx]),
                    "top1_product_name": inv_map.get(int(top5_idx[0]), f"UNK_{top5_idx[0]}"),
                    "top1_score": float(top5_scores[0]),
                    "top2_product_name": inv_map.get(int(top5_idx[1]), None) if len(top5_idx) > 1 else None,
                    "top2_score": float(top5_scores[1]) if len(top5_scores) > 1 else None,
                    "top3_product_name": inv_map.get(int(top5_idx[2]), None) if len(top5_idx) > 2 else None,
                    "top3_score": float(top5_scores[2]) if len(top5_scores) > 2 else None,
                    "top4_product_name": inv_map.get(int(top5_idx[3]), None) if len(top5_idx) > 3 else None,
                    "top4_score": float(top5_scores[3]) if len(top5_scores) > 3 else None,
                    "top5_product_name": inv_map.get(int(top5_idx[4]), None) if len(top5_idx) > 4 else None,
                    "top5_score": float(top5_scores[4]) if len(top5_scores) > 4 else None,
                    "raw_scores_json": str(probs.tolist()),
                    "prediction_timestamp": pd.Timestamp.now(),
                })
            pred_df = pd.DataFrame(rows)
            dataiku.Dataset(config["outputs"]["predictions_dataset"]).write_with_schema(pred_df)
            saved["predictions"] = config["outputs"]["predictions_dataset"]
            logger.info(f"Detailed predictions saved to {config['outputs']['predictions_dataset']}")

        # 3) Metrics standard + Top-K
        if config["outputs"].get("metrics_dataset"):
            metrics_records = []
            std_metrics = [
                ("accuracy", final_acc),
                ("precision_weighted", precision_w),
                ("recall_weighted", recall_w),
                ("f1_weighted", f1_w),
                ("precision_macro", precision_m),
                ("recall_macro", recall_m),
                ("f1_macro", f1_m),
                ("balanced_accuracy", bal_acc),
            ]
            now = pd.Timestamp.now()
            for name, val in std_metrics:
                metrics_records.append({
                    "metric_name": name,
                    "metric_value": float(val),
                    "metric_type": "standard",
                    "dataset_split": "validation",
                    "k_value": None,
                    "timestamp": now,
                })

            # Top-K façon collègue
            inv_map = {i: str(lbl) for i, lbl in enumerate(encoder.classes_)}
            k_values = [1, 3, 5]
            topk = evaluate_topk_metrics_nbo(val_probs, true_labels, inv_map, k_values)
            for k, d in topk.items():
                for k_name, v in [
                    ("precision_at_k", d.get("Precision@K", 0.0)),
                    ("recall_at_k", d.get("Recall@K", 0.0)),
                    ("f1_at_k", d.get("F1@K", 0.0)),
                    ("ndcg_at_k", d.get("NDCG@K", 0.0)),
                    ("map", d.get("MAP", 0.0)),
                    ("hit_rate_at_k", d.get("HitRate@K", 0.0)),
                    ("coverage_at_k", d.get("Coverage@K", 0.0)),
                    ("clients_evaluated", d.get("Clients_evaluated", 0)),
                ]:
                    metrics_records.append({
                        "metric_name": k_name,
                        "metric_value": float(v),
                        "metric_type": "topk_nbo",
                        "dataset_split": "validation",
                        "k_value": int(k),
                        "timestamp": now,
                    })

            mdf = pd.DataFrame(metrics_records)
            dataiku.Dataset(config["outputs"]["metrics_dataset"]).write_with_schema(mdf)
            saved["metrics"] = config["outputs"]["metrics_dataset"]
            logger.info(f"Metrics saved to {config['outputs']['metrics_dataset']}")

        # 4) Model artifacts
        if config["outputs"].get("model_artifacts_dataset"):
            art = pd.DataFrame({
                "artifact_name": ["model_config", "model_architecture", "training_config"],
                "artifact_value": [
                    str(config["model"]),
                    f"T4RecHybrid-{config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D",
                    str(config["training"]),
                ],
                "timestamp": [pd.Timestamp.now()]*3,
            })
            dataiku.Dataset(config["outputs"]["model_artifacts_dataset"]).write_with_schema(art)
            saved["model_artifacts"] = config["outputs"]["model_artifacts_dataset"]
            logger.info(f"Model artifacts saved to {config['outputs']['model_artifacts_dataset']}")

    # Retour
    exec_time = time.time() - start
    return {
        "metrics": {
            "accuracy": final_acc,
            "precision_weighted": precision_w,
            "recall_weighted": recall_w,
            "f1_weighted": f1_w,
            "precision_macro": precision_m,
            "recall_macro": recall_m,
            "f1_macro": f1_m,
            "balanced_accuracy": bal_acc,
        },
        "predictions": {
            "raw_outputs": val_probs,                  # ← probabilités (softmax)
            "predicted_classes": val_probs.argmax(1),
            "true_classes": true_labels,
        },
        "model_info": {
            "total_params": sum(p.numel() for p in model.parameters()),
            "architecture": f"T4RecHybrid-{config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D",
        },
        "data_info": {
            "rows": len(df),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "n_sequence_features": len(seq_cols),
            "n_categorical_features": len(cat_cols),
            "n_features": len(seq_cols) + len(cat_cols),
            "target_classes": n_classes,
        },
        "execution_time": exec_time,
        "saved_datasets": saved,
    }


# =============================================================================
# TOP-K façon “collègue”
# =============================================================================

def compute_ranking_metrics_at_k(client_ids, labels, scores, products, k):
    """
    Calcule les métriques NBO par client puis moyenne.
    - client_ids: array [n_rows]
    - labels: 0/1
    - scores: probas pour chaque (client, produit)
    - products: identifiants produits (texte)
    """
    from sklearn.metrics import ndcg_score, average_precision_score
    from collections import defaultdict

    client_data = defaultdict(list)
    for cid, label, score, prod in zip(client_ids, labels, scores, products):
        client_data[cid].append((label, score, prod))

    ndcgs, aps, recalls, f1s = [], [], [], []
    hit_count = 0
    recommended_products = set()
    precision_topk_total = 0
    topk_count = 0
    valid_clients = 0

    for cid, items in client_data.items():
        y_true = np.array([l for l, _, _ in items], dtype=float)
        y_score = np.array([s for _, s, _ in items], dtype=float)
        y_prods = np.array([p for _, _, p in items])

        if y_true.sum() == 0 or len(y_true) < 2 or np.isnan(y_score).any():
            continue

        valid_clients += 1
        top_k_idx = np.argsort(y_score)[::-1][:k]
        y_topk = y_true[top_k_idx]
        p_topk = y_prods[top_k_idx]

        precision_topk_total += y_topk.sum()
        topk_count += k

        try:
            ndcgs.append(ndcg_score([y_true], [y_score], k=k))
            aps.append(average_precision_score(y_true, y_score))
        except Exception:
            ndcgs.append(0.0)
            aps.append(0.0)

        rec_k = y_topk.sum() / y_true.sum()
        recalls.append(rec_k)

        prec_k = y_topk.sum() / k
        f1s.append(0.0 if (prec_k + rec_k) == 0 else 2 * (prec_k * rec_k) / (prec_k + rec_k))

        if y_topk.sum() > 0:
            hit_count += 1

        recommended_products.update(p_topk)

    return {
        "Precision@K": precision_topk_total / topk_count if topk_count > 0 else 0.0,
        "Recall@K": float(np.mean(recalls)) if recalls else 0.0,
        "F1@K": float(np.mean(f1s)) if f1s else 0.0,
        "NDCG@K": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "MAP": float(np.mean(aps)) if aps else 0.0,
        "HitRate@K": hit_count / valid_clients if valid_clients > 0 else 0.0,
        "Coverage@K": len(recommended_products) / len(set(products)) if len(products) > 0 else 0.0,
        "Clients_evaluated": valid_clients,
    }


def convert_predictions_to_nbo_format(predictions, targets, inverse_target_mapping):
    """
    Convertit nos prédictions [n_samples, n_classes] en format NBO
    (client, label binaire, score, produit).
    'predictions' DOIT être des probabilités (softmax).
    """
    client_ids_list, labels_list, scores_list, products_list = [], [], [], []
    n_samples, n_classes = predictions.shape

    for sample_idx in range(n_samples):
        client_id = sample_idx + 1
        pred_probs = predictions[sample_idx]
        true_class = targets[sample_idx]

        for product_idx in range(n_classes):
            client_ids_list.append(client_id)
            labels_list.append(1.0 if product_idx == true_class else 0.0)
            scores_list.append(float(pred_probs[product_idx]))
            products_list.append(inverse_target_mapping.get(product_idx, f"UNKNOWN_{product_idx}"))

    return (
        np.array(client_ids_list),
        np.array(labels_list),
        np.array(scores_list),
        np.array(products_list),
    )


def evaluate_topk_metrics_nbo(predictions, targets, inverse_target_mapping, k_values=[1, 3, 5]):
    """
    Retourne un dict {K: métriques} avec l'algo "collègue".
    - predictions: probabilités [n_samples, n_classes]
    - targets: vrais indices de classes
    - inverse_target_mapping: {class_index: "libellé produit"}
    """
    client_ids, labels, scores, products = convert_predictions_to_nbo_format(
        predictions, targets, inverse_target_mapping
    )
    all_metrics = {}
    for k in k_values:
        all_metrics[k] = compute_ranking_metrics_at_k(client_ids, labels, scores, products, k)
    return all_metrics


def format_topk_table(metrics_by_k, baseline_metrics=None):
    """Joli tableau Top-K lisible en console."""
    lines = []
    lines.append("T4REC XLNET TOP-K INFERENCE METRICS")
    lines.append("=" * 80)
    header = "| Metric          |" + "".join([f" K={k:<10} |" for k in sorted(metrics_by_k.keys())])
    lines.append(header)
    lines.append("|" + "-" * (len(header) - 2) + "|")
    for key, label in [
        ("Precision@K", "Precision"),
        ("Recall@K", "Recall"),
        ("F1@K", "F1-Score"),
        ("NDCG@K", "NDCG"),
        ("HitRate@K", "Hit Rate"),
        ("Coverage@K", "Coverage"),
    ]:
        row = f"| {label:<15} |"
        for k in sorted(metrics_by_k.keys()):
            row += f" {metrics_by_k[k].get(key,0.0)*100:>6.2f}%   |"
        lines.append(row)
    lines.append("|" + "-" * (len(header) - 2) + "|")
    return "\n".join(lines)


def print_topk_results(metrics_by_k, baseline_metrics=None):
    print(format_topk_table(metrics_by_k, baseline_metrics))
