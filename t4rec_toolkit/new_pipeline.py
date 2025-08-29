
# t4rec_toolkit/pipeline_core.py
# -*- coding: utf-8 -*-
"""
PIPELINE HYBRIDE T4Rec + PyTorch avec DIMENSION TEMPORELLE RÉELLE
------------------------------------------------------------------
Ce module :
  1) charge les données (profil + events) depuis Dataiku,
  2) construit de VRAIES séquences temporelles par client (12-24 mois),
  3) prépare les features (catégorielles / séquentielles) pour T4Rec,
  4) entraîne un Transformer PyTorch sur ces séquences,
  5) calcule les métriques + sauvegarde datasets (features/pred/metrics/model).

Points clés :
  - Compatible T4Rec 23.04.00 (on n'utilise que les embeddings)
  - Exploite enfin la dimension temporelle (le vrai "+")
  - Regroupement des classes ultra-rares → "AUTRES_PRODUITS" (log explicite)
"""

from __future__ import annotations
import logging, time, json, math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============== T4Rec (embeddings uniquement) =====================
try:
    import transformers4rec.torch as tr
    from transformers4rec.torch.features.embedding import (
        EmbeddingFeatures,
        FeatureConfig,
        TableConfig,
    )
    from transformers4rec.torch.features.sequence import SequenceEmbeddingFeatures
    _HAS_T4REC = True
    print("T4Rec import OK")
except Exception as e:
    _HAS_T4REC = False
    raise ImportError("T4Rec requis. Installez: pip install transformers4rec[torch]")

# ============== Dataiku (optionnel) ================================
try:
    import dataiku
    _HAS_DATAIKU = True
except Exception:
    _HAS_DATAIKU = False

# ============== TQDM (optionnel) ==================================
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ============== Transfos locales ==================================
from .transformers.sequence_transformer import SequenceTransformer  # corrigé
from .transformers.categorical_transformer import CategoricalTransformer  # corrigé
from .utils.sequence_builder import SequenceBuilder, SequenceBuilderConfig  # nouveau

# ============== Sklearn (label encoding / metrics) =================
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support


# ------------------------------------------------------------------
# 1) CONFIGS & HELPERS
# ------------------------------------------------------------------

@dataclass
class T4RecConfig:
    """Config du modèle (dimensions principales)"""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    max_sequence_length: int = 24
    vocab_size: int = 2000


def blank_config() -> Dict[str, Any]:
    """
    Gabarit de configuration.
    NOTE: nouvelles clés "events_*" pour activer la dimension temporelle.
    """
    return {
        "data": {
            "dataset_name": "",            # table statique (profil) - optionnelle
            "events_dataset": "",          # table évènements (obligatoire pour les séquences)
            "sample_size": 100_000,        # nb max clients
            "limit": None,
            "chunk_size": 50_000,
            "partitions": None,
            "temporal_split": None,
            # Colonnes (events)
            "client_id_col": "CLIENT_ID",
            "event_time_col": "DATE_EVENT",       # doit être parsable datetime
            "product_col": "PRODUCT_CODE",        # identifiant produit
            # Optionnels : d'autres attributs d'événement (canal, montant, etc.)
            "event_extra_cols": [],               # ex: ["CANAL", "MONTANT"]
            # Colonnes profil (si dataset profil est fourni)
            "profile_join_key": "CLIENT_ID",
            "profile_categorical_cols": [],       # ex: ["SEGMENT", "REGION"]
            "profile_sequence_cols": [],          # ex: ["AGE", "REVENU"]
        },
        "sequence": {
            # Fenêtre temporelle / sampling
            "months_lookback": 24,         # longueur de séquence (derniers n mois)
            "min_events_per_client": 1,    # garde les clients avec au moins 1 event
            "time_granularity": "M",       # "M" = mois, "W" = semaine (support de base)
            "target_horizon": 1,           # prédire l'item du mois suivant (horizon=1)
            "pad_value": 0,                # padding index pour les séquences
            "build_target_from_events": True,  # True: label=produit du "mois cible"
        },
        "features": {
            "sequence_cols": [],           # (rempli automatiquement pour events)
            "categorical_cols": [],        # (profil ou events extra)
            "target_col": "TARGET_PRODUCT",# label final (si pas build auto)
            "exclude_target_values": ["aucune_souscription"],
            "merge_rare_threshold": 200,   # <200 exemples → "AUTRES_PRODUITS"
            "other_class_name": "AUTRES_PRODUITS",
        },
        "model": {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 4,
            "dropout": 0.1,
            "max_sequence_length": 24,
            "vocab_size": 2000,            # base pour embeddings
        },
        "training": {
            "batch_size": 64,
            "num_epochs": 20,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "val_split": 0.2,
            "class_weighting": True,       # pondérer CrossEntropy selon fréquences
            "gradient_clip": 1.0,
            "optimizer": "adamw",
        },
        "outputs": {
            "features_dataset": "T4REC_FEATURES",
            "predictions_dataset": "T4REC_PREDICTIONS",
            "metrics_dataset": "T4REC_METRICS",
            "model_artifacts_dataset": "T4REC_MODEL",
            "local_dir": "output",
        },
        "runtime": {
            "verbose": True,
            "progress": True,
            "seed": 42,
        },
    }


def _setup_logging(verbose: bool) -> None:
    """Logging simple."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


# ------------------------------------------------------------------
# 2) CHARGEMENT & PRÉPARATION DONNÉES
# ------------------------------------------------------------------

def _load_events_df(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Charge le dataset d'événements (obligatoire pour la dimension temporelle).
    Le dataset doit contenir au minimum :
      - client_id_col
      - event_time_col (datetime-like)
      - product_col (id produit)
    """
    if not _HAS_DATAIKU:
        raise RuntimeError("Dataiku nécessaire pour charger les datasets.")

    logger = logging.getLogger(__name__)
    dcfg = cfg["data"]
    ds_name = dcfg["events_dataset"]
    if not ds_name:
        raise ValueError("data.events_dataset est requis (séquences temporelles).")

    ds = dataiku.Dataset(ds_name)
    df = ds.get_dataframe(limit=dcfg.get("limit"))
    logger.info(f"Events loaded: {ds_name} → {df.shape}")

    # Renommer pour simplifier le code interne
    cid = dcfg["client_id_col"]
    tcol = dcfg["event_time_col"]
    pcol = dcfg["product_col"]
    if tcol not in df.columns or cid not in df.columns or pcol not in df.columns:
        raise ValueError(
            f"Colonnes requises manquantes dans events: "
            f"{cid}, {tcol}, {pcol}."
        )

    # Ensure datetime
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol])  # retire lignes non datées
    return df


def _load_profile_df_if_any(cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Charge la table profil si "dataset_name" est fourni.
    (Optionnelle : utile pour features catégorielles/statique par client)
    """
    if not _HAS_DATAIKU:
        return None

    logger = logging.getLogger(__name__)
    dcfg = cfg["data"]
    ds_name = dcfg.get("dataset_name")
    if not ds_name:
        return None

    ds = dataiku.Dataset(ds_name)
    df = ds.get_dataframe(limit=dcfg.get("limit"))
    logger.info(f"Profile loaded: {ds_name} → {df.shape}")
    return df


def merge_rare_classes(series: pd.Series, min_count: int, other_name: str) -> Tuple[pd.Series, Dict[str, str]]:
    """
    Regroupe les classes rares (< min_count) en 'other_name'.
    Retourne la série transformée + le mapping classe_originale -> 'other_name'.
    """
    vc = series.value_counts()
    rare = vc[vc < min_count].index
    mapping = {str(x): other_name for x in rare}
    new_series = series.astype(str).where(~series.astype(str).isin(rare), other_name)
    return new_series, mapping


# ------------------------------------------------------------------
# 3) MODÈLE HYBRIDE
# ------------------------------------------------------------------

class T4RecTemporalModel(nn.Module):
    """
    Modèle hybride :
      - Embeddings T4Rec pour les features (séquence item_id + extras)
      - TransformerEncoder PyTorch qui lit toute la séquence
      - Head de prédiction du prochain produit (multi-classes)
    """

    def __init__(
        self,
        embedding_module,           # SequenceEmbeddingFeatures T4Rec
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        n_classes: int,
        max_seq_len: int,
        proj_in_dim: Optional[int] = None,
    ):
        super().__init__()
        self.t4rec_embeddings = embedding_module

        # Si la concat T4Rec ne fait pas "pile" d_model, on projette
        self.proj = None
        if proj_in_dim is not None and proj_in_dim != d_model:
            self.proj = nn.Linear(proj_in_dim, d_model)

        # Positional encoding simple (appris)
        self.positional = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_classes),
        )

        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self, batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch_inputs = dict de tenseurs Long [B, T] pour toutes les features
        """
        # 1) Embeddings T4Rec (concat par feature → [B, T, F_concat])
        x = self.t4rec_embeddings(batch_inputs)  # [B, T, F_concat]

        # 2) Projection éventuelle pour matcher d_model
        if self.proj is not None:
            x = self.proj(x)  # [B, T, d_model]

        # 3) + positional
        T = x.shape[1]
        pos = self.positional[:, :T, :]
        x = x + pos

        # 4) Transformer
        x = self.encoder(x)  # [B, T, d_model]

        # 5) On prend le dernier pas de temps comme "résumé" (next-item)
        out = x[:, -1, :]  # [B, d_model]

        # 6) Logits multi-classes
        logits = self.head(out)  # [B, n_classes]
        return logits


# ------------------------------------------------------------------
# 4) CRÉATION EMBEDDINGS T4REC
# ------------------------------------------------------------------

def _build_t4rec_embedding_module(
    feature_dims: Dict[str, int],
    d_model: int,
    max_seq_len: int,
) -> Tuple[SequenceEmbeddingFeatures, int]:
    """
    Construit un SequenceEmbeddingFeatures où chaque feature a sa TableConfig.
    Retourne (module, concat_dim).
    - feature_dims: dict {feature_name: vocab_size}
      Ex: {"item_id": 5000, "canal": 10}
    - Chaque table a une dimension d_model // nb_features (simple & robuste)
    """
    assert _HAS_T4REC
    n_feats = len(feature_dims)
    per_dim = max(8, d_model // max(1, n_feats))  # au moins 8

    feature_cfgs = {}
    for feat, vocab in feature_dims.items():
        tbl = TableConfig(vocabulary_size=vocab, dim=per_dim, name=f"{feat}_table")
        feature_cfgs[feat] = FeatureConfig(
            table=tbl, max_sequence_length=max_seq_len, name=feat
        )

    # item_id = 1ère feature par convention (obligatoire pour T4Rec)
    item_id = list(feature_dims.keys())[0]
    emb = SequenceEmbeddingFeatures(feature_config=feature_cfgs, item_id=item_id, aggregation="concat")
    concat_dim = per_dim * n_feats
    return emb, concat_dim


# ------------------------------------------------------------------
# 5) ENTRAÎNEMENT
# ------------------------------------------------------------------

def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pipeline complet :
      - construction des séquences temporelles (SequenceBuilder),
      - encodage des features (cat/seq),
      - embeddings T4Rec + Transformer,
      - training/validation,
      - métriques + sauvegardes.
    """
    t0 = time.time()
    _setup_logging(config["runtime"]["verbose"])
    logger = logging.getLogger(__name__)

    # Seeds
    seed = config["runtime"].get("seed")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 1) Charger données
    events_df = _load_events_df(config)
    profile_df = _load_profile_df_if_any(config)

    # 2) Construire séquences temporelles
    sb_cfg = SequenceBuilderConfig(
        client_id_col=config["data"]["client_id_col"],
        time_col=config["data"]["event_time_col"],
        item_col=config["data"]["product_col"],
        extra_event_cols=config["data"].get("event_extra_cols", []),
        months_lookback=config["sequence"]["months_lookback"],
        time_granularity=config["sequence"]["time_granularity"],
        min_events_per_client=config["sequence"]["min_events_per_client"],
        target_horizon=config["sequence"]["target_horizon"],
        pad_value=config["sequence"]["pad_value"],
        build_target_from_events=config["sequence"]["build_target_from_events"],
    )
    builder = SequenceBuilder(sb_cfg)
    seq_pack = builder.fit_transform(events_df)
    # seq_pack contient :
    #  - "X_seq": dict {feature_name: np.array [N_clients, T] d'ids}
    #  - "y": np.array (cible item à T+horizon) si build_target_from_events=True
    #  - "client_ids": ordre des clients
    #  - "vocab_sizes": dict {feature_name: vocab_size}
    logger.info(f"[SequenceBuilder] X_seq keys={list(seq_pack['X_seq'].keys())}, shape={next(iter(seq_pack['X_seq'].values())).shape}")

    # 3) Créer/ajouter features profil (facultatif)
    #    → on traite les colonnes profil (catégorielles / "sequence_cols" mono-pas)
    X_cat = {}
    X_seq_extra = {}
    if profile_df is not None:
        # Joindre sur client_id
        pid = config["data"]["profile_join_key"]
        prof = profile_df.set_index(pid).reindex(seq_pack["client_ids"]).reset_index()
        prof.columns = [pid] + list(prof.columns[1:])

        # CATEGORIELLES profil → encodage entiers
        cat_cols = config["data"].get("profile_categorical_cols", [])
        if len(cat_cols) > 0:
            cat_tr = CategoricalTransformer(handle_unknown="encode", unknown_value=1, name="ProfileCategorical")
            cat_tr.fit(prof, cat_cols)
            cat_res = cat_tr.transform(prof)
            # Chaque col devient <col>_encoded → vecteur [N_clients]
            for k, arr in cat_res.data.items():
                # Pour compat T4Rec (qui attend [B,T]), on répète sur T et on mettra petite table
                X_cat[k.replace("_encoded", "")] = arr.astype(np.int64)

        # SEQ "mono-pas" profil (ex: âge) → on en fait une "séquence plate" (répétée sur T)
        seq_cols = config["data"].get("profile_sequence_cols", [])
        if len(seq_cols) > 0:
            seq_tr = SequenceTransformer(name="ProfileSeq")
            seq_tr.fit(prof, seq_cols)
            seq_res = seq_tr.transform(prof)
            for k, arr in seq_res.data.items():
                # arr est float [N], on va le discrétiser pour embeddings → indices int dans [0, vocab-1]
                # on choisit un vocab "profil" petit (ex 200) et un simple scale
                vocab_p = 200
                arr01 = np.clip(arr, 0.0, 1.0)
                X_seq_extra[k.replace("_seq", "")] = (arr01 * (vocab_p - 1)).astype(np.int64)

    # 4) Cible (y)
    if config["sequence"]["build_target_from_events"]:
        y_raw = seq_pack["y"]  # items (str ou id) à T+h
        y_series = pd.Series(y_raw).astype(str)
    else:
        # Alternative : target fournie dans une table (moins recommandé ici)
        raise ValueError("Pour la version temporelle, utilisez build_target_from_events=True.")

    # 5) Regroupement classes rares
    min_count = config["features"]["merge_rare_threshold"]
    other_name = config["features"]["other_class_name"]
    y_merged, rare_map = merge_rare_classes(y_series, min_count=min_count, other_name=other_name)

    if len(rare_map) > 0:
        logger.info("=== Regroupement classes rares → 'AUTRES_PRODUITS' ===")
        # On logue un résumé compact
        tops = list(rare_map.keys())[:20]
        logger.info(f"  {len(rare_map)} classes fusionnées. Exemples: {tops} → {other_name}")
        # On sauvegarde aussi le mapping complet en JSON dans les artifacts plus tard
    else:
        logger.info("Aucune classe fusionnée (pas de classe < seuil).")

    # 6) Encodage cible (LabelEncoder)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_merged.values)
    n_classes = len(encoder.classes_)
    logger.info(f"Cible: {n_classes} classes après fusion éventuelle.")

    # 7) Préparer les features pour T4Rec → tout doit être des IDs entiers dans [0, vocab-1]
    #   7.1) features séquentielles principales (events) déjà en IDs int via SequenceBuilder
    #        (ex: "item_id" + autres extras d’événements optionnels)
    X_seq_dict = {}
    for feat, mat in seq_pack["X_seq"].items():  # mat [N, T] int64
        X_seq_dict[feat] = torch.tensor(mat, dtype=torch.long)

    #   7.2) features profil "seq_extra" (répéter par T)
    for feat, vec in X_seq_extra.items():  # vec [N]
        repeated = np.repeat(vec[:, None], repeats=X_seq_dict[list(X_seq_dict.keys())[0]].shape[1], axis=1)
        X_seq_dict[feat] = torch.tensor(repeated, dtype=torch.long)

    #   7.3) features profil catégorielles (non séquentielles) → répéter sur T
    for feat, vec in X_cat.items():
        repeated = np.repeat(vec[:, None], repeats=X_seq_dict[list(X_seq_dict.keys())[0]].shape[1], axis=1)
        X_seq_dict[feat] = torch.tensor(repeated.astype(np.int64), dtype=torch.long)

    # 8) Vocab sizes (pour embeddings)
    vocab_sizes = dict(seq_pack["vocab_sizes"])  # events
    # ajouter profils "seq_extra"
    for feat in X_seq_extra.keys():
        vocab_sizes[feat] = vocab_sizes.get(feat, 200)
    # ajouter profils catégoriels
    for feat in X_cat.keys():
        # approx : 1 + max val observée (sinon 50 par défaut)
        vmax = int(np.max(X_cat[feat])) if len(X_cat[feat]) > 0 else 1
        vocab_sizes[feat] = max(vmax + 1, 10)

    # 9) Split train/val
    N = len(y)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(N * (1.0 - config["training"]["val_split"]))
    tr_idx, va_idx = idx[:split], idx[split:]

    def take_idx(d: Dict[str, torch.Tensor], ids: np.ndarray) -> Dict[str, torch.Tensor]:
        return {k: v[ids] for k, v in d.items()}

    Xtr = take_idx(X_seq_dict, tr_idx)
    Xva = take_idx(X_seq_dict, va_idx)
    ytr = torch.tensor(y[tr_idx], dtype=torch.long)
    yva = torch.tensor(y[va_idx], dtype=torch.long)

    # 10) Module d'embeddings T4Rec
    d_model = config["model"]["d_model"]
    max_T = config["model"]["max_sequence_length"]
    emb_mod, concat_dim = _build_t4rec_embedding_module(
        feature_dims=vocab_sizes,
        d_model=d_model,
        max_seq_len=max_T,
    )

    # 11) Modèle
    model = T4RecTemporalModel(
        embedding_module=emb_mod,
        d_model=d_model,
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
        n_classes=n_classes,
        max_seq_len=max_T,
        proj_in_dim=concat_dim,
    )
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 12) Optim / Loss
    if config["training"]["optimizer"].lower() == "adamw":
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

    # Class weights (optionnel)
    if config["training"]["class_weighting"]:
        counts = np.bincount(ytr.numpy(), minlength=n_classes).astype(np.float32)
        inv = 1.0 / np.clip(counts, 1.0, None)
        weights = inv / inv.sum() * n_classes
        class_w = torch.tensor(weights, dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=class_w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # 13) Training loop (full-batch pour simplicité ; Dataiku → mini-batches possible)
    num_epochs = config["training"]["num_epochs"]
    bs = config["training"]["batch_size"]
    grad_clip = config["training"].get("gradient_clip", None)

    def iterate_batches(X: Dict[str, torch.Tensor], y: torch.Tensor, batch_size: int):
        N = len(y)
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            yield {k: v[s:e] for k, v in X.items()}, y[s:e]

    pbar = tqdm(range(num_epochs), desc="Training") if (_HAS_TQDM and config["runtime"]["progress"]) else range(num_epochs)
    history = []

    for epoch in pbar:
        model.train()
        tr_loss = 0.0; n_steps = 0
        for xb, yb in iterate_batches(Xtr, ytr, bs):
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            tr_loss += loss.item(); n_steps += 1
        tr_loss /= max(1, n_steps)

        # Validation
        model.eval()
        with torch.no_grad():
            # full val (simple)
            logits = model(Xva)
            va_loss = loss_fn(logits, yva).item()
            preds = torch.argmax(logits, dim=1)
            acc = (preds == yva).float().mean().item()

        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": va_loss, "val_accuracy": acc})
        if _HAS_TQDM and config["runtime"]["progress"]:
            pbar.set_postfix({"loss": f"{tr_loss:.4f}", "val_acc": f"{acc:.4f}"})
        logger.info(f"Epoch {epoch+1}/{num_epochs} | loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={acc:.4f}")

    # 14) Final metrics
    model.eval()
    with torch.no_grad():
        logits = model(Xva)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == yva).float().mean().item()
        precision, recall, f1, _ = precision_recall_fscore_support(
            yva.numpy(), preds.numpy(), average="weighted", zero_division=0
        )

    # 15) Top-K (format collègue)
    prediction_scores = logits.cpu().numpy()
    true_labels = yva.cpu().numpy()

    # mapping inverse enc -> classe (str)
    inverse_target_mapping = {i: cls for i, cls in enumerate(encoder.classes_)}

    # On réutilise tes helpers (inchangés)
    topk_metrics, _ignored = evaluate_topk_metrics_nbo(
        predictions=prediction_scores,
        targets=true_labels,
        inverse_target_mapping=inverse_target_mapping,
        k_values=[1, 3, 5, 10],
    )

    # 16) Sauvegardes Dataiku (best effort)
    saved = {}
    if _HAS_DATAIKU:
        try:
            # PREDICTIONS (client, top1..top5)
            topk = 5
            pred_list = []
            for i, (score, ytrue) in enumerate(zip(prediction_scores, true_labels)):
                top_idx = np.argsort(score)[-topk:][::-1]
                row = {
                    "client_id": int(seq_pack["client_ids"][va_idx[i]]),
                    "true_label": str(inverse_target_mapping.get(int(ytrue), "UNK")),
                    "pred_top1": str(inverse_target_mapping.get(int(top_idx[0]), "UNK")),
                    "pred_top1_score": float(score[top_idx[0]]),
                }
                # facultatif : top2..top5
                for k in range(1, topk):
                    ki = top_idx[k]
                    row[f"pred_top{k+1}"] = str(inverse_target_mapping.get(int(ki), "UNK"))
                    row[f"pred_top{k+1}_score"] = float(score[ki])
                pred_list.append(row)

            pred_df = pd.DataFrame(pred_list)
            if config["outputs"].get("predictions_dataset"):
                dataiku.Dataset(config["outputs"]["predictions_dataset"]).write_with_schema(pred_df)
                saved["predictions"] = config["outputs"]["predictions_dataset"]

            # METRICS
            metrics_rows = []
            metrics_rows += [
                {"metric_name": "accuracy", "metric_value": float(acc), "metric_type": "standard", "dataset_split": "validation"},
                {"metric_name": "precision", "metric_value": float(precision), "metric_type": "standard", "dataset_split": "validation"},
                {"metric_name": "recall", "metric_value": float(recall), "metric_type": "standard", "dataset_split": "validation"},
                {"metric_name": "f1", "metric_value": float(f1), "metric_type": "standard", "dataset_split": "validation"},
            ]
            # top-k NBO
            for k, d in topk_metrics.items():
                metrics_rows += [
                    {"metric_name": "Precision@K", "metric_value": float(d.get("Precision@K", 0.0)), "metric_type": "topk_nbo", "dataset_split": "validation", "k_value": k},
                    {"metric_name": "Recall@K", "metric_value": float(d.get("Recall@K", 0.0)), "metric_type": "topk_nbo", "dataset_split": "validation", "k_value": k},
                    {"metric_name": "F1@K", "metric_value": float(d.get("F1@K", 0.0)), "metric_type": "topk_nbo", "dataset_split": "validation", "k_value": k},
                    {"metric_name": "NDCG@K", "metric_value": float(d.get("NDCG@K", 0.0)), "metric_type": "topk_nbo", "dataset_split": "validation", "k_value": k},
                    {"metric_name": "MAP", "metric_value": float(d.get("MAP", 0.0)), "metric_type": "topk_nbo", "dataset_split": "validation", "k_value": k},
                    {"metric_name": "HitRate@K", "metric_value": float(d.get("HitRate@K", 0.0)), "metric_type": "topk_nbo", "dataset_split": "validation", "k_value": k},
                    {"metric_name": "Coverage@K", "metric_value": float(d.get("Coverage@K", 0.0)), "metric_type": "topk_nbo", "dataset_split": "validation", "k_value": k},
                    {"metric_name": "Clients_evaluated", "metric_value": float(d.get("Clients_evaluated", 0.0)), "metric_type": "topk_nbo", "dataset_split": "validation", "k_value": k},
                ]
            met_df = pd.DataFrame(metrics_rows)
            if config["outputs"].get("metrics_dataset"):
                dataiku.Dataset(config["outputs"]["metrics_dataset"]).write_with_schema(met_df)
                saved["metrics"] = config["outputs"]["metrics_dataset"]

            # ARTIFACTS (inclure mapping rarité)
            artifacts = pd.DataFrame({
                "artifact_name": ["model_config", "sequence_builder_config", "rare_class_mapping_json"],
                "artifact_value": [
                    json.dumps(config["model"]),
                    json.dumps(sb_cfg.__dict__),
                    json.dumps(rare_map, ensure_ascii=False),
                ],
                "timestamp": [pd.Timestamp.now()]*3,
            })
            if config["outputs"].get("model_artifacts_dataset"):
                dataiku.Dataset(config["outputs"]["model_artifacts_dataset"]).write_with_schema(artifacts)
                saved["model_artifacts"] = config["outputs"]["model_artifacts_dataset"]

        except Exception as e:
            logger.warning(f"Sauvegarde Dataiku échouée (non bloquant): {e}")

    # 17) Retour
    out = {
        "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1},
        "predictions": {
            "raw_outputs": prediction_scores,
            "predicted_classes": preds.cpu().numpy(),
            "true_classes": true_labels,
        },
        "model_info": {
            "total_params": int(sum(p.numel() for p in model.parameters())),
            "architecture": f"Hybrid-T4Rec-Transformer {config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D",
        },
        "data_info": {
            "n_clients": int(len(y)),
            "seq_len": int(next(iter(X_seq_dict.values())).shape[1]),
            "features": list(X_seq_dict.keys()),
            "n_classes": int(n_classes),
        },
        "execution_time": float(time.time() - t0),
        "saved_datasets": saved,
    }
    return out


# ------------------------------------------------------------------
# 6) TOP-K (inchangé, repris de ta version)
# ------------------------------------------------------------------

def compute_ranking_metrics_at_k(client_ids, labels, scores, products, k):
    """
    (copié / simplifié de ta version ; inchangé dans l'esprit)
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
        ndcgs.append(ndcg_score([y_true], [y_score], k=k))
        aps.append(average_precision_score(y_true, y_score))
        rec_k = y_topk.sum() / y_true.sum()
        recalls.append(rec_k)
        prec_k = y_topk.sum() / k
        if (prec_k + rec_k) > 0:
            f1s.append(2 * (prec_k * rec_k) / (prec_k + rec_k))
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
    Idem ta version.
    """
    client_ids_list, labels_list, scores_list, products_list = [], [], [], []
    n_samples, n_classes = predictions.shape
    for i in range(n_samples):
        pred = predictions[i]
        true_class = targets[i]
        probs = np.exp(pred) / np.sum(np.exp(pred))
        for j in range(n_classes):
            client_ids_list.append(i + 1)
            labels_list.append(1.0 if j == true_class else 0.0)
            scores_list.append(probs[j])
            products_list.append(inverse_target_mapping.get(j, f"UNK_{j}"))
    return (np.array(client_ids_list), np.array(labels_list), np.array(scores_list), np.array(products_list))


def evaluate_topk_metrics_nbo(predictions, targets, inverse_target_mapping, k_values=[1, 3, 5]):
    """
    Idem ta version : renvoie (metrics_dict, None)
    """
    client_ids, labels, scores, products = convert_predictions_to_nbo_format(
        predictions, targets, inverse_target_mapping
    )
    all_metrics = {}
    for k in k_values:
        all_metrics[k] = compute_ranking_metrics_at_k(client_ids, labels, scores, products, k)
    return all_metrics, None
