
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
            "exclude_target_values": ["Aucune_Proposition"],
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
    Charge le dataset d'événements depuis :
      - cfg['data']['events_df'] (DataFrame en mémoire) si fourni
      - sinon un dataset Dataiku (cfg['data']['events_dataset'])
    """
    logger = logging.getLogger(__name__)
    dcfg = cfg["data"]

    # 1) Chemin "no-IO" : DataFrame déjà construit en mémoire
    if dcfg.get("events_df") is not None:
        df = dcfg["events_df"]
        if not isinstance(df, pd.DataFrame):
            raise ValueError("data.events_df doit être un pandas.DataFrame")
        logger.info(f"Events (in-memory) loaded: shape={df.shape}")
    else:
        # 2) Chemin Dataiku classique
        if not _HAS_DATAIKU:
            raise RuntimeError("Dataiku requis pour charger events_dataset.")
        ds_name = dcfg.get("events_dataset")
        if not ds_name:
            raise ValueError("data.events_dataset est requis (séquences temporelles).")
        ds = dataiku.Dataset(ds_name)
        df = ds.get_dataframe(limit=dcfg.get("limit"))
        logger.info(f"Events loaded: {ds_name} → {df.shape}")

    # Renommer/valider colonnes
    cid = dcfg["client_id_col"]
    tcol = dcfg["event_time_col"]
    pcol = dcfg["product_col"]
    for c in (cid, tcol, pcol):
        if c not in df.columns:
            raise ValueError(f"Colonne manquante dans events: '{c}'")

    # Date en datetime
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol])
    return df




def _load_profile_df_if_any(cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Charge la table profil :
      1) si cfg["data"]["profile_df"] (DataFrame en mémoire) est fourni → priorité
      2) sinon lit le dataset Dataiku cfg["data"]["dataset_name"] si présent
      3) sinon None
    """
    logger = logging.getLogger(__name__)
    dcfg = cfg["data"]

    # ---- 1) Mode mémoire ----
    if dcfg.get("profile_df") is not None:
        df = dcfg["profile_df"]
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning("data.profile_df fourni mais vide ou non-DataFrame → ignoré.")
            return None
        logger.info(f"[profile_df] (in-memory) → {df.shape}")
        return df

    # ---- 2) Mode Dataiku ----
    if not _HAS_DATAIKU:
        return None

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
    Pipeline complet (RAM/VRAM friendly) :
      - construction des séquences temporelles (SequenceBuilder),
      - encodage des features (cat/seq),
      - embeddings T4Rec + Transformer,
      - training/validation en MINI-BATCHES,
      - métriques + sauvegardes.
    """
    import gc
    from torch.cuda.amp import autocast, GradScaler

    t0 = time.time()
    _setup_logging(config["runtime"]["verbose"])
    logger = logging.getLogger(__name__)

    # Device + AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

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
    logger.info(f"[SequenceBuilder] X_seq keys={list(seq_pack['X_seq'].keys())}, shape={next(iter(seq_pack['X_seq'].values())).shape}")

    # 3) Features profil (facultatif)
    X_cat = {}
    X_seq_extra = {}
    if profile_df is not None:
        pid = config["data"]["profile_join_key"]
        prof = profile_df.set_index(pid).reindex(seq_pack["client_ids"]).reset_index()
        prof.columns = [pid] + list(prof.columns[1:])

        # Catégorielles
        cat_cols = config["data"].get("profile_categorical_cols", [])
        if len(cat_cols) > 0:
            cat_tr = CategoricalTransformer(handle_unknown="encode", unknown_value=1, name="ProfileCategorical")
            cat_tr.fit(prof, cat_cols)
            cat_res = cat_tr.transform(prof)
            for k, arr in cat_res.data.items():
                X_cat[k.replace("_encoded", "")] = arr.astype(np.int64)

        # Séquentielles "mono-pas" → discrétisation simple
        seq_cols = config["data"].get("profile_sequence_cols", [])
        if len(seq_cols) > 0:
            seq_tr = SequenceTransformer(name="ProfileSeq")
            seq_tr.fit(prof, seq_cols)
            seq_res = seq_tr.transform(prof)
            for k, arr in seq_res.data.items():
                vocab_p = 200
                arr01 = np.clip(arr, 0.0, 1.0)
                X_seq_extra[k.replace("_seq", "")] = (arr01 * (vocab_p - 1)).astype(np.int64)

    # 4) Cible (y) + exclusion explicite
    if config["sequence"]["build_target_from_events"]:
        y_series = pd.Series(seq_pack["y"]).astype(str)
        exclude_vals = config.get("features", {}).get("exclude_target_values", [])
        exclude_vals = [str(x).strip() for x in (exclude_vals or [])]
        y_series_norm = y_series.str.strip()
        mask = ~y_series_norm.isin(exclude_vals)
        n_drop = int((~mask).sum())
        if n_drop > 0:
            logger.info(f"[Target exclusion] {n_drop}/{len(mask)} lignes exclues (valeurs: {exclude_vals})")
            y_series = y_series[mask]
            for k in list(seq_pack["X_seq"].keys()):
                seq_pack["X_seq"][k] = seq_pack["X_seq"][k][mask.values]
            seq_pack["client_ids"] = np.asarray(seq_pack["client_ids"])[mask.values]
            if X_cat:
                for k in list(X_cat.keys()):
                    X_cat[k] = X_cat[k][mask.values]
            if X_seq_extra:
                for k in list(X_seq_extra.keys()):
                    X_seq_extra[k] = X_seq_extra[k][mask.values]
        else:
            logger.info("[Target exclusion] aucune ligne exclue")
    else:
        raise ValueError("Pour la version temporelle, utilisez build_target_from_events=True.")

    # 5) Regroupement classes rares
    min_count = config["features"]["merge_rare_threshold"]
    other_name = config["features"]["other_class_name"]
    y_merged, rare_map = merge_rare_classes(y_series, min_count=min_count, other_name=other_name)

    if len(rare_map) > 0:
        tops = list(rare_map.keys())[:20]
        logger.info(f"=== Classes rares fusionnées → '{other_name}' ({len(rare_map)}). Ex: {tops}")
    else:
        logger.info("Aucune classe fusionnée.")

    # 6) Encodage cible
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_merged.values)
    n_classes = len(encoder.classes_)
    logger.info(f"Cible: {n_classes} classes.")

    # 7) Tenseurs features (CPU) — on ne pousse que les mini-batches sur GPU
    X_seq_dict = {}
    T_len = next(iter(seq_pack["X_seq"].values())).shape[1]
    for feat, mat in seq_pack["X_seq"].items():
        X_seq_dict[feat] = torch.tensor(mat, dtype=torch.long)  # reste sur CPU

    for feat, vec in X_seq_extra.items():
        repeated = np.repeat(vec[:, None], repeats=T_len, axis=1)
        X_seq_dict[feat] = torch.tensor(repeated, dtype=torch.long)

    for feat, vec in X_cat.items():
        repeated = np.repeat(vec[:, None], repeats=T_len, axis=1)
        X_seq_dict[feat] = torch.tensor(repeated.astype(np.int64), dtype=torch.long)

    # 8) Vocab sizes
    vocab_sizes = dict(seq_pack["vocab_sizes"])
    for feat in X_seq_extra.keys():
        vocab_sizes[feat] = vocab_sizes.get(feat, 200)
    for feat in X_cat.keys():
        vmax = int(np.max(X_cat[feat])) if len(X_cat[feat]) > 0 else 1
        vocab_sizes[feat] = max(vmax + 1, 10)

    # 9) Split train/val (aléatoire client-level)
    N = len(y)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(N * (1.0 - config["training"]["val_split"]))
    tr_idx, va_idx = idx[:split], idx[split:]

    def slice_dict(d: Dict[str, torch.Tensor], ids: np.ndarray) -> Dict[str, torch.Tensor]:
        return {k: v[ids] for k, v in d.items()}

    Xtr_cpu = slice_dict(X_seq_dict, tr_idx)
    Xva_cpu = slice_dict(X_seq_dict, va_idx)
    ytr_cpu = torch.tensor(y[tr_idx], dtype=torch.long)
    yva_cpu = torch.tensor(y[va_idx], dtype=torch.long)

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
    ).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 12) Optim / Loss
    optimizer = (torch.optim.AdamW if config["training"]["optimizer"].lower() == "adamw" else torch.optim.Adam)(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    if config["training"]["class_weighting"]:
        counts = np.bincount(ytr_cpu.numpy(), minlength=n_classes).astype(np.float32)
        inv = 1.0 / np.clip(counts, 1.0, None)
        weights = inv / inv.sum() * n_classes
        class_w = torch.tensor(weights, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=class_w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # 13) Data iterator
    bs = int(config["training"]["batch_size"])
    bs_eval = max(1, min(bs, 256))  # batch val (petit pour éviter OOM)
    grad_clip = config["training"].get("gradient_clip", None)

    def batches(Xcpu: Dict[str, torch.Tensor], ycpu: torch.Tensor, batch_size: int):
        Nloc = len(ycpu)
        for s in range(0, Nloc, batch_size):
            e = min(s + batch_size, Nloc)
            xb = {k: v[s:e].to(device, non_blocking=False) for k, v in Xcpu.items()}
            yb = ycpu[s:e].to(device, non_blocking=False)
            yield xb, yb

    # 14) Training loop (mini-batches + AMP)
    num_epochs = config["training"]["num_epochs"]
    pbar = tqdm(range(num_epochs), desc="Training") if (_HAS_TQDM and config["runtime"]["progress"]) else range(num_epochs)
    history = []

    for epoch in pbar:
        model.train()
        tr_loss = 0.0; n_steps = 0
        for xb, yb in batches(Xtr_cpu, ytr_cpu, bs):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            tr_loss += float(loss.item()); n_steps += 1

        tr_loss /= max(1, n_steps)

        # Validation (mini-batches + pas d'allocation géante)
        model.eval()
        va_loss, n_va, correct = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in batches(Xva_cpu, yva_cpu, bs_eval):
                with autocast(enabled=use_amp):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                va_loss += float(loss.item()) * yb.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == yb).sum().item())
                n_va += yb.size(0)
        va_loss /= max(1, n_va)
        acc = correct / max(1, n_va)

        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": va_loss, "val_accuracy": acc})
        if _HAS_TQDM and config["runtime"]["progress"]:
            pbar.set_postfix({"loss": f"{tr_loss:.4f}", "val_acc": f"{acc:.4f}"})
        logger.info(f"Epoch {epoch+1}/{num_epochs} | loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={acc:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # 15) Final metrics (mini-batches)
    model.eval()
    all_logits = []
    all_preds  = []
    with torch.no_grad():
        for xb, yb in batches(Xva_cpu, yva_cpu, bs_eval):
            with autocast(enabled=use_amp):
                logits = model(xb)
            all_logits.append(logits.detach().cpu())
            all_preds.append(torch.argmax(logits, dim=1).detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    preds_cat  = torch.cat(all_preds,  dim=0)
    acc = (preds_cat.numpy() == yva_cpu.numpy()).mean().item() if hasattr(np, "mean") else float((preds_cat == yva_cpu).float().mean().item())
    precision, recall, f1, _ = precision_recall_fscore_support(
        yva_cpu.numpy(), preds_cat.numpy(), average="weighted", zero_division=0
    )

    # 16) Top-K (utilise logits en CPU déjà batchés)
    prediction_scores = logits_cat.numpy()
    true_labels = yva_cpu.numpy()
    inverse_target_mapping = {i: cls for i, cls in enumerate(encoder.classes_)}
    topk_metrics, _ignored = evaluate_topk_metrics_nbo(
        predictions=prediction_scores,
        targets=true_labels,
        inverse_target_mapping=inverse_target_mapping,
        k_values=[1, 3, 5, 10],
    )

    # 17) Sauvegardes Dataiku (best effort)
    saved = {}
    if _HAS_DATAIKU:
        try:
            topk = 5
            pred_list = []
            # Pour récupérer l’alignement client_id sur la validation
            va_client_ids = np.asarray(seq_pack["client_ids"])[va_idx]
            for i, (score, ytrue) in enumerate(zip(prediction_scores, true_labels)):
                top_idx = np.argsort(score)[-topk:][::-1]
                row = {
                    "client_id": int(va_client_ids[i]),
                    "true_label": str(inverse_target_mapping.get(int(ytrue), "UNK")),
                    "pred_top1": str(inverse_target_mapping.get(int(top_idx[0]), "UNK")),
                    "pred_top1_score": float(score[top_idx[0]]),
                }
                for k in range(1, topk):
                    ki = top_idx[k]
                    row[f"pred_top{k+1}"] = str(inverse_target_mapping.get(int(ki), "UNK"))
                    row[f"pred_top{k+1}_score"] = float(score[ki])
                pred_list.append(row)

            if config["outputs"].get("predictions_dataset"):
                pred_df = pd.DataFrame(pred_list)
                dataiku.Dataset(config["outputs"]["predictions_dataset"]).write_with_schema(pred_df)
                saved["predictions"] = config["outputs"]["predictions_dataset"]

            metrics_rows = [
                {"metric_name": "accuracy", "metric_value": float(acc), "metric_type": "standard", "dataset_split": "validation"},
                {"metric_name": "precision", "metric_value": float(precision), "metric_type": "standard", "dataset_split": "validation"},
                {"metric_name": "recall", "metric_value": float(recall), "metric_type": "standard", "dataset_split": "validation"},
                {"metric_name": "f1", "metric_value": float(f1), "metric_type": "standard", "dataset_split": "validation"},
            ]
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
            if config["outputs"].get("metrics_dataset"):
                met_df = pd.DataFrame(metrics_rows)
                dataiku.Dataset(config["outputs"]["metrics_dataset"]).write_with_schema(met_df)
                saved["metrics"] = config["outputs"]["metrics_dataset"]

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

    # 18) Retour
    out = {
        "metrics": {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)},
        "predictions": {
            "raw_outputs": prediction_scores,
            "predicted_classes": preds_cat.numpy(),
            "true_classes": true_labels,
        },
        "model_info": {
            "total_params": int(sum(p.numel() for p in model.parameters())),
            "architecture": f"Hybrid-T4Rec-Transformer {config['model']['n_layers']}L-{config['model']['n_heads']}H-{config['model']['d_model']}D",
        },
        "data_info": {
            "n_clients": int(len(y)),
            "seq_len": int(T_len),
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

# =====================================================================
# Validation de la configuration
# =====================================================================

def get_config_schema() -> Dict[str, Any]:
    """
    Schéma minimal pour valider la config.
    NOTE: events_dataset OU events_df doivent être fournis (au moins un).
    """
    return {
        "data": {
            # Au moins l'un des deux :
            "events_dataset": {"type": (str, type(None)), "required": False},
            "events_df": {"type": (pd.DataFrame, type(None)), "required": False},

            # Profil en mémoire (optionnel)
            "profile_df": {"type": (pd.DataFrame, type(None)), "required": False},

            # Requis pour interpréter les données
            "client_id_col": {"type": str, "required": True},
            "event_time_col": {"type": str, "required": True},
            "product_col": {"type": str, "required": True},

            # Optionnels
            "dataset_name": {"type": (str, type(None)), "required": False},
            "sample_size": {"type": int, "required": False},
            "limit": {"type": (int, type(None)), "required": False},
            "chunk_size": {"type": int, "required": False},
            "partitions": {"type": (str, list, type(None)), "required": False},
            "temporal_split": {"type": (dict, type(None)), "required": False},
            "event_extra_cols": {"type": list, "required": False},
            "profile_join_key": {"type": (str, type(None)), "required": False},
            "profile_categorical_cols": {"type": list, "required": False},
            "profile_sequence_cols": {"type": list, "required": False},
        },
        "sequence": {
            "months_lookback": {"type": int, "required": True},
            "time_granularity": {"type": str, "required": True},
            "min_events_per_client": {"type": int, "required": True},
            "target_horizon": {"type": int, "required": True},
            "pad_value": {"type": int, "required": True},
            "build_target_from_events": {"type": bool, "required": True},
        },
        "features": {
            "exclude_target_values": {"type": list, "required": False},
            "merge_rare_threshold": {"type": int, "required": True},
            "other_class_name": {"type": str, "required": True},
            "target_col": {"type": (str, type(None)), "required": False},
        },
        "model": {
            "d_model": {"type": int, "required": True},
            "n_heads": {"type": int, "required": True},
            "n_layers": {"type": int, "required": True},
            "dropout": {"type": float, "required": True},
            "max_sequence_length": {"type": int, "required": True},
            "vocab_size": {"type": int, "required": True},
        },
        "training": {
            "batch_size": {"type": int, "required": True},
            "num_epochs": {"type": int, "required": True},
            "learning_rate": {"type": float, "required": True},
            "weight_decay": {"type": float, "required": True},
            "val_split": {"type": float, "required": True},
            "class_weighting": {"type": bool, "required": True},
            "gradient_clip": {"type": (float, type(None)), "required": False},
            "optimizer": {"type": str, "required": True},
        },
        "outputs": {
            "features_dataset": {"type": (str, type(None)), "required": False},
            "predictions_dataset": {"type": (str, type(None)), "required": False},
            "metrics_dataset": {"type": (str, type(None)), "required": False},
            "model_artifacts_dataset": {"type": (str, type(None)), "required": False},
            "local_dir": {"type": str, "required": True},
        },
        "runtime": {
            "verbose": {"type": bool, "required": True},
            "progress": {"type": bool, "required": True},
            "seed": {"type": (int, type(None)), "required": True},
        },
    }



def validate_config(config: Dict[str, Any], strict: bool = False) -> List[str]:
    """
    Valide la config avant entraînement.
    - Vérifie la présence et le type de clés essentielles
    - Vérifie contraintes simples (ex: d_model % n_heads == 0)
    
    Retour:
        Liste d'erreurs (vide si tout est OK)
    """
    errors: List[str] = []
    schema = get_config_schema()

    # Helper pour naviguer dans le dict selon le schéma
    def _check_block(block_name: str, block_schema: Dict[str, Any]):
        if block_name not in config:
            errors.append(f"Bloc '{block_name}' manquant dans la config")
            return
        block = config[block_name]
        for key, spec in block_schema.items():
            required = spec.get("required", False)
            expected_type = spec.get("type", object)
            if required and key not in block:
                errors.append(f"{block_name}.{key} est requis")
                continue
            if key in block and expected_type is not None:
                if not isinstance(block[key], expected_type):
                    errors.append(
                        f"{block_name}.{key} doit être de type {expected_type} (actuel: {type(block[key])})"
                    )

    for section, sect_schema in schema.items():
        _check_block(section, sect_schema)

    # Contraintes spécifiques modèle
    if "model" in config:
        m = config["model"]
        if isinstance(m.get("d_model"), int) and isinstance(m.get("n_heads"), int):
            if m["d_model"] % m["n_heads"] != 0:
                errors.append("model.d_model doit être divisible par model.n_heads")

        # Séquence maximale ≥ fenêtre temporelle demandée
        if "max_sequence_length" in m and "sequence" in config:
            if m["max_sequence_length"] < config["sequence"]["months_lookback"]:
                errors.append(
                    "model.max_sequence_length doit être >= sequence.months_lookback"
                )


    if "data" in config:
        d = config["data"]

        # Au moins l'un des deux : events_df (DataFrame) OU events_dataset (str non vide)
        has_mem = isinstance(d.get("events_df"), pd.DataFrame) and not d.get("events_df").empty
        has_ds  = isinstance(d.get("events_dataset"), str) and len(d.get("events_dataset")) > 0
        if not (has_mem or has_ds):
            errors.append("data.events_df (DataFrame) ou data.events_dataset (nom de dataset) doit être fourni")

        # Colonnes requises non vides
        for col_key in ["client_id_col", "event_time_col", "product_col"]:
            if not d.get(col_key):
                errors.append(f"data.{col_key} est requis (non vide)")

    return errors

# =====================================================================
# Affichage des métriques Top-K (pour le notebook)
# =====================================================================

def format_topk_table(metrics_by_k: Dict[int, Dict[str, float]], baseline_metrics: Optional[Dict[int, Dict[str, float]]] = None) -> str:
    """
    Formate joliment les métriques Top-K.

    metrics_by_k: {K: {"precision":..., "recall":..., "f1":..., "ndcg":..., "hit_rate":..., "coverage":..., ...}}
    baseline_metrics: métriques de référence (optionnel) pour comparaison.
    """
    lines = []
    lines.append("T4REC XLNET - MÉTRIQUES TOP-K")
    lines.append("=" * 80)

    # Entête
    header = "| Metric          |"
    for k in sorted(metrics_by_k.keys()):
        header += f" K={k:<8} |"
    lines.append(header)
    lines.append("|" + "-" * (len(header) - 2) + "|")

    # Liste des clés à afficher (on mappe depuis nos noms NBO)
    mapping = [
        ("Precision@K", "Precision"),
        ("Recall@K",    "Recall"),
        ("F1@K",        "F1-Score"),
        ("NDCG@K",      "NDCG"),
        ("HitRate@K",   "Hit Rate"),
        ("Coverage@K",  "Coverage"),
    ]

    for key, label in mapping:
        row = f"| {label:<15} |"
        for k in sorted(metrics_by_k.keys()):
            val = metrics_by_k[k].get(key, 0.0) * 100.0
            row += f" {val:>7.2f}% |"
        lines.append(row)

    lines.append("|" + "-" * (len(header) - 2) + "|")
    lines.append("")
    lines.append("📊 INTERPRÉTATION BUSINESS :")

    # Petite synthèse "meilleur K" par métrique
    def _best_k_for(metric_key: str) -> Tuple[int, float]:
        best_k = max(metrics_by_k.keys(), key=lambda kk: metrics_by_k[kk].get(metric_key, 0.0))
        return best_k, metrics_by_k[best_k].get(metric_key, 0.0)

    for key, label in mapping:
        bk, bv = _best_k_for(key)
        lines.append(f"   → {label} max à K={bk}: {bv*100:.2f}%")

    lines.append("")
    lines.append("ℹ️  Définitions :")
    lines.append("   • Precision@K : % d’items recommandés qui sont pertinents")
    lines.append("   • Recall@K    : % d’items pertinents retrouvés dans le Top-K")
    lines.append("   • F1@K        : moyenne harmonique précision / rappel")
    lines.append("   • NDCG@K      : qualité de l’ordre du ranking")
    lines.append("   • Hit Rate@K  : % de clients avec ≥1 bon produit dans le Top-K")
    lines.append("   • Coverage@K  : % des produits couverts par les reco")
    lines.append("")
    lines.append("   ✅ Powered by T4Rec XLNet")

    return "\n".join(lines)


def print_topk_results(metrics_by_k: Dict[int, Dict[str, float]], baseline_metrics: Optional[Dict[int, Dict[str, float]]] = None) -> None:
    """
    Affiche le tableau Top-K en console (utilisé par le notebook).
    """
    print(format_topk_table(metrics_by_k, baseline_metrics))

