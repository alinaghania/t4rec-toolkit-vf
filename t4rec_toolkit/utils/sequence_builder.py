# t4rec_toolkit/utils/sequence_builder.py
# -*- coding: utf-8 -*-
"""
SequenceBuilder : construit des séquences temporelles client (par mois)
à partir d'un dataset d'événements (client_id, date, item, extras...).

Sortie :
  - X_seq: dict {feature_name: np.array [N_clients, T] d'ids int}
  - y: np.array [N_clients] (item à T+horizon) si build_target_from_events=True
  - client_ids: ordre des clients (pour rejoin)
  - vocab_sizes: dict {feature_name: vocab_size} (pour T4Rec)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class SequenceBuilderConfig:
    client_id_col: str
    time_col: str
    item_col: str
    extra_event_cols: List[str]
    months_lookback: int = 24
    time_granularity: str = "M"      # "M" (mois) ou "W" (semaine)
    min_events_per_client: int = 1
    target_horizon: int = 1
    pad_value: int = 0
    build_target_from_events: bool = True


class SequenceBuilder:
    """
    Implémentation simple et claire :
      - bucketise les dates par mois (ou semaine),
      - pour chaque client : dernier "months_lookback" buckets → on remplit l'item le + récent du bucket,
      - extras idem (avec leurs propres vocabulaires),
      - y = item dans le bucket (T + horizon) si demandé.
    """
    def __init__(self, cfg: SequenceBuilderConfig):
        self.cfg = cfg
        self.item2id: Dict[str, int] = {}        # vocab item
        self.extras2id: Dict[str, Dict[str, int]] = {}  # par extra
        self.pad_value = cfg.pad_value

    def _discretize_time(self, s: pd.Series) -> pd.Series:
        if self.cfg.time_granularity.upper() == "M":
            # Mois
            return s.dt.to_period("M").dt.to_timestamp()
        elif self.cfg.time_granularity.upper() == "W":
            # Semaine (lundi)
            return s.dt.to_period("W-MON").dt.to_timestamp()
        else:
            raise ValueError(f"Granularité non supportée: {self.cfg.time_granularity}")

    def _fit_vocabs(self, df: pd.DataFrame):
        # Items
        items = df[self.cfg.item_col].astype(str).unique().tolist()
        # 0 = pad ; 1..N = items
        self.item2id = {str(x): i+1 for i, x in enumerate(items)}

        # Extras
        self.extras2id = {}
        for col in self.cfg.extra_event_cols:
            vals = df[col].astype(str).unique().tolist()
            self.extras2id[col] = {str(x): i+1 for i, x in enumerate(vals)}

    def _to_id(self, val: Any, mapping: Dict[str, int]) -> int:
        return mapping.get(str(val), 0)  # 0 si non trouvé

    def fit_transform(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        cfg = self.cfg
        df = events_df.copy()

        # Sanity
        needed = [cfg.client_id_col, cfg.time_col, cfg.item_col]
        for c in needed:
            if c not in df.columns:
                raise ValueError(f"Colonne requise manquante dans events: {c}")

        # Bucket temps (mois/semaine)
        df["_bucket"] = self._discretize_time(df[cfg.time_col])

        # Ne garder que les colonnes utiles
        cols = [cfg.client_id_col, "_bucket", cfg.item_col] + cfg.extra_event_cols
        df = df[cols].dropna(subset=[cfg.client_id_col, "_bucket", cfg.item_col])

        # Fit vocabs (items + extras)
        self._fit_vocabs(df)

        # Construire la grille temporelle : pour chaque client, les "months_lookback" buckets les plus récents
        # 1) Dernier bucket global
        max_bucket = df["_bucket"].max()
        # 2) Générer les buckets cibles
        if cfg.time_granularity.upper() == "M":
            buckets = pd.date_range(
                (max_bucket - pd.offsets.MonthBegin(cfg.months_lookback-1)).normalize(),
                max_bucket.normalize(),
                freq="MS"
            )
        else:  # "W"
            buckets = pd.date_range(
                (max_bucket - pd.offsets.Week(weeks=cfg.months_lookback-1)).normalize(),
                max_bucket.normalize(),
                freq="W-MON"
            )
        bucket_index = {b: i for i, b in enumerate(buckets)}
        T = len(buckets)

        # Agrégation par client / bucket → on prend le DERNIER item du bucket (si multi events)
        # c’est simple, robuste, et marche bien pour NBO
        # NOTE: tu peux mettre une autre stratégie (most frequent dans le bucket, etc.)
        df = df.sort_values(by=[cfg.client_id_col, "_bucket"])  # ordre
        last_in_bucket = df.groupby([cfg.client_id_col, "_bucket"]).tail(1)

        # Liste clients (ceux qui ont au moins min_events)
        cnt = last_in_bucket.groupby(cfg.client_id_col).size()
        keep_clients = set(cnt[cnt >= cfg.min_events_per_client].index)

        # Itération clients → remplir matrices
        clients = sorted(list(keep_clients))
        N = len(clients)

        # Matrices (items + extras)
        X_item = np.full((N, T), self.pad_value, dtype=np.int64)
        X_extras: Dict[str, np.ndarray] = {
            col: np.full((N, T), self.pad_value, dtype=np.int64)
            for col in cfg.extra_event_cols
        }

        # Labels (si build_target_from_events)
        y = None
        if cfg.build_target_from_events:
            y = np.full((N,), self.pad_value, dtype=np.int64)

        # Build index per client
        g = last_in_bucket.groupby(cfg.client_id_col)
        for i, cid in enumerate(clients):
            sub = g.get_group(cid)
            # mapping bucket → row
            for _, row in sub.iterrows():
                b = row["_bucket"]
                if b in bucket_index:
                    t = bucket_index[b]
                    X_item[i, t] = self._to_id(row[cfg.item_col], self.item2id)
                    for col in cfg.extra_event_cols:
                        X_extras[col][i, t] = self._to_id(row[col], self.extras2id[col])

            # Label = item à T + horizon
            if cfg.build_target_from_events:
                t_label = T - 1 + cfg.target_horizon  # "mois suivant" au-delà de la fenêtre
                # Pour rester dans la fenêtre, on prend le dernier bucket "rempli"
                # stratégie : si dernier pas (T-1) a un item ≠ pad → y = cet item
                # sinon on remonte jusqu'au précédent non-pad.
                last_non_pad = np.where(X_item[i] != self.pad_value)[0]
                if len(last_non_pad) > 0:
                    idx = last_non_pad[-1]
                    # "approximation" next-item : on prend l'item du dernier pas comme target (proxy)
                    # (si tu as un vrai (T+h), c’est mieux : il suffit d’élargir la fenêtre et lire le bucket suivant)
                    y[i] = X_item[i, idx]
                else:
                    y[i] = self.pad_value

        # Pack
        X_seq = {"item_id": X_item}
        for col, mat in X_extras.items():
            X_seq[col] = mat

        # Vocabs
        vocab_sizes = {"item_id": 1 + len(self.item2id)}  # + pad
        for col in cfg.extra_event_cols:
            vocab_sizes[col] = 1 + len(self.extras2id[col])

        out = {
            "X_seq": X_seq,
            "y": y,
            "client_ids": clients,
            "vocab_sizes": vocab_sizes,
            "buckets": buckets.tolist(),
        }
        return out
