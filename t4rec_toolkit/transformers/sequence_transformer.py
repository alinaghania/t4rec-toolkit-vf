# t4rec_toolkit/transformers/sequence_transformer.py
# -*- coding: utf-8 -*-
"""
Transformer pour données "séquentielles" au sens large MAIS
utilisé ici pour des colonnes de profil "mono-pas", que l'on normalise
et que l'on pourra discrétiser ensuite (dans pipeline).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
import logging

from ..core.base_transformer import BaseTransformer, TransformationResult


class DataQualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class ColumnAnalysis:
    name: str
    dtype: str
    missing_ratio: float
    unique_ratio: float
    numeric_ratio: float
    quality_level: DataQualityLevel
    recommendations: List[str]
    warnings: List[str]
    stats: Dict[str, float]


class SequenceTransformer(BaseTransformer):
    """
    Ici on gère des colonnes numériques "statique" (profil),
    qu'on normalise [0,1] pour ensuite les convertir en IDs (dans pipeline).
    """

    def __init__(
        self,
        max_sequence_length: int = 1,  # non utilisé ici (profil mono-pas)
        vocab_size: int = 1000,        # non utilisé ici
        auto_adjust: bool = True,
        quality_threshold: float = 0.8,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.auto_adjust = auto_adjust
        self.quality_threshold = quality_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.column_analyses: Dict[str, ColumnAnalysis] = {}

    # ---------- Helpers d'analyse (facultatif pour logs) ----------
    def _analyze_column(self, series: pd.Series, name: str) -> ColumnAnalysis:
        missing_ratio = series.isna().mean()
        unique_ratio = series.nunique() / max(1, len(series))
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_ratio = (~numeric_series.isna()).mean()

        stats_dict = {}
        if numeric_ratio > 0:
            v = numeric_series.dropna()
            if len(v) > 0:
                stats_dict = {
                    "mean": float(v.mean()),
                    "std": float(v.std()),
                    "min": float(v.min()),
                    "max": float(v.max()),
                    "skew": float(stats.skew(v)) if len(v) > 2 else 0.0,
                }

        recs, warns = [], []
        if missing_ratio > 0.3:
            warns.append(f"Taux manquants élevé: {missing_ratio:.1%}")
            recs.append("Imputer ou exclure si non critique.")
        if numeric_ratio < 0.8:
            warns.append("Beaucoup de non-numérique.")
            recs.append("Vérifier le format / conversion numérique.")

        if numeric_ratio > 0.95 and missing_ratio < 0.1:
            q = DataQualityLevel.EXCELLENT
        elif numeric_ratio > 0.8 and missing_ratio < 0.2:
            q = DataQualityLevel.GOOD
        elif numeric_ratio > 0.6 and missing_ratio < 0.3:
            q = DataQualityLevel.FAIR
        else:
            q = DataQualityLevel.POOR

        return ColumnAnalysis(
            name=name,
            dtype=str(series.dtype),
            missing_ratio=float(missing_ratio),
            unique_ratio=float(unique_ratio),
            numeric_ratio=float(numeric_ratio),
            quality_level=q,
            recommendations=recs,
            warnings=warns,
            stats=stats_dict,
        )

    def fit(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None, **kwargs) -> "SequenceTransformer":
        cols = self.get_feature_columns(data, feature_columns)
        self.feature_columns = cols

        # Analyse (logs)
        for c in cols:
            ana = self._analyze_column(data[c], c)
            self.column_analyses[c] = ana
            self.logger.info(f"[SeqTransformer] {c}: quality={ana.quality_level.value}, miss={ana.missing_ratio:.1%}, num={ana.numeric_ratio:.1%}")

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> TransformationResult:
        self._check_fitted()
        transformed_data: Dict[str, np.ndarray] = {}
        feature_info: Dict[str, Any] = {}
        original_columns: List[str] = []
        steps: List[str] = []

        for col in self.feature_columns:
            if col not in data.columns:
                raise ValueError(f"Colonne manquante: {col}")
            vals = pd.to_numeric(data[col], errors="coerce").fillna(0.0).astype(np.float32)
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmax > vmin:
                norm = (vals - vmin) / (vmax - vmin)
            else:
                norm = vals

            k = f"{col}_seq"
            transformed_data[k] = norm.values.astype(np.float32)  # [N]
            feature_info[k] = {
                "original_column": col,
                "dtype": "float32",
                "shape": transformed_data[k].shape,
                "is_sequence": False,
                "is_categorical": False,
                "min_value": float(norm.min()),
                "max_value": float(norm.max()),
            }
            original_columns.append(col)
            steps.append(f"normalize_{col}")

        return TransformationResult(
            data=transformed_data,
            feature_info=feature_info,
            original_columns=original_columns,
            transformation_steps=steps,
            config={"n_features_out": len(transformed_data)},
        )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # IMPORTANT : méthode attendue par BaseTransformer (ABSTRACT)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
        """
        Détection simple : colonnes numériques non-vides.
        """
        cols: List[str] = []
        for c in data.columns:
            s = data[c]
            if s.isnull().mean() < 0.5:
                # numériques ou convertibles
                try:
                    pd.to_numeric(s.dropna().head(100))
                    cols.append(c)
                except Exception:
                    continue
        return cols

