# t4rec_toolkit/transformers/sequence_transformer.py
"""
Transformer séquentiel (ici: colonnes numériques scalaires traitées comme séquences de longueur 1).
- fit() : calcule et stocke min/max PAR COLONNE sur le TRAIN
- transform() : réutilise ces min/max (pas de fuite de données)
- sortie : float32 dans [0,1] (sera converti en indices entiers côté pipeline)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
import logging

from ..core.base_transformer import BaseTransformer, TransformationResult
from ..core.exceptions import TransformationError


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
    def __init__(
        self,
        max_sequence_length: int = 1,   # séquence de longueur 1 (scalaires)
        vocab_size: int = 10000,
        auto_adjust: bool = False,
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
        self.feature_columns: List[str] = []
        # >>> Stats calculées au fit (anti-leakage)
        self._fit_min: Dict[str, float] = {}
        self._fit_max: Dict[str, float] = {}

    # ----------------------- Analyse (log uniquement) -----------------------

    def _analyze_column(self, series: pd.Series, name: str) -> ColumnAnalysis:
        missing_ratio = series.isna().mean()
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_ratio = (~numeric_series.isna()).mean()

        column_stats = {}
        if numeric_ratio > 0:
            valid = numeric_series.dropna()
            if len(valid) > 0:
                column_stats.update({
                    "mean": valid.mean(),
                    "std": valid.std(),
                    "min": valid.min(),
                    "max": valid.max(),
                    "skew": stats.skew(valid) if len(valid) > 2 else 0.0,
                    "unique_count": valid.nunique(),
                })

        recs, warns = [], []
        if missing_ratio > 0.3:
            warns.append(f"Taux élevé de valeurs manquantes: {missing_ratio:.1%}")
            recs.append("Imputer les valeurs manquantes ou exclure si non critique")

        if numeric_ratio < 0.9:
            warns.append(f"Données non-numériques détectées: {(1-numeric_ratio):.1%}")
            recs.append("Vérifier le format / coercition numérique")

        if unique_ratio < 0.01:
            warns.append("Très faible variabilité")
            recs.append("Peut-être mieux comme catégorielle")

        if numeric_ratio > 0.95 and missing_ratio < 0.1:
            quality = DataQualityLevel.EXCELLENT
        elif numeric_ratio > 0.8 and missing_ratio < 0.2:
            quality = DataQualityLevel.GOOD
        elif numeric_ratio > 0.6 and missing_ratio < 0.3:
            quality = DataQualityLevel.FAIR
        else:
            quality = DataQualityLevel.POOR

        return ColumnAnalysis(
            name=name,
            dtype=str(series.dtype),
            missing_ratio=missing_ratio,
            unique_ratio=unique_ratio,
            numeric_ratio=numeric_ratio,
            quality_level=quality,
            recommendations=recs,
            warnings=warns,
            stats=column_stats,
        )

    # ----------------------- Fit / Transform -----------------------

    def fit(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None, **kwargs):
        """
        Calcule les min/max sur TRAIN uniquement (pas de fuite).
        """
        self.logger.info("Début de l'analyse des données (SequenceTransformer)...")

        columns = self.get_feature_columns(data, feature_columns)
        self.feature_columns = columns

        for col in columns:
            analysis = self._analyze_column(data[col], col)
            self.column_analyses[col] = analysis
            self.logger.info(f"\nAnalyse de {col}: Qualité={analysis.quality_level.value}")
            for w in analysis.warnings:
                self.logger.warning(f"⚠️ {w}")
            if analysis.recommendations:
                self.logger.info("Recommandations:")
                for i, r in enumerate(analysis.recommendations, 1):
                    self.logger.info(f"  {i}. {r}")

            # >>> stats de normalisation (train only)
            series_num = pd.to_numeric(data[col], errors="coerce")
            col_min = float(series_num.min(skipna=True)) if series_num.notna().any() else 0.0
            col_max = float(series_num.max(skipna=True)) if series_num.notna().any() else 1.0
            if col_min == col_max:
                # éviter division par 0
                col_max = col_min + 1.0
            self._fit_min[col] = col_min
            self._fit_max[col] = col_max

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> TransformationResult:
        """
        Applique la normalisation min-max avec les stats du FIT.
        Sortie: dict {f"{col}_seq": float32 [N]}  dans [0,1]
        """
        self._check_fitted()

        transformed_data: Dict[str, np.ndarray] = {}
        feature_info: Dict[str, Any] = {}
        original_columns: List[str] = []
        steps: List[str] = []

        for col in self.feature_columns:
            if col not in data.columns:
                raise TransformationError(f"Colonne manquante: {col}", transformer_name=self.name, step="transform")

            vals = pd.to_numeric(data[col], errors="coerce").fillna(self._fit_min[col])  # impute simple
            mn, mx = self._fit_min[col], self._fit_max[col]
            if mx == mn:
                norm = np.zeros(len(vals), dtype=np.float32)
            else:
                norm = ((vals - mn) / (mx - mn)).astype(np.float32)
                norm = np.clip(norm, 0.0, 1.0)

            fname = f"{col}_seq"
            transformed_data[fname] = norm
            feature_info[fname] = {
                "original_column": col,
                "dtype": "float32",
                "min_fit": mn,
                "max_fit": mx,
                "shape": norm.shape,
                "is_sequence": True,
            }
            original_columns.append(col)
            steps.append(f"minmax_norm_{col}")

        result = TransformationResult(
            data=transformed_data,
            feature_info=feature_info,
            original_columns=original_columns,
            transformation_steps=steps,
            config={
                "n_features_in": len(self.feature_columns),
                "n_features_out": len(transformed_data),
            },
        )
        return result
