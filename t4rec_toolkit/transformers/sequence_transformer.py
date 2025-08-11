# transformers/sequence_transformer.py
"""
Transformer pour les données séquentielles.

Ce module gère la transformation des colonnes contenant des séquences
au format string vers des arrays numériques encodés, avec gestion
du padding, troncature et vocabulaire.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
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
    """Analyse détaillée d'une colonne de données."""

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
    Transformer intelligent pour les données séquentielles.

    Cette classe analyse les données en profondeur et fournit des
    recommandations adaptées au contexte des données.
    """

    def __init__(
        self,
        max_sequence_length: int = 20,
        vocab_size: int = 10000,
        auto_adjust: bool = True,
        quality_threshold: float = 0.8,
        name: Optional[str] = None,
    ):
        """
        Args:
            max_sequence_length: Longueur maximale des séquences
            vocab_size: Taille du vocabulaire pour l'encodage
            auto_adjust: Ajuster automatiquement les paramètres selon les données
            quality_threshold: Seuil de qualité minimum acceptable
            name: Nom du transformer
        """
        super().__init__(name)
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.auto_adjust = auto_adjust
        self.quality_threshold = quality_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.column_analyses = {}

    def _analyze_column(self, series: pd.Series, name: str) -> ColumnAnalysis:
        """Analyse approfondie d'une colonne avec recommandations."""
        # Statistiques de base
        missing_ratio = series.isna().mean()
        unique_ratio = series.nunique() / len(series)

        # Tentative de conversion numérique
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_ratio = (~numeric_series.isna()).mean()

        # Statistiques si numérique
        column_stats = {}
        if numeric_ratio > 0:
            valid_values = numeric_series.dropna()
            if len(valid_values) > 0:
                column_stats.update(
                    {
                        "mean": valid_values.mean(),
                        "std": valid_values.std(),
                        "min": valid_values.min(),
                        "max": valid_values.max(),
                        "skew": stats.skew(valid_values)
                        if len(valid_values) > 2
                        else 0,
                    }
                )

        # Évaluation de la qualité
        recommendations = []
        warnings = []

        if missing_ratio > 0.3:
            warnings.append(f"Taux élevé de valeurs manquantes: {missing_ratio:.1%}")
            recommendations.append(
                "Considérer un prétraitement des valeurs manquantes ou "
                "exclure cette colonne si non critique"
            )

        if numeric_ratio < 0.9:
            warnings.append(
                f"Données non-numériques détectées: {1 - numeric_ratio:.1%}"
            )
            recommendations.append(
                "Vérifier le format des données. Exemples non-numériques: "
                f"{series[~numeric_series.notna()].head().tolist()}"
            )

        if unique_ratio < 0.01:
            warnings.append("Très faible variabilité des données")
            recommendations.append(
                "Cette colonne pourrait être mieux traitée comme catégorielle"
            )

        # Recommandations sur la configuration
        if column_stats and "std" in column_stats:
            if column_stats["std"] == 0:
                warnings.append("Colonne constante")
                recommendations.append("Considérer l'exclusion de cette colonne")
            elif column_stats["std"] < 0.1:
                recommendations.append(
                    "Considérer une normalisation ou standardisation "
                    "due à la faible variance"
                )

        # Déterminer le niveau de qualité
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
            recommendations=recommendations,
            warnings=warnings,
            stats=column_stats,
        )

    def _suggest_parameters(
        self, analyses: Dict[str, ColumnAnalysis]
    ) -> Dict[str, Any]:
        """Suggère des paramètres optimaux basés sur l'analyse des données."""
        suggestions = {}

        # Analyser les longueurs de séquence
        sequence_lengths = [
            len(str(val).split(","))
            for analysis in analyses.values()
            for val in analysis.stats.values()
            if isinstance(val, (list, str))
        ]

        if sequence_lengths:
            p95_length = np.percentile(sequence_lengths, 95)
            suggestions["max_sequence_length"] = min(
                max(int(p95_length * 1.2), 5),  # Au moins 5
                100,  # Maximum raisonnable
            )

        # Analyser la cardinalité des valeurs
        unique_values = sum(
            analysis.stats.get("unique_count", 0) for analysis in analyses.values()
        )

        if unique_values > 0:
            suggestions["vocab_size"] = min(
                max(int(unique_values * 1.5), 100),  # Au moins 100
                50000,  # Maximum raisonnable
            )

        return suggestions

    def fit(
        self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None, **kwargs
    ) -> "SequenceTransformer":
        """
        Ajuste le transformer avec analyse et recommandations.

        Args:
            data: DataFrame d'entraînement
            feature_columns: Colonnes à utiliser
            **kwargs: Paramètres supplémentaires
        """
        self.logger.info("Début de l'analyse des données...")

        # Obtenir les colonnes
        columns = self.get_feature_columns(data, feature_columns)
        self.feature_columns = columns  # Stocker pour utilisation dans transform()

        # Analyser chaque colonne
        analyses = {}
        for col in columns:
            analysis = self._analyze_column(data[col], col)
            analyses[col] = analysis

            # Logger les résultats
            self.logger.info(f"\nAnalyse de {col}:")
            self.logger.info(f"Qualité: {analysis.quality_level.value}")

            if analysis.warnings:
                for warning in analysis.warnings:
                    self.logger.warning(f"⚠️ {warning}")

            if analysis.recommendations:
                self.logger.info("Recommandations:")
                for i, rec in enumerate(analysis.recommendations, 1):
                    self.logger.info(f"  {i}. {rec}")

        # Suggérer des paramètres optimaux
        if self.auto_adjust:
            suggestions = self._suggest_parameters(analyses)
            if suggestions:
                self.logger.info("\nParamètres suggérés:")
                for param, value in suggestions.items():
                    current = getattr(self, param)
                    if current != value:
                        self.logger.info(
                            f"  {param}: {current} → {value} (modification suggérée)"
                        )
                        if self.auto_adjust:
                            setattr(self, param, value)
                            self.logger.info("  → Paramètre ajusté automatiquement")

        # Stocker les analyses
        self.column_analyses = analyses

        # Vérifier la qualité globale
        poor_quality_cols = [
            col
            for col, analysis in analyses.items()
            if analysis.quality_level in (DataQualityLevel.POOR, DataQualityLevel.FAIR)
        ]

        if poor_quality_cols:
            msg = (
                f"Colonnes de qualité insuffisante détectées: {poor_quality_cols}\n"
                "Considérer un prétraitement ou l'exclusion de ces colonnes."
            )
            if len(poor_quality_cols) / len(columns) > 0.5:
                raise TransformationError(
                    msg, transformer_name=self.name, step="quality_validation"
                )
            else:
                self.logger.warning(msg)

        # Marquer comme fitted et retourner self pour le chaînage
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> TransformationResult:
        """
        Transforme les données avec validation de qualité.

        Args:
            data: DataFrame à transformer
        """
        self._check_fitted()

        transformed_data = {}
        feature_info = {}
        original_columns = []
        transformation_steps = []

        for col in self.feature_columns:
            if col not in data.columns:
                raise TransformationError(
                    f"Colonne manquante pour la transformation: {col}",
                    transformer_name=self.name,
                    step="transform",
                )

            # Transformer la colonne (simple normalisation pour l'instant)
            values = pd.to_numeric(data[col], errors="coerce").fillna(0)
            # Normalisation min-max simple
            if values.max() != values.min():
                normalized_values = (values - values.min()) / (
                    values.max() - values.min()
                )
            else:
                normalized_values = values

            # Stocker les résultats
            feature_name = f"{col}_seq"
            transformed_data[feature_name] = normalized_values.astype("float32")

            feature_info[feature_name] = {
                "original_column": col,
                "dtype": "float32",
                "shape": normalized_values.shape,
                "is_sequence": True,
                "is_categorical": False,
                "min_value": float(normalized_values.min()),
                "max_value": float(normalized_values.max()),
            }

            original_columns.append(col)
            transformation_steps.append(f"normalize_sequence_{col}")

        # Ajouter les métriques de qualité
        quality_metrics = {}
        for col in self.feature_columns:
            if col in self.column_analyses:
                analysis = self.column_analyses[col]
                quality_metrics[col] = {
                    "quality_level": analysis.quality_level.value,
                    "numeric_ratio": analysis.numeric_ratio,
                    "missing_ratio": analysis.missing_ratio,
                    "recommendations": analysis.recommendations,
                }

                # Créer le résultat
        from t4rec_toolkit.core.base_transformer import TransformationResult

        result = TransformationResult(
            data=transformed_data,
            feature_info=feature_info,
            original_columns=original_columns,
            transformation_steps=transformation_steps,
            config={
                "quality_metrics": quality_metrics,
                "n_features_in": len(self.feature_columns),
                "n_features_out": len(transformed_data),
            },
        )

        return result

    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé détaillé des transformations et analyses.

        Returns:
            Dictionnaire avec le résumé des transformations
        """
        return {
            "parameters": {
                "max_sequence_length": self.max_sequence_length,
                "vocab_size": self.vocab_size,
                "auto_adjust": self.auto_adjust,
            },
            "column_analyses": {
                name: {
                    "quality_level": analysis.quality_level.value,
                    "warnings": analysis.warnings,
                    "recommendations": analysis.recommendations,
                    "stats": analysis.stats,
                }
                for name, analysis in self.column_analyses.items()
            },
        }

    def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
        """
        Détecte automatiquement les colonnes appropriées pour la transformation séquentielle.

        Args:
            data: DataFrame à analyser

        Returns:
            Liste des noms de colonnes adaptées à la transformation séquentielle
        """
        suitable_columns = []

        for col in data.columns:
            try:
                # Vérifier si la colonne contient des données numériques ou convertibles
                series = data[col]

                # Ignorer les colonnes avec trop de valeurs manquantes
                if series.isnull().sum() / len(series) > 0.5:
                    continue

                # Vérifier si la colonne est numérique ou peut être convertie
                if pd.api.types.is_numeric_dtype(series):
                    suitable_columns.append(col)
                elif pd.api.types.is_object_dtype(series):
                    # Essayer de convertir en numérique
                    try:
                        pd.to_numeric(series.dropna().head(100))
                        suitable_columns.append(col)
                    except (ValueError, TypeError):
                        # Vérifier si c'est une séquence de nombres (format string)
                        sample_values = series.dropna().head(10)
                        if len(sample_values) > 0:
                            first_val = str(sample_values.iloc[0])
                            # Chercher des patterns de séquences : [1,2,3] ou "1 2 3"
                            if any(char in first_val for char in ["[", ",", " "]):
                                suitable_columns.append(col)

            except Exception:
                continue

        return suitable_columns
