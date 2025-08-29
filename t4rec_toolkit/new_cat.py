
# t4rec_toolkit/transformers/categorical_transformer.py
"""
Transformer catégoriel:
- fit() construit un mapping catégorie -> id (train only)
- transform(): applique ce mapping; valeurs inconnues → unknown_value
- évite la collision entre 'unknown_value' et la 1ère catégorie réelle
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..core.base_transformer import BaseTransformer, TransformationResult
from ..core.exceptions import TransformationError


class CategoricalTransformer(BaseTransformer):
    def __init__(
        self,
        max_categories: int = 1000,
        min_frequency: int = 1,
        handle_unknown: str = "encode",
        unknown_value: int = 1,
        rare_category_threshold: float = 0.01,
        drop_first: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.rare_category_threshold = rare_category_threshold
        self.drop_first = drop_first

        self.category_mappings: Dict[str, Dict[Any, int]] = {}
        self.vocabulary_sizes: Dict[str, int] = {}
        self.category_frequencies: Dict[str, Dict[Any, int]] = {}
        self.feature_mappings: Dict[str, str] = {}

    def fit(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None, **kwargs):
        cols = self.get_feature_columns(data, feature_columns)
        if not cols:
            raise TransformationError("Aucune colonne catégorielle détectée", transformer_name=self.name, step="fit")

        for col in cols:
            self._fit_column(data[col], col)
            self.feature_mappings[col] = f"{col}_encoded"

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> TransformationResult:
        self._check_fitted()
        transformed_data: Dict[str, np.ndarray] = {}
        feature_info: Dict[str, Any] = {}
        original_columns: List[str] = []
        steps: List[str] = []

        for col, mapping in self.category_mappings.items():
            if col not in data.columns:
                raise TransformationError(f"Colonne manquante: {col}", transformer_name=self.name, step="transform")
            enc = self._transform_column(data[col], col)
            fname = self.feature_mappings[col]
            transformed_data[fname] = enc
            feature_info[fname] = {
                "original_column": col,
                "dtype": "int32",
                "shape": enc.shape,
                "vocab_size": self.vocabulary_sizes[col],
                "is_sequence": False,
                "is_categorical": True,
            }
            original_columns.append(col)
            steps.append(f"encode_{col}")

        return TransformationResult(
            data=transformed_data,
            feature_info=feature_info,
            config={},
            original_columns=original_columns,
            transformation_steps=steps,
        )

    # ------------------- Détails internes -------------------

    def _fit_column(self, series: pd.Series, col_name: str):
        vc = series.value_counts(dropna=False)
        total = len(series)
        rare_thr = max(self.min_frequency, int(total * self.rare_category_threshold))
        rare_cats = set(vc[vc < rare_thr].index)

        mapping: Dict[Any, int] = {}

        # Réserver explicitement la valeur 'unknown_value' pour l'inconnu
        # → les vraies catégories commenceront APRES
        next_id = max(self.unknown_value + 1, 1)

        # Trier par fréquence décroissante
        sorted_cats = vc.index.tolist()

        kept = []
        for c in sorted_cats:
            if c not in rare_cats:
                kept.append(c)
            if len(kept) >= self.max_categories:
                break

        # drop_first optionnel
        if self.drop_first and kept:
            kept = kept[1:]

        # Assigner les ids
        for c in kept:
            mapping[c] = next_id
            next_id += 1

        # Map catégories rares (si on veut les distinguer) → id dédié
        # Ici on choisit de les envoyer aussi vers unknown_value pour simplifier.
        # Si tu veux un id rare distinct, dé-commente:
        # rare_id = next_id; next_id += 1
        # for rc in rare_cats: mapping[rc] = rare_id
        for rc in rare_cats:
            mapping[rc] = self.unknown_value

        self.category_mappings[col_name] = mapping
        self.vocabulary_sizes[col_name] = next_id  # taille du vocab = dernier id + 1 approx
        self.category_frequencies[col_name] = vc.to_dict()

    def _transform_column(self, series: pd.Series, col_name: str) -> np.ndarray:
        mapping = self.category_mappings[col_name]
        out = np.empty(len(series), dtype=np.int32)
        for i, v in enumerate(series):
            if pd.isna(v):
                out[i] = self.unknown_value if self.handle_unknown == "encode" else 0
            else:
                out[i] = mapping.get(v, self.unknown_value if self.handle_unknown == "encode" else 0)
        return out

    # auto-detect: garde uniquement des colonnes plausibles (object / category / basse cardinalité)
    def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
        cols: List[str] = []
        for col in data.columns:
            s = data[col]
            if s.dtype == "object" or s.dtype.name == "category":
                cols.append(col)
            elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s):
                # cardinalité raisonnable
                nu = s.nunique(dropna=True)
                if nu <= 100 and nu / max(len(s.dropna()), 1) < 0.2:
                    cols.append(col)
        return cols
