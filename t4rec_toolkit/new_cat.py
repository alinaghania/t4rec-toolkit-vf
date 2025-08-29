# t4rec_toolkit/transformers/categorical_transformer.py
# -*- coding: utf-8 -*-
"""
Encodage catégoriel simple → IDs entiers (avec unknown_value si besoin).
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..core.base_transformer import BaseTransformer, TransformationResult


class CategoricalTransformer(BaseTransformer):
    def __init__(
        self,
        max_categories: int = 1000,
        min_frequency: int = 1,
        handle_unknown: str = "encode",
        unknown_value: int = 1,
        rare_category_threshold: float = 0.0,  # désactivé par défaut
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
        self.feature_mappings: Dict[str, str] = {}

    def _is_categorical_column(self, s: pd.Series, col_name: str) -> bool:
        if s.dtype.name in ("object", "category"):
            return True
        # numériques avec faible cardinalité
        if pd.api.types.is_numeric_dtype(s):
            n_u = s.nunique(dropna=True)
            return n_u <= 50
        return False

    def _auto_detect_features(self, data: pd.DataFrame) -> List[str]:
        return [c for c in data.columns if self._is_categorical_column(data[c], c)]

    def fit(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None, **kwargs) -> "CategoricalTransformer":
        cols = self.get_feature_columns(data, feature_columns)
        for col in cols:
            vc = data[col].astype(str).value_counts(dropna=False)
            cats = vc.index.tolist()
            # éventuellement tronquer
            cats = cats[: self.max_categories]
            mapping: Dict[Any, int] = {}
            cur = 0
            if self.handle_unknown == "encode":
                cur = max(cur, self.unknown_value + 1)  # réserver unknown_value (ex 1)
            for c in cats:
                if self.drop_first and cur == 0:
                    # on saute la 1ère catégorie si demandé
                    self.drop_first = False
                    continue
                mapping[c] = cur
                cur += 1
            self.category_mappings[col] = mapping
            self.vocabulary_sizes[col] = max(cur, self.unknown_value + 1)
            self.feature_mappings[col] = f"{col}_encoded"
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> TransformationResult:
        self._check_fitted()
        out: Dict[str, np.ndarray] = {}
        finfo: Dict[str, Any] = {}
        orig: List[str] = []
        steps: List[str] = []

        for col, mapping in self.category_mappings.items():
            if col not in data.columns:
                raise ValueError(f"Colonne manquante: {col}")
            arr = np.zeros(len(data), dtype=np.int32)
            s = data[col].astype(str)
            for i, v in enumerate(s):
                if v in mapping:
                    arr[i] = mapping[v]
                else:
                    arr[i] = self.unknown_value if self.handle_unknown == "encode" else 0

            k = self.feature_mappings[col]
            out[k] = arr
            finfo[k] = {
                "original_column": col,
                "dtype": "int32",
                "shape": arr.shape,
                "vocab_size": int(self.vocabulary_sizes[col]),
                "is_categorical": True,
            }
            orig.append(col)
            steps.append(f"encode_{col}")

        return TransformationResult(
            data=out,
            feature_info=finfo,
            original_columns=orig,
            transformation_steps=steps,
            config={"n_features_out": len(out)},
        )

