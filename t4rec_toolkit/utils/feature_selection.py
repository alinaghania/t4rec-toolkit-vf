"""
Sélection intelligente de variables pour T4Rec (XLNet séquentiel).

Objectif
--------
Limiter le coût mémoire/latence et améliorer la stabilité/performance des modèles T4Rec
en ne retenant qu'un sous-ensemble pertinent de variables (~12–15) adaptées à la dynamique
séquentielle. Chaque feature catégorielle ou séquentielle induit des embeddings, dont le coût
est approximativement proportionnel à nb_features × longueur_de_séquence × dimension_d'embed.
Réduire les features redondantes/peu informatives réduit significativement ce coût.

Approche
--------
1) Détection du type de cible (binaire / multiclasse / continue)
2) Mesures d'association adaptées au type (Pearson/Spearman/MI pour numériques,
   Chi²/Cramer's V/MI pour catégorielles)
3) Contrôle de la multicolinéarité (corrélation forte entre numériques) pour éviter
   les embeddings redondants
4) Importance modèle via RandomForest + validation croisée (option GroupKFold)
5) Agrégation normalisée en score composite, top-K par type et cap global
6) Rapport détaillé avec raisons de drop, flags de sélection et journalisation
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore

    _HAVE_TQDM = True
except Exception:  # pragma: no cover
    _tqdm = None  # type: ignore
    _HAVE_TQDM = False

try:
    # Dataiku optional import
    from ..adapters import DataikuAdapter

    exists_dataiku = True
except Exception:  # pragma: no cover
    exists_dataiku = False
    DataikuAdapter = None  # type: ignore


@dataclass
class FeatureSelectorConfig:
    """Configuration pour la sélection de features T4Rec."""

    # Sélection
    top_k_sequence: int = 20
    top_k_categorical: int = 20
    total_feature_cap: Optional[int] = 15

    # Échantillonnage et qualité
    sample_size: int = 20000
    random_state: int = 42
    corr_threshold: float = 0.9

    # Performance et mémoire
    compute_model_importance: bool = True
    rf_n_estimators: Optional[int] = None
    max_cat_cramers_levels: int = 100

    # NOUVELLES OPTIMISATIONS MÉMOIRE
    chunk_size: int = 10000  # Traitement par chunks pour gros datasets
    downcast_dtypes: bool = True  # Réduction automatique des types
    correlation_batch_size: int = 50  # Calcul corrélation par batch
    gc_frequency: int = 100  # Garbage collection fréquent

    # Validation croisée
    group_col: Optional[str] = None
    class_weight_balanced: bool = True

    # Outputs
    report_output_dir: Optional[str] = "output/feature_selection"
    report_dataset: Optional[str] = None

    # Runtime
    verbose: bool = True
    progress: bool = True


@dataclass
class FeatureSelectionResult:
    """Résultat de la sélection de features."""

    target_type: str
    numeric_candidates: List[str]
    categorical_candidates: List[str]
    selected_sequence_cols: List[str]
    selected_categorical_cols: List[str]
    report: pd.DataFrame
    config: FeatureSelectorConfig
    execution_time_seconds: float


class FeatureSelector:
    """
    Sélecteur intelligent de features pour T4Rec.

    Cette classe implémente une pipeline complète de sélection de variables
    adaptée aux contraintes des modèles T4Rec (XLNet séquentiel), avec
    contrôle des coûts mémoire et optimisation de la performance.
    """

    def __init__(
        self, config: Optional[FeatureSelectorConfig] = None, name: Optional[str] = None
    ):
        """
        Initialise le sélecteur de features.

        Args:
            config: Configuration de sélection
            name: Nom du sélecteur (optionnel)
        """
        self.name = name or self.__class__.__name__
        self.config = config or FeatureSelectorConfig()
        self.logger = logging.getLogger(self.name)
        self.is_fitted = False

        # Résultats conservés après fit
        self.result_: Optional[FeatureSelectionResult] = None

        # Progress bar state
        self._progress = self.config.progress

    def _pbar(self, iterable, desc: Optional[str] = None, total: Optional[int] = None):
        """Helper pour barres de progression conditionnelles."""
        if self._progress and _HAVE_TQDM:
            try:
                return _tqdm(iterable, desc=desc, total=total, leave=False)
            except Exception:
                return iterable
        return iterable

    def _detect_target_type(self, y: pd.Series) -> str:
        """Détecte le type de cible.

        Pourquoi (T4Rec):
        - Le choix des mesures d'association et du protocole CV dépend du type de cible
          (ex. MI classification vs régression, StratifiedKFold vs KFold).
        """
        if pd.api.types.is_numeric_dtype(y):
            unique_vals = y.dropna().nunique()
            if unique_vals <= 2:
                return "binary"
            elif unique_vals <= 50:
                return "multiclass"
            else:
                return "continuous"
        else:
            unique_vals = y.dropna().nunique()
            if unique_vals == 2:
                return "binary"
            elif unique_vals <= 100:
                return "multiclass"
            else:
                # OPTIMISATION: Très nombreuses classes (>100) = traiter comme régression
                # pour éviter OOM et améliorer performance
                return "high_cardinality_multiclass"

    def _safe_numeric_cols(self, df: pd.DataFrame) -> List[str]:
        """Liste des colonnes numériques détectées via dtypes pandas."""
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def _safe_categorical_cols(
        self, df: pd.DataFrame, exclude: Optional[List[str]] = None
    ) -> List[str]:
        """Liste des colonnes catégorielles/texte/booléennes (hors exclusions)."""
        exclude = set(exclude or [])
        cols: List[str] = []
        for c in df.columns:
            if c in exclude:
                continue
            if (
                pd.api.types.is_object_dtype(df[c])
                or pd.api.types.is_categorical_dtype(df[c])
                or pd.api.types.is_bool_dtype(df[c])
            ):
                cols.append(c)
            elif not pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        return cols

    def _pearson_spearman_for_numeric(
        self, df: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Calcule |Pearson| et |Spearman| par feature numérique vs cible."""
        rows = []

        # OPTIMISATION: Pre-filtrage des colonnes constantes pour éviter warnings
        valid_cols = []
        for col in df.columns:
            series = df[col]
            if series.nunique(dropna=True) > 1 and series.notna().sum() > 2:
                valid_cols.append(col)
            else:
                # Log colonnes constantes/invalides
                if self.config.verbose:
                    self.logger.debug(
                        f"[FeatureSelection] Skipping constant/invalid column: {col}"
                    )
                rows.append(
                    {
                        "feature": col,
                        "pearson_abs": np.nan,
                        "spearman_abs": np.nan,
                    }
                )

        # Traitement des colonnes valides
        for col in self._pbar(valid_cols, desc="Assoc num (Pearson/Spearman)"):
            series = df[col]
            try:
                valid = series.notna() & y.notna()
                n_valid = valid.sum()

                if n_valid > 2:
                    # OPTIMISATION: Vérification variance avant calcul
                    series_valid = series[valid]
                    y_valid = y[valid]

                    if series_valid.std() > 1e-10 and y_valid.std() > 1e-10:
                        r, _ = stats.pearsonr(series_valid, y_valid)
                        rho, _ = stats.spearmanr(series_valid, y_valid)
                    else:
                        r, rho = np.nan, np.nan
                else:
                    r, rho = np.nan, np.nan

            except Exception as e:
                if self.config.verbose:
                    self.logger.debug(
                        f"[FeatureSelection] Error computing correlation for {col}: {e}"
                    )
                r, rho = np.nan, np.nan

            rows.append(
                {
                    "feature": col,
                    "pearson_abs": abs(r) if pd.notna(r) else np.nan,
                    "spearman_abs": abs(rho) if pd.notna(rho) else np.nan,
                }
            )

        return pd.DataFrame(rows).set_index("feature")

    def _chi2_cramersv_for_categorical(
        self, df: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Cramer's V (depuis chi²) pour features catégorielles vs cible."""
        rows = []
        y_str = y.fillna("__NA__").astype(str)
        for col in self._pbar(df.columns, desc="Assoc cat (Cramer's V)"):
            try:
                x_str = df[col].fillna("__NA__").astype(str)
                cont = pd.crosstab(x_str, y_str)
                chi2_val = stats.chi2_contingency(cont, correction=False)[0]
                n = cont.values.sum()
                phi2 = chi2_val / max(n, 1)
                r, k = cont.shape
                phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
                kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
                rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
                cramers_v = np.sqrt(phi2corr / max(min(kcorr - 1, rcorr - 1), 1))
            except Exception:
                cramers_v = np.nan
            rows.append({"feature": col, "cramers_v": cramers_v})
        return pd.DataFrame(rows).set_index("feature")

    def _mutual_info_numeric(
        self, df: pd.DataFrame, y: pd.Series, target_type: str
    ) -> pd.Series:
        """Information mutuelle pour variables numériques."""
        try:
            X = df.values
            if target_type in ("binary", "multiclass"):
                mi = mutual_info_classif(X, y, random_state=self.config.random_state)
            else:
                mi = mutual_info_regression(X, y, random_state=self.config.random_state)
            return pd.Series(mi, index=df.columns, name="mutual_info")
        except Exception:
            return pd.Series(
                [np.nan] * len(df.columns), index=df.columns, name="mutual_info"
            )

    def _mutual_info_categorical(
        self, df: pd.DataFrame, y: pd.Series, target_type: str
    ) -> pd.Series:
        """Information mutuelle pour variables catégorielles."""
        try:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X = enc.fit_transform(df.fillna("__NA__").astype(str))
            if target_type in ("binary", "multiclass"):
                mi = mutual_info_classif(X, y, random_state=self.config.random_state)
            else:
                mi = mutual_info_regression(X, y, random_state=self.config.random_state)
            return pd.Series(mi, index=df.columns, name="mutual_info")
        except Exception:
            return pd.Series(
                [np.nan] * len(df.columns), index=df.columns, name="mutual_info"
            )

    def _numeric_multicollinearity_filter(
        self, df_numeric: pd.DataFrame
    ) -> Tuple[List[str], List[Tuple[str, str, float]]]:
        """Filtre glouton de multicolinéarité sur variables numériques."""
        if df_numeric.shape[1] <= 1:
            return list(df_numeric.columns), []
        corr = df_numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop: set = set()
        drop_pairs: List[Tuple[str, str, float]] = []
        for col in self._pbar(upper.columns, desc="Filtre corrélation (num)"):
            if col in to_drop:
                continue
            for row, v in upper[col].dropna().items():
                if v >= self.config.corr_threshold and row not in to_drop:
                    to_drop.add(row)
                    drop_pairs.append((col, row, float(v)))
        kept = [c for c in df_numeric.columns if c not in to_drop]
        return kept, drop_pairs

    def _rf_importance_with_cv(
        self,
        X_num: pd.DataFrame,
        X_cat: pd.DataFrame,
        y: pd.Series,
        target_type: str,
        groups: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Importance des features via RandomForest et validation croisée."""

        # CORRECTION: Compatibilité OneHotEncoder versions sklearn
        try:
            # Test sklearn version pour paramètre sparse vs sparse_output
            from sklearn import __version__ as sklearn_version

            sklearn_major_version = int(sklearn_version.split(".")[0])
            sklearn_minor_version = int(sklearn_version.split(".")[1])

            # sklearn >= 1.2 utilise sparse_output au lieu de sparse
            if sklearn_major_version > 1 or (
                sklearn_major_version == 1 and sklearn_minor_version >= 2
            ):
                onehot_params = {"handle_unknown": "ignore", "sparse_output": False}
            else:
                onehot_params = {"handle_unknown": "ignore", "sparse": False}

        except Exception:
            # Fallback: essayer d'abord sparse_output, puis sparse
            try:
                # Test avec sparse_output (version récente)
                test_encoder = OneHotEncoder(sparse_output=False)
                onehot_params = {"handle_unknown": "ignore", "sparse_output": False}
            except TypeError:
                # Fallback vers sparse (version ancienne)
                onehot_params = {"handle_unknown": "ignore", "sparse": False}

        # Preprocessors
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(**onehot_params)),
            ]
        )
        pre = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    list(X_num.columns) if X_num is not None else [],
                ),
                (
                    "cat",
                    categorical_pipeline,
                    list(X_cat.columns) if X_cat is not None else [],
                ),
            ],
            sparse_threshold=0.0,
        )

        n_estimators_cls = (
            self.config.rf_n_estimators
            if self.config.rf_n_estimators is not None
            else 150
        )
        n_estimators_reg = (
            self.config.rf_n_estimators
            if self.config.rf_n_estimators is not None
            else 200
        )

        if target_type in ("binary", "multiclass"):
            model = RandomForestClassifier(
                n_estimators=n_estimators_cls,
                max_depth=None,
                random_state=self.config.random_state,
                n_jobs=-1,
                class_weight=(
                    "balanced" if self.config.class_weight_balanced else None
                ),
            )
            cv = (
                GroupKFold(n_splits=3)
                if groups is not None
                else StratifiedKFold(
                    n_splits=3, shuffle=True, random_state=self.config.random_state
                )
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators_reg,
                max_depth=None,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
            cv = (
                GroupKFold(n_splits=3)
                if groups is not None
                else KFold(
                    n_splits=3, shuffle=True, random_state=self.config.random_state
                )
            )

        feature_importance_acc: Dict[str, List[float]] = {
            f: [] for f in list(X_num.columns) + list(X_cat.columns)
        }

        split_iter = (
            cv.split(pd.DataFrame(index=y.index), y, groups=groups)
            if groups is not None
            else cv.split(pd.DataFrame(index=y.index), y)
        )
        for train_idx, valid_idx in self._pbar(split_iter, desc="RF-CV", total=3):
            Xn_tr = (
                X_num.iloc[train_idx]
                if X_num is not None and not X_num.empty
                else pd.DataFrame(index=train_idx)
            )
            Xc_tr = (
                X_cat.iloc[train_idx]
                if X_cat is not None and not X_cat.empty
                else pd.DataFrame(index=train_idx)
            )
            y_tr = y.iloc[train_idx]

            pipe = Pipeline(steps=[("pre", pre), ("model", model)])
            pipe.fit(pd.concat([Xn_tr, Xc_tr], axis=1), y_tr)

            try:
                importances = getattr(
                    pipe.named_steps["model"], "feature_importances_", None
                )
                if importances is None:
                    continue
                importances = np.asarray(importances)
            except Exception:
                continue

            pointer = 0
            num_feats = list(X_num.columns) if X_num is not None else []
            cat_feats = list(X_cat.columns) if X_cat is not None else []

            for f in num_feats:
                if pointer >= len(importances):
                    break
                feature_importance_acc[f].append(float(importances[pointer]))
                pointer += 1

            if cat_feats:
                onehot: OneHotEncoder = (
                    pipe.named_steps["pre"]
                    .named_transformers_["cat"]
                    .named_steps["onehot"]
                )
                categories = onehot.categories_
                for i, f in enumerate(cat_feats):
                    n_levels = len(categories[i])
                    seg = importances[pointer : pointer + n_levels]
                    feature_importance_acc[f].append(float(seg.sum()))
                    pointer += n_levels

        mean_importance = {
            f: (np.mean(v) if len(v) > 0 else 0.0)
            for f, v in feature_importance_acc.items()
        }
        return pd.Series(mean_importance)

    def fit(
        self, dataset_name: Optional[str], df: Optional[pd.DataFrame], target_col: str
    ) -> "FeatureSelector":
        """
        Ajuste le sélecteur sur les données et effectue la sélection.

        Args:
            dataset_name: nom du dataset Dataiku (si None, utiliser df)
            df: DataFrame local (si None, charger dataset_name)
            target_col: nom de la variable cible

        Returns:
            self (pour method chaining)
        """
        if self.config.verbose:
            self.logger.info("[FeatureSelection] Starting feature selection...")
        t_global0 = time.time()

        # Chargement des données
        if df is None:
            if dataset_name is None:
                raise ValueError("Either df or dataset_name must be provided")
            if not exists_dataiku:
                raise RuntimeError(
                    "Dataiku environment not available; please pass a DataFrame df instead."
                )
            adapter = DataikuAdapter()
            df = adapter.load_dataset(dataset_name, limit=self.config.sample_size)
        else:
            if (
                self.config.sample_size is not None
                and len(df) > self.config.sample_size
            ):
                df = df.sample(
                    n=self.config.sample_size, random_state=self.config.random_state
                )

        if self.config.verbose:
            self.logger.info(
                f"[FeatureSelection] Data loaded: {df.shape[0]:,} rows × {df.shape[1]} cols (sample_size={self.config.sample_size})"
            )

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in DataFrame")

        y = df[target_col]
        X = df.drop(columns=[target_col])
        target_type = self._detect_target_type(y)

        if self.config.verbose:
            self.logger.info(
                f"[FeatureSelection] Target '{target_col}' detected as: {target_type}"
            )

        numeric_cols = self._safe_numeric_cols(X)
        categorical_cols = self._safe_categorical_cols(X, exclude=numeric_cols)

        if self.config.verbose:
            self.logger.info(
                f"[FeatureSelection] Candidates — numeric: {len(numeric_cols)}, categorical: {len(categorical_cols)}"
            )

        # Associations statistiques
        reports: List[pd.DataFrame] = []
        t_assoc0 = time.time()

        if target_type == "continuous":
            num_assoc = self._pearson_spearman_for_numeric(X[numeric_cols], y)
            num_assoc["mutual_info"] = self._mutual_info_numeric(
                X[numeric_cols], y, target_type
            )
            reports.append(num_assoc.add_prefix("num_"))
        elif target_type == "high_cardinality_multiclass":
            # OPTIMISATION: Pour 150+ classes, traiter comme problème de régression
            # pour éviter OOM et améliorer performance
            if self.config.verbose:
                self.logger.info(
                    f"[FeatureSelection] High cardinality target ({y.nunique()} classes) - using regression approach"
                )

            # Encoder la cible numériquement pour corrélation
            y_encoded = pd.Categorical(y).codes

            num_assoc = self._pearson_spearman_for_numeric(X[numeric_cols], y_encoded)
            num_assoc["mutual_info"] = self._mutual_info_numeric(
                X[numeric_cols],
                y_encoded,
                "continuous",  # Traiter comme continu
            )
            reports.append(num_assoc.add_prefix("num_"))

            # Pour catégorielles: utiliser MI uniquement (pas Cramer's V qui est coûteux)
            if len(categorical_cols) > 0:
                cat_assoc = pd.DataFrame(index=categorical_cols)
                cat_assoc["mutual_info"] = self._mutual_info_categorical(
                    X[categorical_cols], y_encoded, "continuous"
                )
                reports.append(cat_assoc.add_prefix("cat_"))
        else:
            # Cas normal: binary ou multiclass < 100 classes
            num_assoc = self._pearson_spearman_for_numeric(
                X[numeric_cols], y.astype("category").cat.codes
            )
            num_assoc["mutual_info"] = self._mutual_info_numeric(
                X[numeric_cols], y, target_type
            )
            reports.append(num_assoc.add_prefix("num_"))

            # Limiter Cramer's V aux colonnes à faible cardinalité
            cat_low_card = [
                c
                for c in categorical_cols
                if X[c].nunique(dropna=False) <= self.config.max_cat_cramers_levels
            ]
            if self.config.verbose and len(cat_low_card) < len(categorical_cols):
                self.logger.info(
                    f"[FeatureSelection] Skipping Cramer's V for {len(categorical_cols) - len(cat_low_card)} high-cardinality cats (> {self.config.max_cat_cramers_levels} levels)"
                )

            cat_assoc1 = (
                self._chi2_cramersv_for_categorical(X[cat_low_card], y)
                if cat_low_card
                else pd.DataFrame()
            )
            cat_assoc2 = self._mutual_info_categorical(
                X[categorical_cols], y, target_type
            )
            cat_assoc = (
                (
                    cat_assoc1.add_prefix("")
                    if not cat_assoc1.empty
                    else pd.DataFrame()
                ).join(cat_assoc2, how="outer")
                if not cat_assoc2.empty
                else cat_assoc1
            )
            reports.append(cat_assoc.add_prefix("cat_"))

        if self.config.verbose:
            self.logger.info(
                f"[FeatureSelection] Associations computed in {time.time() - t_assoc0:.2f}s"
            )

        # Multicolinéarité
        t_mc0 = time.time()
        if len(numeric_cols) > 0:
            kept_numeric, drop_pairs = self._numeric_multicollinearity_filter(
                X[numeric_cols]
            )
            dropped_numeric = [d for _, d, _ in drop_pairs]
            if self.config.verbose:
                self.logger.info(
                    f"[FeatureSelection] Multicollinearity — kept: {len(kept_numeric)}, dropped: {len(dropped_numeric)} (threshold={self.config.corr_threshold}) in {time.time() - t_mc0:.2f}s"
                )
                for i, (keep_f, drop_f, r) in enumerate(drop_pairs[:10]):
                    self.logger.info(
                        f"  drop '{drop_f}' due to high corr with '{keep_f}' (|r|={r:.3f})"
                    )
        else:
            kept_numeric, drop_pairs = [], []

        # RF-CV (optionnel)
        if self.config.compute_model_importance:
            if self.config.verbose:
                self.logger.info(
                    "[FeatureSelection] Computing model-based importances (CV)..."
                )
            t_rf0 = time.time()
            groups = (
                df[self.config.group_col]
                if (
                    self.config.group_col is not None
                    and self.config.group_col in df.columns
                )
                else None
            )

            # OPTIMISATION: Pour high cardinality, ajuster le target pour RF
            rf_target = y
            rf_target_type = target_type
            if target_type == "high_cardinality_multiclass":
                # Utiliser l'encodage numérique pour RandomForest
                rf_target = pd.Categorical(y).codes
                rf_target_type = "continuous"  # Traiter comme régression
                if self.config.verbose:
                    self.logger.info(
                        f"[FeatureSelection] Using regression approach for RF with {y.nunique()} classes"
                    )

            imp_series = self._rf_importance_with_cv(
                X_num=X[kept_numeric] if kept_numeric else pd.DataFrame(index=X.index),
                X_cat=X[categorical_cols]
                if categorical_cols
                else pd.DataFrame(index=X.index),
                y=rf_target,
                target_type=rf_target_type,
                groups=groups,
            )
            if self.config.verbose:
                self.logger.info(
                    f"[FeatureSelection] RF-CV importances computed in {time.time() - t_rf0:.2f}s"
                )
        else:
            imp_series = pd.Series(0.0, index=X.columns)
            if self.config.verbose:
                self.logger.info(
                    "[FeatureSelection] Skipping RF-CV (compute_model_importance=False)"
                )

        # Scoring composite
        scores = pd.DataFrame(index=list(X.columns))
        if len(numeric_cols) > 0:
            for col in ["num_pearson_abs", "num_spearman_abs", "num_mutual_info"]:
                all_reports = pd.concat(reports, axis=1)
                if col in all_reports.columns:
                    scores[col] = all_reports[col]
        if len(categorical_cols) > 0:
            for col in ["cat_cramers_v", "cat_mutual_info"]:
                all_reports = pd.concat(reports, axis=1)
                if col in all_reports.columns:
                    scores[col] = all_reports[col]

        scores = scores.fillna(0.0)
        # Normalisation
        for c in scores.columns:
            mx = scores[c].max()
            scores[c] = (
                scores[c] / mx if mx and np.isfinite(mx) and mx > 0 else scores[c]
            )

        scores["model_importance"] = imp_series.reindex(scores.index).fillna(0.0)
        mi_max = scores["model_importance"].max()
        scores["model_importance"] = (
            scores["model_importance"] / mi_max
            if mi_max and mi_max > 0
            else scores["model_importance"]
        )

        # Pondérations selon type de cible
        if target_type in ("binary", "multiclass"):
            weights = {
                "num_pearson_abs": 0.2,
                "num_spearman_abs": 0.2,
                "num_mutual_info": 0.3,
                "cat_cramers_v": 0.2,
                "cat_mutual_info": 0.3,
                "model_importance": 0.4,
            }
        elif target_type == "high_cardinality_multiclass":
            # OPTIMISATION: Pour 150+ classes, privilégier MI et RF
            weights = {
                "num_pearson_abs": 0.15,
                "num_spearman_abs": 0.15,
                "num_mutual_info": 0.4,  # Plus important pour multiclass
                "cat_mutual_info": 0.4,  # Pas de Cramer's V
                "model_importance": 0.5,  # RandomForest très important
            }
        else:
            weights = {
                "num_pearson_abs": 0.4,
                "num_spearman_abs": 0.3,
                "num_mutual_info": 0.3,
                "model_importance": 0.4,
            }

        present = [k for k in weights if k in scores.columns]
        if not present:
            present = ["model_importance"]
            weights = {"model_importance": 1.0}
        w_sum = sum(weights[k] for k in present)
        for k in present:
            weights[k] = weights[k] / w_sum if w_sum > 0 else 0

        if self.config.verbose:
            self.logger.info(
                f"[FeatureSelection] Composite weights used (present only): {{k: round(weights[k], 3) for k in present}}"
            )

        scores["composite_score"] = sum(
            scores.get(k, pd.Series(0.0, index=scores.index)) * weights[k]
            for k in present
        )

        # Flags et raisons
        reasons = pd.Series("", index=scores.index)
        if drop_pairs:
            for keep_f, drop_f, r in drop_pairs:
                reasons[drop_f] = f"dropped_corr_with:{keep_f}|r={r:.3f}"

        # Sélection finale
        numeric_ranked = (
            scores.loc[kept_numeric]
            .sort_values("composite_score", ascending=False)
            .head(self.config.top_k_sequence)
            .index.tolist()
            if len(kept_numeric) > 0
            else []
        )
        categorical_ranked = (
            scores.loc[categorical_cols]
            .sort_values("composite_score", ascending=False)
            .head(self.config.top_k_categorical)
            .index.tolist()
            if len(categorical_cols) > 0
            else []
        )

        if self.config.total_feature_cap is not None:
            combined = pd.concat(
                [
                    scores.loc[numeric_ranked, ["composite_score"]].assign(
                        _type="numeric"
                    ),
                    scores.loc[categorical_ranked, ["composite_score"]].assign(
                        _type="categorical"
                    ),
                ]
            ).sort_values("composite_score", ascending=False)
            combined = combined.head(self.config.total_feature_cap)
            numeric_ranked = [
                f for f, row in combined.iterrows() if row["_type"] == "numeric"
            ]
            categorical_ranked = [
                f for f, row in combined.iterrows() if row["_type"] == "categorical"
            ]

        selected_final_set = set(numeric_ranked + categorical_ranked)
        if self.config.verbose:
            self.logger.info(
                f"[FeatureSelection] Selected — numeric: {len(numeric_ranked)}, categorical: {len(categorical_ranked)}, total: {len(selected_final_set)}"
            )
            self.logger.info(
                f"  numeric: {numeric_ranked[:5]}{'...' if len(numeric_ranked) > 5 else ''}"
            )
            self.logger.info(
                f"  categorical: {categorical_ranked[:5]}{'...' if len(categorical_ranked) > 5 else ''}"
            )

        # Construction du rapport
        report_df = scores.copy()
        report_df.insert(
            0,
            "feature_type",
            [
                "numeric"
                if f in numeric_cols
                else ("categorical" if f in categorical_cols else "other")
                for f in report_df.index
            ],
        )
        report_df["kept_after_corr_filter"] = False
        if len(kept_numeric) > 0:
            report_df.loc[kept_numeric, "kept_after_corr_filter"] = True
        report_df["drop_reason"] = reasons.reindex(report_df.index).fillna("")
        report_df["selected_final"] = report_df.index.isin(selected_final_set)
        non_selected_mask = (~report_df["selected_final"]) & (
            report_df["drop_reason"] == ""
        )
        report_df.loc[non_selected_mask, "drop_reason"] = "below_topK_or_cap"
        report_df = report_df.sort_values(
            ["feature_type", "selected_final", "composite_score"],
            ascending=[True, False, False],
        )

        # Sauvegarde
        self._save_report(report_df, dataset_name)

        # Finalisation
        execution_time = time.time() - t_global0
        self.result_ = FeatureSelectionResult(
            target_type=target_type,
            numeric_candidates=numeric_cols,
            categorical_candidates=categorical_cols,
            selected_sequence_cols=numeric_ranked,
            selected_categorical_cols=categorical_ranked,
            report=report_df,
            config=self.config,
            execution_time_seconds=execution_time,
        )

        self.is_fitted = True
        if self.config.verbose:
            self.logger.info(f"[FeatureSelection] Done in {execution_time:.2f}s")

        return self

    def _save_report(self, report_df: pd.DataFrame, dataset_name: Optional[str]):
        """Sauvegarde le rapport localement et/ou sur Dataiku."""

        # PRIORITÉ 1: Sauvegarde locale TOUJOURS
        local_saved = False
        if self.config.report_output_dir:
            try:
                import os

                os.makedirs(self.config.report_output_dir, exist_ok=True)
                name = dataset_name or "dataframe"
                report_path = os.path.join(
                    self.config.report_output_dir,
                    f"feature_selection_report_{name}.csv",
                )
                report_df.to_csv(report_path, index=True)
                local_saved = True
                if self.config.verbose:
                    self.logger.info(
                        f"[FeatureSelection] ✅ LOCAL SAVE SUCCESS: {report_path}"
                    )
            except Exception as e:
                self.logger.warning(f"[FeatureSelection] Local save failed: {e}")

        # PRIORITÉ 2: Tentative Dataiku (optionnelle)
        if self.config.report_dataset and exists_dataiku:
            try:
                # AMÉLIORATION: Gestion automatique des outputs Dataiku
                try:
                    # Méthode 1: Output dataset déclaré (recette Python)
                    import dataiku

                    output_dataset = dataiku.Dataset(self.config.report_dataset)

                    # Préparation données
                    report_to_save = report_df.reset_index().rename(
                        columns={"index": "feature"}
                    )

                    # Ajout métadonnées pour éviter erreurs Hive
                    report_to_save["processing_timestamp"] = pd.Timestamp.now()
                    report_to_save["dataset_source"] = dataset_name or "unknown"
                    report_to_save["total_features_analyzed"] = len(report_df)

                    # Assurer que le DataFrame n'est pas vide
                    if len(report_to_save) == 0:
                        dummy_row = {
                            "feature": "dummy_feature",
                            "feature_type": "system",
                            "composite_score": 0.0,
                            "selected_final": False,
                            "drop_reason": "system_dummy_row",
                            "processing_timestamp": pd.Timestamp.now(),
                            "dataset_source": dataset_name or "unknown",
                            "total_features_analyzed": 0,
                        }
                        report_to_save = pd.DataFrame([dummy_row])
                        self.logger.warning(
                            "[FeatureSelection] Empty report detected, adding dummy row"
                        )

                    # Écriture via output dataset
                    output_dataset.write_with_schema(report_to_save)

                    if self.config.verbose:
                        self.logger.info(
                            f"[FeatureSelection] ✅ DATAIKU OUTPUT SUCCESS: {self.config.report_dataset}"
                        )

                except Exception as e1:
                    # Méthode 2: Fallback vers DataikuAdapter classique
                    self.logger.warning(
                        f"[FeatureSelection] Output dataset method failed: {e1}, trying adapter..."
                    )

                    adapter = DataikuAdapter()

                    # Préparation données (même code que méthode 1)
                    report_to_save = report_df.reset_index().rename(
                        columns={"index": "feature"}
                    )

                    report_to_save["processing_timestamp"] = pd.Timestamp.now()
                    report_to_save["dataset_source"] = dataset_name or "unknown"
                    report_to_save["total_features_analyzed"] = len(report_df)

                    if len(report_to_save) == 0:
                        dummy_row = {
                            "feature": "dummy_feature",
                            "feature_type": "system",
                            "composite_score": 0.0,
                            "selected_final": False,
                            "drop_reason": "system_dummy_row",
                            "processing_timestamp": pd.Timestamp.now(),
                            "dataset_source": dataset_name or "unknown",
                            "total_features_analyzed": 0,
                        }
                        report_to_save = pd.DataFrame([dummy_row])

                    adapter.write_dataframe(report_to_save, self.config.report_dataset)

                    if self.config.verbose:
                        self.logger.info(
                            f"[FeatureSelection] ✅ DATAIKU ADAPTER SUCCESS: {self.config.report_dataset}"
                        )

            except Exception as e:
                self.logger.warning(
                    f"[FeatureSelection] ⚠️ DATAIKU SAVE FAILED (non critique): {e}"
                )

        # PRIORITÉ 3: Fallback CSV supplémentaire si Dataiku échoue ET pas de sauvegarde locale
        if not local_saved and not self.config.report_output_dir:
            try:
                fallback_path = (
                    f"feature_selection_fallback_{dataset_name or 'unknown'}.csv"
                )
                report_df.to_csv(fallback_path, index=True)
                self.logger.info(
                    f"[FeatureSelection] ✅ FALLBACK SAVE SUCCESS: {fallback_path}"
                )
            except Exception as e2:
                self.logger.error(
                    f"[FeatureSelection] ❌ ALL SAVE METHODS FAILED: {e2}"
                )

        # Résumé final
        if self.config.verbose:
            self.logger.info(
                f"[FeatureSelection] 📊 Report contains {len(report_df)} features"
            )

    def get_selected_features(self) -> Dict[str, List[str]]:
        """
        Retourne les features sélectionnées.

        Returns:
            Dict avec 'sequence_cols' et 'categorical_cols'

        Raises:
            RuntimeError: Si fit() n'a pas encore été appelé
        """
        if not self.is_fitted or self.result_ is None:
            raise RuntimeError(
                "FeatureSelector must be fitted before calling get_selected_features()"
            )

        return {
            "sequence_cols": self.result_.selected_sequence_cols,
            "categorical_cols": self.result_.selected_categorical_cols,
        }

    def get_report(self) -> pd.DataFrame:
        """
        Retourne le rapport détaillé de sélection.

        Returns:
            DataFrame du rapport

        Raises:
            RuntimeError: Si fit() n'a pas encore été appelé
        """
        if not self.is_fitted or self.result_ is None:
            raise RuntimeError(
                "FeatureSelector must be fitted before calling get_report()"
            )

        return self.result_.report

    def get_result(self) -> FeatureSelectionResult:
        """
        Retourne le résultat complet de sélection.

        Returns:
            FeatureSelectionResult complet

        Raises:
            RuntimeError: Si fit() n'a pas encore été appelé
        """
        if not self.is_fitted or self.result_ is None:
            raise RuntimeError(
                "FeatureSelector must be fitted before calling get_result()"
            )

        return self.result_


# Fonction de compatibilité (wrapper pour l'API fonctionnelle existante)
def select_features_for_t4rec(
    dataset_name: Optional[str],
    df: Optional[pd.DataFrame],
    target_col: str,
    top_k_sequence: int = 20,
    top_k_categorical: int = 20,
    total_feature_cap: Optional[int] = 15,
    sample_size: int = 20000,
    random_state: int = 42,
    corr_threshold: float = 0.9,
    report_output_dir: Optional[str] = "output/feature_selection",
    report_dataset: Optional[str] = None,
    verbose: bool = True,
    group_col: Optional[str] = None,
    class_weight_balanced: bool = True,
    progress: bool = True,
    compute_model_importance: bool = True,
    rf_n_estimators: Optional[int] = None,
    max_cat_cramers_levels: int = 100,
) -> Dict[str, object]:
    """Wrapper de compatibilité pour l'API fonctionnelle existante."""
    config = FeatureSelectorConfig(
        top_k_sequence=top_k_sequence,
        top_k_categorical=top_k_categorical,
        total_feature_cap=total_feature_cap,
        sample_size=sample_size,
        random_state=random_state,
        corr_threshold=corr_threshold,
        compute_model_importance=compute_model_importance,
        rf_n_estimators=rf_n_estimators,
        max_cat_cramers_levels=max_cat_cramers_levels,
        report_output_dir=report_output_dir,
        report_dataset=report_dataset,
        verbose=verbose,
        group_col=group_col,
        class_weight_balanced=class_weight_balanced,
        progress=progress,
    )

    selector = FeatureSelector(config)
    selector.fit(dataset_name, df, target_col)
    result = selector.get_result()

    # Format de retour compatible avec l'ancienne API
    return {
        "target_type": result.target_type,
        "numeric_candidates": result.numeric_candidates,
        "categorical_candidates": result.categorical_candidates,
        "selected_sequence_cols": result.selected_sequence_cols,
        "selected_categorical_cols": result.selected_categorical_cols,
        "report": result.report,
    }

