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
import time
 
try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
 
    _HAVE_TQDM = True
except Exception:  # pragma: no cover
    _tqdm = None  # type: ignore
    _HAVE_TQDM = False
 
# Progress toggle (set in select_features_for_t4rec)
_PROGRESS: bool = False
 
 
def _pbar(iterable, desc: Optional[str] = None, total: Optional[int] = None):
    if _PROGRESS and _HAVE_TQDM:
        try:
            return _tqdm(iterable, desc=desc, total=total, leave=False)
        except Exception:
            return iterable
    return iterable
 
 
try:
    # Dataiku optional import
    from ..adapters import DataikuAdapter
 
    exists_dataiku = True
except Exception:  # pragma: no cover
    exists_dataiku = False
    DataikuAdapter = None  # type: ignore
 
 
logger = logging.getLogger(__name__)
 
 
def _detect_target_type(y: pd.Series) -> str:
    """Détecte le type de cible.
 
    Pourquoi (T4Rec):
    - Le choix des mesures d'association et du protocole CV dépend du type de cible
      (ex. MI classification vs régression, StratifiedKFold vs KFold).
 
    Paramètres
    - y: Série pandas de la variable cible.
 
    Retour
    - 'binary' | 'multiclass' | 'continuous'
    """
    if pd.api.types.is_numeric_dtype(y):
        # Heuristic: if many unique values -> continuous; else categorical
        unique_vals = y.dropna().nunique()
        if unique_vals <= 2:
            return "binary"
            # If few classes but numeric labels, treat as multiclass
        elif unique_vals <= 50:
            return "multiclass"
        else:
            return "continuous"
    else:
        unique_vals = y.dropna().nunique()
        return "binary" if unique_vals == 2 else "multiclass"
 
 
def _safe_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Liste des colonnes numériques (entiers/flottants) détectées via dtypes pandas.
 
    Pourquoi (T4Rec):
    - Les variables numériques sont de bonnes candidates pour des signaux séquentiels
      (ex. montants, fréquences), et nécessitent un contrôle de corrélation.
    """
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
 
 
def _safe_categorical_cols(
    df: pd.DataFrame, exclude: Optional[List[str]] = None
) -> List[str]:
    """Liste des colonnes catégorielles/texte/booléennes (hors exclusions).
 
    Pourquoi (T4Rec):
    - Les features catégorielles entraînent des embeddings; leur nombre doit donc
      être maîtrisé pour contenir les coûts.
 
    Paramètres
    - df: DataFrame d'entrée (sans la cible).
    - exclude: colonnes à ignorer (ex. déjà retenues comme numériques).
 
    Retour
    - Liste de noms de colonnes catégorielles.
    """
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
 
 
def _pearson_spearman_for_numeric(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Calcule |Pearson| et |Spearman| par feature numérique vs cible.
 
    Pourquoi (T4Rec):
    - Pearson mesure la relation linéaire, Spearman la relation monotone. Ces signaux
      complémentaires aident à prioriser des variables numériques «sequence-friendly».
 
    Paramètres
    - df: sous-DataFrame des variables numériques.
    - y: cible (encodée si nécessaire).
 
    Retour
    - DataFrame indexé par 'feature' avec colonnes 'pearson_abs', 'spearman_abs'.
    """
    rows = []
    for col in _pbar(df.columns, desc="Assoc num (Pearson/Spearman)"):
        series = df[col]
        try:
            valid = series.notna() & y.notna()
            r, _ = (
                stats.pearsonr(series[valid], y[valid])
                if valid.sum() > 2
                else (np.nan, np.nan)
            )
            rho, _ = (
                stats.spearmanr(series[valid], y[valid])
                if valid.sum() > 2
                else (np.nan, np.nan)
            )
        except Exception:
            r, rho = np.nan, np.nan
        rows.append(
            {
                "feature": col,
                "pearson_abs": abs(r) if pd.notna(r) else np.nan,
                "spearman_abs": abs(rho) if pd.notna(rho) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("feature")
 
 
def _anova_kruskal_for_numeric_vs_categorical(
    df: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """ANOVA et Kruskal-Wallis pour numériques vs cible catégorielle.
 
    Pourquoi (T4Rec):
    - En présence de cibles discrètes, tester si la distribution d'une variable
      numérique varie selon les classes aide à identifier un signal utile.
 
    Paramètres
    - df: sous-DataFrame numérique.
    - y: cible catégorielle.
 
    Retour
    - DataFrame indexé 'feature' avec 'anova_f', 'anova_p', 'kruskal_h', 'kruskal_p'.
    """
    rows = []
    classes = y.dropna().unique()
    if len(classes) > 50:
        le_target = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded_y = pd.Series(
            le_target.fit_transform(
                y.fillna("__NA__").astype(str).values.reshape(-1, 1)
            ).ravel(),
            index=y.index,
        )
        return _pearson_spearman_for_numeric(df, encoded_y)
    for col in _pbar(df.columns, desc="ANOVA/Kruskal (num vs cat)"):
        series = df[col]
        try:
            groups = []
            for cls in classes:
                mask = (y == cls) & series.notna()
                groups.append(series[mask].values)
            f_stat, p_anova = (
                stats.f_oneway(*groups)
                if all(len(g) > 1 for g in groups)
                else (np.nan, np.nan)
            )
            h_stat, p_kruskal = (
                stats.kruskal(*groups)
                if all(len(g) > 1 for g in groups)
                else (np.nan, np.nan)
            )
        except Exception:
            f_stat = p_anova = h_stat = p_kruskal = np.nan
        rows.append(
            {
                "feature": col,
                "anova_f": f_stat,
                "anova_p": p_anova,
                "kruskal_h": h_stat,
                "kruskal_p": p_kruskal,
            }
        )
    return pd.DataFrame(rows).set_index("feature")
 
 
def _mutual_info_numeric(
    df: pd.DataFrame, y: pd.Series, target_type: str, random_state: int
) -> pd.Series:
    """Information mutuelle pour variables numériques.
 
    Pourquoi (T4Rec):
    - Capture des relations non linéaires avec la cible, utile quand Pearson/Spearman
      sont faibles mais que le signal est présent.
 
    Paramètres
    - df: numériques
    - y: cible
    - target_type: 'binary'/'multiclass'/'continuous' (pilote le choix classif/régression)
    - random_state: reproductibilité
 
    Retour
    - Series nommée 'mutual_info' indexée par feature.
    """
    try:
        X = df.values
        if target_type in ("binary", "multiclass"):
            mi = mutual_info_classif(X, y, random_state=random_state)
        else:
            mi = mutual_info_regression(X, y, random_state=random_state)
        return pd.Series(mi, index=df.columns, name="mutual_info")
    except Exception:
        return pd.Series(
            [np.nan] * len(df.columns), index=df.columns, name="mutual_info"
        )
 
 
def _chi2_cramersv_for_categorical(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cramer's V (depuis chi²) pour features catégorielles vs cible.
 
    Pourquoi (T4Rec):
    - Mesure d'association robuste pour variables qualitatives; utile pour prioriser
      les embeddings catégoriels à conserver.
 
    Retour
    - DataFrame indexé 'feature' avec colonne 'cramers_v'.
    """
    rows = []
    y_str = y.fillna("__NA__").astype(str)
    for col in _pbar(df.columns, desc="Assoc cat (Cramer's V)"):
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
 
 
def _mutual_info_categorical(
    df: pd.DataFrame, y: pd.Series, target_type: str, random_state: int
) -> pd.Series:
    """Information mutuelle pour variables catégorielles.
 
    Pourquoi (T4Rec):
    - Détecte des dépendances non linéaires entre catégories et cible.
 
    Retour
    - Series nommée 'mutual_info' indexée par feature.
    """
    try:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X = enc.fit_transform(df.fillna("__NA__").astype(str))
        if target_type in ("binary", "multiclass"):
            mi = mutual_info_classif(X, y, random_state=random_state)
        else:
            mi = mutual_info_regression(X, y, random_state=random_state)
        return pd.Series(mi, index=df.columns, name="mutual_info")
    except Exception:
        return pd.Series(
            [np.nan] * len(df.columns), index=df.columns, name="mutual_info"
        )
 
 
def _numeric_multicollinearity_filter(
    df_numeric: pd.DataFrame,
    threshold: float = 0.9,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """Filtre glouton de multicolinéarité sur variables numériques.
 
    Pourquoi (T4Rec):
    - Deux variables très corrélées impliquent des embeddings et un coût redondants;
      en garder une seule réduit la charge mémoire/latence et le risque d'overfit.
 
    Paramètres
    - df_numeric: sous-DataFrame de variables numériques
    - threshold: seuil de corrélation absolue au-delà duquel on droppe la variable redondante
 
    Retour
    - (kept, pairs): kept = liste des variables conservées; pairs = [(keep, drop, |r|)]
    """
    if df_numeric.shape[1] <= 1:
        return list(df_numeric.columns), []
    corr = df_numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: set = set()
    drop_pairs: List[Tuple[str, str, float]] = []
    for col in _pbar(upper.columns, desc="Filtre corrélation (num)"):
        if col in to_drop:
            continue
        for row, v in upper[col].dropna().items():
            if v >= threshold and row not in to_drop:
                to_drop.add(row)
                drop_pairs.append((col, row, float(v)))
    kept = [c for c in df_numeric.columns if c not in to_drop]
    return kept, drop_pairs
 
 
def _rf_importance_with_cv(
    X_num: pd.DataFrame,
    X_cat: pd.DataFrame,
    y: pd.Series,
    target_type: str,
    n_splits: int,
    random_state: int,
    class_weight_balanced: bool = False,
    groups: Optional[pd.Series] = None,
) -> pd.Series:
    """Importance des features via RandomForest et validation croisée.
 
    Pourquoi (T4Rec):
    - Donne une vision «modèle» des variables utiles après prétraitements, et
      renforce la robustesse via agrégation sur plusieurs folds (Stratified/GroupKFold).
 
    Paramètres
    - X_num: features numériques
    - X_cat: features catégorielles
    - y: cible
    - target_type: binaire/multiclasse/continue
    - n_splits: nombre de folds
    - random_state: graine de reproductibilité
    - class_weight_balanced: pondération des classes (utile en cas de déséquilibre)
    - groups: groupes pour GroupKFold (ex: client_id) pour éviter la fuite de données
 
    Retour
    - Series des importances moyennes par feature (agrégée par feature d'origine)
    """
    # Define preprocessors
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(X_num.columns) if X_num is not None else []),
            (
                "cat",
                categorical_pipeline,
                list(X_cat.columns) if X_cat is not None else [],
            ),
        ],
        sparse_threshold=0.3,
    )
 
    if target_type in ("binary", "multiclass"):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
            class_weight=("balanced" if class_weight_balanced else None),
        )
        cv = (
            GroupKFold(n_splits=n_splits)
            if groups is not None
            else StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        )
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
        cv = (
            GroupKFold(n_splits=n_splits)
            if groups is not None
            else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        )
 
    feature_importance_acc: Dict[str, List[float]] = {
        f: [] for f in list(X_num.columns) + list(X_cat.columns)
    }
 
    split_iter = (
        cv.split(pd.DataFrame(index=y.index), y, groups=groups)
        if groups is not None
        else cv.split(pd.DataFrame(index=y.index), y)
    )
    for _ in _pbar(range(0), desc=""):  # no-op to satisfy linter when tqdm missing
        pass
    for train_valid in _pbar(split_iter, desc="RF-CV", total=n_splits):
        train_idx, valid_idx = train_valid
        Xn_tr = (
            X_num.iloc[train_idx]
            if X_num is not None and not X_num.empty
            else pd.DataFrame(index=train_idx)
        )
        Xn_va = (
            X_num.iloc[valid_idx]
            if X_num is not None and not X_num.empty
            else pd.DataFrame(index=valid_idx)
        )
        Xc_tr = (
            X_cat.iloc[train_idx]
            if X_cat is not None and not X_cat.empty
            else pd.DataFrame(index=train_idx)
        )
        Xc_va = (
            X_cat.iloc[valid_idx]
            if X_cat is not None and not X_cat.empty
            else pd.DataFrame(index=valid_idx)
        )
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
 
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(pd.concat([Xn_tr, Xc_tr], axis=1), y_tr)
 
        preproc: ColumnTransformer = pipe.named_steps["pre"]
        model_fitted = pipe.named_steps["model"]
 
        num_feats = list(X_num.columns) if X_num is not None else []
        cat_feats = list(X_cat.columns) if X_cat is not None else []
        try:
            ct_feature_names: List[str] = []
            if num_feats:
                ct_feature_names.extend(num_feats)
            if cat_feats:
                onehot: OneHotEncoder = preproc.named_transformers_["cat"].named_steps[
                    "onehot"
                ]
                onehot_names = list(onehot.get_feature_names_out(cat_feats))
                ct_feature_names.extend(onehot_names)
        except Exception:
            continue
 
        try:
            importances = getattr(model_fitted, "feature_importances_", None)
            if importances is None:
                continue
            importances = np.asarray(importances)
        except Exception:
            continue
 
        pointer = 0
        for f in num_feats:
            if pointer >= len(importances):
                break
            feature_importance_acc[f].append(float(importances[pointer]))
            pointer += 1
        if cat_feats:
            onehot: OneHotEncoder = preproc.named_transformers_["cat"].named_steps[
                "onehot"
            ]
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
) -> Dict[str, object]:
    """Pipeline complète de sélection de variables pour T4Rec.
 
    Pourquoi (T4Rec):
    - Limiter le nombre de features pour contenir le coût des embeddings et améliorer la stabilité.
    - Privilégier des variables porteuses d'un signal temporel fort et réduire la redondance.
 
    Paramètres
    - dataset_name: nom du dataset Dataiku (si None, utiliser df)
    - df: DataFrame local (si None, charger dataset_name)
    - target_col: nom de la cible
    - top_k_sequence: nombre max de variables numériques retenues avant cap global
    - top_k_categorical: nombre max de variables catégorielles retenues avant cap global
    - total_feature_cap: plafond global (toutes variables confondues), ex. 15
    - sample_size: taille d'échantillon pour accélérer l'analyse
    - random_state: graine de reproductibilité
    - corr_threshold: seuil de corrélation absolue pour dropper (multicolinéarité)
    - report_output_dir: dossier de sortie pour le CSV de rapport
    - report_dataset: nom du dataset Dataiku pour écrire le rapport
    - verbose: activer les logs détaillés
    - group_col: colonne de groupe (ex. client_id) pour GroupKFold
    - class_weight_balanced: activer la pondération des classes pour RF (classification)
    - progress: afficher des barres de progression (tqdm) si disponible
 
    Retour
    - dict avec:
      - 'target_type': type de cible
      - 'numeric_candidates' / 'categorical_candidates': candidats initiaux
      - 'selected_sequence_cols' / 'selected_categorical_cols': sélection finale
      - 'report': DataFrame du rapport détaillé (scores, flags, raisons)
    """
    global _PROGRESS
    _PROGRESS = bool(progress)
    if verbose:
        logger.info("[FeatureSelection] Starting feature selection...")
    t_global0 = time.time()
 
    if df is None:
        if dataset_name is None:
            raise ValueError("Either df or dataset_name must be provided")
        if not exists_dataiku:
            raise RuntimeError(
                "Dataiku environment not available; please pass a DataFrame df instead."
            )
        adapter = DataikuAdapter()
        df = adapter.load_dataset(dataset_name)
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state)
    else:
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state)
 
    if verbose:
        logger.info(
            f"[FeatureSelection] Data loaded: {df.shape[0]:,} rows × {df.shape[1]} cols (sample_size={sample_size})"
        )
 
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame")
 
    y = df[target_col]
    X = df.drop(columns=[target_col])
    target_type = _detect_target_type(y)
    if verbose:
        logger.info(
            f"[FeatureSelection] Target '{target_col}' detected as: {target_type}"
        )
 
    numeric_cols = _safe_numeric_cols(X)
    categorical_cols = _safe_categorical_cols(X, exclude=numeric_cols)
    if verbose:
        logger.info(
            f"[FeatureSelection] Candidates — numeric: {len(numeric_cols)}, categorical: {len(categorical_cols)}"
        )
 
    reports: List[pd.DataFrame] = []
    t_assoc0 = time.time()
    if target_type == "continuous":
        num_assoc = _pearson_spearman_for_numeric(X[numeric_cols], y)
        num_assoc["mutual_info"] = _mutual_info_numeric(
            X[numeric_cols], y, target_type, random_state
        )
        reports.append(num_assoc.add_prefix("num_"))
        _ = _anova_kruskal_for_numeric_vs_categorical(X[numeric_cols], y)
    else:
        num_assoc = _pearson_spearman_for_numeric(
            X[numeric_cols], y.astype("category").cat.codes
        )
        num_assoc["mutual_info"] = _mutual_info_numeric(
            X[numeric_cols], y, target_type, random_state
        )
        reports.append(num_assoc.add_prefix("num_"))
        cat_assoc1 = _chi2_cramersv_for_categorical(X[categorical_cols], y)
        cat_assoc2 = _mutual_info_categorical(
            X[categorical_cols], y, target_type, random_state
        )
        cat_assoc = cat_assoc1.join(cat_assoc2, how="outer")
        reports.append(cat_assoc.add_prefix("cat_"))
    if verbose:
        logger.info(
            f"[FeatureSelection] Associations computed in {time.time() - t_assoc0:.2f}s"
        )
 
    # Multicollinearity (numeric) with details
    t_mc0 = time.time()
    if len(numeric_cols) > 0:
        kept_numeric, drop_pairs = _numeric_multicollinearity_filter(
            X[numeric_cols], threshold=corr_threshold
        )
        dropped_numeric = [d for _, d, _ in drop_pairs]
        if verbose:
            logger.info(
                f"[FeatureSelection] Multicollinearity — kept: {len(kept_numeric)}, dropped: {len(dropped_numeric)} (threshold={corr_threshold}) in {time.time() - t_mc0:.2f}s"
            )
            for i, (keep_f, drop_f, r) in enumerate(drop_pairs[:10]):
                logger.info(
                    f"  drop '{drop_f}' due to high corr with '{keep_f}' (|r|={r:.3f})"
                )
    else:
        kept_numeric, drop_pairs = [], []
        dropped_numeric = []
 
    if verbose:
        logger.info("[FeatureSelection] Computing model-based importances (CV)...")
    t_rf0 = time.time()
    groups = (
        df[group_col] if (group_col is not None and group_col in df.columns) else None
    )
    imp_series = _rf_importance_with_cv(
        X_num=X[kept_numeric] if kept_numeric else pd.DataFrame(index=X.index),
        X_cat=X[categorical_cols] if categorical_cols else pd.DataFrame(index=X.index),
        y=y,
        target_type=target_type,
        n_splits=3,
        random_state=random_state,
        class_weight_balanced=class_weight_balanced,
        groups=groups,
    )
    if verbose:
        logger.info(
            f"[FeatureSelection] RF-CV importances computed in {time.time() - t_rf0:.2f}s"
        )
 
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
    for c in scores.columns:
        mx = scores[c].max()
        scores[c] = scores[c] / mx if mx and np.isfinite(mx) and mx > 0 else scores[c]
    scores["model_importance"] = imp_series.reindex(scores.index).fillna(0.0)
    mi_max = scores["model_importance"].max()
    scores["model_importance"] = (
        scores["model_importance"] / mi_max
        if mi_max and mi_max > 0
        else scores["model_importance"]
    )
 
    if target_type in ("binary", "multiclass"):
        weights = {
            "num_pearson_abs": 0.2,
            "num_spearman_abs": 0.2,
            "num_mutual_info": 0.3,
            "cat_cramers_v": 0.2,
            "cat_mutual_info": 0.3,
            "model_importance": 0.4,
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
    if verbose:
        logger.info(
            f"[FeatureSelection] Composite weights used (present only): { {k: round(weights[k], 3) for k in present} }"
        )
 
    scores["composite_score"] = sum(
        scores.get(k, pd.Series(0.0, index=scores.index)) * weights[k] for k in present
    )
 
    # Flags for reasons
    reasons = pd.Series("", index=scores.index)
    if dropped_numeric:
        for keep_f, drop_f, r in drop_pairs:
            reasons[drop_f] = f"dropped_corr_with:{keep_f}|r={r:.3f}"
 
    numeric_ranked = (
        scores.loc[kept_numeric]
        .sort_values("composite_score", ascending=False)
        .head(top_k_sequence)
        .index.tolist()
        if len(kept_numeric) > 0
        else []
    )
    categorical_ranked = (
        scores.loc[categorical_cols]
        .sort_values("composite_score", ascending=False)
        .head(top_k_categorical)
        .index.tolist()
        if len(categorical_cols) > 0
        else []
    )
 
    if total_feature_cap is not None:
        combined = pd.concat(
            [
                scores.loc[numeric_ranked, ["composite_score"]].assign(_type="numeric"),
                scores.loc[categorical_ranked, ["composite_score"]].assign(
                    _type="categorical"
                ),
            ]
        ).sort_values("composite_score", ascending=False)
        combined = combined.head(total_feature_cap)
        numeric_ranked = [
            f for f, row in combined.iterrows() if row["_type"] == "numeric"
        ]
        categorical_ranked = [
            f for f, row in combined.iterrows() if row["_type"] == "categorical"
        ]
 
    selected_final_set = set(numeric_ranked + categorical_ranked)
    if verbose:
        logger.info(
            f"[FeatureSelection] Selected — numeric: {len(numeric_ranked)}, categorical: {len(categorical_ranked)}, total: {len(selected_final_set)}"
        )
        logger.info(
            f"  numeric: {numeric_ranked[:5]}{'...' if len(numeric_ranked) > 5 else ''}"
        )
        logger.info(
            f"  categorical: {categorical_ranked[:5]}{'...' if len(categorical_ranked) > 5 else ''}"
        )
 
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
    # Add kept after corr filter flag for numeric
    report_df["kept_after_corr_filter"] = False
    if len(kept_numeric) > 0:
        report_df.loc[kept_numeric, "kept_after_corr_filter"] = True
    # Drop reasons and final selection flags
    report_df["drop_reason"] = reasons.reindex(report_df.index).fillna("")
    report_df["selected_final"] = report_df.index.isin(selected_final_set)
    # For non-selected without explicit reason, mark as not in top/cap
    non_selected_mask = (~report_df["selected_final"]) & (
        report_df["drop_reason"] == ""
    )
    report_df.loc[non_selected_mask, "drop_reason"] = "below_topK_or_cap"
 
    report_df = report_df.sort_values(
        ["feature_type", "selected_final", "composite_score"],
        ascending=[True, False, False],
    )
 
    if report_output_dir:
        try:
            import os
 
            os.makedirs(report_output_dir, exist_ok=True)
            name = dataset_name or "dataframe"
            report_path = os.path.join(
                report_output_dir, f"feature_selection_report_{name}.csv"
            )
            report_df.to_csv(report_path, index=True)
            if verbose:
                logger.info(f"[FeatureSelection] Saved report to {report_path}")
        except Exception as e:
            logger.warning(f"[FeatureSelection] Could not save local report: {e}")
 
    if report_dataset and exists_dataiku:
        try:
            adapter = DataikuAdapter()
            adapter.write_dataframe(
                report_df.reset_index().rename(columns={"index": "feature"}),
                report_dataset,
            )
            if verbose:
                logger.info(
                    f"[FeatureSelection] Saved report to Dataiku dataset {report_dataset}"
                )
        except Exception as e:
            logger.warning(
                f"[FeatureSelection] Could not save to Dataiku dataset '{report_dataset}': {e}"
            )
 
    if verbose:
        logger.info(f"[FeatureSelection] Done in {time.time() - t_global0:.2f}s")
 
    return {
        "target_type": target_type,
        "numeric_candidates": numeric_cols,
        "categorical_candidates": categorical_cols,
        "selected_sequence_cols": numeric_ranked,
        "selected_categorical_cols": categorical_ranked,
        "report": report_df,
    }
 
 
