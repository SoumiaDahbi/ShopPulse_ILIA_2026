"""
Source des décisions : notebook 03_outliers.ipynb (statut de toutes les décisions : « à valider (CV) »).

DÉCISIONS APPLIQUÉES
    - AUCUNE suppression d'observation, aucune valeur extrême traitée comme une erreur.
    - Pas de winsorisation / clipping, pas de CeilingIndicator, pas d'indicateur >0.
    - log1p uniquement sur LOG1P_COLS ; BounceRate, ExitRate, PromoProximity restent telles quelles.
    - Seules les valeurs IMPOSSIBLES selon le dictionnaire de données (HARD_CONSTRAINTS) deviennent NaN.
    - Imputation = filet de sécurité (NaN existants / créés par le feature engineering / valeurs impossibles),
      paramètres appris sur le train de chaque fold.

CONTRAT
    Entrée  : pandas.DataFrame contenant les colonnes de NUMERIC_COLS, en échelle BRUTE (non transformées).
              Les autres colonnes (catégorielles, Weekend, ...) traversent sans modification.
    Sortie  : pandas.DataFrame, mêmes lignes et même index (rien n'est supprimé), log1p appliqué sur LOG1P_COLS,
              colonnes numériques sans NaN si impute=True. (Avec impute=True l'ordre des colonnes peut changer :
              colonnes imputées d'abord, puis les autres ; les noms sont conservés.)

ORDRE OBLIGATOIRE DANS LE PIPELINE FINAL
    InvalidToNaN -> feature engineering (ratios, safe_divide) -> OutlierHandler -> SimpleImputer -> encodage/scaler -> modèle
    - Les ratios se calculent sur l'échelle brute, donc AVANT log1p (un ratio de logs n'a pas de sens).
    - InvalidToNaN passe AVANT log1p pour qu'une valeur impossible ne fasse pas échouer safe_log1p.
    - L'imputer passe APRÈS log1p et APRÈS le feature engineering, et doit couvrir les colonnes dérivées.

"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "DataQualityWarning", "DECISIONS", "NUMERIC_COLS", "LOG1P_COLS", "HARD_CONSTRAINTS",
    "safe_log1p", "safe_divide", "InvalidToNaN", "OutlierHandler"
]


class DataQualityWarning(UserWarning):
    """Signal de qualité / dérive des données. Ne signifie jamais qu'un extrême plausible a été modifié."""

DECISIONS = {
    "AdminPages":      dict(outliers="Conserver (aucun traitement)",          transfo="log1p", winsor_test=False, statut="à valider (CV)"),
    "AdminTimeSec":    dict(outliers="Conserver (aucun traitement)",          transfo="log1p", winsor_test=False, statut="à valider (CV)"),
    "InfoPages":       dict(outliers="Conserver : Tukey dégénéré (IQR=0)",    transfo="log1p", winsor_test=False, statut="à valider (CV)"),
    "InfoTimeSec":     dict(outliers="Conserver : Tukey dégénéré (IQR=0)",    transfo="log1p", winsor_test=False, statut="à valider (CV)"),
    "CatalogPages":    dict(outliers="Conserver ; extrêmes sans suppression", transfo="log1p", winsor_test=False, statut="à valider (CV)"),
    "CatalogTimeSec":  dict(outliers="Conserver ; extrêmes sans suppression", transfo="log1p", winsor_test=False, statut="à valider (CV)"),
    "BounceRate":      dict(outliers="Conserver telle quelle",                transfo=None,    winsor_test=False, statut="à valider (CV)"),
    "ExitRate":        dict(outliers="Conserver telle quelle",                transfo=None,    winsor_test=False, statut="à valider (CV)"),
    "EngagementValue": dict(outliers="Conserver",                             transfo="log1p", winsor_test=False, statut="à valider (CV)"),
    "PromoProximity":  dict(outliers="Conserver telle quelle",                transfo=None,    winsor_test=False, statut="à valider (CV)"),
}
NUMERIC_COLS = list(DECISIONS)
LOG1P_COLS = [c for c, d in DECISIONS.items() if d["transfo"] == "log1p"]

# Contraintes DURES issues du DICTIONNAIRE DE DONNÉES (min/max logiques par construction ; jamais un seuil
# statistique de type Tukey). Seules les valeurs qui les violent peuvent devenir NaN ; toute valeur dans ces
# bornes est conservée, même extrême. max=None : pas de borne supérieure dans le dictionnaire.
HARD_CONSTRAINTS = {
    "AdminPages":      dict(min=0, max=27,   integer=True),
    "InfoPages":       dict(min=0, max=24,   integer=True),
    "CatalogPages":    dict(min=0, max=650,  integer=True),
    "AdminTimeSec":    dict(min=0, max=None, integer=False),
    "InfoTimeSec":     dict(min=0, max=None, integer=False),
    "CatalogTimeSec":  dict(min=0, max=None, integer=False),
    "BounceRate":      dict(min=0, max=0.2,  integer=False),
    "ExitRate":        dict(min=0, max=0.2,  integer=False),
    "EngagementValue": dict(min=0, max=None, integer=False),
    "PromoProximity":  dict(min=0, max=1,    integer=False),
}



def safe_log1p(X):
    """log1p exigeant x > -1 ; échoue explicitement (jamais de NaN silencieux). Les NaN existants sont conservés."""
    arr = np.asarray(X, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and finite.min() <= -1:
        raise ValueError("log1p impossible : valeur <= -1 détectée (placer InvalidToNaN avant OutlierHandler)")
    return np.log1p(arr)


def safe_divide(num, den, default=0.0):
    """Division sans NaN ni inf : renvoie `default` là où den == 0 ou une valeur n'est pas finie.
    Ex. : safe_divide(AdminTimeSec, AdminPages) -> secondes par page (0 page -> 0 ; décision métier à documenter)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out = np.full(np.broadcast(num, den).shape, float(default))
    np.divide(num, den, out=out, where=ok)
    return out


# 3. InvalidToNaN : valeurs IMPOSSIBLES uniquement
class InvalidToNaN(BaseEstimator, TransformerMixin):
    """Met en NaN les valeurs violant une contrainte DURE du dictionnaire (ou non finies : ±inf).

    Sans état appris, ne supprime aucune ligne, travaille sur une copie. Après transform :
    `n_invalid_` = {colonne: nombre de valeurs mises en NaN} ; un DataQualityWarning est émis si > 0,
    pour qu'une dérive des données ne soit jamais masquée.

    Paramètres
        constraints : dict {col: {min, max, integer}} ou None -> HARD_CONSTRAINTS.
        warn        : émettre l'avertissement (True par défaut).
    """

    def __init__(self, constraints=None, warn=True):
        self.constraints = constraints
        self.warn = warn

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.constraints_ = dict(HARD_CONSTRAINTS if self.constraints is None else self.constraints)
        absent = [c for c in self.constraints_ if c not in X.columns]
        if absent and self.warn:
            warnings.warn(f"InvalidToNaN : colonnes contraintes absentes (ignorées) : {absent}",
                          DataQualityWarning, stacklevel=2)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "constraints_")
        X = pd.DataFrame(X).copy()
        counts = {}
        for c, r in self.constraints_.items():
            if c not in X.columns:
                continue
            s = pd.to_numeric(X[c], errors="coerce").astype(float)
            bad = s.notna() & ~np.isfinite(s)                                   # ±inf
            if r.get("min") is not None:
                bad |= s < r["min"]
            if r.get("max") is not None:
                bad |= s > r["max"]
            if r.get("integer"):
                bad |= s.notna() & np.isfinite(s) & (s % 1 != 0)
            bad = bad.fillna(False)
            if bad.any():
                counts[c] = int(bad.sum())
            X[c] = s.mask(bad)
        self.n_invalid_ = counts
        if counts and self.warn:
            warnings.warn(f"InvalidToNaN : valeurs impossibles mises en NaN {counts}",
                          DataQualityWarning, stacklevel=2)
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "constraints_")
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)



# 4. OutlierHandler : 
class OutlierHandler(BaseEstimator, TransformerMixin):

    def __init__(self, log_cols=None, monitor=True, strict=True):
        self.log_cols = log_cols
        self.monitor = monitor
        self.strict = strict

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = list(LOG1P_COLS if self.log_cols is None else self.log_cols)
        missing = [c for c in cols if c not in X.columns]
        if missing and self.strict:
            raise KeyError(f"Colonnes absentes : {missing}")
        self.log_cols_ = [c for c in cols if c in X.columns]
        self.train_max_ = {c: float(pd.to_numeric(X[c], errors="coerce").max())
                           for c in X.columns if pd.api.types.is_numeric_dtype(X[c])}
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "log_cols_")
        X = pd.DataFrame(X).copy()
        missing = [c for c in self.log_cols_ if c not in X.columns]
        if missing:
            raise KeyError(f"Colonnes absentes à la transformation : {missing}")
        if self.monitor:
            drift = {}
            for c, mx in self.train_max_.items():
                if c in X.columns and pd.notna(mx):
                    n = int((pd.to_numeric(X[c], errors="coerce") > mx).sum())
                    if n:
                        drift[c] = n
            self.n_above_train_max_ = drift
            if drift:
                warnings.warn(f"OutlierHandler : valeurs > max du train (conservées, non modifiées) {drift}",
                              DataQualityWarning, stacklevel=2)
        for c in self.log_cols_:
            X[c] = safe_log1p(pd.to_numeric(X[c], errors="coerce").to_numpy())
        return X

    def inverse_transform(self, X):
        check_is_fitted(self, "log_cols_")
        X = pd.DataFrame(X).copy()
        for c in self.log_cols_:
            X[c] = np.expm1(X[c].to_numpy(dtype=float))
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "log_cols_")
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)


