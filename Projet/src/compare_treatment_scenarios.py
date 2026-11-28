"""
Comparaison empirique des scénarios de traitement 
Scénarios (les seuls compatibles avec les décisions : pas de winsorisation, pas d'indicateurs) :
    S0_baseline        : InvalidToNaN + OutlierHandler(log_cols=[]) -> AUCUN log1p (référence)
    S1_log1p           : InvalidToNaN + OutlierHandler (log1p sur LOG1P_COLS)
    S4_sans_Engagement : S1 sans EngagementValue (sensibilité : dépendance / fuite possible)
Modèles : logreg (sensible à l'échelle) et hgb (invariant aux transformations monotones : garde-fou).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import NUMERIC_COLS, DataQualityWarning, InvalidToNaN, OutlierHandler

SCENARIOS = {
    "S0_baseline":        dict(log_cols=[]),
    "S1_log1p":           dict(log_cols=None),
    "S4_sans_Engagement": dict(log_cols=None, drop=("EngagementValue",)),
}
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


def build_pipeline(model, cat_cols=(), passthrough_cols=(), log_cols=None, drop=(), random_state=0):
    num = [c for c in NUMERIC_COLS if c not in drop]
    if cat_cols:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=(model != "hgb")
        )
        pre = [("cat", encoder, list(cat_cols))]
    else:
        pre = []
    if model == "logreg":
        pre.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                                     ("sc", StandardScaler())]), num))
        clf = LogisticRegression(max_iter=3000, random_state=random_state)
    else:   # hgb : gère les NaN nativement, pas d'imputer ni de scaler
        pre.append(("num", "passthrough", num))
        clf = HistGradientBoostingClassifier(random_state=random_state)
    if passthrough_cols:
        pre.append(("pass", "passthrough", list(passthrough_cols)))
    return Pipeline([("invalid", InvalidToNaN()), ("outliers", OutlierHandler(log_cols=log_cols)),
                     ("prep", ColumnTransformer(pre, remainder="drop")), ("clf", clf)])


def compare_scenarios(X_train, y_train, cat_cols=(), passthrough_cols=(), models=("logreg", "hgb"),
                      n_splits=5, n_repeats=3, random_state=0, n_jobs=-1):
    """Retourne (summary, paired). `paired` = différences par fold vs S0 (même fold, même modèle)."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        for sc, cfg in SCENARIOS.items():
            for m in models:
                pipe = build_pipeline(m, cat_cols, passthrough_cols, random_state=random_state, **cfg)
                results[(sc, m)] = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=n_jobs)

    summary = pd.DataFrame({k: {"roc_auc_moy": r["test_roc_auc"].mean(), "roc_auc_std": r["test_roc_auc"].std(),
                                "pr_auc_moy": r["test_pr_auc"].mean(), "pr_auc_std": r["test_pr_auc"].std()}
                            for k, r in results.items()}).T
    summary.index.names = ["scénario", "modèle"]

    rows = []
    for m in models:
        for sc in ("S1_log1p", "S4_sans_Engagement"):
            ref = "S0_baseline" if sc == "S1_log1p" else "S1_log1p"      # S4 est comparé à S1
            for metric in ("roc_auc", "pr_auc"):
                d = results[(sc, m)][f"test_{metric}"] - results[(ref, m)][f"test_{metric}"]
                rows.append({"modèle": m, "scénario": sc, "référence": ref, "métrique": metric,
                             "delta_moyen": d.mean(), "delta_std": d.std(), "part_folds_meilleurs": float((d > 0).mean())})
    paired = pd.DataFrame(rows)
    # Règle de décision du notebook : delta > bruit ET >= 80 % de folds meilleurs
    paired["utile"] = (paired["delta_moyen"] > paired["delta_std"]) & (paired["part_folds_meilleurs"] >= 0.8)
    return summary, paired
if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train_final.csv"   # Projet/data/train_final.csv
    TARGET = "Purchased"

    df = pd.read_csv(DATA_PATH)
    X_train = df.drop(columns=[TARGET])
    y_train = df[TARGET]

    summary, paired = compare_scenarios(
        X_train,
        y_train,
        cat_cols=["CampaignType", "CustomerType", "DeviceOS", "BrowserCode", "GeoZone", "EntryChannel"],
        passthrough_cols=["Weekend"],
        random_state=42,
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n=== SUMMARY ===")
    print(summary.round(4))
    print("\n=== PAIRED COMPARISON ===")
    print(paired.round(4).to_string())