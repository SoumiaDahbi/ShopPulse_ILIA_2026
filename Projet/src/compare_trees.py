import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,HistGradientBoostingClassifier)
from src.features import construction

train = pd.read_csv("data/train_final.csv")
val = pd.read_csv("data/validation.csv")
X, y = train.drop(columns="Purchased"), train["Purchased"]
Xv, yv = val.drop(columns="Purchased"), val["Purchased"]

def prep():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01), make_column_selector(dtype_exclude="number")),
    ], sparse_threshold=0)   # sortie dense (nécessaire pour HistGradientBoosting)

modeles = {
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lignes = []
for niveau in [0, 1, 2]:
    for nom, modele in modeles.items():
        pipe = Pipeline([("ajout", construction(niveau)), ("prep", prep()), ("modele", modele)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
        pipe.fit(X, y)
        ap_val = average_precision_score(yv, pipe.predict_proba(Xv)[:, 1])
        lignes.append(f"| {niveau} | {nom} | {scores.mean():.4f} ± {scores.std():.4f} | {ap_val:.4f} |")
        print(lignes[-1])

os.makedirs("reports", exist_ok=True)
with open("reports/resultats_arbres.md", "w", encoding="utf-8") as f:
    f.write("| Niveau features | Modèle | AP en CV | AP validation |\n|---|---|---|---|\n" + "\n".join(lignes))