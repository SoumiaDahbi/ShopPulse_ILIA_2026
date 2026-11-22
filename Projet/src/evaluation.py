# manipuler les donnees sos forme de dataframe 
import pandas as pd

# clone : creer une copie independante de modele à chaque fold 
from sklearn.base import clone
# startifiedkfold divise les donnees en 5 folds en conservant les memes proportions des classes 
from sklearn.model_selection import StratifiedKFold
# les metriques d'evaluation
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


def evaluate_cv(model, X, y):
    """
    Évalue un modèle avec (la validation croisee) Stratified K-Fold (5 folds).

    Paramètres :
        model : modèle de classification à évaluer
        X     : variables explicatives
        y     : variable cible

    Retour :
        DataFrame contenant les résultats de chaque fold
    """

    # 1. Créer la stratégie de validation croisée (train_final)
    skf = StratifiedKFold(
        # 5 folds = 4 pour l'entrainement et 1 pour l'evaluation
        n_splits=5,
        # Mélanger les données avant la division pour éviter de dépendre
        # de leur ordre initial        
        shuffle=True,
        random_state=42
    )

    # 2. Stocker les résultats de chaque iteration  dans une liste
    results = []

    # 3. Parcourir les 5 folds
    # skf.split(x,y) divise les donnees en 5 folds 
    # et retourne les indices d'entrainement train_idx 
    # et de validation val_idx 
    # enumerate permet de numeroter les iterations a partir de 1 
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):

        # Séparer les données d'entraînement et de validation
        # recuperer les donnees avec les indices 
        if hasattr(X, "iloc"):
            # si X est un DataFrame, on utilise .iloc pour accéder aux lignes par index
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            # si X est un tableau numpy, on utilise l'indexation directe
            X_train, X_val = X[train_idx], X[val_idx]

        if hasattr(y, "iloc"):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_train, y_val = y[train_idx], y[val_idx]

        # 4. Créer une copie du modèle pour chaque fold doit commencer avec un modele neuf 
        model_fold = clone(model)

        # 5. Entraîner le modèle
        model_fold.fit(X_train, y_train)

        # 6. Prédire les classes
        y_pred = model_fold.predict(X_val)

        # 7. Obtenir les scores pour calculer Average Precision
        # predict_proba retourne les probabilites de chaque classe
        # si le modele possede predict_proba()
        if hasattr(model_fold, "predict_proba"):
            y_scores = model_fold.predict_proba(X_val)[:, 1]
            # [:, 1] ( toutes les lignes et le colonne 1) pour prendre
            #  la probabilite de la classe positive (1) pour le calcul de l'AP
        else:
            # decision_function retourne les scores de decision pour 
            # chaque classe qui permet de classer les observations 
            y_scores = model_fold.decision_function(X_val)

        # 8. Calculer les métriques
        results.append({
            "Fold": fold,
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "Recall": recall_score(y_val, y_pred, zero_division=0),
            "F1": f1_score(y_val, y_pred, zero_division=0),
            "AP": average_precision_score(y_val, y_scores)
        })

    # 9. Retourner les résultats
    return pd.DataFrame(results)
