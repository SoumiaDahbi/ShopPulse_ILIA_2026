import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from Projet.src.preprocessing import build_preprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Réentraînement sur tout le Train et évaluation finale unique sur Test."
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Exécute l'évaluation finale unique sur le jeu de test.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("Projet/data/train_final.csv"),
        help="Chemin vers le CSV d'entraînement.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("Projet/data/test_interne.csv"),
        help="Chemin vers le CSV de test.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="Purchased",
        help="Nom de la colonne cible.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("Projet/models/final_pipeline.joblib"),
        help="Chemin de sauvegarde du pipeline entraîné.",
    )
    return parser.parse_args()


def infer_column_groups(X: pd.DataFrame):
    """Sépare les colonnes en numériques continues, catégorielles et binaires."""
    bin_cols = []
    cat_cols = []
    num_cols = []

    for col in X.columns:
        s = X[col]
        unique_non_null = s.dropna().unique()
        if pd.api.types.is_bool_dtype(s) or len(unique_non_null) <= 2:
            bin_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    return num_cols, cat_cols, bin_cols


def build_training_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols, cat_cols, bin_cols = infer_column_groups(X)
    preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols, bin_cols=bin_cols)

    # Remplace RandomForestClassifier par le modèle final retenu par votre équipe si différent
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    args = parse_args()

    if not args.train_path.exists():
        raise FileNotFoundError(f"Fichier train introuvable : {args.train_path}")

    train_df = pd.read_csv(args.train_path)
    if args.target_col not in train_df.columns:
        raise ValueError(f"Colonne cible '{args.target_col}' absente de {args.train_path}")

    X_train = train_df.drop(columns=[args.target_col])
    y_train = train_df[args.target_col]

    pipeline = build_training_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)
    print(f"Pipeline entraîné sur tout le Train et sauvegardé dans : {args.model_out}")

    if args.final:
        if not args.test_path.exists():
            raise FileNotFoundError(f"Fichier test introuvable : {args.test_path}")

        test_df = pd.read_csv(args.test_path)
        X_test = test_df.drop(columns=[args.target_col])
        y_test = test_df[args.target_col]

        y_pred = pipeline.predict(X_test)
        y_proba = (
            pipeline.predict_proba(X_test)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else None
        )

        print("\n=== ÉVALUATION FINALE UNIQUE SUR TEST ===")
        print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
        if y_proba is not None:
            print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
