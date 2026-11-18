import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessing_pipeline():
    """
    Crée et retourne le ColumnTransformer pour le prétraitement des données ShopPulse.
    - Variables numériques : StandardScaler
    - Variables catégorielles : OneHotEncoder
    """
    
    # 1. Définition des colonnes (b nafs les noms li f train.csv)
    num_cols = [
        'AdminPages', 'AdminTimeSec', 'InfoPages', 'InfoTimeSec',
        'CatalogPages', 'CatalogTimeSec', 'BounceRate', 'ExitRate',
        'EngagementValue', 'PromoProximity'
    ]
    
    cat_cols = [
        'CampaignType', 'CustomerType', 'DeviceOS', 'BrowserCode',
        'GeoZone', 'EntryChannel', 'Weekend'
    ]
    
    # 2. Construction du ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop' # Kay-supprimi ay colonne okhra ma mchmoulaش
    )
    
    return preprocessor