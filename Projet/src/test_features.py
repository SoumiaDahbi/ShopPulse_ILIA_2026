import numpy as np
import pandas as pd
from src.features import ajouter

def petit_df():
    return pd.DataFrame({
        "AdminPages": [0, 2], "InfoPages": [0, 1], "CatalogPages": [0, 3],
        "AdminTimeSec": [0.0, 10.0], "InfoTimeSec": [0.0, 5.0], "CatalogTimeSec": [0.0, 15.0],
        "EngagementValue": [0.0, 6.0], "ExitRate": [0.1, 0.2], "BounceRate": [0.0, 0.1],
        "CampaignType": ["Regular", "Holiday"], "DeviceOS": [1, 2], "BrowserCode": [1, 2],
        "GeoZone": [1, 2], "EntryChannel": [1, 2], "CustomerType": ["Returning", "New"],
    })

def test_ratios_et_categories():
    Z = ajouter(petit_df(), niveau=2)
    assert np.isnan(Z.loc[0, "time_per_page"])       # 0 page -> NaN
    assert np.isnan(Z.loc[0, "exit_sur_bounce"])     # BounceRate = 0 -> NaN
    assert Z.loc[1, "total_pages"] == 6
    assert Z.loc[1, "time_per_page"] == 5.0
    assert Z.loc[1, "DeviceOS"] == "2"               # code converti en catégorie

def test_niveaux():
    X = petit_df()
    assert "total_pages" not in ajouter(X, niveau=0).columns
    assert "total_pages" in ajouter(X, niveau=1).columns
    assert "time_per_page" not in ajouter(X, niveau=1).columns