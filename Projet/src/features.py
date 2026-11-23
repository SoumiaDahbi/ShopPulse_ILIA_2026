
import numpy as np
from sklearn.preprocessing import FunctionTransformer

CAT_COLS = ["CampaignType", "DeviceOS", "BrowserCode", "GeoZone", "EntryChannel", "CustomerType"]

def ajouter(X, niveau=2):
    X = X.copy()

    # catégories :  codes entiers -> texte (pour le One-Hot)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)

    if niveau >= 1:
        X["total_pages"] = X["AdminPages"] + X["InfoPages"] + X["CatalogPages"]
        X["total_time"]  = X["AdminTimeSec"] + X["InfoTimeSec"] + X["CatalogTimeSec"]
        pages = X["total_pages"]
        temps = X["total_time"]

    if niveau >= 2:
        X["time_per_page"] = temps / pages.replace(0, np.nan) # division par zéro -> NaN
        X["catalog_share"] = X["CatalogPages"] / pages.replace(0, np.nan)
        X["time_per_page_catalog"] = X["CatalogTimeSec"] / X["CatalogPages"].replace(0, np.nan)
        X["engagement_par_page"] = X["EngagementValue"] / pages.replace(0, np.nan)
        X["exit_sur_bounce"] = X["ExitRate"] / X["BounceRate"].replace(0, np.nan)
    return X

def construction(niveau=2):
    return FunctionTransformer(ajouter, kw_args={"niveau": niveau})