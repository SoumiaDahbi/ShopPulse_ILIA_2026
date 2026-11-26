from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(num_cols, cat_cols, bin_cols):
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if bin_cols:
        transformers.append(("bin", "passthrough", bin_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")
