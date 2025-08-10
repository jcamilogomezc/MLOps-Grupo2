from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

def _clean_category_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.lower()
    return s.fillna("desconocido").replace({"nan": "desconocido"})

class CategoryCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        # Para modelos viejos que no guardaron este atributo, usamos default
        self.columns = columns

    def __setstate__(self, state):
        # Al unpickle, asegúrate de tener 'columns'
        self.__dict__.update(state)
        if "columns" not in self.__dict__:
            self.columns = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Si el modelo viejo no tenía atributo, getattr evita el error
        cols = getattr(self, "columns", None)
        if cols is None:
            cols = [
                c for c in X.columns
                if X[c].dtype == "object" or pd.api.types.is_string_dtype(X[c])
            ]
        for c in cols:
            X[c] = _clean_category_series(X[c])
        return X