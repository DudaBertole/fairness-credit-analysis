import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_categorical_encoder(method='none'):
    if method == 'one-hot':
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    elif method == 'label':
        return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    elif method == 'none' or method is None:
        return 'passthrough'

def get_numeric_scaler(method='none'):
    if method == 'standardization':
        return StandardScaler()
    elif method == 'none' or method is None:
        return 'passthrough'

def build_preprocessor(numeric_cols, categorical_cols, scaling_method=None, encoding_method=None):
    """
    Builds a ColumnTransformer that:
    - applies scaling to numerical features
    - applies encoding to categorical features
    - drops any columns not explicitly listed

    Args:
        numeric_cols (list): List of numerical feature names.
        categorical_cols (list): List of categorical feature names.
        scaling_method (str): 'standardization' or 'none'.
        encoding_method (str): 'one-hot', 'label', or 'none'.

    Returns:
        ColumnTransformer: A transformer ready to be used in a scikit-learn Pipeline.
    """
    num_cols_to_transform = list(numeric_cols)
    cat_cols_to_transform = list(categorical_cols)

    numeric_transformer = Pipeline(steps=[
        ('scaler', get_numeric_scaler(scaling_method))
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', get_categorical_encoder(encoding_method))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols_to_transform),
            ('cat', categorical_transformer, cat_cols_to_transform)
        ],
        verbose_feature_names_out=False
    )
    
    return preprocessor