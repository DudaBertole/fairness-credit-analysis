import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_categorical_encoder(method='none'):
    """
    Retorna o transformador adequado para variáveis categóricas.
    """
    if method == 'one-hot':
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    elif method == 'label':
        return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    elif method == 'none' or method is None:
        return 'passthrough'

def get_numeric_scaler(method='none'):
    """
    Retorna o transformador adequado para variáveis numéricas.
    """
    if method == 'standardization':
        return StandardScaler()
    elif method == 'none' or method is None:
        return 'passthrough'

def build_preprocessor(numeric_cols, categorical_cols, scaling_method=None, encoding_method=None):
    """
    Constrói um ColumnTransformer que:
      - aplica scaler nas numéricas
      - aplica encoder nas categóricas
      - REMOVE a coluna sensível (se fornecida)
      - descarta quaisquer colunas que não estejam listadas 
    
    Args:
        numeric_cols (list): Lista com o nome das colunas numéricas.
        categorical_cols (list): Lista com o nome das colunas categóricas.
        sensitive_attribute (str): Nome da coluna do atributo sensível que não deve ser alterada.
        scaling_method (str): 'standardization' ou 'none'.
        encoding_method (str): 'one-hot', 'label' ou 'none'.
        
    Returns:
        ColumnTransformer: Objeto pronto para ser inserido em um Pipeline do sklearn.
    """
    
    # Criamos cópias das listas para não alterar as variáveis originais fora da função
    num_cols_to_transform = list(numeric_cols)
    cat_cols_to_transform = list(categorical_cols)

    numeric_transformer = Pipeline(steps=[
        ('scaler', get_numeric_scaler(scaling_method))
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', get_categorical_encoder(encoding_method))
    ])

    # Combina tudo em um ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols_to_transform),
            ('cat', categorical_transformer, cat_cols_to_transform)
        ],
    )
    
    return preprocessor