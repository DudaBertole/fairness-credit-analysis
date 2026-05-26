from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def apply_tuning(method, estimator, grid_params=None, random_params=None, 
                 cv=5, n_iter=25, scoring='roc_auc', random_state=42):
    """
    Roteador para aplicar a técnica de otimização de hiperparâmetros escolhida.
    Retorna o objeto pronto para receber o .fit(X, y).
    
    Args:
        method (str): 'none', 'grid' ou 'random'.
        estimator: Instância do modelo do scikit-learn.
        grid_params (dict): Dicionário com os parâmetros para o GridSearch.
        random_params (dict): Dicionário com as distribuições para o RandomSearch.
        cv (int): Número de folds para a validação cruzada.
        n_iter (int): Número de iterações (usado apenas no RandomSearch).
        scoring (str): Métrica a ser otimizada (ex: 'accuracy', 'f1', 'roc_auc').
        random_state (int): Semente de aleatoriedade para o RandomSearch.
        
    Returns:
        Estimator: O modelo original ou envelopado em um GridSearchCV / RandomizedSearchCV.
    """
    
    if method == 'none' or method is None:
        print("Tuning desativado. Utilizando os parâmetros padrão do modelo.")
        return estimator
        
    # Verifica se o modelo suporta tuning (ex: AIF360 in-processing retorna dicionários vazios)
    if method == 'grid':
        if not grid_params:
            print(f"Aviso: Parâmetros de Grid Search vazios. Retornando modelo base.")
            return estimator
            
        print(f"Configurando GridSearchCV (cv={cv}, scoring='{scoring}')...")
        return GridSearchCV(
            estimator=estimator,
            param_grid=grid_params,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=3
        )
        
    elif method == 'random':
        if not random_params:
            print(f"Aviso: Parâmetros de Random Search vazios. Retornando modelo base.")
            return estimator
            
        print(f"Configurando RandomizedSearchCV (n_iter={n_iter}, cv={cv}, scoring='{scoring}')...")
        return RandomizedSearchCV(
            estimator=estimator,
            param_distributions=random_params,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=3
        )
