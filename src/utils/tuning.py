import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import get_scorer

def apply_tuning(method, estimator, grid_params=None, random_params=None,
                 cv=5, n_iter=25, scoring='roc_auc', random_state=2,
                 strict_weight_routing=True):
    """
    Apply the selected hyperparameter tuning method.

    Args:
        method (str): 'none', 'grid', or 'random'.
        estimator: scikit-learn estimator instance.
        grid_params (dict): GridSearch parameters.
        random_params (dict): RandomSearch distributions.
        cv (int): Number of cross-validation folds.
        n_iter (int): Number of random search iterations.
        scoring (str): Metric to optimize.
        random_state (int): RandomSearch seed.
        strict_weight_routing (bool): Enable explicit sample_weight routing.

    Returns:
        Estimator: The original model or a search wrapper.
    """

    if method == 'none' or method is None:
        # Log prints ---------------------------------------------------------
        print("Tuning disabled. Using the model default parameters.")
        return estimator

    if method not in ('grid', 'random'):
        raise ValueError(f"Unknown tuning method: {method!r}")

    params = grid_params if method == 'grid' else random_params
    if not params:
        # Log prints ---------------------------------------------------------
        print(f"Warning: Empty {'Grid' if method == 'grid' else 'Random'} Search parameters. "
              f"Returning base model.")
        return estimator

    scorer = scoring
    if strict_weight_routing:
        # Enable explicit metadata routing in the current process.
        sklearn.set_config(enable_metadata_routing=True)
        if hasattr(estimator, "set_fit_request"):
            estimator = estimator.set_fit_request(sample_weight=True)
        scorer = get_scorer(scoring)
        if hasattr(scorer, "set_score_request"):
            scorer = scorer.set_score_request(sample_weight=True)

    if method == 'grid':
        # Log prints ---------------------------------------------------------
        print(f"Setting up GridSearchCV (cv={cv}, scoring='{scoring}', "
              f"strict_weight_routing={strict_weight_routing})...")
        return GridSearchCV(
            estimator=estimator,
            param_grid=params,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            verbose=3
        )
    
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        random_state=random_state,
        verbose=3
    )
