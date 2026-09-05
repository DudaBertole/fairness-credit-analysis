from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
from scipy.stats import loguniform, randint, uniform

def get_model(model_name, random_state=2):
    """
    Returns the instance of the selected model.
    
    Args:
        model_name (str): 'logistic_regression', 'random_forest', or 'gradient_boosting'
        random_state (int): seed
                  
    Returns:
        Estimator: Instance of the sklearn model
    """
    if model_name == 'logistic_regression':
        return LogisticRegression(max_iter=2000, random_state=random_state)
        
    elif model_name == 'random_forest':
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
        
    elif model_name == 'gradient_boosting':
        return GradientBoostingClassifier(random_state=random_state)


def get_hyperparameters(model_name):
    if model_name == 'logistic_regression':
        grid_params = {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
            "class_weight": [None, "balanced"]
        }
        random_params = {
            "C": loguniform(1e-3, 1e2),
            "solver": ["lbfgs", "liblinear"],
            "class_weight": [None, "balanced"]
        }
        return grid_params, random_params

    elif model_name == 'random_forest':
        grid_params = {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "class_weight": [None, "balanced"]
        }
        random_params = {
            "n_estimators": randint(200, 700),
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 6),
            "class_weight": [None, "balanced"]
        }
        return grid_params, random_params

    elif model_name == 'gradient_boosting':
        grid_params = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5]
        }
        random_params = {
            "n_estimators": randint(50, 301),
            "learning_rate": loguniform(1e-2, 2e-1),
            "max_depth": randint(2, 5),
            "subsample": uniform(0.7, 0.3)
        }
        return grid_params, random_params