from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import RejectOptionClassification

def _apply_calibrated_equalized_odds(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):
    cpp = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                         unprivileged_groups=unprivileged_groups,
                                         cost_constraint='fpr', 
                                         seed=2) 
    
    cpp = cpp.fit(dataset_true, dataset_pred)
    
    return cpp.predict(dataset_pred)

def _apply_reject_option_classification(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):
    roc = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                     num_class_thresh=100, num_ROC_margin=50,
                                     metric_name="Statistical parity difference",
                                     metric_ub=0.05, metric_lb=-0.05)
    
    roc = roc.fit(dataset_true, dataset_pred)
    return roc.predict(dataset_pred)

def apply_bias_postprocessing(method, dataset_true, dataset_pred, 
                              unprivileged_groups, privileged_groups):
    """
    Main function that routes which post-processing algorithm to apply.
    
    Args:
        method (str): 'none', 'calibrate_equalized_odds' or 'reject_option_classification'.
        dataset_true (BinaryLabelDataset): Dataset with the TRUE labels (ground truth).
        dataset_pred (BinaryLabelDataset): Dataset with the SCORES/PROBABILITIES predicted by the model.
        unprivileged_groups (list): Unprivileged group.
        privileged_groups (list): Privileged group.
        
    Returns:
        BinaryLabelDataset: Dataset with the final adjusted (mitigated) labels.
    """
    if method == 'none' or method is None:
        # If there is no post-processing, we simply transform the continuous scores
        # into classic binary labels (0.5 threshold) so that the metrics work.
        dataset_final = dataset_pred.copy()
        import numpy as np
        dataset_final.labels = np.where(dataset_final.scores >= 0.5, 
                                        dataset_final.favorable_label, 
                                        dataset_final.unfavorable_label)
        return dataset_final
        
    elif method == 'calibrate_equalized_odds':
        return _apply_calibrated_equalized_odds(dataset_true, dataset_pred, 
                                                unprivileged_groups, privileged_groups)
        
    elif method == 'reject_option_classification':
        return _apply_reject_option_classification(dataset_true, dataset_pred, 
                                                   unprivileged_groups, privileged_groups)