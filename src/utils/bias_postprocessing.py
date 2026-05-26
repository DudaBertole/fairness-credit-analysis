from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import RejectOptionClassification

def _apply_calibrated_equalized_odds(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):
    """
    Aplica Calibrated Equalized Odds: otimiza as taxas de falsos negativos/positivos 
    para torná-las iguais entre os grupos, mantendo a calibração do classificador.
    """
    print("Aplicando técnica de mitigação (Pós-processamento): Calibrated Equalized Odds...")
    
    # cost_constraint pode ser 'fpr', 'fnr' ou 'weighted'; testar com novos valores
    cpp = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                         unprivileged_groups=unprivileged_groups,
                                         cost_constraint='fnr', 
                                         seed=42)
    
    cpp = cpp.fit(dataset_true, dataset_pred)
    
    return cpp.predict(dataset_pred)

def _apply_reject_option_classification(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):
    """
    Aplica Reject Option Classification (ROC): dá resultados favoráveis para o grupo 
    não-privilegiado e desfavoráveis para o privilegiado em uma "zona de incerteza" 
    próxima ao limiar de decisão (ex: predições entre 0.45 e 0.55).
    """
    print("Aplicando técnica de mitigação (Pós-processamento): Reject Option Classification...")
    
    # metric_name pode ser "Statistical parity difference" ou "Average odds difference"; testar com novos valores
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
    Função principal que roteia qual algoritmo de pós-processamento aplicar.
    
    Args:
        method (str): 'none', 'calibrate_equalized_odds' ou 'reject_option_classification'.
        dataset_true (BinaryLabelDataset): Dataset com os rótulos REAIS (ground truth).
        dataset_pred (BinaryLabelDataset): Dataset com os SCORES/PROBABILIDADES preditos pelo modelo.
        unprivileged_groups (list): Grupo não-privilegiado.
        privileged_groups (list): Grupo privilegiado.
        
    Returns:
        BinaryLabelDataset: Dataset com os rótulos finais ajustados (mitigados).
    """
    if method == 'none' or method is None:
        print("Nenhuma técnica de mitigação de viés (pós-processamento) selecionada.")
        
        # Se não houver pós-processamento, apenas transformamos os scores contínuos 
        # em labels binários clássicos (threshold de 0.5) para que as métricas funcionem.
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