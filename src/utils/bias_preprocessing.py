from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover

def _apply_reweighing(dataset, unprivileged_groups, privileged_groups):
    """
    Aplica Reweighing: calcula pesos (instance_weights) diferentes para instâncias 
    no dataset de treino para garantir a equidade antes da classificação.
    """
    print("Aplicando técnica de mitigação: Reweighing...")
    rw = Reweighing(unprivileged_groups=unprivileged_groups, 
                    privileged_groups=privileged_groups)
    
    return rw.fit_transform(dataset)

def _apply_disparate_impact_remover(dataset, sensitive_attribute, repair_level=1.0):
    """
    Aplica Disparate Impact Remover: edita os valores das features para aumentar 
    a justiça do grupo, preservando a ordenação dentro dos grupos.
    """
    print(f"Aplicando técnica de mitigação: Disparate Impact Remover (repair_level={repair_level})...")
    
    dir = DisparateImpactRemover(repair_level=repair_level, 
                                      sensitive_attribute=sensitive_attribute)
    
    return dir.fit_transform(dataset)

def apply_bias_preprocessing(method, dataset, sensitive_attribute=None, 
                             unprivileged_groups=None, privileged_groups=None):
    """
    Função principal que roteia qual algoritmo de mitigação de viés aplicar
    com base na escolha do usuário (string).
    
    Args:
        method (str): 'none', 'reweighing' ou 'disparate-impact-remover'.
        dataset (BinaryLabelDataset): Dataset AIF360 original (geralmente o de treino).
        sensitive_attribute (str): Nome da coluna do atributo sensível (necessário para o DIR).
        unprivileged_groups (list): Grupo não-privilegiado (necessário para Reweighing).
        privileged_groups (list): Grupo privilegiado (necessário para Reweighing).
        
    Returns:
        BinaryLabelDataset: Dataset transformado (ou original se method=='none').
    """
    if method == 'none' or method is None:
        return dataset.copy()
        
    elif method == 'reweighing':
        return _apply_reweighing(dataset, unprivileged_groups, privileged_groups)
        
    elif method == 'disparate-impact-remover':
        return _apply_disparate_impact_remover(dataset, sensitive_attribute)