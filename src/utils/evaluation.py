import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aif360.metrics import ClassificationMetric

def evaluate_performance(y_true, y_pred, favorable_label=1.0):
    """
    Calcula as métricas clássicas de performance de Machine Learning.
    
    Args:
        y_true (array-like): Rótulos reais (ground truth).
        y_pred (array-like): Rótulos previstos pelo modelo.
        favorable_label (float/int): O valor que representa a classe positiva.
        
    Returns:
        dict: Dicionário contendo Accuracy, Precision, Recall e F1-Score.
    """
    # pos_label garante que as métricas sejam calculadas em relação à classe favorável
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=favorable_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=favorable_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=favorable_label, zero_division=0)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }

def evaluate_fairness(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):
    """
    Calcula as métricas de justiça (Fairness) utilizando o AIF360.
    
    Args:
        dataset_true (BinaryLabelDataset): Dataset com os rótulos reais.
        dataset_pred (BinaryLabelDataset): Dataset com os rótulos preditos.
        unprivileged_groups (list): Definição do grupo não-privilegiado.
        privileged_groups (list): Definição do grupo privilegiado.
        
    Returns:
        dict: Dicionário contendo as métricas de disparidade.
    """
    # Inicializa o objeto de métricas do AIF360
    metric = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    
    # Demographic Parity Difference (ideal: 0.0)
    dp_diff = metric.statistical_parity_difference()
    
    # Equal Opportunity Difference (ideal: 0.0)
    eo_diff = metric.equal_opportunity_difference()
    
    # Predictive Parity Difference (PPV_unpriv - PPV_priv, ideal: 0.0)
    ppv_diff = metric.positive_predictive_value(privileged=False) - metric.positive_predictive_value(privileged=True)
    
    # Negative Predictive Value Difference (NPV_unpriv - NPV_priv)
    npv_diff = metric.negative_predictive_value(privileged=False) - metric.negative_predictive_value(privileged=True)
    
    # Average Predictive Value Difference (Média entre PPV e NPV)
    avg_pv_diff = (ppv_diff + npv_diff) / 2.0
    
    # Average Odds Difference
    avg_odds_diff = metric.average_odds_difference()
    
    return {
        "Demographic Parity Diff.": dp_diff,
        "Equal Opportunity Diff.": eo_diff,
        "Predictive Parity Diff.": ppv_diff,
        "Average Predictive Value Diff.": avg_pv_diff,
        "Average Odds Diff.": avg_odds_diff
    }

def evaluate_pipeline(dataset_true, dataset_pred, unprivileged_groups, privileged_groups, pipeline_name="Pipeline"):
    """
    Função principal que consolida as métricas de Performance e Fairness 
    e imprime um relatório formatado.
    
    Args:
        dataset_true (BinaryLabelDataset): Dataset AIF360 de teste com rótulos reais.
        dataset_pred (BinaryLabelDataset): Dataset AIF360 de teste com rótulos preditos.
        unprivileged_groups (list): Grupo não-privilegiado.
        privileged_groups (list): Grupo privilegiado.
        pipeline_name (str): Nome do pipeline para exibição.
        
    Returns:
        pd.DataFrame: Um DataFrame de uma linha contendo todas as métricas (útil para concatenar resultados).
    """
    print(f"\n{'='*50}")
    print(f" RESULTADOS: {pipeline_name.upper()}")
    print(f"{'='*50}")
    
    # Extrai os arrays numpy para o sklearn
    y_true = dataset_true.labels.ravel()
    y_pred = dataset_pred.labels.ravel()
    favorable_label = dataset_true.favorable_label
    
    # Calcula as métricas
    perf_metrics = evaluate_performance(y_true, y_pred, favorable_label)
    fair_metrics = evaluate_fairness(dataset_true, dataset_pred, unprivileged_groups, privileged_groups)
    
    # Imprime Performance
    print("\n--- Métricas de Performance ---")
    for k, v in perf_metrics.items():
        print(f"{k+':':<20} {v:.4f}")
        
    # Imprime Fairness
    print("\n--- Métricas de Fairness (Ideal próximo a 0.0) ---")
    for k, v in fair_metrics.items():
        print(f"{k+':':<30} {v:.4f}")
        
    print(f"{'='*50}\n")
    
    # Combina tudo em um único dicionário e retorna como DataFrame
    all_metrics = {"Pipeline": pipeline_name}
    all_metrics.update(perf_metrics)
    all_metrics.update(fair_metrics)
    
    return pd.DataFrame([all_metrics])