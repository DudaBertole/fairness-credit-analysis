from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset


def _apply_reweighing(dataset, unprivileged_groups, privileged_groups):

    print("Aplicando técnica de mitigação: Reweighing...")
    rw = Reweighing(unprivileged_groups=unprivileged_groups, 
                    privileged_groups=privileged_groups)
    
    return rw.fit_transform(dataset)


def _apply_disparate_impact_remover(
    dataset,
    sensitive_attribute,
    repair_level=0.5,
    features_to_repair=None
):
    """
    Applies the Disparate Impact Remover (DIR) to the selected features.

    Args:
        dataset (BinaryLabelDataset): Dataset to be repaired.
        sensitive_attribute (str): Name of the protected attribute used by DIR.
        repair_level (float): Repair strength, ranging from 0 to 1.
        features_to_repair (list[str]): Feature names (present in
            dataset.feature_names) to be repaired.

    Returns:
        BinaryLabelDataset: A copy of the dataset with the selected features
        repaired.
    """

    if features_to_repair is None:
        raise ValueError(
            "'features_to_repair' must be provided when using "
            "Disparate Impact Remover."
        )

    feature_names = dataset.feature_names
    missing = [c for c in features_to_repair if c not in feature_names]
    if missing:
        raise ValueError(
            f"Features not found in dataset.feature_names: {missing}"
        )

    repair_idx = [feature_names.index(c) for c in features_to_repair]

    # Create a sub-dataset containing only the features to repair and the
    # protected attribute required by DIR.
    cols_for_dir = list(features_to_repair)
    if sensitive_attribute not in cols_for_dir:
        cols_for_dir.append(sensitive_attribute)

    df_full, _ = dataset.convert_to_dataframe()
    label_col = dataset.label_names[0]
    df_sub = df_full[cols_for_dir + [label_col]]

    ds_sub = BinaryLabelDataset(
        df=df_sub,
        label_names=[label_col],
        protected_attribute_names=[sensitive_attribute],
        favorable_label=dataset.favorable_label,
        unfavorable_label=dataset.unfavorable_label,
    )

    dir_transformer = DisparateImpactRemover(
        repair_level=repair_level,
        sensitive_attribute=sensitive_attribute,
    )

    ds_sub_repaired = dir_transformer.fit_transform(ds_sub)

    sub_feature_names = ds_sub_repaired.feature_names
    repaired_values = ds_sub_repaired.features[
        :, [sub_feature_names.index(c) for c in features_to_repair]
    ]

    dataset_out = dataset.copy(deepcopy=True)
    dataset_out.features[:, repair_idx] = repaired_values

    return dataset_out


def apply_bias_preprocessing(method, dataset, sensitive_attribute=None, 
                             unprivileged_groups=None, privileged_groups=None,
                             features_to_repair=None):

    if method == 'none' or method is None:
        return dataset.copy()
        
    elif method == 'reweighing':
        return _apply_reweighing(dataset, unprivileged_groups, privileged_groups)
        
    elif method == 'disparate-impact-remover':
        return _apply_disparate_impact_remover(
            dataset, sensitive_attribute, features_to_repair=features_to_repair
        )

    raise ValueError(f"Método de bias pre-processing desconhecido: {method!r}")
