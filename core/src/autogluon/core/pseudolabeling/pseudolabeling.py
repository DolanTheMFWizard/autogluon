import numpy as np
import pandas as pd
from autogluon.core.constants import PROBLEM_TYPES_CLASSIFICATION


def filter_pseudo(y_pred_proba_og, problem_type,
                  min_proportion_prob: float = 0.05, max_proportion_prob: float = 0.6,
                  threshold: float = 0.95, proportion_sample: float = 0.3):
    """
    Takes in the predicted probabilities of the model and chooses the indices that meet
    a criteria to incorporate into training data. Criteria is determined by problem_type.
    If multiclass or binary will choose all rows with max prob over threshold. For regression
    chooses 30% of the labeled data randomly. This filter is used pseudo labeled data.

    Parameters:
    -----------
    y_pred_proba_og: The predicted probabilities from the current best model. If problem is
        'binary' or 'multiclass' then it's Panda series of predictive probs, if it's 'regression'
        then it's a scalar. Binary probs should be set to multiclass.
    min_proportion_prob: Minimum proportion of indices in y_pred_proba_og to select. The filter
        threshold will be automatically adjusted until at least min_proportion_prob of the predictions
        in y_pred_proba_og pass the filter. This ensures we return at least min_proportion_prob of the
        pseudolabeled data to augment the training set in pseudolabeling.
    max_proportion_prob: Maximum proportion of indices in y_pred_proba_og to select. The filter threshold
        will be automatically adjusted until at most max_proportion_prob of the predictions in y_pred_proba_og
        pass the filter. This ensures we return at most max_proportion_prob of the pseudolabeled data to augment
        the training set in pseudolabeling.
    threshold: This filter will only return those indices of y_pred_proba_og where the probability
        of the most likely class exceeds the given threshold value.
    proportion_sample: When problem_type is regression this is percent of pseudo data
        to incorporate into train. Rows selected randomly.

    Returns:
    --------
    pd.Series of indices that met pseudolabeling requirements
    """
    if problem_type in PROBLEM_TYPES_CLASSIFICATION:
        y_pred_proba_max = y_pred_proba_og.max(axis=1)
        curr_threshold = threshold
        curr_percentage = (y_pred_proba_max >= curr_threshold).mean()
        num_rows = len(y_pred_proba_max)

        if curr_percentage > max_proportion_prob or curr_percentage < min_proportion_prob:
            if curr_percentage > max_proportion_prob:
                num_rows_threshold = max(np.ceil(max_proportion_prob * num_rows), 1)
            else:
                num_rows_threshold = max(np.ceil(min_proportion_prob * num_rows), 1)
            curr_threshold = y_pred_proba_max.sort_values(ascending=False).iloc[int(num_rows_threshold) - 1]

        test_pseudo_indices = (y_pred_proba_max >= curr_threshold)
    else:
        test_pseudo_indices = pd.Series(data=False, index=y_pred_proba_og.index)
        test_pseudo_indices_true = test_pseudo_indices.sample(frac=proportion_sample, random_state=0)
        test_pseudo_indices[test_pseudo_indices_true.index] = True

    test_pseudo_indices = test_pseudo_indices[test_pseudo_indices]

    return test_pseudo_indices


def filter_pseudo_std_regression(predictor, test_data: pd.DataFrame, k: int = 5,
                                 z_score_threshold: float = 0.25):
    top_k_models_list = list(predictor._trainer.leaderboard()['model'][:k])
    pred_proba_top_k = None
    for model in top_k_models_list:
        y_test_pred = predictor.predict(data=test_data, model=model)
        if model == top_k_models_list[0]:
            pred_proba_top_k = y_test_pred
        else:
            pred_proba_top_k = pd.concat([pred_proba_top_k, y_test_pred], axis=1)
    pred_proba_top_k = pred_proba_top_k.to_numpy()

    pred_sd = pd.Series(data=np.std(pred_proba_top_k, axis=1), index=test_data.index)
    pred_z_score = (pred_sd - pred_sd.mean()) / pred_sd.std()

    df_filtered = pred_z_score.between(-1 * z_score_threshold, z_score_threshold)

    return df_filtered[df_filtered]


def filter_pseudo_ECE(y_pred_proba: pd.DataFrame, val_pred_proba: pd.DataFrame, val_label: pd.Series,
                      threshold: float = 0.95, anneal_frac: float = 0.5):
    """
        Takes in the predicted probabilities of the model and chooses the indices that meet
        a criteria to incorporate into training data. Criteria predictive probability that is
        above the threshold - class calibration * anneal_frac
        Parameters:
        -----------
        y_pred_proba_og: The predicted probabilities from the current best model. If problem is
        'binary' or 'multiclass' then it's Panda series of predictive probs, if it's 'regression'
        then it's a scalar
        val_pred_proba: The predicted probability from the model for the validation set
        val_label: The validation set labels
        threshold: The predictive probability percent that must be exceeded in order to be
        incorporated into the next round of training
        anneal_frac: How much to scale the calculated class calibration when subtracting from
        threshold
        Returns:
        --------
        pd Dataframe of selected indices
    """
    predictions = val_pred_proba.idxmax(axis=1)
    y_pred_proba_max = val_pred_proba.max(axis=1)
    y_predicts = y_pred_proba.idxmax(axis=1)
    y_max_probs = y_pred_proba.max(axis=1)
    classes = predictions.unique()
    pseudo_indexes = pd.Series(data=False, index=y_pred_proba.index)

    for c in classes:
        predicted_as_c_idxes = predictions[predictions == c].index
        predicted_probs_as_c = y_pred_proba_max.loc[predicted_as_c_idxes]
        val_labels_as_c = val_label.loc[predicted_as_c_idxes]

        accuracy = len(val_labels_as_c[val_labels_as_c == c]) / len(val_labels_as_c)
        confidence = predicted_probs_as_c.mean()
        class_calibration = accuracy - confidence

        class_threshold = threshold - (anneal_frac * class_calibration)

        holdout_as_c_idxes = y_predicts[y_predicts == c].index
        holdout_c_probs = y_max_probs.loc[holdout_as_c_idxes]

        above_thres = (holdout_c_probs >= class_threshold)
        pseudo_indexes.loc[above_thres[above_thres].index] = True

    return pseudo_indexes[pseudo_indexes]


def ensemble_classification_filter(unlabeled_data: pd.DataFrame, predictor, top_k: int = 5, threshold: float = 0.95):
    y_pred_proba_ensemble = None
    top_k_model_names = predictor._trainer.leaderboard().head(top_k)['model']

    for model_name in top_k_model_names:
        y_pred_proba_curr_model = predictor.predict_proba(data=unlabeled_data, model=model_name)

        if y_pred_proba_ensemble is None:
            y_pred_proba_ensemble = y_pred_proba_curr_model
        else:
            y_pred_proba_ensemble += y_pred_proba_curr_model

    y_pred_proba_ensemble /= top_k
    y_max_prob = y_pred_proba_ensemble.max(axis=1)
    pseudo_indexes = (y_max_prob >= threshold)
    y_pred_ensemble = y_pred_proba_ensemble.idxmax(axis=1)

    pseudo_idxmax = y_pred_proba_ensemble[pseudo_indexes].idxmax(axis=1)
    pseudo_value_counts = pseudo_idxmax.value_counts()
    min_count = pseudo_value_counts.min()
    pseudo_keys = list(pseudo_value_counts.keys())

    new_test_pseudo_indices = None
    for k in pseudo_keys:
        k_pseudo_idxes = pseudo_idxmax == k
        selected_rows = k_pseudo_idxes[k_pseudo_idxes].head(min_count)

        if new_test_pseudo_indices is None:
            new_test_pseudo_indices = selected_rows.index
        else:
            new_test_pseudo_indices = new_test_pseudo_indices.append(selected_rows.index)

    test_pseudo_indices = pd.Series(data=False, index=y_pred_proba_ensemble.index)
    test_pseudo_indices.loc[new_test_pseudo_indices] = True

    return test_pseudo_indices, y_pred_proba_ensemble, y_pred_ensemble
