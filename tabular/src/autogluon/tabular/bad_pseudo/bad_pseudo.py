import unittest
import pandas as pd
import numpy as np
from ..predictor.predictor import TabularPredictor


def fit_pseudo_end_to_end(train_data, test_data, validation_data, label, init_kwargs=None, fit_kwargs=None,
                          max_iter: bool = 1, reuse_pred_test: bool = False, threshold: float = 0.9):
    if init_kwargs is None:
        init_kwargs = dict()
    if fit_kwargs is None:
        fit_kwargs = dict()

    predictor = TabularPredictor(label=init_kwargs['label'], **init_kwargs).fit(train_data, **fit_kwargs)

    y_pred_proba_og = predictor.predict_proba(test_data)
    y_pred_og = predictor.predict(test_data)

    y_pred_proba, best_model, total_iter = fit_pseudo_given_preds(train_data=train_data,
                                                                  test_data=test_data,
                                                                  y_pred_proba_og=y_pred_proba_og,
                                                                  y_pred_og=y_pred_og,
                                                                  problem_type=predictor.problem_type,
                                                                  label=label,
                                                                  validation_data=validation_data,
                                                                  init_kwargs=init_kwargs,
                                                                  fit_kwargs=fit_kwargs,
                                                                  max_iter=max_iter,
                                                                  reuse_pred_test=reuse_pred_test,
                                                                  threshold=threshold)

    #######
    # score_og = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred_proba_og)
    # score_ps = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred_proba)
    # print(f'score_og: {score_og}')
    # print(f'score_ps: {score_ps}')
    #######

    return y_pred_proba, best_model, total_iter


def fit_pseudo_given_preds(train_data, validation_data, test_data, y_pred_proba_og, y_pred_og, problem_type, label,
                           init_kwargs=None,
                           fit_kwargs=None, max_iter=1, reuse_pred_test: bool = False, threshold: float = .9):
    if init_kwargs is None:
        init_kwargs = dict()
    if fit_kwargs is None:
        fit_kwargs = dict()

    y_pred = y_pred_og.copy()
    y_pred_proba = y_pred_proba_og.copy()
    y_pred_proba_holdout = y_pred_proba.copy()
    previous_score = float('-inf')
    best_model = None

    for i in range(max_iter):
        # Finds pseudo labeling rows that are above threshold
        test_pseudo_indices_true = filter_pseudo(y_pred_proba_holdout, problem_type=problem_type, threshold=threshold)
        test_pseudo_indices = pd.Series(data=False, index=y_pred_proba_holdout.index)
        test_pseudo_indices[test_pseudo_indices_true.index] = True

        # Copy test data and impute labels then select indices that are above threshold
        test_data_pseudo = test_data.copy()
        test_data_pseudo[label] = y_pred
        test_data_pseudo = test_data_pseudo.loc[test_pseudo_indices_true.index]

        if len(test_data_pseudo) > 0:
            curr_train_data = pd.concat([train_data, test_data_pseudo], ignore_index=True)
            # test_data_holdout is data that should not be added into train because didn't meet threshold
            test_data_holdout = test_data.copy()
            test_data_holdout = test_data_holdout.loc[test_pseudo_indices[~test_pseudo_indices].index]
            # predictor_pseudo = TabularPredictor(label=label, **init_kwargs).fit(train_data = train_data, **fit_kwargs)
            predictor_pseudo = TabularPredictor(label=label, **init_kwargs).fit(train_data=curr_train_data,
                                                                                tuning_data=validation_data,
                                                                                **fit_kwargs)
            curr_score = predictor_pseudo.info()['best_model_score_val']

            if curr_score > previous_score:
                previous_score = curr_score
                best_model = predictor_pseudo
            else:
                break

            y_pred_proba_holdout = predictor_pseudo.predict_proba(test_data_holdout)
            # Sets predicted probs for heldout data
            y_pred_proba.loc[test_data_holdout.index] = y_pred_proba_holdout
            y_pred = predictor_pseudo.predict(test_data)

            # No repeat runs on pseudo-labeled test
            if reuse_pred_test:
                test_data = test_data.loc[~test_pseudo_indices]
                unittest.TestCase.assertTrue(np.all(test_data.index == test_pseudo_indices[~test_pseudo_indices].index),
                                             'Test data indices and pseudo labels do not match')

            #######
            # score_ps = predictor_pseudo.evaluate_predictions(y_true=test_data[label], y_pred=y_pred_proba)
            # print(f'score_ps {i}: {score_ps}')
            #######
        else:
            break

    return best_model


def filter_pseudo(y_pred_proba_og, problem_type, min_percentage: float = 0.05, max_percentage: float = 0.6,
                  threshold: float = 0.9):
    if problem_type in ['binary', 'multiclass']:
        y_pred_proba_max = y_pred_proba_og.max(axis=1)
        curr_threshold = threshold
        # Percent of rows above threshold
        curr_percentage = (y_pred_proba_max >= curr_threshold).mean()
        num_rows = len(y_pred_proba_max)

        if curr_percentage > max_percentage or curr_percentage < min_percentage:
            if curr_percentage > max_percentage:
                num_rows_threshold = max(np.ceil(max_percentage * num_rows), 1)
            else:
                num_rows_threshold = max(np.ceil(min_percentage * num_rows), 1)
            y_pred_proba_max_sorted = y_pred_proba_max.sort_values(ascending=False, ignore_index=True)
            # Set current threshold to num_rows_threshold - 1
            curr_threshold = y_pred_proba_max_sorted[num_rows_threshold - 1]

        # Pseudo indices greater than threshold of 0.95
        test_pseudo_indices = (y_pred_proba_max >= curr_threshold)
    else:
        # Select a random 30% of the data to use as pseudo
        test_pseudo_indices = pd.Series(data=False, index=y_pred_proba_og.index)
        test_pseudo_indices_true = test_pseudo_indices.sample(frac=0.3, random_state=0)
        test_pseudo_indices[test_pseudo_indices_true.index] = True

    test_pseudo_indices = test_pseudo_indices[test_pseudo_indices]

    return test_pseudo_indices