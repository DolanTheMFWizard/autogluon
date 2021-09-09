import argparse
import unittest

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

from openml_meta import generate_openml_meta


def fit_pseudo_end_to_end(train_data, test_data, validation_data, label, init_kwargs=None, fit_kwargs=None,
                          max_iter: bool = 1, reuse_pred_test: bool = False, threshold: float = 0.9,
                          test_only: bool = False, full_test_data: pd.DataFrame = None):
    if init_kwargs is None:
        init_kwargs = dict()
    if fit_kwargs is None:
        fit_kwargs = dict()

    predictor = TabularPredictor(label=label, **init_kwargs).fit(train_data, **fit_kwargs)

    y_pred_proba_og = predictor.predict_proba(test_data)
    y_pred_og = predictor.predict(test_data)

    y_pred_proba, best_model = fit_pseudo_given_preds(train_data=train_data,
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
                                                      threshold=threshold,
                                                      test_only=test_only,
                                                      full_test_data=full_test_data)

    #######
    # score_og = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred_proba_og)
    # score_ps = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred_proba)
    # print(f'score_og: {score_og}')
    # print(f'score_ps: {score_ps}')
    #######

    return y_pred_proba, best_model


def fit_pseudo_given_preds(train_data, validation_data, test_data, y_pred_proba_og, y_pred_og, problem_type, label,
                           init_kwargs=None,
                           fit_kwargs=None, max_iter=1, reuse_pred_test: bool = False, threshold: float = .9,
                           test_only: bool = False, full_test_data: pd.DataFrame = None):
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

        if full_test_data is None:
            test_data_pseudo = test_data.copy()
            test_data_pseudo[label] = y_pred
            test_data_pseudo = test_data_pseudo.loc[test_pseudo_indices_true.index]
        else:
            test_data_pseudo = full_test_data.loc[test_pseudo_indices_true.index]

        if len(test_data_pseudo) > 0:
            if test_only:
                curr_train_data = test_data_pseudo
            else:
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

    return y_pred_proba, best_model


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


class Metrics:
    def __init__(self):
        self.procedure_list = list()
        self.best_model_list = list()
        self.accuracy = list()
        self.eval_p_list = list()
        self.threshold_list = list()
        self.is_reuse_list = list()
        self.total_iter_list = list()

    def add(self, model, accuracy, eval_p, is_reuse: bool = None, threshold: float = None, total_iter: int = None):
        self.eval_p_list.append(eval_p)
        self.best_model_list.append(model.get_model_best())
        self.accuracy.append(accuracy)
        self.threshold_list.append(threshold)
        self.is_reuse_list.append(is_reuse)
        self.total_iter_list.append((total_iter))

        if is_reuse is None and threshold is None:
            self.procedure_list.append('Vanilla')
        else:
            self.procedure_list.append('Pseudo-Label')

    def generate_csv(self, path):
        pd.DataFrame.from_dict(
            dict(procedure=self.procedure_list, best_model=self.best_model_list, accuracy=self.accuracy,
                 eval_percentage=self.eval_p_list, threshold=self.threshold_list, is_reuse=self.is_reuse_list,
                 total_iter=self.total_iter_list)).to_csv(
            path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--openml_id', type=int, help='OpenML id to run on', default=32)
    parser.add_argument('--test_only', type=bool, help='Use test data only during refit', default=False)
    parser.add_argument('--cheat', type=bool, help='Model should get the true test labels', default=False)
    parser.add_argument('--max_iter', type=int, help='Max number of iterations of pseudo-labeling', default=1)
    # parser.add_argument('--id', type=str, help='id given to this runs results', default='no_name')
    # parser.add_argument('--threshold', type=float,
    #                     help='Predictive probability threshold to be above in order to use for pseudo-labeling',
    #                     default=0.85)
    # parser.add_argument('--max_iter', type=int, help='Max number of iterations that pseudo-labeling can take',
    #                     default=1)
    # parser.add_argument('--percent_eval', type=float, help='Percent of data to use for evaluation', default=0.2)
    # parser.add_argument('--reuse', type=bool, help='Whether to rerun inference on old pseudo-labelled points',
    #                     default=True)
    # parser.add_argument('--validation_percent', type=float, help='What percent of train should be used for validation',
    #                     default=0.15)
    args = parser.parse_args()

    generate_openml_meta(args.openml_id)
    val_percent_list = [0.2]
    eval_percent_list = [0.25, 0.5, 0.75, 0.95]
    threshold_list = [0.5, 0.75, 0.9, 0.95]
    max_iter = args.max_iter

    if max_iter > 1:
        reuse_list = [False]
    else:
        reuse_list = [True, False]

    openml_id = args.openml_id
    data = fetch_openml(data_id=openml_id, as_frame=True)
    features = data['data']
    target = data['target']
    label = data['target_names'][0]
    df = features.join(target)
    num_rows = len(df)
    score_tracker = Metrics()

    for vp in val_percent_list:
        for ep in eval_percent_list:
            test_split = df.sample(frac=ep, random_state=1)
            true_test = list()
            train_split = df.drop(test_split.index)
            validation_data = train_split.sample(frac=vp, random_state=1)
            train_data = train_split.drop(validation_data.index)

            if args.cheat:
                true_test = test_split.sample(frac=0.2, random_state=1)
                test_split = test_split.drop(true_test.index)

            test_data = test_split.drop(columns=label)

            train_len = len(train_data)
            test_len = len(test_data)
            val_len = len(validation_data)
            true_test_len = len(true_test)

            assert (train_len + test_len + val_len + true_test_len) == num_rows
            assert not train_data.index.equals(test_data.index)
            assert not test_data.index.equals(validation_data.index)
            assert not train_data.index.equals(validation_data.index)

            if args.cheat:
                assert not train_data.index.equals(true_test.index)
                assert not test_data.index.equals(true_test.index)
                assert not validation_data.index.equals(true_test.index)

            agp = TabularPredictor(label=label).fit(train_data=train_data, tuning_data=validation_data)

            if args.cheat:
                agp_pred = agp.predict(true_test)
                vanilla_acc = accuracy_score(agp_pred.to_numpy(), true_test[label].to_numpy())
            else:
                agp_pred = agp.predict(test_data)
                vanilla_acc = accuracy_score(agp_pred.to_numpy(), test_split[label].to_numpy())

            score_tracker.add(agp, vanilla_acc, ep)

            for is_reuse in reuse_list:
                for t in threshold_list:
                    # final_predict, best_model = TabularPredictor(label=label).bad_pseudo_fit(train_data=train_data,
                    #                                                                          test_data=test_data,
                    #                                                                          validation_data=validation_data)
                    if args.cheat:
                        test_pred, best_model = fit_pseudo_end_to_end(train_data=train_data,
                                                                      validation_data=validation_data,
                                                                      test_data=test_data, label=label,
                                                                      max_iter=max_iter,
                                                                      reuse_pred_test=is_reuse, threshold=t,
                                                                      test_only=args.test_only,
                                                                      full_test_data=test_split)
                        final_predict = best_model.predict(true_test.drop(columns=label))
                        pseudo_label_acc = accuracy_score(final_predict.to_numpy(), true_test[label].to_numpy())

                    else:
                        test_pred, best_model = fit_pseudo_end_to_end(train_data=train_data,
                                                                      validation_data=validation_data,
                                                                      test_data=test_data, label=label,
                                                                      max_iter=max_iter,
                                                                      reuse_pred_test=is_reuse, threshold=t,
                                                                      test_only=args.test_only)
                        final_predict = test_pred.idxmax(axis=1)
                        pseudo_label_acc = accuracy_score(final_predict.to_numpy(), test_split[label].to_numpy())

                    score_tracker.add(best_model, pseudo_label_acc, ep, is_reuse, t)

    test_only_label = '_testonly' if args.test_only else ''
    cheat_label = '/cheat' if args.cheat else ''
    score_tracker.generate_csv(f'./results{cheat_label}/openml{openml_id}_results_iter{max_iter}{test_only_label}.csv')
