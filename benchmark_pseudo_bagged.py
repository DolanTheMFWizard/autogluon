import argparse

import autogluon.core.metrics as metrics
import numpy as np
import pandas as pd
from autogluon.core.data import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.core.utils import infer_problem_type
from autogluon.core.utils import set_logger_verbosity
from autogluon.core.utils.utils import default_holdout_frac
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset
from autogluon.tabular.models import LGBModel
from autogluon.tabular.predictor.predictor import TabularPredictor
from sklearn.datasets import fetch_openml

RE = 'regression'
BI = 'binary'
MU = 'multiclass'

PROBLEM_TYPES = [RE, BI, MU]
CLASSIFICATION = [BI, MU]


class Open_ML_Metrics:
    def __init__(self):
        self.model_name_list = list()
        self.openml_id_list = list()

        self.accuracy = list()
        self.auc_list = list()
        self.neg_logloss_list = list()
        self.neg_MSE_list = list()
        self.MAE_list = list()
        self.result_list = list()
        self.model_score = list()

        self.threshold_list = list()
        self.metric_list = list()
        self.num_iter_list = list()
        self.best_val_list = list()
        self.problem_type_list = list()

    def add(self, model_name: str, eval_p: float, openml_id: int, metric: str,
            accuracy: float = None, auc: float = None, neg_logloss: float = None,
            neg_MSE: float = None, MAE: float = None, model_score: float = None,
            threshold: float = None, result: float = None, iter: int = None,
            problem_type: str = None):
        self.best_val_list.append(eval_p)
        self.model_name_list.append(model_name)
        self.openml_id_list.append(openml_id)

        self.accuracy.append(accuracy)
        self.auc_list.append(auc)
        self.neg_logloss_list.append(neg_logloss)
        self.neg_MSE_list.append(neg_MSE)
        self.MAE_list.append(MAE)
        self.result_list.append(result)
        self.model_score.append(model_score)

        self.threshold_list.append(threshold)
        self.metric_list.append(metric)
        self.num_iter_list.append(iter)
        self.problem_type_list.append(problem_type)

    def generate_csv(self, path):
        pd.DataFrame.from_dict(
            dict(model=self.model_name_list, openml_id=self.openml_id_list, problem_type=self.problem_type_list,
                 accuracy=self.accuracy, auc=self.auc_list, neg_logloss=self.neg_logloss_list,
                 neg_MSE=self.neg_MSE_list, MAE=self.MAE_list, result=self.result_list, model_score=self.model_score,
                 best_val_score=self.best_val_list, threshold=self.threshold_list, metric=self.metric_list,
                 max_num_ter=self.num_iter_list)).to_csv(
            path)

    def get_csv(self):
        return pd.DataFrame.from_dict(
            dict(model=self.model_name_list, openml_id=self.openml_id_list,
                 accuracy=self.accuracy, auc=self.auc_list, neg_logloss=self.neg_logloss_list,
                 neg_MSE=self.neg_MSE_list, neg_MAE=self.MAE_list, result=self.result_list,
                 model_score=self.model_score,
                 best_val_score=self.best_val_list, threshold=self.threshold_list, metric=self.metric_list,
                 max_num_ter=self.num_iter_list))


def get_bagged_model_val_score_avg(bagged_model):
    children_info_dict = bagged_model.get_info()['children_info']
    val_score_folds_list = list()
    for child_info in children_info_dict.keys():
        val_score_folds_list.append(children_info_dict[child_info]['val_score'])

    return np.mean(val_score_folds_list)


def clean_data(train_data, test_data, label):
    # Separate features and labels
    X = train_data.drop(columns=[label])
    y = train_data[label]
    X_test = test_data.drop(columns=[label])
    y_test = test_data[label]

    # Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
    problem_type = infer_problem_type(y=y)  # Infer problem type (or else specify directly)
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    y_clean = label_cleaner.transform(y)

    set_logger_verbosity(2)  # Set logger so more detailed logging is shown for tutorial

    feature_generator = AutoMLPipelineFeatureGenerator()
    X_clean = feature_generator.fit_transform(X)

    X_test_clean = feature_generator.transform(X_test)
    y_test_clean = label_cleaner.transform(y_test)

    return X_clean, y_clean, X_test_clean, y_test_clean, problem_type, label_cleaner


def get_test_score(y_test_clean, y_pred, y_pred_proba, problem_type):
    auc, acc, neg_mae, neg_mse, result, neg_log_loss = None, None, None, None, None, None,

    if problem_type in CLASSIFICATION:
        acc = metrics.accuracy(y_test_clean, y_pred)

    if problem_type == BI:
        auc = metrics.roc_auc(y_test_clean, y_pred)
        result = auc
    elif problem_type == MU:
        neg_log_loss = metrics.log_loss(y_test_clean, y_pred_proba)
        result = neg_log_loss
    else:
        neg_mse = metrics.mean_squared_error(y_test_clean, y_pred)
        neg_mae = metrics.mean_absolute_error(y_test_clean, y_pred)
        result = neg_mse

    return result, auc, neg_log_loss, acc, neg_mae, neg_mse


def convert_np_pred_prob(y_pred_proba: np.ndarray, label_cleaner: LabelCleaner, problem_type: str,
                         y: pd.DataFrame):
    if problem_type == BI:
        y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
        y_pred_proba = pd.DataFrame(y_pred_proba, columns=list(label_cleaner.inv_map.keys()),
                                    index=y.index)
    elif problem_type == MU:
        y_pred_proba = pd.DataFrame(y_pred_proba, columns=list(label_cleaner.inv_map.keys()),
                                    index=y.index)
    else:
        y_pred_proba = pd.Series(y_pred_proba, index=y.index, name=y.name)

    return y_pred_proba


def get_metric(problem_type):
    if problem_type == BI:
        return metrics.roc_auc
    elif problem_type == MU:
        return metrics.log_loss
    elif problem_type == RE:
        return metrics.root_mean_squared_error


def ECE_filter_pseudo(y_pred_proba: pd.DataFrame, val_pred_proba: pd.DataFrame, val_label: pd.Series, threshold: float):
    predictions = val_pred_proba.idxmax(axis=1)
    prediction_probs = val_pred_proba.max(axis=1)
    y_predicts = y_pred_proba.idxmax(axis=1)
    y_max_probs = y_pred_proba.max(axis=1)
    classes = predictions.unique()
    pseudo_indexes = pd.Series(data=False, index=y_pred_proba.index)

    for c in classes:
        predicted_as_c_idxes = predictions[predictions == c].index
        predicted_c_probs = prediction_probs.loc[predicted_as_c_idxes]
        val_labels_as_c = val_label.loc[predicted_as_c_idxes]

        accuracy = len(val_labels_as_c[val_labels_as_c == c]) / len(val_labels_as_c)
        confidence = np.mean(predicted_c_probs.to_numpy())
        calibration = accuracy - confidence

        class_threshold = threshold - (0.5 * calibration)

        holdout_as_c_idxes = y_predicts[y_predicts == c].index
        holdout_c_probs = y_max_probs.loc[holdout_as_c_idxes]

        pseudo_indexes.loc[(holdout_c_probs >= class_threshold).index] = True

    return pseudo_indexes[pseudo_indexes == True]


def ECE_filter_pseudo(y_pred_proba: pd.DataFrame, val_pred_proba: pd.DataFrame, val_label: pd.Series, threshold: float):
    predictions = val_pred_proba.idxmax(axis=1)
    prediction_probs = val_pred_proba.max(axis=1)
    y_predicts = y_pred_proba.idxmax(axis=1)
    y_max_probs = y_pred_proba.max(axis=1)
    classes = predictions.unique()
    pseudo_indexes = pd.Series(data=False, index=y_pred_proba.index)

    for c in classes:
        predicted_as_c_idxes = predictions[predictions == c].index
        predicted_c_probs = prediction_probs.loc[predicted_as_c_idxes]
        val_labels_as_c = val_label.loc[predicted_as_c_idxes]

        accuracy = len(val_labels_as_c[val_labels_as_c == c]) / len(val_labels_as_c)
        confidence = np.mean(predicted_c_probs.to_numpy())
        calibration = accuracy - confidence

        class_threshold = threshold - (0.5 * calibration)

        holdout_as_c_idxes = y_predicts[y_predicts == c].index
        holdout_c_probs = y_max_probs.loc[holdout_as_c_idxes]

        pseudo_indexes.loc[(holdout_c_probs >= class_threshold).index] = True

    return pseudo_indexes[pseudo_indexes == True]


def Jonas_filter(y_pred_proba: pd.DataFrame, val_pred_proba: pd.DataFrame, val_label: pd.Series, train_label: pd.Series,
                 threshold: float):
    predictions = val_pred_proba.idxmax(axis=1)
    prediction_probs = val_pred_proba.max(axis=1)
    y_predicts = y_pred_proba.idxmax(axis=1)
    y_max_probs = y_pred_proba.max(axis=1)
    classes = predictions.unique()
    pseudo_indexes = pd.Series(data=False, index=y_pred_proba.index)
    class_idxes_dict = dict()
    class_lens_dict = dict()

    for c in classes:
        predicted_as_c_idxes = predictions[predictions == c].index
        predicted_c_probs = prediction_probs.loc[predicted_as_c_idxes]
        val_labels_as_c = val_label.loc[predicted_as_c_idxes]

        accuracy = len(val_labels_as_c[val_labels_as_c == c]) / len(val_labels_as_c)
        confidence = np.mean(predicted_c_probs.to_numpy())
        calibration = accuracy - confidence

        class_threshold = threshold - (0.5 * calibration)

        holdout_as_c_idxes = y_predicts[y_predicts == c].index
        holdout_c_probs = y_max_probs.loc[holdout_as_c_idxes]

        class_idxes_dict[c] = (holdout_c_probs >= class_threshold).index
        class_lens_dict[c] = len(class_idxes_dict[c])

    train_val_fracs = train_label.value_counts(normalize=True)
    class_lens_series = pd.Series(class_lens_dict)
    min_class = class_lens_series.idxmin()
    min_class_num = class_lens_series.loc[min_class]

    return None


def run_pseudo_label(best_model: BaggedEnsembleModel,
                     X_clean: pd.DataFrame, y_clean: pd.Series,
                     X_test_clean: pd.DataFrame, y_test_clean: pd.Series,
                     previous_val_score: float, y_pred: pd.Series, y_pred_proba: pd.DataFrame,
                     max_iter: int, use_ECE: bool, problem_type: str, label_cleaner: LabelCleaner, threshold: float,
                     eval_metric, open_ml_id: int,
                     open_ml_metrics: Open_ML_Metrics, use_Jonas: bool):
    X_test_clean_og = X_test_clean.copy()
    y_test_clean_og = y_test_clean.copy()
    y_pred_proba_og = y_pred_proba.copy()

    result, auc, neg_log_loss, acc, mae, neg_mse = get_test_score(y_test_clean=y_test_clean_og,
                                                                  y_pred=y_pred,
                                                                  y_pred_proba=y_pred_proba,
                                                                  problem_type=problem_type)

    # Run Regular Pseudo
    for i in range(max_iter):
        if use_ECE and problem_type in CLASSIFICATION:
            val_pred_proba = best_model.get_oof_pred_proba()
            val_pred_proba = convert_np_pred_prob(y_pred_proba=val_pred_proba, label_cleaner=label_cleaner,
                                                  problem_type=problem_type, y=y_clean)

            assert val_pred_proba.index.equals(y_clean.index)

            if not use_Jonas:
                test_pseudo_idxes_true = ECE_filter_pseudo(y_pred_proba=y_pred_proba, val_label=y_clean,
                                                           val_pred_proba=val_pred_proba,
                                                           threshold=threshold)
            else:
                test_pseudo_idxes_true = Jonas_filter(y_pred_proba=y_pred_proba, val_label=y_clean,
                                                      val_pred_proba=val_pred_proba,
                                                      threshold=threshold,
                                                      train_label=label_cleaner.inverse_transform(y_clean))
        else:
            test_pseudo_idxes_true = TabularPredictor.filter_pseudo(None, y_pred_proba_og=y_pred_proba,
                                                                    problem_type=problem_type, threshold=threshold)

        test_pseudo_idxes = pd.Series(data=False, index=y_pred_proba.index)
        test_pseudo_idxes.loc[test_pseudo_idxes_true.index] = True
        y_pred_proba.loc[test_pseudo_idxes_true.index] = y_pred_proba.loc[test_pseudo_idxes_true.index]

        assert X_test_clean.index.identical(test_pseudo_idxes.index)

        if len(test_pseudo_idxes_true) > 0 and not test_pseudo_idxes_true.index.equals(y_pred_proba.index):
            X_pseudo = X_test_clean.loc[test_pseudo_idxes_true.index]
            y_pseudo = y_pred.loc[test_pseudo_idxes_true.index]

            model_pseudo = BaggedEnsembleModel(LGBModel(eval_metric=eval_metric))
            model_pseudo.fit(X=X_clean, y=y_clean, X_pseudo=X_pseudo, y_pseudo=y_pseudo,
                             k_fold=10)  # Perform 10-fold bagging

            test_not_pseudo_idxes = test_pseudo_idxes[test_pseudo_idxes == False].index

            assert not test_not_pseudo_idxes.isin(test_pseudo_idxes_true.index).any()

            X_test_clean = X_test_clean.loc[test_not_pseudo_idxes]
            y_test_clean = y_test_clean.loc[test_not_pseudo_idxes]

            assert X_test_clean.index.identical(y_test_clean.index)
            assert not (test_pseudo_idxes_true.index.isin(X_test_clean.index)).any()

            y_pred.loc[test_pseudo_idxes_true.index] = y_pseudo

            y_pred_proba_np = model_pseudo.predict_proba(X_test_clean)
            y_pred_proba = convert_np_pred_prob(y_pred_proba=y_pred_proba_np, label_cleaner=label_cleaner,
                                                problem_type=problem_type, y=y_test_clean)

            assert y_test_clean_og.index.identical(y_pred.index)

            if i > 0:
                assert not y_pred_proba.equals(y_pred_proba_og)

            curr_score = get_bagged_model_val_score_avg(model_pseudo)

            if curr_score > previous_val_score:
                previous_val_score = curr_score
                result, auc, neg_log_loss, acc, mae, neg_mse = get_test_score(y_test_clean=y_test_clean_og,
                                                                              y_pred=y_pred,
                                                                              y_pred_proba=y_pred_proba,
                                                                              problem_type=problem_type)
                best_model = model_pseudo
            else:
                break
        else:
            break

    name = 'ECE Pseudo Label' if use_ECE else 'Pseudo Label'
    name = 'Jonas ' + name if use_Jonas else name
    open_ml_metrics.add(model_name=name, eval_p=previous_val_score, openml_id=open_ml_id,
                        accuracy=acc, auc=auc, neg_logloss=neg_log_loss, MAE=mae, neg_MSE=neg_mse, result=result,
                        iter=i, metric=eval_metric.name, threshold=threshold, problem_type=problem_type,
                        model_score=best_model.score(X_test_clean_og, y_test_clean_og))


def run(openml_id: int, threshold: float, max_iter: int, openml_metrics: Open_ML_Metrics):
    try:
        data = fetch_openml(data_id=openml_id, as_frame=True)
    except Exception as e:
        return

    features = data['data']
    target = data['target']
    label = data['target_names'][0]
    df = features.join(target)

    test_frac = default_holdout_frac(len(features))
    test_data = df.sample(frac=test_frac, random_state=1)
    train_data = df.drop(test_data.index)
    train_data = TabularDataset(train_data)  # can be local CSV file as well, returns Pandas DataFrame
    test_data = TabularDataset(test_data)  # another Pandas DataFrame
    X_clean, y_clean, X_test_clean, y_test_clean, problem_type, label_cleaner = clean_data(train_data=train_data,
                                                                                           test_data=test_data,
                                                                                           label=label)

    if problem_type not in CLASSIFICATION:
        return None

    eval_metric = get_metric(problem_type=problem_type)

    if eval_metric == metrics.log_loss:
        threshold = .99

    model_vanilla = BaggedEnsembleModel(LGBModel(eval_metric=eval_metric))
    model_vanilla.fit(X=X_clean, y=y_clean, k_fold=10)  # Perform 10-fold bagging

    y_pred_vanilla_np = model_vanilla.predict(X_test_clean)
    y_pred_vanilla_series = pd.Series(y_pred_vanilla_np, index=X_test_clean.index)

    val_score_vanilla = get_bagged_model_val_score_avg(bagged_model=model_vanilla)

    assert y_test_clean.index.equals(X_test_clean.index)

    y_test_pred_proba_np = model_vanilla.predict_proba(X_test_clean)

    assert len(y_test_pred_proba_np) == len(X_test_clean)

    y_test_pred_proba_df = convert_np_pred_prob(y_pred_proba=y_test_pred_proba_np, label_cleaner=label_cleaner,
                                                problem_type=problem_type, y=y_test_clean)

    result, auc, neg_log_loss, acc, mae, neg_mse = get_test_score(y_test_clean=y_test_clean,
                                                                  y_pred=y_pred_vanilla_series,
                                                                  y_pred_proba=y_test_pred_proba_df,
                                                                  problem_type=problem_type)

    openml_metrics.add(model_name='Vanilla', eval_p=val_score_vanilla, openml_id=openml_id,
                       accuracy=acc, auc=auc, neg_logloss=neg_log_loss, MAE=mae, neg_MSE=neg_mse, result=result, iter=0,
                       metric=eval_metric.name, model_score=model_vanilla.score(X_test_clean, y_test_clean), problem_type=problem_type)

    if problem_type in CLASSIFICATION:
        # run_pseudo_label(best_model=model_vanilla, X_clean=X_clean.copy(), y_clean=y_clean.copy(),
        #                  X_test_clean=X_test_clean.copy(), y_test_clean=y_test_clean.copy(),
        #                  previous_val_score=val_score_vanilla, y_pred=y_pred_vanilla_series.copy(),
        #                  y_pred_proba=y_test_pred_proba_df.copy(), max_iter=max_iter, use_ECE=True,
        #                  problem_type=problem_type, label_cleaner=label_cleaner, threshold=threshold,
        #                  eval_metric=eval_metric, open_ml_id=openml_id, open_ml_metrics=openml_metrics,
        #                  use_Jonas=True)

        run_pseudo_label(best_model=model_vanilla, X_clean=X_clean.copy(), y_clean=y_clean.copy(),
                         X_test_clean=X_test_clean.copy(), y_test_clean=y_test_clean.copy(),
                         previous_val_score=val_score_vanilla, y_pred=y_pred_vanilla_series.copy(),
                         y_pred_proba=y_test_pred_proba_df.copy(), max_iter=max_iter, use_ECE=True,
                         problem_type=problem_type, label_cleaner=label_cleaner, threshold=threshold,
                         eval_metric=eval_metric, open_ml_id=openml_id, open_ml_metrics=openml_metrics,
                         use_Jonas=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_path', type=str, nargs='+', help='Path to save results CSV', default='./results.csv')
    parser.add_argument('--threshold', type=float, nargs='+', help='Threshold for using pseudo labeling or not',
                        default=0.95)
    args = parser.parse_args()

    benchmark = [1468, 1596, 40981, 40984, 40975, 41163, 41147, 1111, 41164, 1169, 1486, 41143, 1461, 41167, 40668,
                 23512, 41146, 41169, 41027, 23517, 40685, 41165, 41161, 41159, 4135, 40996, 41138, 41166, 1464, 41168,
                 41150, 1489, 41142, 3, 12, 31, 1067, 54, 1590]
    openml_metrics = Open_ML_Metrics()

    # run(openml_id=31, threshold=args.threshold, max_iter=5, openml_metrics=openml_metrics)

    for id in benchmark:
        run(openml_id=id, threshold=args.threshold, max_iter=5, openml_metrics=openml_metrics)
        openml_metrics.generate_csv(args.save_path)
