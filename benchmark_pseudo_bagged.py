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


def filter_bagged_regression_pseudo(bagged_model: BaggedEnsembleModel, X_test_data):
    list_child_models = bagged_model.models
    first_child_model = bagged_model.load_child(list_child_models[0])
    X_preprocessed = bagged_model.preprocess(X=X_test_data, model=first_child_model)
    pred_proba = first_child_model.predict_proba(X=X_preprocessed, preprocess_nonadaptive=False)
    for child in list_child_models[1:]:
        curr_model = bagged_model.load_child(child)
        pred_proba = np.column_stack(
            [pred_proba, curr_model.predict_proba(X=X_preprocessed, preprocess_nonadaptive=False)])

    pred_sd = pd.Series(data=np.std(pred_proba, axis=1), index=X_test_data.index)
    pred_sd_z = (pred_sd - pred_sd.mean()) / pred_sd.std()

    # Sample 30% with lowest variance
    # pred_sd = pred_sd.sort_values(ascending=True)
    # num_sample = int(.3 * len(pred_sd))
    # df_filtered = pred_sd.head(num_sample)
    # return pd.Series(data=True, index=df_filtered.index)

    threshold = 0.5
    df_filtered = pred_sd_z.between(-1 * threshold, threshold)

    return df_filtered[df_filtered == True]


def ECE_filter_pseudo(y_pred_proba: pd.DataFrame, val_pred_proba: pd.DataFrame, val_label: pd.Series, threshold: float,
                      anneal_frac: float = 0.25):
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

        class_threshold = threshold - (anneal_frac * calibration)

        holdout_as_c_idxes = y_predicts[y_predicts == c].index
        holdout_c_probs = y_max_probs.loc[holdout_as_c_idxes]

        above_thres = (holdout_c_probs >= class_threshold)
        pseudo_indexes.loc[above_thres[above_thres == True].index] = True

    return pseudo_indexes[pseudo_indexes == True]


def scale_ece_filter(y_pred_proba: pd.DataFrame, val_pred_proba: pd.DataFrame, val_label: pd.Series,
                     threshold: float, anneal_frac: float = 0.25):
    predictions = val_pred_proba.idxmax(axis=1)
    prediction_probs = val_pred_proba.max(axis=1)
    y_predicts = y_pred_proba.idxmax(axis=1)
    y_max_probs = y_pred_proba.max(axis=1)
    classes = y_pred_proba.columns
    pseudo_indexes = pd.Series(data=False, index=y_pred_proba.index)
    class_idxes_dict = dict()
    class_lens_dict = dict()
    class_threshold_dict = dict()

    for c in classes:
        predicted_as_c_idxes = predictions[predictions == c].index
        predicted_c_probs = prediction_probs.loc[predicted_as_c_idxes]
        val_labels_as_c = val_label.loc[predicted_as_c_idxes]

        accuracy = len(val_labels_as_c[val_labels_as_c == c]) / len(val_labels_as_c)
        confidence = np.mean(predicted_c_probs.to_numpy())
        calibration = accuracy - confidence

        class_threshold = threshold - (anneal_frac * calibration)

        holdout_as_c_idxes = y_predicts[y_predicts == c].index
        holdout_c_probs = y_max_probs.loc[holdout_as_c_idxes]

        above_thres_bool = holdout_c_probs >= class_threshold
        class_idxes_dict[c] = above_thres_bool[above_thres_bool == True].index
        class_lens_dict[c] = len(class_idxes_dict[c])
        class_threshold_dict[c] = class_threshold

    class_lens_series = pd.Series(class_lens_dict)
    min_class = class_lens_series.idxmin()
    min_class_num = class_lens_series.loc[min_class]

    if min_class_num == 0:
        return pd.Series([])

    for c, idxes in class_idxes_dict.items():
        curr_probs = y_max_probs.loc[idxes]
        pseudo_indexes.loc[curr_probs.sort_values().head(min_class_num).index] = True

    return pseudo_indexes


def run_pseudo_label(best_model: BaggedEnsembleModel,
                     X_clean: pd.DataFrame, y_clean: pd.Series,
                     X_test_clean: pd.DataFrame, y_test_clean: pd.Series,
                     previous_val_score: float, y_pred: pd.Series, y_pred_proba: pd.DataFrame,
                     max_iter: int, use_ECE: bool, problem_type: str, label_cleaner: LabelCleaner, threshold: float,
                     eval_metric, open_ml_id: int,
                     open_ml_metrics: Open_ML_Metrics, use_Jonas: bool, use_new_regression: bool):
    X_test_clean_og = X_test_clean.copy()
    y_test_clean_og = y_test_clean.copy()
    y_pred_og = y_pred.copy()
    y_pred_proba_og = y_pred_proba.copy()

    result, auc, neg_log_loss, acc, mae, neg_mse = get_test_score(y_test_clean=y_test_clean_og,
                                                                  y_pred=y_pred,
                                                                  y_pred_proba=y_pred_proba,
                                                                  problem_type=problem_type)
    is_classification = problem_type in CLASSIFICATION
    # Run Regular Pseudo
    for i in range(max_iter):
        if use_ECE and is_classification:
            val_pred_proba = best_model.get_oof_pred_proba()
            val_pred_proba = convert_np_pred_prob(y_pred_proba=val_pred_proba, label_cleaner=label_cleaner,
                                                  problem_type=problem_type, y=y_clean)

            if not use_Jonas:
                test_pseudo_idxes_true = ECE_filter_pseudo(y_pred_proba=y_pred_proba,
                                                           val_label=label_cleaner.inverse_transform(y_clean),
                                                           val_pred_proba=val_pred_proba,
                                                           threshold=threshold)
            else:
                test_pseudo_idxes_true = scale_ece_filter(y_pred_proba=y_pred_proba, val_pred_proba=val_pred_proba,
                                                          val_label=label_cleaner.inverse_transform(y_clean),
                                                          threshold=threshold)
        elif not is_classification and use_new_regression:
            test_pseudo_idxes_true = filter_bagged_regression_pseudo(bagged_model=best_model, X_test_data=X_test_clean)
        else:
            test_pseudo_idxes_true = TabularPredictor.filter_pseudo(None, y_pred_proba_og=y_pred_proba,
                                                                    problem_type=problem_type, threshold=threshold)

        test_pseudo_idxes = pd.Series(data=False, index=y_pred_proba.index)
        test_pseudo_idxes.loc[test_pseudo_idxes_true.index] = True

        if len(test_pseudo_idxes_true) > 0 and len(y_pred_proba) > len(test_pseudo_idxes_true):
            # Get current round of pseudo using selector
            X_new_pseudo = X_test_clean.loc[test_pseudo_idxes_true.index]
            y_new_pseudo = y_pred.loc[test_pseudo_idxes_true.index]

            # Append or create pseudo data if iteration is 0
            if i == 0:
                X_pseudo = X_new_pseudo
                y_pseudo = y_new_pseudo
            else:
                X_pseudo = X_pseudo.append(X_new_pseudo, verify_integrity=True)
                y_pseudo = y_pseudo.append(y_new_pseudo, verify_integrity=True)

            # Cut down test data to remove pseudo-labeled data. Don't want to run inference on seen data
            test_not_pseudo_idxes = test_pseudo_idxes[test_pseudo_idxes == False].index
            X_test_clean = X_test_clean.loc[test_not_pseudo_idxes]
            y_test_clean = y_test_clean.loc[test_not_pseudo_idxes]

            # Fit model on new data
            model_pseudo = BaggedEnsembleModel(LGBModel(eval_metric=eval_metric))
            model_pseudo.fit(X=X_clean, y=y_clean, X_pseudo=X_pseudo, y_pseudo=y_pseudo,
                             k_fold=10)  # Perform 10-fold bagging

            # Get new predictions
            y_pred_proba_np = model_pseudo.predict_proba(X_test_clean)
            y_pred_proba = convert_np_pred_prob(y_pred_proba=y_pred_proba_np, label_cleaner=label_cleaner,
                                                problem_type=problem_type, y=y_test_clean)
            y_pred = pd.Series(model_pseudo.predict(X_test_clean), name=y_test_clean.name, index=X_test_clean.index)

            # Compare validation score
            curr_score = get_bagged_model_val_score_avg(model_pseudo)
            if curr_score > previous_val_score:
                previous_val_score = curr_score
                y_pred_og.loc[X_test_clean.index] = y_pred
                y_pred_proba_og.loc[X_test_clean.index] = y_pred_proba
                result, auc, neg_log_loss, acc, mae, neg_mse = get_test_score(y_test_clean=y_test_clean_og,
                                                                              y_pred=y_pred_og,
                                                                              y_pred_proba=y_pred_proba_og,
                                                                              problem_type=problem_type)
                best_model = model_pseudo
            else:
                break
        else:
            break

    name = 'ECE Pseudo Label' if use_ECE and is_classification else 'Pseudo Label'
    name = 'Jonas ' + name if use_Jonas and is_classification else name
    name = 'New Reg Method' if not is_classification and use_new_regression else name
    open_ml_metrics.add(model_name=name, eval_p=previous_val_score, openml_id=open_ml_id,
                        accuracy=acc, auc=auc, neg_logloss=neg_log_loss, MAE=mae, neg_MSE=neg_mse, result=result,
                        iter=i, metric=eval_metric.name, threshold=threshold, problem_type=problem_type,
                        model_score=best_model.score(X_test_clean_og, y_test_clean_og))


def run(openml_id: int, threshold: float, max_iter: int, open_ml_metrics: Open_ML_Metrics, percent_test: float = None):
    try:
        data = fetch_openml(data_id=openml_id, as_frame=True)
    except Exception as e:
        return None

    features = data['data']
    target = data['target']
    label = data['target_names'][0]
    df = features.join(target)

    test_frac = default_holdout_frac(len(features)) if percent_test is None else percent_test
    test_data = df.sample(frac=test_frac, random_state=1)
    train_data = df.drop(test_data.index)
    train_data = TabularDataset(train_data)  # can be local CSV file as well, returns Pandas DataFrame
    test_data = TabularDataset(test_data)  # another Pandas DataFrame
    X_clean, y_clean, X_test_clean, y_test_clean, problem_type, label_cleaner = clean_data(train_data=train_data,
                                                                                           test_data=test_data,
                                                                                           label=label)

    if len(features) > 40000 or problem_type not in CLASSIFICATION:
        return None

    eval_metric = get_metric(problem_type=problem_type)

    # if eval_metric == metrics.log_loss:
    #     threshold = .98

    model_vanilla = BaggedEnsembleModel(LGBModel(eval_metric=eval_metric))
    model_vanilla.fit(X=X_clean, y=y_clean, k_fold=10)  # Perform 10-fold bagging

    y_pred_vanilla_np = model_vanilla.predict(X_test_clean)
    y_pred_vanilla_series = pd.Series(y_pred_vanilla_np, index=X_test_clean.index)
    val_score_vanilla = get_bagged_model_val_score_avg(bagged_model=model_vanilla)
    y_test_pred_proba_np = model_vanilla.predict_proba(X_test_clean)
    y_test_pred_proba_df = convert_np_pred_prob(y_pred_proba=y_test_pred_proba_np, label_cleaner=label_cleaner,
                                                problem_type=problem_type, y=y_test_clean)

    is_classification = problem_type in CLASSIFICATION

    if is_classification:
        run_pseudo_label(best_model=model_vanilla, X_clean=X_clean.copy(), y_clean=y_clean.copy(),
                         X_test_clean=X_test_clean.copy(), y_test_clean=y_test_clean.copy(),
                         previous_val_score=val_score_vanilla, y_pred=y_pred_vanilla_series.copy(),
                         y_pred_proba=y_test_pred_proba_df.copy(), max_iter=max_iter, use_ECE=True,
                         problem_type=problem_type, label_cleaner=label_cleaner, threshold=threshold,
                         eval_metric=eval_metric, open_ml_id=openml_id, open_ml_metrics=open_ml_metrics,
                         use_Jonas=True, use_new_regression=False)

        run_pseudo_label(best_model=model_vanilla, X_clean=X_clean.copy(), y_clean=y_clean.copy(),
                         X_test_clean=X_test_clean.copy(), y_test_clean=y_test_clean.copy(),
                         previous_val_score=val_score_vanilla, y_pred=y_pred_vanilla_series.copy(),
                         y_pred_proba=y_test_pred_proba_df.copy(), max_iter=max_iter, use_ECE=True,
                         problem_type=problem_type, label_cleaner=label_cleaner, threshold=threshold,
                         eval_metric=eval_metric, open_ml_id=openml_id, open_ml_metrics=open_ml_metrics,
                         use_Jonas=False, use_new_regression=False)
    else:
        run_pseudo_label(best_model=model_vanilla, X_clean=X_clean.copy(), y_clean=y_clean.copy(),
                         X_test_clean=X_test_clean.copy(), y_test_clean=y_test_clean.copy(),
                         previous_val_score=val_score_vanilla, y_pred=y_pred_vanilla_series.copy(),
                         y_pred_proba=y_test_pred_proba_df.copy(), max_iter=max_iter, use_ECE=True,
                         problem_type=problem_type, label_cleaner=label_cleaner, threshold=threshold,
                         eval_metric=eval_metric, open_ml_id=openml_id, open_ml_metrics=open_ml_metrics,
                         use_Jonas=False, use_new_regression=True)

    run_pseudo_label(best_model=model_vanilla, X_clean=X_clean.copy(), y_clean=y_clean.copy(),
                     X_test_clean=X_test_clean.copy(), y_test_clean=y_test_clean.copy(),
                     previous_val_score=val_score_vanilla, y_pred=y_pred_vanilla_series.copy(),
                     y_pred_proba=y_test_pred_proba_df.copy(), max_iter=max_iter, use_ECE=False,
                     problem_type=problem_type, label_cleaner=label_cleaner, threshold=threshold,
                     eval_metric=eval_metric, open_ml_id=openml_id, open_ml_metrics=open_ml_metrics,
                     use_Jonas=False, use_new_regression=False)

    result, auc, neg_log_loss, acc, mae, neg_mse = get_test_score(y_test_clean=y_test_clean,
                                                                  y_pred=y_pred_vanilla_series,
                                                                  y_pred_proba=y_test_pred_proba_df,
                                                                  problem_type=problem_type)

    open_ml_metrics.add(model_name='Vanilla', eval_p=val_score_vanilla, openml_id=openml_id,
                        accuracy=acc, auc=auc, neg_logloss=neg_log_loss, MAE=mae, neg_MSE=neg_mse, result=result,
                        iter=0,
                        metric=eval_metric.name, model_score=model_vanilla.score(X_test_clean, y_test_clean),
                        problem_type=problem_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_path', type=str, nargs='?', help='Path to save results CSV',
                        default='./classification_annealp25ECE.csv')
    parser.add_argument('--threshold', type=float, nargs='?', help='Threshold for using pseudo labeling or not',
                        default=0.95)
    parser.add_argument('--test_percent', type=float, nargs='?', help='Percent of total data used for testing',
                        default=None)
    args = parser.parse_args()

    # AutoML Benchmark
    benchmark = [1468, 1596, 40981, 40984, 40975, 41163, 41147, 1111, 41164, 1169, 1486, 41143, 1461, 41167, 40668,
                 23512, 41146, 41169, 41027, 23517, 40685, 41165, 41161, 41159, 4135, 40996, 41138, 41166, 1464, 41168,
                 41150, 1489, 41142, 3, 12, 31, 1067, 54, 1590]

    # Regressions
    # benchmark = [215, 216, 558, 564, 565, 574, 503, 505, 507, 528, 531, 537, 541, 546, 547, 549, 512, 497, 344, 308,
    #              287, 405, 41021, 550, 495, 227, 223, 189, 196, 183]

    openml_metrics = Open_ML_Metrics()
    percent_test = '_' + str(int(args.test_percent * 100)) if args.test_percent is not None else ''
    path = args.save_path[:-4] + f'_{int(args.threshold * 100)}Threshold' + percent_test + args.save_path[-4:]

    # run(openml_id=41021, threshold=args.threshold, max_iter=5, open_ml_metrics=openml_metrics,
    #     percent_test=args.test_percent)

    for id in benchmark:
        run(openml_id=id, threshold=args.threshold, max_iter=5, open_ml_metrics=openml_metrics,
            percent_test=args.test_percent)
        openml_metrics.generate_csv(path)
