import argparse

import autogluon.core.metrics as metrics
import numpy as np
import pandas as pd
import scipy.special
import torch
from autogluon.core.utils import infer_problem_type
from autogluon.core.utils.utils import default_holdout_frac
from autogluon.tabular.predictor.predictor import TabularPredictor
from sklearn.datasets import fetch_openml

BI = 'binary'
MU = 'multiclass'
CLASSIFICATION = [BI, MU]


def augmented_filter(X_aug, y_aug, y_real, problem_type):
    """ Filters out certain points from the augmented dataset so that it better matches the real data """
    indices_to_drop = []
    y_aug_hard = pd.Series(y_aug.idxmax(axis="columns"))
    if problem_type == MU:
        y_aug = pd.DataFrame(y_aug)
    else:
        y_aug = pd.Series(y_aug[y_aug.columns[-1]])

    p_y = y_real.value_counts(sort=False).sort_index() / len(y_real)
    desired_class_cnts = p_y * len(y_aug_hard)
    y_aug_cnts = y_aug_hard.value_counts(sort=False).sort_index()
    if len(y_aug_cnts) != len(p_y):  # some classes were never predicted, so cannot match label distributions
        return None

    scaling = np.min(y_aug_cnts / desired_class_cnts)
    desired_class_cnts = scaling * desired_class_cnts
    desired_class_cnts = np.maximum(np.floor(desired_class_cnts), 1).astype(int)
    num_to_drop = np.maximum(0, y_aug_cnts - desired_class_cnts)
    for clss in p_y.index.values.tolist():
        if clss in y_aug_cnts.index:
            print("clss", clss)
            clss_inds = y_aug_hard[y_aug_hard == clss].index.tolist()
            print("clss_inds", clss_inds)
            indices_to_drop += clss_inds[:num_to_drop[clss]]

    if len(indices_to_drop) == 0:
        return None

    y_aug.drop(indices_to_drop)
    print(f"Augmented training dataset has {len(y_aug)} datapoints after augmented_filter")
    return y_aug


def get_test_score(y_test_clean, y_pred, y_pred_proba, problem_type):
    auc, acc, neg_mae, neg_mse, result, neg_log_loss = None, None, None, None, None, None

    if problem_type in CLASSIFICATION:
        acc = metrics.accuracy(y_test_clean, y_pred)

    if problem_type == BI:
        auc = metrics.roc_auc(y_test_clean, y_pred)
        result = auc
    elif problem_type == MU:
        try:
            neg_log_loss = metrics.log_loss(y_test_clean, y_pred_proba)
        except Exception as e:
            from sklearn.metrics import log_loss
            neg_log_loss = -1 * log_loss(y_test_clean, y_pred_proba)

        result = neg_log_loss
    else:
        neg_mse = metrics.mean_squared_error(y_test_clean, y_pred)
        neg_mae = metrics.mean_absolute_error(y_test_clean, y_pred)
        result = neg_mse

    return result, auc, neg_log_loss, acc, neg_mae, neg_mse


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
            threshold: float = None, result: float = None, iter: int = 0,
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


def inverse_softmax(y_pred_proba: pd.DataFrame):
    return np.log2(y_pred_proba)


def balance_pseudo(y_pred_proba_og: pd.DataFrame, X_test: pd.DataFrame, y_label: pd.DataFrame, predictor):
    y_pred_proba = augmented_filter(X_aug=X_test.copy(), y_aug=y_pred_proba_og.copy(), y_real=y_label.copy())

    if y_pred_proba is None:
        return y_pred_proba_og, y_pred_proba_og.idxmax(axis=1), predictor

    y_pred_proba_holdout = y_pred_proba_og.copy()
    y_pred = y_pred_proba.idxmax(axis=1)
    X = X_test.loc[y_pred.index]
    X[predictor.label] = y_pred
    pseudo_indices = pd.Series(data=False, index=X_test.index)
    pseudo_indices.loc[y_pred.index] = True
    PL_predictor = TabularPredictor(label=predictor.label, eval_metric=predictor.eval_metric).fit(train_data=X)
    train_data = X_test.loc[pseudo_indices[pseudo_indices == False].index]

    if len(train_data) == 0:
        y_pred_proba = PL_predictor.predict_proba(data=X_test)
    else:
        y_pred_proba = PL_predictor.predict_proba(data=X_test.loc[pseudo_indices[pseudo_indices == False].index])

    y_pred_proba_holdout.loc[y_pred_proba.index] = y_pred_proba

    return y_pred_proba_holdout, y_pred_proba_holdout.idxmax(axis=1), PL_predictor


def balance_pseudo_no_holdouts(y_pred_proba_og: pd.DataFrame, X_test: pd.DataFrame, y_label: pd.DataFrame, predictor):
    y_pred_proba = augmented_filter(X_aug=X_test.copy(), y_aug=y_pred_proba_og.copy(), y_real=y_label.copy())

    if y_pred_proba is None:
        return y_pred_proba_og, y_pred_proba_og.idxmax(axis=1), predictor

    y_pred = y_pred_proba.idxmax(axis=1)
    X = X_test.loc[y_pred.index]
    X[predictor.label] = y_pred
    pseudo_indices = pd.Series(data=False, index=X_test.index)
    pseudo_indices.loc[y_pred.index] = True
    PL_predictor = TabularPredictor(label=predictor.label, eval_metric=predictor.eval_metric).fit(train_data=X)
    y_pred_proba = PL_predictor.predict_proba(data=X_test)

    return y_pred_proba, y_pred_proba.idxmax(axis=1), PL_predictor


def temperature_scale(predictor: TabularPredictor, y_pred_proba: pd.DataFrame,
                      validation_data: pd.DataFrame, y_label: pd.DataFrame):
    label = y_label.name
    temperature_param = torch.nn.Parameter(torch.ones(1))
    X_validation_data = validation_data.drop(columns=[label])
    y_validation_data = validation_data[label]
    val_pred_proba = predictor.predict_proba(data=X_validation_data)
    logits_df = inverse_softmax(y_pred_proba=val_pred_proba)
    logits = torch.tensor(logits_df.values)
    y_validation_data = predictor._learner.label_cleaner.transform(y_validation_data)
    y = torch.tensor(y_validation_data.values)

    nll_criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.LBFGS([temperature_param], lr=0.01, max_iter=1000)

    def run():
        optimizer.zero_grad()
        temperature = temperature_param.unsqueeze(1).expand(logits.size(0), logits.size(1))
        curr_probs = logits / temperature
        loss = -1 * nll_criterion(curr_probs, y)
        loss.backward()
        return loss

    optimizer.step(run)

    logits = inverse_softmax(y_pred_proba)
    output = logits / temperature_param[0].item()
    output = scipy.special.softmax(output, axis=1)

    return output, output.idxmax(axis=1)


def epsilon_shift(predictor: TabularPredictor, y_pred_proba: pd.DataFrame,
                  validation_data: pd.DataFrame, y_label: pd.DataFrame):
    label = y_label.name
    epsilon_param = torch.nn.Parameter(torch.ones(1))
    X_validation_data = validation_data.drop(columns=[label])
    y_validation_data = validation_data[label]
    val_pred_proba = predictor.predict_proba(data=X_validation_data)
    logits_df = inverse_softmax(y_pred_proba=val_pred_proba)
    logits = torch.tensor(logits_df.values)
    y_validation_data = predictor._learner.label_cleaner.transform(y_validation_data)
    y = torch.tensor(y_validation_data.values)

    nll_criterion = torch.nn.NLLLoss().cuda()
    optimizer = torch.optim.LBFGS([epsilon_param], lr=0.01, max_iter=1000)

    def run():
        optimizer.zero_grad()
        epsilon = epsilon_param.unsqueeze(1).expand(logits.size(0), logits.size(1))
        new_logits = (logits + epsilon)
        sum_new_logits = torch.sum(new_logits, dim=1)
        loss = nll_criterion(new_logits / sum_new_logits[:, None], y)
        loss.backward()
        return loss

    optimizer.step(run)

    output = y_pred_proba + epsilon_param[0].item()
    output = output / output.sum()

    return output, output.idxmax(axis=1)


def run(open_ml_data, open_ml_metrics):
    features = open_ml_data['data']
    target = open_ml_data['target']
    label = open_ml_data['target_names'][0]

    problem_type = infer_problem_type(target)

    if len(features) > 10000:
        print(f'Id: {id} is over 10000 rows')
        return

    if problem_type != MU:
        print(f'Id: {id} is not multiclass')
        return

    if problem_type is BI:
        eval_metric = 'roc_auc'
    else:
        eval_metric = 'nll'

    df = features.join(target)
    test_frac = default_holdout_frac(len(features)) if percent_test is None else percent_test
    test_data = df.sample(frac=test_frac, random_state=1)
    y = test_data[label]
    test_data = test_data.drop(columns=label)
    train_data = df.drop(test_data.index)
    validation_frac = default_holdout_frac(len(train_data))
    validation_data = train_data.sample(frac=validation_frac, random_state=1)
    train_data = train_data.drop(validation_data.index)

    predictor = TabularPredictor(label=label, eval_metric=eval_metric).fit(train_data=train_data,
                                                                           tuning_data=validation_data)
    y_pred = predictor.predict(data=test_data)
    y_pred_proba = predictor.predict_proba(data=test_data)
    result, auc, neg_log_loss, acc, neg_mae, neg_mse = get_test_score(y_test_clean=y, y_pred=y_pred,
                                                                      y_pred_proba=y_pred_proba,
                                                                      problem_type=problem_type)

    open_ml_metrics.add(model_name='Vanilla', eval_p=predictor._trainer.leaderboard()['score_val'][0], openml_id=id,
                        accuracy=acc, auc=auc, neg_logloss=neg_log_loss, MAE=neg_mae, neg_MSE=neg_mse, result=result,
                        metric=eval_metric, problem_type=problem_type)

    y_pred_proba_temp, y_pred_temp = temperature_scale(predictor=predictor, y_pred_proba=y_pred_proba,
                                                       validation_data=validation_data, y_label=y)

    result, auc, neg_log_loss, acc, neg_mae, neg_mse = get_test_score(y_test_clean=y, y_pred=y_pred_temp,
                                                                      y_pred_proba=y_pred_proba_temp,
                                                                      problem_type=problem_type)

    open_ml_metrics.add(model_name='Temperature', eval_p=predictor._trainer.leaderboard()['score_val'][0],
                        openml_id=id, accuracy=acc, auc=auc, neg_logloss=neg_log_loss, MAE=neg_mae, neg_MSE=neg_mse,
                        result=result, metric=eval_metric, problem_type=problem_type)

    y_pred_proba_ep, y_pred_ep = epsilon_shift(predictor=predictor, y_pred_proba=y_pred_proba,
                                               validation_data=validation_data, y_label=y)

    result, auc, neg_log_loss, acc, neg_mae, neg_mse = get_test_score(y_test_clean=y, y_pred=y_pred_ep,
                                                                      y_pred_proba=y_pred_proba_ep,
                                                                      problem_type=problem_type)

    open_ml_metrics.add(model_name='Epsilon', eval_p=predictor._trainer.leaderboard()['score_val'][0],
                        openml_id=id, accuracy=acc, auc=auc, neg_logloss=neg_log_loss, MAE=neg_mae, neg_MSE=neg_mse,
                        result=result, metric=eval_metric, problem_type=problem_type)

    y_pred_proba_Jonas, y_pred_Jonas, PL_predictor = balance_pseudo(y_pred_proba_og=y_pred_proba, X_test=test_data,
                                                                    y_label=y, predictor=predictor)

    result, auc, neg_log_loss, acc, neg_mae, neg_mse = get_test_score(y_test_clean=y, y_pred=y_pred_Jonas,
                                                                      y_pred_proba=y_pred_proba_Jonas,
                                                                      problem_type=problem_type)

    open_ml_metrics.add(model_name='Jonas', eval_p=PL_predictor._trainer.leaderboard()['score_val'][0],
                        openml_id=id, accuracy=acc, auc=auc, neg_logloss=neg_log_loss, MAE=neg_mae, neg_MSE=neg_mse,
                        result=result, metric=eval_metric, problem_type=problem_type)

    y_pred_proba_Jonas, y_pred_Jonas, PL_predictor = balance_pseudo_no_holdouts(y_pred_proba_og=y_pred_proba,
                                                                                X_test=test_data,
                                                                                y_label=y, predictor=predictor)

    result, auc, neg_log_loss, acc, neg_mae, neg_mse = get_test_score(y_test_clean=y, y_pred=y_pred_Jonas,
                                                                      y_pred_proba=y_pred_proba_Jonas,
                                                                      problem_type=problem_type)

    open_ml_metrics.add(model_name='Jonas_no_holdout', eval_p=PL_predictor._trainer.leaderboard()['score_val'][0],
                        openml_id=id, accuracy=acc, auc=auc, neg_logloss=neg_log_loss, MAE=neg_mae, neg_MSE=neg_mse,
                        result=result, metric=eval_metric, problem_type=problem_type)

    open_ml_metrics.generate_csv(path=args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_path', type=str, nargs='?', help='Path to save results CSV',
                        default='./classification.csv')
    parser.add_argument('--test_percent', type=float, nargs='?', help='Percent of total data used for testing',
                        default=None)
    args = parser.parse_args()

    # AutoML Benchmark
    benchmark = [1468, 1596, 40981, 40984, 40975, 41163, 41147, 1111, 41164, 1169, 1486, 41143, 1461, 41167, 40668,
                 23512, 41146, 41169, 41027, 23517, 40685, 41165, 41161, 41159, 4135, 40996, 41138, 41166, 1464, 41168,
                 41150, 1489, 41142, 3, 12, 31, 1067, 54, 1590] + [23381, 1476, 1459, 23380, 40496, 40971, 1515, 1467,
                                                                   1479, 40499, 40966, 40982, 1485, 6332, 1462,
                                                                   1480, 1510, 40994, 40983, 40978, 1468, 40670, 40981,
                                                                   40984, 40701, 40975, 1486, 1461, 40668, 41027,
                                                                   1464, 40536, 1590, 23517, 15, 11, 37, 29, 334, 335,
                                                                   333, 50, 451, 1038, 1046, 188, 42, 307, 54, 470,
                                                                   469, 151, 458, 377, 375, 1063]

    metrics_object = Open_ML_Metrics()
    percent_test = args.test_percent

    for id in benchmark:
        try:
            data = fetch_openml(data_id=id, as_frame=True)
            print(f'Running open ml Id: {id}')
        except Exception as e:
            print(e)
            continue

        try:
            run(data, metrics_object)
        except Exception as e:
            print(e)
            continue
