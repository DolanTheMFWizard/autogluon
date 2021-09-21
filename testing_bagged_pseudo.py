import autogluon.core.metrics as metrics
import numpy as np
import pandas as pd
from autogluon.core.data import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.core.utils import infer_problem_type
from autogluon.core.utils import set_logger_verbosity
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset
from autogluon.tabular.models import LGBModel
from autogluon.tabular.predictor.predictor import TabularPredictor
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from autogluon.core.utils import generate_train_test_split
from autogluon.core.utils.utils import default_holdout_frac

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

    print(f'Labels cleaned: {label_cleaner.inv_map}')
    print(f'inferred problem type as: {problem_type}')
    print('Cleaned label values:')
    print(y_clean.head(5))

    set_logger_verbosity(2)  # Set logger so more detailed logging is shown for tutorial

    feature_generator = AutoMLPipelineFeatureGenerator()
    X_clean = feature_generator.fit_transform(X)

    print(X_clean.head(5))

    # custom_model = LGBModel()
    # We could also specify hyperparameters to override defaults
    # custom_model = CustomRandomForestModel(hyperparameters={'max_depth': 10})
    # custom_model.fit(X=X_clean, y=y_clean)  # Fit custom model

    X_test_clean = feature_generator.transform(X_test)
    y_test_clean = label_cleaner.transform(y_test)

    return X_clean, y_clean, X_test_clean, y_test_clean, problem_type, label_cleaner


def get_test_score(y_test_clean, y_pred, problem_type):
    if problem_type == 'binary':
        return roc_auc_score(y_test_clean, y_pred)


def convert_np_pred_prob(y_test_pred_proba, label_cleaner, problem_type, y_test):
    if problem_type == 'binary':
        y_test_pred_proba = np.column_stack([1 - y_test_pred_proba, y_test_pred_proba])
        y_test_pred_proba = pd.DataFrame(y_test_pred_proba, columns=list(label_cleaner.inv_map.keys()),
                                         index=y_test.index)

    return y_test_pred_proba


def get_metric(problem_type):
    if problem_type == 'binary':
        return metrics.roc_auc


def run(openml_id: int, threshold: float, max_iter: int):
    data = fetch_openml(data_id=31, as_frame=True)
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

    eval_metric = get_metric(problem_type=problem_type)
    model_vanilla = BaggedEnsembleModel(LGBModel(eval_metric=eval_metric))
    model_vanilla.fit(X=X_clean, y=y_clean, k_fold=10)  # Perform 10-fold bagging
    y_pred = model_vanilla.predict(X_test_clean)

    test_score_vanilla = get_test_score(y_test_clean=y_test_clean, y_pred=y_pred, problem_type=problem_type)
    val_score_vanilla = get_bagged_model_val_score_avg(bagged_model=model_vanilla)

    y_test_pred_proba = model_vanilla.predict_proba(X_test_clean)
    y_test_pred_series = pd.Series(y_pred, index=X_test_clean.index)
    y_test_pred_proba_og = convert_np_pred_prob(y_test_pred_proba=y_test_pred_proba, label_cleaner=label_cleaner,
                                                problem_type=problem_type, y_test=y_test_clean)

    y_test_pred_proba = y_test_pred_proba_og.copy()

    best_val_score_pseudo, previous_score = val_score_vanilla, val_score_vanilla
    best_test_score_pseudo = test_score_vanilla

    y_test_clean_og = y_test_clean.copy()
    X_test_clean_og = X_test_clean.copy()

    # Run Regular Pseudo
    for i in range(max_iter):
        test_pseudo_idxes_true = TabularPredictor.filter_pseudo(None, y_pred_proba_og=y_test_pred_proba,
                                                           problem_type=problem_type, threshold=threshold)
        test_pseudo_idxes = pd.Series(data=False, index=y_test_pred_proba.index)
        test_pseudo_idxes.loc[test_pseudo_idxes_true.index] = True

        if len(test_pseudo_idxes) > 0:
            X_pseudo = X_test_clean.loc[test_pseudo_idxes.index]
            y_pseudo = y_test_pred_series.loc[test_pseudo_idxes.index]

            model_pseudo = BaggedEnsembleModel(LGBModel(eval_metric=eval_metric))
            model_pseudo.fit(X=X_clean, y=y_clean, X_pseudo=X_pseudo, y_pseudo=y_pseudo,
                             k_fold=10)  # Perform 10-fold bagging

            X_test_clean = X_test_clean.loc[test_pseudo_idxes[~test_pseudo_idxes].index]
            y_test_clean = y_test_clean.loc[test_pseudo_idxes[~test_pseudo_idxes].index]

            if len(X_test_clean) < 1:
                break

            y_pred = model_pseudo.predict(X_test_clean)
            y_pred = pd.Series(y_pred, index=X_test_clean.index)
            y_pred.loc[test_pseudo_idxes.index] = y_test_pred_series

            test_score = get_test_score(y_test_clean=y_test_clean, y_pred=y_pred, problem_type=problem_type)
            curr_score = get_bagged_model_val_score_avg(model_pseudo)

            if curr_score > previous_score:
                best_test_score_pseudo = test_score
                previous_score = curr_score
                best_val_score_pseudo = curr_score
            else:
                break

    print(f'Vanilla val score: {val_score_vanilla}, test score: {test_score_vanilla}')
    print(f'Pseudo val score: {best_val_score_pseudo}, test score: {best_test_score_pseudo}, iterations:{i}')

if __name__ == "__main__":
    benchmarck = [1468, 1596, 40981, 40984, 40975, 41163, 41147, 1111, 41164, 1169, 1486, 41143, 1461, 41167, 40668, 23512, 41146, 41169, 41027, 23517, 40685, 41165, 41161, 41159, 4135, 40996, 41138, 41166, 1464, 41168, 41150, 1489, 41142, 3, 12, 31, 1067, 54, 1590]
    run(openml_id=40668, threshold=0.95, max_iter=5)