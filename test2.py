import numpy as np
import pandas as pd
from autogluon.core.constants import BINARY
from autogluon.tabular import TabularPredictor

from tabular.tests.unittests.test_tabular import load_data

train_file = 'train_data.csv'
test_file = 'test_data.csv'
directory_prefix = './datasets/'
dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip',
           'name': 'AdultIncomeBinaryClassification',
           'problem_type': BINARY,
           'label': 'class'}
label = dataset['label']

train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file,
                                  name=dataset['name'], url=dataset['url'])

test_data = test_data.drop(columns=[label])

predictor_og = TabularPredictor(label=label).fit(train_data, time_limit=10)
X, y, X_val, y_val, X_unlabeled, holdout_frac, num_bag_folds, groups = predictor_og._learner.general_data_processing(
    train_data, None, test_data, 0, 1)

train_data = X.copy()
y = y.reset_index(drop=True)
train_data[label] = y

numerical_features = train_data.columns[train_data.dtypes != 'category']
categorical_features = train_data.columns[train_data.dtypes == 'category']

if predictor_og.problem_type == 'regression':
    for feat in categorical_features:
        train_data[feat] = pd.to_numeric(train_data[feat])

    for feat in categorical_features:
        X_unlabeled[feat] = pd.to_numeric(X_unlabeled[feat])

    num_samples = int(len(train_data) / 2)
    train_sample_1 = train_data.sample(num_samples).reset_index(drop=True)
    train_sample_2 = train_data.sample(num_samples).reset_index(drop=True)
    lam = np.random.beta(0.4, 0.4, num_samples)[:, None].repeat(len(train_data.columns), axis=1)

    train_data_mixed = lam * train_sample_1 + (1 - lam) * train_sample_2

    train_data.append(train_data_mixed).reset_index(drop=True)
else:
    numerical_features = train_data.columns[train_data.dtypes != 'category']
    categorical_features = train_data.columns[train_data.dtypes == 'category']

    if not categorical_features.empty:
        grouped_df = train_data.groupby(by=list(categorical_features))
        mixed_rows_df = None
        for key, value in grouped_df.groups.items():
            num_rows = len(value)
            if num_rows < 2:
                continue

            if num_rows % 2 != 0:
                num_rows -= 1

            selected_rows = train_data.loc[value[:num_rows]]

            half_num_rows = int(num_rows/2)

            sample_1_df = selected_rows.iloc[:half_num_rows].reset_index(drop=True)
            sample_2_df = selected_rows.iloc[half_num_rows:].reset_index(drop=True)

            lam = np.random.beta(0.4, 0.4, half_num_rows)[:, None].repeat(len(numerical_features), axis=1)

            new_mixed_rows_df = lam * sample_1_df[numerical_features] + (1 - lam) * sample_2_df[numerical_features]

            if mixed_rows_df is not None:
                mixed_rows_df = mixed_rows_df.append(new_mixed_rows_df, ignore_index=True)
            else:
                mixed_rows_df = new_mixed_rows_df

        train_data = train_data.append(mixed_rows_df, ignore_index=True).reset_index(drop=True)


predictor = TabularPredictor(label=label).fit(train_data)
y_pred = predictor.predict_proba(X_unlabeled)

print('')
