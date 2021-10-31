import numpy as np
from autogluon.core.constants import REGRESSION
from autogluon.tabular import TabularPredictor

from tabular.tests.unittests.test_tabular import load_data

train_file = 'train_data.csv'
test_file = 'test_data.csv'
directory_prefix = './datasets/'
dataset = {'url': 'https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression.zip',
           'name': 'AmesHousingPriceRegression',
           'problem_type': REGRESSION,
           'label': 'SalePrice',
           'performance_val': 0.076}
label = dataset['label']

train_data, test_data = load_data(directory_prefix=directory_prefix, train_file=train_file, test_file=test_file,
                                  name=dataset['name'], url=dataset['url'])

test_data = test_data.drop(columns=[label])

predictor = TabularPredictor(label=label).fit(train_data, time_limit=10)
X, y, X_val, y_val, X_unlabeled, holdout_frac, num_bag_folds, groups = predictor._learner.general_data_processing(
    train_data, None, test_data, 0, 1)

train_data = X.copy()
y = y.reset_index(drop=True)
train_data[label] = y

numerical_features = train_data.columns[train_data.dtypes != 'category']
categorical_features = train_data.columns[train_data.dtypes == 'category']

train_data_mixed = None

for index, row in train_data.iterrows():
    train_data_subset = train_data.drop(index=index, axis='rows')
    row_equal_cate = (row[categorical_features] == train_data_subset[categorical_features]).all(axis='columns').copy()
    row_eq_true = row_equal_cate[row_equal_cate]

    if row_eq_true.empty:
        continue
    else:
        lam = np.random.beta(0.4, 0.4)
        sampled_idx = row_eq_true.sample(1)
        sampled_row = train_data_subset.loc[sampled_idx.index]

        x_curr_mixed = lam * sampled_row[numerical_features] + (1 - lam) * row[numerical_features]
        x_curr_mixed[categorical_features] = row[categorical_features]

        if train_data_mixed is not None:
            train_data_mixed = train_data_mixed.append(x_curr_mixed, ignore_index=True)
        else:
            train_data_mixed = x_curr_mixed

train_data_mixed = train_data_mixed.drop_duplicates()
train_data.append(train_data_mixed).reset_index(drop=True)

print('')
