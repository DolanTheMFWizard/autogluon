import numpy as np
import pandas as pd
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

for feat in categorical_features:
    train_data[feat] = pd.to_numeric(train_data[feat])

for feat in categorical_features:
    X_unlabeled[feat] = pd.to_numeric(X_unlabeled[feat])

num_samples = int(len(train_data)/2)
train_sample_1 = train_data.sample(num_samples).reset_index(drop=True)
train_sample_2 = train_data.sample(num_samples).reset_index(drop=True)
lam = np.random.beta(0.4, 0.4, num_samples)[:, None].repeat(len(train_data.columns), axis=1)

train_data_mixed = lam * train_sample_1 + (1 - lam) * train_sample_2

train_data.append(train_data_mixed).reset_index(drop=True)

predictor = TabularPredictor(label=label).fit(train_data)
predictor.predict(X_unlabeled)

print('')
