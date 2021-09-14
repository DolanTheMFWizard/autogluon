import os
import json
import unittest
import argparse
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularPredictor

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--id', type=str, help='id given to this runs results', default='no_name')
parser.add_argument('--threshold', type=float,
                    help='Predictive probability threshold to be above in order to use for pseudo-labeling',
                    default=0.95)
parser.add_argument('--max_iter', type=int, help='Max number of iterations that pseudo-labeling can take', default=1)
parser.add_argument('--min_p', type=float,
                    help='Min percentage of data that has to be below threshold to use min_p percentage of data',
                    default=0.05)
parser.add_argument('--max_p', type=float,
                    help='Max percentage of data that has to exceed threshold to use max_p percentage of data',
                    default=0.8)
parser.add_argument('--p_eval', type=float, help='Percent of data to use for evaluation', default=0.2)
parser.add_argument('--reuse', type=bool, help='Whether to rerun inference on old pseudo-labelled points', default=True)
parser.add_argument('--validation_percent', type=float, help='What percent of train should be used for validation',
                    default=0.2)
args = parser.parse_args()

THRESHOLD = args.threshold
MAX_ITER = args.max_iter
MIN_P = args.min_p
MAX_P = args.max_p
IS_REUSE = args.reuse
EVAL_P = args.p_eval
VAL_P = args.validation_percent
results_path = './results'
run_id = args.id

# id = 1590
id = 31
data = fetch_openml(data_id=id, as_frame=True)
features = data['data']
target = data['target']
label = 'class'
df = features.join(target)
num_rows = len(df)

vanilla_accs = list()
pseudo_accs = list()
best_pseudo_model = list()

split_idx = int((1 - VAL_P) * num_rows)
train_split = df.iloc[:split_idx]
test_split = df.iloc[split_idx:]
test_split_no_label = test_split.drop(columns=label)
validation_idxes = np.random.choice(train_split.index, int(VAL_P * len(train_split)))
validation_data = train_split.iloc[validation_idxes]
new_train_data = train_split.iloc[~validation_idxes]

agp, y_pred_prob = TabularPredictor(label=label).bad_pseudo_fit(train_data=new_train_data, validation_data=validation_data, test_data=test_split_no_label)
# test_pred, best_model = fit_pseudo_end_to_end(train_split, test_split_no_label, label)
# final_predict = test_pred.idxmax(axis=1)

# agp = TabularPredictor(label=label).fit(train_data=new_train_data, tuning_data=validation_data)
# agp_pred = agp.predict(test_split_no_label)

# vanilla_acc = accuracy_score(agp_pred.to_numpy(), test_split[label].to_numpy())
# pseudo_label_acc = accuracy_score(final_predict.to_numpy(), test_split[label].to_numpy())
#
# vanilla_accs.append(vanilla_acc)
# pseudo_accs.append(pseudo_label_acc)
# best_pseudo_model.append(best_model.get_model_best())
#
# df = pd.DataFrame.from_dict(dict(percentage=, vanilla_scores=vanilla_accs, pseudo_scores=pseudo_accs,
#                                  best_pseudo_model=best_pseudo_model))
# label = 'reuse' if IS_REUSE else 'no_reuse'
#
# dir_path = os.path.join(results_path, run_id)
# os.mkdir(dir_path)
# df.to_csv(os.path.join(dir_path, 'results.csv'))
#
# with open(os.path.join(dir_path, 'config.json'), 'w') as outfile:
#     data = vars(args)
#     data['open_ml_id'] = id
#     json.dump(data, outfile)
