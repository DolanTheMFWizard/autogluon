import os
import argparse
import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt


def make_key(threshold: float, reuse: bool):
    reuse_label = 'reuse' if reuse else 'no reuse'
    t_label = str(threshold)

    return reuse_label + '_' + t_label


def reverse_key(key: str):
    attr_list = key.split('_')
    is_reuse = True if attr_list[0] == 'reuse' else False
    threshold = float(attr_list[1])

    return is_reuse, threshold


parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, help='Path to CSV file to run inference on')
parser.add_argument('--top_k', type=int, help='Number of top averaging accuracy models to add to plot', default=None)
args = parser.parse_args()

threshold_list = [0.5, 0.75, 0.9, 0.95]
reuse_list = [True, False]
threshold_list_len = len(threshold_list)
reuse_list_len = len(reuse_list)

args.top_k = None if (threshold_list_len * reuse_list_len) < args.top_k else args.top_k

df = pd.read_csv(args.file_path)
fig = plt.figure()

avg_accuracy_dict = dict()
val = df[df.procedure == 'Vanilla']
vanilla_X = np.array(val.eval_percentage)
vanilla_y = np.array(val['accuracy'])
plt.plot(vanilla_X, vanilla_y, label='Vanilla', linewidth=4)

if args.top_k is None:
    for t in threshold_list:
        for r in reuse_list:
            value = df[df.threshold == t][df.is_reuse == r]
            X = np.array(value.eval_percentage)
            y = np.array(value['accuracy'])

            reuse_label = 'reuse' if r else 'no reuse'
            plt.plot(X, y, label=f'Pseudo threshold={t} {reuse_label}')
else:
    for t in threshold_list:
        for r in reuse_list:
            value = df[df.threshold == t][df.is_reuse == r]
            acc = np.array(value['accuracy'])
            key = make_key(t, r)

            avg_accuracy_dict[key] = np.mean(acc)

    top_k_models = dict(sorted(avg_accuracy_dict.items(), key=itemgetter(1), reverse=True)[:args.top_k])
    topk_models_key = list(top_k_models.keys())
    for i in range(len(topk_models_key)):
        key = topk_models_key[i]
        is_reuse, threshold = reverse_key(key)
        value = df[df.threshold == threshold][df.is_reuse == is_reuse]
        X = np.array(value.eval_percentage)
        y = np.array(value['accuracy'])

        reuse_label = 'reuse' if is_reuse else 'no reuse'
        plt.plot(X, y, label=f'Pseudo threshold={threshold} {reuse_label} rank={i + 1}')

plt.xlabel('% of OpenML Data Used for Evaluation')
plt.title(os.path.basename(args.file_path))
plt.ylabel('Accuracy')
plt.xticks(vanilla_X)
plt.legend()
plt.show()
