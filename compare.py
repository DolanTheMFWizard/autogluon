import argparse

import numpy as np
import pandas as pd


def run(path, metric):
    df = pd.read_csv(path)
    open_ml_ids = df['openml_id'].unique()
    num_openml_ids = len(open_ml_ids)
    rank_sums = None
    score_sums = None
    num_iter_sums = None

    model_list = list(df['model'].unique())
    model_wins_dict = dict()
    for m in model_list:
        model_wins_dict[m] = 0

    for id in open_ml_ids:
        df_openml = df[df.openml_id == id]
        # df_openml = df_openml[df_openml.model != 'Pseudo Label']
        ranks = df_openml[metric].rank(axis=0, method='average', ascending=False).to_numpy()
        scores = df_openml[metric].to_numpy()

        # models = df_openml['model'].to_numpy()

        # if scores[0] != scores[1]:
        #     model_winner = models[ranks.argmin()]
        #
        #     model_wins_dict[model_winner] += 1

        num_iter = df_openml['max_num_ter'].to_numpy()

        if id == open_ml_ids[0]:
            rank_sums = ranks
            score_sums = scores
            num_iter_sums = num_iter
        else:
            rank_sums += ranks
            score_sums += scores
            num_iter_sums += num_iter

    print('Model names:')
    print(df_openml['model'].to_numpy())
    print('Average ranks:')
    print(rank_sums / num_openml_ids)
    print(f'Average {metric} score:')
    print(score_sums / num_openml_ids)
    print('Average number of iterations:')
    print(num_iter_sums / num_openml_ids)
    print('Models 1st place counts:')
    print(model_wins_dict)
    print('Open ML Ids:')
    print(open_ml_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-metric', help='Metric to evaluate models by', default='result', type=str)
    parser.add_argument('-path', help='Path to file', default='./classification_annealp25ECE_95Threshold.csv', type=str)
    args = parser.parse_args()

    run(args.path, args.metric)
