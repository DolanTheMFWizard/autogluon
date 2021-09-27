import argparse

import pandas as pd


def run(path, metric):
    df = pd.read_csv(path)
    open_ml_ids = df['openml_id'].unique()
    num_openml_ids = len(open_ml_ids)
    rank_sums = None
    score_sums = None

    for id in open_ml_ids:
        df_openml = df[df.openml_id == id]
        ranks = df_openml[metric].rank(axis=0, method='average', ascending=False).to_numpy()
        scores = df_openml[metric].to_numpy()

        if id == open_ml_ids[0]:
            rank_sums = ranks
            score_sums = scores
        else:
            rank_sums += ranks
            score_sums += scores

    print('Model names:')
    print(df_openml['model'].to_numpy())
    print('Average ranks:')
    print(rank_sums/num_openml_ids)
    print(f'Average {metric} score:')
    print(score_sums/num_openml_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-metric', help='Metric to evaluate models by', default='result', type=str)
    parser.add_argument('-path', help='Path to file', default='./results_95.csv', type=str)
    args = parser.parse_args()

    run(args.path, args.metric)