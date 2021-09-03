import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def generate_openml_meta(openml_id: int):
    data = fetch_openml(data_id=openml_id, as_frame=True)
    column_list = data['feature_names']
    num_rows = len(data['target'])
    num_classes = data['target'].nunique()
    num_unique_nominal_feat = list()
    num_unique_numeric_feat = list()

    for col in column_list:
        column_data = data['data'][col]
        dtype = str(column_data.dtype)
        n_unique = column_data.nunique()

        if dtype == 'category':
            num_unique_nominal_feat.append(n_unique)
        else:
            num_unique_numeric_feat.append(n_unique)

    df_columns = ['id', 'mean unique nominal feat', 'mean unique numeric feat', 'num nominal feat', 'num numeric feat', 'num classes', 'num rows']
    df = pd.DataFrame(columns=df_columns)
    df.loc[0] = [openml_id, np.mean(num_unique_nominal_feat), np.mean(num_unique_numeric_feat), len(num_unique_nominal_feat), len(num_unique_numeric_feat), num_classes, num_rows]
    df.to_csv(f'./results/meta/openml{openml_id}_meta.csv', index=False)

    # pd.DataFrame.from_dict(dict(id=list(openml_id), mean_unique_nominal_feat=list(np.mean(num_unique_nominal_feat)),
    #                             mean_unique_numeric_feat=list(np.mean(num_unique_numeric_feat)),
    #                             num_nominal_feat=list(len(num_unique_nominal_feat)),
    #                             num_numeric_feat=list(len(num_unique_numeric_feat)),
    #                             num_classes=list(num_classes),
    #                             num_rows=list(num_rows))).to_csv(f'./results/meta/openml{openml_id}_meta.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--openml_id', type=int, help='OpenML id to run on', default=32)
    args = parser.parse_args()

    generate_openml_meta(args.openml_id)
