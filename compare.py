import pandas as pd

path = './results_95.csv'

df = pd.read_csv(path)
open_ml_ids = df['openml_id'].unique()

all = None

for id in open_ml_ids:
    df_openml = df[df.openml_id == id]
    results = df_openml['result'].rank(axis=0, method='average', ascending=False).to_numpy()

    if id == open_ml_ids[0]:
        all = results
    else:
        all += results

print(df_openml['model'].to_numpy())
print(all/18)
