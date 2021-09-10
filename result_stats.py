import argparse
import numpy as np
import os
from numpy.core.defchararray import upper
import pandas as pd
import pdb


def add_dataset_info(results_raw: pd.DataFrame, task_metadata: pd.DataFrame):
    results_raw['tid'] = [int(x.split('/')[-1]) for x in results_raw['id']]
    task_metadata['ClassRatio'] = task_metadata['MinorityClassSize'] / task_metadata['NumberOfInstances']
    results_raw = results_raw.merge(task_metadata, on=['tid'])
    return results_raw


def mean_score(df: pd.DataFrame, column: str = 'result'):
    return round(df[column].mean(), 4)


def filter_type(df: pd.DataFrame):
    return df[df['type'] == 'binary'], df[df['type'] == 'multiclass'], df[df['type'] == 'regression']


def filter_samples(df, samples=100000, lower=True):
    return df[df['NumberOfInstances'] < samples] if lower else df[df['NumberOfInstances'] >= samples]


def filter_features(df, features=100, lower=True):
    return df[df['NumberOfFeatures'] < features] if lower else df[df['NumberOfFeatures'] >= features]


def filter_duration(df, duration=40000):
    return df[df['duration'] < duration]


def compute_win_lose(base, tie, prune):
    total = max(len(base) + len(prune) + len(tie), 1)
    return round((len(prune) + 0.5 * len(tie)) / total, 4), round((len(base) + 0.5 * len(tie)) / total, 4)


def print_inference_speedup(df1, df2):
    relative_speedups = []
    absolute_speedups = []
    for task in df1['task'].unique():
        df1_rows = df1[df1["task"] == task]
        df2_rows = df2[df2["task"] == task]
        for fold in df1_rows["fold"].unique():
            row1, row2 = df1_rows[df1_rows["fold"] == fold], df2_rows[df2_rows["fold"] == fold]
            if len(row1) == 0 or len(row2) == 0 or row1['predict_duration'].isna().item() or row2['predict_duration'].isna().item():
                continue
            df1_time, df2_time = row1['predict_duration'].item(), row2['predict_duration'].item()
            if df1_time == 0 or df2_time == 0:
                continue
            relative_speedups.append((df1_time - df2_time)/min(df1_time, df2_time))
            absolute_speedups.append(df1_time - df2_time)
    print(f"Average Relative Speedup: {round(np.mean(relative_speedups), 4)}, Average Absolute Speedup: {round(np.mean(absolute_speedups), 4)}")


def compare_dfs_improvement(df1, df2):
    metric = "result"
    binary, multiclass, regression = [], [], []
    for task in df1['task'].unique():
        df1_rows = df1[df1["task"] == task]
        df2_rows = df2[df2["task"] == task]

        for fold in df1_rows["fold"].unique():
            row1, row2 = df1_rows[df1_rows["fold"] == fold], df2_rows[df2_rows["fold"] == fold]
            if len(row1) == 0 or len(row2) == 0 or row1[metric].isna().item() or row2[metric].isna().item():
                continue
            df1_score, df2_score = row1[metric].item(), row2[metric].item()
            problem_type = df1_rows.iloc[0]['type']
            if problem_type == "binary":
                score = (df2_score - df1_score) / df1_score if df1_score > df2_score else (df2_score - df1_score) / df2_score
                binary.append(score)
            elif problem_type == "multiclass":
                score = (df1_score - df2_score) / df1_score if df1_score < df2_score else (df1_score - df2_score) / df2_score
                multiclass.append(score)
            else:
                score = (df1_score - df2_score) / df1_score if df1_score < df2_score else (df1_score - df2_score) / df2_score
                regression.append(score)

    binary_improvement = round(np.mean(binary), 4)
    multiclass_improvement = round(np.mean(multiclass), 4)
    regression_improvement = round(np.mean(regression), 4)
    total_improvement = round(np.mean(binary + multiclass + regression), 4)
    return total_improvement, binary_improvement, multiclass_improvement, regression_improvement


def compare_dfs(df1, df2, grouped=False):
    df1_better, equal_performance, df2_better = [], [], []
    metric = "result"
    for task in df1['task'].unique():
        df1_rows = df1[df1["task"] == task]
        df2_rows = df2[df2["task"] == task]
        if grouped:
            if len(df1_rows) > 0:
                df1_score = df1_rows[metric].dropna().mean()
                if df1_score != df1_score:
                    continue
            else:
                continue
            if len(df2_rows) > 0:
                df2_score = df2_rows[metric].dropna().mean()
                if df2_score != df2_score:
                    continue
            else:
                continue
            if df1_score > df2_score:
                df1_better.append(task)
            elif df1_score < df2_score:
                df2_better.append(task)
            else:
                equal_performance.append(task)
        else:
            for fold in df1_rows["fold"].unique():
                row1, row2 = df1_rows[df1_rows["fold"] == fold], df2_rows[df2_rows["fold"] == fold]
                if len(row1) == 0 or len(row2) == 0 or row1[metric].isna().item() or row2[metric].isna().item():
                    continue
                score1, score2 = row1[metric].item(), row2[metric].item()
                if score1 > score2:
                    df1_better.append(task+f"_{fold}")
                elif score1 < score2:
                    df2_better.append(task+f"_{fold}")
                else:
                    equal_performance.append(task+f"_{fold}")
    return df1_better, equal_performance, df2_better


def print_miscellaneous(df1, df2):
    metric = "result"
    score_diffs = []
    for task in df1['task'].unique():
        df1_rows = df1[df1["task"] == task]
        df2_rows = df2[df2["task"] == task]
        if len(df1_rows) > 0:
            df1_score = df1_rows[metric].dropna().mean()
            if df1_score != df1_score:
                continue
        else:
            continue
        if len(df2_rows) > 0:
            df2_score = df2_rows[metric].dropna().mean()
            if df2_score != df2_score:
                continue
        else:
            continue
        problem_type = df1_rows.iloc[0]['type']
        if problem_type == "binary":
            score = (df2_score - df1_score) / df1_score if df1_score > df2_score else (df2_score - df1_score) / df2_score
        elif problem_type == "multiclass":
            score = (df1_score - df2_score) / df1_score if df1_score < df2_score else (df1_score - df2_score) / df2_score
        else:
            score = (df1_score - df2_score) / df1_score if df1_score < df2_score else (df1_score - df2_score) / df2_score
        score_diffs.append((task, score))
    score_diffs = sorted(score_diffs, key=lambda info: info[1])
    score_diffs = [diff[1] for diff in score_diffs]
    print(f"Relative Error Reduction Info: {round(np.mean(score_diffs), 4)} ± {round(np.std(score_diffs), 4)}, ({round(score_diffs[0], 4)}, {round(score_diffs[-1], 4)})")
    lower_quantile, upper_quantile = np.quantile(score_diffs, 0.025), np.quantile(score_diffs, 0.975)
    score_diffs = [diff for diff in score_diffs if lower_quantile < diff < upper_quantile]
    print(f"Relative Error Reduction Info (mean ± 2 * sigma): {round(np.mean(score_diffs), 4)} ± {round(np.std(score_diffs), 4)}, ({round(score_diffs[0], 4)}, {round(score_diffs[-1], 4)})")
    print(f"Number of Errored Runs (Base/Prune): {len(base[~base['info'].isna()])}/{len(prune[~prune['info'].isna()])}")


def print_automl_comparisons(base: pd.DataFrame, prune: pd.DataFrame, others: pd.DataFrame):
    print("==============================================================================")
    rows = []
    for framework in others['framework'].unique():
        other = others[others['framework'] == framework]
        first_better, equal_performance, second_better = compare_dfs(other, base)
        base_win, base_lose = compute_win_lose(first_better, equal_performance, second_better)
        first_better, equal_performance, second_better = compare_dfs(other, prune)
        prune_win, prune_lose = compute_win_lose(first_better, equal_performance, second_better)
        base_improvement, _, _, _ = compare_dfs_improvement(other, base)
        prune_improvement, _, _, _ = compare_dfs_improvement(other, prune)

        rows.append({'Framework': framework, 'Base Win Rate': base_win, 'Prune Win Rate': prune_win,
                     'Base Error Reduction': base_improvement, 'Prune Error Reduction': prune_improvement,
                     'Win Rate Improvement': prune_win - base_win, 'Error Reduction Improvement': prune_improvement - base_improvement})
    df = pd.DataFrame(rows)
    print(df)
    print("==============================================================================")


def print_suite_result(base: pd.DataFrame, prune: pd.DataFrame, indepth=True, grouped=False):
    baselow = filter_samples(base, samples=args.sample_low)
    prunelow = filter_samples(prune, samples=args.sample_low)
    basemed = filter_samples(filter_samples(base, samples=args.sample_low, lower=False), samples=args.sample_med)
    prunemed = filter_samples(filter_samples(prune, samples=args.sample_low, lower=False), samples=args.sample_med)
    basehigh = filter_samples(base, samples=args.sample_med, lower=False)
    prunehigh = filter_samples(prune, samples=args.sample_med, lower=False)
    first_better, equal_performance, second_better = compare_dfs(base, prune, grouped=grouped)
    num_total = len(first_better) + len(second_better) + len(equal_performance)
    print("==============================================================================")
    print(f"Mean Improvement Ratio: {compare_dfs_improvement(base, prune)[0]}")
    print(f"Win Rate: {round((len(second_better) + 0.5 * len(equal_performance))/ num_total, 4)}, Lose Rate: {round((len(first_better) + 0.5 * len(equal_performance)) / num_total, 4)}")
    print(f"All Run Base Win: {len(first_better)}, Prune Win: {len(second_better)}, Tie: {len(equal_performance)}")

    rows = []
    win, lose = compute_win_lose(first_better, equal_performance, second_better)
    rows.append({'Sample Size': 'ALL', 'Feature Size': 'ALL', 'Base Win': len(first_better), 'Prune Win': len(second_better), 'Tie': len(equal_performance), 'Win Rate': win, 'Lose Rate': lose})
    ss_base = filter_features(baselow, features=args.feature_low)
    ss_prune = filter_features(prunelow, features=args.feature_low)
    first_better, equal_performance, second_better = compare_dfs(ss_base, ss_prune, grouped=grouped)
    win, lose = compute_win_lose(first_better, equal_performance, second_better)
    rows.append({'Sample Size': 'S', 'Feature Size': 'S', 'Base Win': len(first_better), 'Prune Win': len(second_better), 'Tie': len(equal_performance), 'Win Rate': win, 'Lose Rate': lose})
    sm_base = filter_features(filter_features(baselow, features=args.feature_low, lower=False), features=args.feature_med)
    sm_prune = filter_features(filter_features(prunelow, features=args.feature_low, lower=False), features=args.feature_med)
    first_better, equal_performance, second_better = compare_dfs(sm_base, sm_prune, grouped=grouped)
    win, lose = compute_win_lose(first_better, equal_performance, second_better)
    rows.append({'Sample Size': 'S', 'Feature Size': 'M', 'Base Win': len(first_better), 'Prune Win': len(second_better), 'Tie': len(equal_performance), 'Win Rate': win, 'Lose Rate': lose})
    sl_base = filter_features(baselow, features=args.feature_med, lower=False)
    sl_prune = filter_features(prunelow, features=args.feature_med, lower=False)
    first_better, equal_performance, second_better = compare_dfs(sl_base, sl_prune, grouped=grouped)
    win, lose = compute_win_lose(first_better, equal_performance, second_better)
    rows.append({'Sample Size': 'S', 'Feature Size': 'L', 'Base Win': len(first_better), 'Prune Win': len(second_better), 'Tie': len(equal_performance), 'Win Rate': win, 'Lose Rate': lose})

    ms_base = filter_features(basemed, features=args.feature_low)
    ms_prune = filter_features(prunemed, features=args.feature_low)
    first_better, equal_performance, second_better = compare_dfs(ms_base, ms_prune, grouped=grouped)
    win, lose = compute_win_lose(first_better, equal_performance, second_better)
    rows.append({'Sample Size': 'M', 'Feature Size': 'S', 'Base Win': len(first_better), 'Prune Win': len(second_better), 'Tie': len(equal_performance), 'Win Rate': win, 'Lose Rate': lose})
    mm_base = filter_features(filter_features(basemed, features=args.feature_low, lower=False), features=args.feature_med)
    mm_prune = filter_features(filter_features(prunemed, features=args.feature_low, lower=False), features=args.feature_med)
    first_better, equal_performance, second_better = compare_dfs(mm_base, mm_prune, grouped=grouped)
    win, lose = compute_win_lose(first_better, equal_performance, second_better)
    rows.append({'Sample Size': 'M', 'Feature Size': 'M', 'Base Win': len(first_better), 'Prune Win': len(second_better), 'Tie': len(equal_performance), 'Win Rate': win, 'Lose Rate': lose})
    ml_base = filter_features(basemed, features=args.feature_med, lower=False)
    ml_prune = filter_features(prunemed, features=args.feature_med, lower=False)
    first_better, equal_performance, second_better = compare_dfs(ml_base, ml_prune, grouped=grouped)
    win, lose = compute_win_lose(first_better, equal_performance, second_better)
    rows.append({'Sample Size': 'M', 'Feature Size': 'L', 'Base Win': len(first_better), 'Prune Win': len(second_better), 'Tie': len(equal_performance), 'Win Rate': win, 'Lose Rate': lose})

    ls_base = filter_features(basehigh, features=args.feature_low)
    ls_prune = filter_features(prunehigh, features=args.feature_low)
    first_better, equal_performance, second_better = compare_dfs(ls_base, ls_prune, grouped=grouped)
    win, lose = compute_win_lose(first_better, equal_performance, second_better)
    rows.append({'Sample Size': 'L', 'Feature Size': 'S', 'Base Win': len(first_better), 'Prune Win': len(second_better), 'Tie': len(equal_performance), 'Win Rate': win, 'Lose Rate': lose})
    lm_base = filter_features(filter_features(basehigh, features=args.feature_low, lower=False), features=args.feature_med)
    lm_prune = filter_features(filter_features(prunehigh, features=args.feature_low, lower=False), features=args.feature_med)
    first_better, equal_performance, second_better = compare_dfs(lm_base, lm_prune, grouped=grouped)
    win, lose = compute_win_lose(first_better, equal_performance, second_better)
    rows.append({'Sample Size': 'L', 'Feature Size': 'M', 'Base Win': len(first_better), 'Prune Win': len(second_better), 'Tie': len(equal_performance), 'Win Rate': win, 'Lose Rate': lose})
    ll_base = filter_features(basehigh, features=args.feature_med, lower=False)
    ll_prune = filter_features(prunehigh, features=args.feature_med, lower=False)
    first_better, equal_performance, second_better = compare_dfs(ll_base, ll_prune, grouped=grouped)
    win, lose = compute_win_lose(first_better, equal_performance, second_better)
    rows.append({'Sample Size': 'L', 'Feature Size': 'L', 'Base Win': len(first_better), 'Prune Win': len(second_better), 'Tie': len(equal_performance), 'Win Rate': win, 'Lose Rate': lose})

    df = pd.DataFrame(rows)
    print(df)
    print("==============================================================================")


# ======= NEW ======= #

# 1h
# base = "result/baseline/1hmed/results_automlbenchmark_1h8c_autogluon.ag.1h8c.aws.20210827T163031.csv"
# prune = "result/best/1hmed/results_automlbenchmark_1h8c_prune_med.ag.1h8c.aws.20210828T182000.csv"

# base = "result/baseline/1hhigh/results_automlbenchmark_1h8c_autogluon_high.ag.1h8c.aws.20210829T224457.csv"
# prune = "result/best/1hhigh/results_automlbenchmark_1h8c_prune_high.ag.1h8c.aws.20210829T224459.csv"

# base = "result/baseline/1hnorepeat/results_automlbenchmark_1h8c_autogluon_norepeat.ag.1h8c.aws.20210827T202558.csv"
# prune = "result/best/1hnorepeat/results_automlbenchmark_1h8c_prune_norepeat.ag.1h8c.aws.20210829T224516.csv"

# base = "result/baseline/1hbest/results_automlbenchmark_1h8c_autogluon_bestquality.ag.1h8c.aws.20210830T230714.csv"
# base = "~/Downloads/results_automlbenchmark_1h8c_2021_08_29_knn.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_autogluon_bestquality.ag.1h8c.aws.20210902T175142.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210902T175139.csv"

# norepeat
# base = "~/Downloads/results_automlbenchmark_1h8c_autogluon_norepeat.ag.1h8c.aws.20210908T151458.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune_norepeat.ag.1h8c.aws.20210905T230349.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune_norepeat.ag.1h8c.aws.20210906T213059.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune_norepeat.ag.1h8c.aws.20210908T035241.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune_norepeat.ag.1h8c.aws.20210909T090238.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune_norepeat.ag.1h8c.aws.20210909T202902.csv"

# full
base = "~/Downloads/results_automlbenchmark_1h8c_2021_09_02.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210904T011959(1).csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210905T192540.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210906T202118.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune_minimprovement.ag.1h8c.aws.20210906T213112.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210907T084943.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune_stoppinground.ag.1h8c.aws.20210907T104958.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210907T175902.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune_replacebag.ag.1h8c.aws.20210907T175858.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210907T221927.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune_replacebag.ag.1h8c.aws.20210907T222005.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210908T012205.csv" # slide result
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune_minimprovement.ag.1h8c.aws.20210908T094645.csv"
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210908T182233.csv" # no 300 cap
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210908T235902.csv" # removed feature metadata bug
# prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210909T070228.csv" # experimental
prune = "~/Downloads/results_automlbenchmark_1h8c_prune.ag.1h8c.aws.20210909T192854.csv" # final

# 4h
# base = "result/baseline/4hmed/results_automlbenchmark_4h8c_autogluon.ag.4h8c.aws.20210827T163032.csv"
# prune = "result/best/4hmed/results_automlbenchmark_4h8c_prune_med.ag.4h8c.aws.20210828T210007.csv"

# base = "result/baseline/4hhigh/results_automlbenchmark_4h8c_autogluon_high.ag.4h8c.aws.20210830T073353.csv"
# prune = "result/best/4hhigh/results_automlbenchmark_4h8c_prune_high.ag.4h8c.aws.20210830T073352.csv"

# base = "result/baseline/4hnorepeat/results_automlbenchmark_4h8c_autogluon_norepeat.ag.4h8c.aws.20210827T062721.csv"
# prune = "result/best/4hnorepeat/results_automlbenchmark_4h8c_prune_norepeat.ag.4h8c.aws.20210828T210006.csv"
# prune = "result/best/4hnorepeat/results_automlbenchmark_4h8c_prune_norepeat.ag.4h8c.aws.20210829T060616.csv"

# base = "result/baseline/4hbest/results_automlbenchmark_4h8c_autogluon_bestquality.ag.4h8c.aws.20210827T062731.csv"
# base = "result/best/4hbest/results_automlbenchmark_4h8c_prune.ag.4h8c.aws.20210828T210005.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune_improvementthreshold.ag.4h8c.aws.20210829T060617.csv"

# # norepeat
# base = "~/Downloads/results_automlbenchmark_4h8c_autogluon_norepeat.ag.4h8c.aws.20210908T145253.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune_norepeat.ag.4h8c.aws.20210905T230350.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune_norepeat.ag.4h8c.aws.20210907T001718.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune_norepeat.ag.4h8c.aws.20210908T062913.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune_norepeat.ag.4h8c.aws.20210909T090240.csv"

# # full
# base = "~/Downloads/results_automlbenchmark_4h8c_2021_09_02.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune.ag.4h8c.aws.20210902T062359(1).csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune.ag.4h8c.aws.20210905T192543.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune.ag.4h8c.aws.20210906T095323.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune.ag.4h8c.aws.20210906T202121.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune.ag.4h8c.aws.20210907T131305.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune.ag.4h8c.aws.20210908T012206.csv" # slide result
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune_minimprovement.ag.4h8c.aws.20210908T074624.csv"
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune.ag.4h8c.aws.20210908T182235.csv" # no 300 cap
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune.ag.4h8c.aws.20210908T235905.csv" # no feature metadata bag
# prune = "~/Downloads/results_automlbenchmark_4h8c_prune.ag.4h8c.aws.20210909T070232.csv" # experimental


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--sample_low', help='upper boundary that determines whether a dataset is small sized', default=5000, type=int)
    parser.add_argument('-b', '--sample_med', help='upper boundary that determines whether a dataset is medium sized', default=50000, type=int)
    parser.add_argument('-c', '--feature_low', help='upper boundary that determines whether a dataset feature space is small sized', default=20, type=int)
    parser.add_argument('-d', '--feature_med', help='upper boundary that determines whether a dataset feature space is medium sized', default=50, type=int)
    parser.add_argument('-e', '--duration', help='determines before what hour the job must have finished', default=4, type=float)
    parser.add_argument('-f', '--framework', help='whether to compare against other frameworks', default=False, type=bool)
    parser.add_argument('-g', '--done_only', help='whether to display results for only datasets that finished', default=False, type=bool)
    args = parser.parse_args()

    print(f"{os.path.basename(base)} vs {os.path.basename(prune)}")
    base = pd.read_csv(base)
    prune = pd.read_csv(prune)
    base = base[base['framework'] == 'AutoGluon_bestquality']  # FIXME !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # prune = prune[prune['framework'] == 'AutoGluon_bestquality']  # FIXME !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # prune = prune[prune['framework'] == 'AutoGluon_bestquality_prune']  # FIXME !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # PRUNE TRIGGERED ONLY
    # # import pdb; pdb.set_trace()
    # debug = pd.read_csv('~/Downloads/debug_info(166).csv')
    # untriggered_rows = debug[(~debug['pruned']) & (debug['total_prune_time'] == 0)]
    # untriggered_task_fold = untriggered_rows['name'] + untriggered_rows['fold'].astype(str)
    # base['taskfold'] = base['task'] + base['fold'].astype(str)
    # prune['taskfold'] = prune['task'] + prune['fold'].astype(str)
    # base = base[~base['taskfold'].isin(untriggered_task_fold)]
    # prune = prune[~prune['taskfold'].isin(untriggered_task_fold)]

    others = pd.read_csv(f"result/baseline/{int(args.duration)}hbest/other_systems.csv")
    task_metadata = pd.read_csv('result/task_metadata.csv')
    base = add_dataset_info(base, task_metadata)
    prune = add_dataset_info(prune, task_metadata)

    DURATION = args.duration * 3600
    basedone = filter_duration(base, duration=DURATION)
    prunedone = filter_duration(prune, duration=DURATION)

    try:
        print("==============================================================================")
        if not args.done_only:
            print("ALL")
            print_suite_result(base, prune, grouped=False)
            print("ALL (Grouped)")
            print_suite_result(base, prune, grouped=True)
        else:
            print("DONE ONLY")
            print_suite_result(basedone, prunedone, grouped=False)
            print("DONE ONLY (Grouped)")
            print_suite_result(basedone, prunedone, grouped=True)
        if args.framework:
            print("OTHERS")
            print_automl_comparisons(base, prune, others)
        print("MISCELLANEOUS")
        print_inference_speedup(base, prune)
        print_miscellaneous(base, prune)
    except Exception as e:
        pdb.post_mortem()

    # pdb.set_trace()
    print("Done")
