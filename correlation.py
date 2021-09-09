import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_key(threshold: float, eval_percent: float, is_pseudo: bool):
    label = 'pseudo' if is_pseudo else 'vanilla'
    return label + '_' + str(threshold) + '_' + str(eval_percent)


def reverse_key(key):
    parts = key.split('_')
    # Return label, thres, eval p
    return parts[0], parts[1], parts[2]


def get_all_files_with_ext(dir: str, ext: str):
    files_list = list()
    for file in os.listdir(dir):
        if file.endswith(ext):
            files_list.append(os.path.join(dir, file))

    return files_list


def get_meta_df(file: str):
    openml_name = os.path.basename(file).split('_')[0]
    return pd.read_csv(f'./results/meta/{openml_name}_meta.csv').fillna(0)


def get_meta_dict(file: str):
    df = get_meta_df(file)
    meta_dict = df.to_dict()
    for key in meta_dict.keys():
        meta_dict[key] = meta_dict[key][0]
    return meta_dict


def calculate_correlations_of_pseudo(eval_percent_list: list, threshold_list: list, files_list: list):
    results = dict()
    for eval_p in eval_percent_list:
        for threshold in threshold_list:
            for file in files_list:
                df = pd.read_csv(file)
                relevant_rows = df[df.threshold == threshold][df.eval_percentage == eval_p]
                key = generate_key(threshold, eval_p, True)
                meta_dict = get_meta_dict(file)
                temp_result = dict(accuracy=np.mean(relevant_rows['accuracy']), threshold=threshold,
                                   evaluation_p=eval_p)
                temp_result.update(meta_dict)
                results[key] = temp_result
    return results


def calculate_correlations_of_vanilla(eval_percent_list: list, threshold_list: list, files_list: list):
    results = dict()
    for eval_p in eval_percent_list:
        for file in files_list:
            df = pd.read_csv(file)
            relevant_rows = df[df.procedure == 'Vanilla'][df.eval_percentage == eval_p]
            key = generate_key('', eval_p, False)
            meta_dict = get_meta_dict(file)
            temp_result = dict(accuracy=np.mean(relevant_rows['accuracy']), threshold=0,
                               evaluation_p=eval_p)
            temp_result.update(meta_dict)
            results[key] = temp_result
    return results


def get_data_across_all_csv(eval_percent: float, threshold: float, is_vanilla: bool):
    files_list = get_all_files_with_ext('./results', '.csv')
    result = None
    for file in files_list:
        df = pd.read_csv(file)

        vanilla_row = df[df.procedure == 'Vanilla'][df.eval_percentage == eval_percent]
        selected_row = df[df.threshold == threshold][df.eval_percentage == eval_percent][df.is_reuse == False]
        diff = selected_row['accuracy'].values[0] - vanilla_row['accuracy'].values[0]

        meta_df = get_meta_df(file)
        new_row = pd.concat(
            [selected_row.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
        new_row['diff'] = diff

        if file == files_list[0]:
            result = new_row
        else:
            result = pd.concat([result, new_row])

    fig = plt.figure(figsize=(9, 9), dpi=100)
    ax = fig.add_subplot()
    cax = ax.matshow(result.corr(), interpolation='nearest')
    fig.colorbar(cax)

    num_vars = len(result.corr().columns)
    ax.set_xticks(list(range(num_vars)))
    ax.set_yticks(list(range(num_vars)))
    ax.set_xticklabels(list(result.corr().columns))
    ax.set_yticklabels(list(result.corr().columns))
    ax.tick_params(axis='x', which='major', labelsize=5)
    ax.tick_params(axis='y', which='major', labelsize=5)

    for i in range(num_vars):
        for j in range(num_vars):
            text = ax.text(j, i, round(np.array(result.corr())[i, j], 2),
                           ha="center", va="center", color="w")

    plt.show()


def calculate_correlations(eval_percent_list: list, threshold_list: list):
    files_list = get_all_files_with_ext('./results', '.csv')

    results = calculate_correlations_of_pseudo(eval_percent_list, threshold_list, files_list)
    results.update(calculate_correlations_of_vanilla(eval_percent_list, threshold_list, files_list))

    model_name_list = list()
    accuracy_list = list()
    threshold_list = list()
    evaluation_p_list = list()
    mu_uniq_nom_feat_list = list()
    mu_uniq_num_feat_list = list()
    num_nom_feat_list = list()
    num_num_feat_list = list()
    num_classes_list = list()
    num_rows_list = list()

    for key in results.keys():
        curr_data = results[key]
        model_name_list.append(key)
        accuracy_list.append(curr_data['accuracy'])
        threshold_list.append(curr_data['threshold'])
        evaluation_p_list.append(curr_data['evaluation_p'])
        mu_uniq_nom_feat_list.append(curr_data['mean unique nominal feat'])
        mu_uniq_num_feat_list.append(curr_data['mean unique numeric feat'])
        num_nom_feat_list.append(curr_data['num nominal feat'])
        num_num_feat_list.append(curr_data['num numeric feat'])
        num_classes_list.append(curr_data['num classes'])
        num_rows_list.append(curr_data['num rows'])

    df = pd.DataFrame(dict(model_name=model_name_list, accuracy=accuracy_list, threshold=threshold_list,
                           evaluation=evaluation_p_list, mu_uniq_nom_feat=mu_uniq_nom_feat_list,
                           mu_uniq_num_feat=mu_uniq_num_feat_list, num_nom_feat=num_nom_feat_list,
                           num_num_feat=num_num_feat_list, num_classes=num_classes_list, num_rows=num_rows_list))

    print('')


if __name__ == "__main__":
    eval_percent_list = [0.75, 0.95]  # [0.25, 0.5, 0.75, 0.95]
    threshold_list = [0.9, 0.95]  # [0.5, 0.75, 0.9, 0.95]
    # calculate_correlations(eval_percent_list, threshold_list)
    get_data_across_all_csv(0.75, 0.9, False)
