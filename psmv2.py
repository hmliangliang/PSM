# -*-coding: utf-8 -*-
# @Time    : 2024/9/13 16:46
# @File    : psmv2.py
# @Software: PyCharm

import datetime
import os
os.system("pip install \"dask[dataframe]\"")
os.system("pip install statsmodels")


import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
from scipy import stats
import dask.dataframe as dd



def cohen_d(x, y):
    """
    计算Cohen's D
    :param x: 第一个组的数据
    :param y: 第二个组的数据
    :return: Cohen's D
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    d = (np.mean(x) - np.mean(y)) / pooled_std
    return d


def standardized_mean_difference(data, treatment, covariates):
    treated = data[data[treatment] == 1]
    control = data[data[treatment] == 0]
    smd = {}
    for covariate in covariates:
        mean_treated = treated[covariate].mean()
        mean_control = control[covariate].mean()
        std_treated = treated[covariate].std()
        std_control = control[covariate].std()
        smd[covariate] = (mean_treated - mean_control) / np.sqrt((std_treated**2 + std_control**2) / 2)
    return smd


if __name__ == "__main__":
    print("注意输入的数据需如下的格式：输入的数据倒数第二列为treatment列(取值为0或1), 最后一列为观测列，其余列为协变量列.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_output', help='Output file path', type=str, default='')
    parser.add_argument('--data_input', help='map_feature_file', type=str, default='')
    parser.add_argument('--smd_print', help='map_feature_file', type=bool, default=False)
    parser.add_argument('--split_flag', help='csv文件数据分隔符', type=str, default=',')
    parser.add_argument('--model_name', help='倾向性得分预测模型(LogisticRegression或GradientBoostingClassifier)', type=str, default='GradientBoostingClassifier')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    # 读取数据
    path = args.data_input.split(',')[0]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    n_files = len(input_files)
    count = 0
    for file in input_files:
        count += 1
        print("一共{}个文件,当前正在处理第{}个文件,文件路径:{}......".format(n_files, count, "cfs:/" + path + "/" + file))
        # 读取数据数据最后一列为序列数据
        if count == 1:
            data = dd.read_csv(os.path.join(path, file), sep=args.split_flag, header=None).astype(float)
            data = data.compute()  # 将 Dask DataFrame 转换为 Pandas DataFrame
        else:
            data_temp = dd.read_csv(os.path.join(path, file), sep=args.split_flag, header=None).astype(float)
            data_temp = data_temp.compute()  # 将 Dask DataFrame 转换为 Pandas DataFrame
            data = pd.concat([data, data_temp], axis=0)
    # 数据的行数与列数
    n, m = data.shape
    data = pd.DataFrame(data)
    data = data.fillna(0)
    # 加一列key
    data.columns = [str(i) for i in range(m-2)] + ['treatment'] + ['outcome']
    print("开始执行psm分析过程! {}".format(datetime.datetime.now()))
    covariates = [str(i) for i in range(m-2)]  # 协变量
    treatment = 'treatment'  # 处理变量

    print("开始计算propensity score! {}".format(datetime.datetime.now()))
    if args.model_name == "LogisticRegression":
        model = LogisticRegression()
    else:
        model = GradientBoostingClassifier()
    model.fit(data[covariates], data[treatment])
    data['propensity_score'] = model.predict_proba(data[covariates])[:, 1]
    # 开始进行匹配
    treated = data[data[treatment] == 1]
    control = data[data[treatment] == 0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    _, indices = nn.kneighbors(treated[['propensity_score']])

    matched_control = control.iloc[indices.flatten()]
    matched_data = pd.concat([treated, matched_control])

    # 计算SMD评估匹配效果
    if args.smd_print:
        smd_before = standardized_mean_difference(data, treatment, covariates)
        smd_after = standardized_mean_difference(matched_data, treatment, covariates)
        print("Standardized Mean Differences Before Matching:", smd_before)
        print("Standardized Mean Differences After Matching:", smd_after)

    outcome = 'outcome'  # 结果变量
    treated_outcome = matched_data[matched_data[treatment] == 1][outcome]
    control_outcome = matched_data[matched_data[treatment] == 0][outcome]
    treatment_effect = treated_outcome.mean() - control_outcome.mean()

    # print("小效应‌：当Cohen's d值约为0.2时，表示两组之间的差异很小 中等效应‌：当Cohen's d值约为0.5时，表示两组之间的差异中等 当Cohen's d值约为0.8时，表示两组之间的差异很大")
    d = cohen_d(treated_outcome, control_outcome)
    if abs(d) < 0.2:
        effect_size = "small"
    elif abs(d) < 0.5:
        effect_size = "small to medium"
    elif abs(d) < 0.8:
        effect_size = "medium to large"
    else:
        effect_size = "large"

    # 进行 t 检验
    _, p_value = stats.ttest_ind(treated_outcome, control_outcome)

    output_data = [
        ["Treated Outcome: ", treated_outcome.mean()],
        ["Control Outcome: ", control_outcome.mean()],
        ["Estimated Treatment Effect: ", treatment_effect],
        ["Cohen's D: ", abs(d)],
        ["Effect Size: ", effect_size],
        ["P-value: ", p_value]
    ]
    # 打印表格
    # 计算每一列的最大宽度
    max_description_length = max(len(row[0]) for row in data)
    max_value_length = max(len(str(row[1])) for row in data)

    # 定义列宽
    column_width = max(max_description_length, max_value_length) + 5  # 加一些额外的空间

    # 打印表格
    print("Causal effect test results:")
    print("+" + "-" * (column_width + 2) + "+" + "-" * (column_width + 2) + "+")  # 顶部分割线
    print("| {:<{}} | {:<{}} |".format("Description", column_width, "Value", column_width))  # 表头
    print("+" + "=" * (column_width + 2) + "+" + "=" * (column_width + 2) + "+")  # 表头分割线

    # 打印数据行
    for row in data:
        print("| {:<{}} | {:<{}} |".format(row[0], column_width, row[1], column_width))
        print("+" + "-" * (column_width + 2) + "+" + "-" * (column_width + 2) + "+")
