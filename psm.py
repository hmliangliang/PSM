# -*-coding: utf-8 -*-
# @Time    : 2024/9/12 14:58
# @File    : psm.py
# @Software: PyCharm
import datetime
# 使用方法：https://github.com/adriennekline/psmpy?tab=readme-ov-file#predict-scores
import os

os.system("pip install psmpy")
os.system("pip install \"dask[dataframe]\"")

import pandas as pd
import numpy as np
from psmpy import PsmPy
import argparse
from psmpy.functions import cohenD
from scipy import stats
import dask.dataframe as dd

if __name__ == "__main__":
    print("注意输入的数据需如下的格式：输入的数据倒数第二列为treatment列(取值为0或1), 最后一列为观测列，其余列为协变量列.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_output', help='Output file path', type=str, default='')
    parser.add_argument('--data_input', help='map_feature_file', type=str, default='')
    parser.add_argument('--split_flag', help='csv文件数据分隔符', type=str, default=',')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    # 读取数据
    path = args.data_input.split(',')[0]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    n_files = len(input_files)
    count = 0
    for file in input_files:
        count += 1
        print("一共{}个文件,当前正在处理第{}个文件,文件路径:{}......".format(n_files, count,
                                                                             "cfs://" + path + "/" + file))
        # 读取数据数据最后一列为序列数据
        if count == 1:
            data = dd.read_csv(os.path.join(path, file), sep=args.split_flag, header=None).astype(float)
            data = data.compute()  # 将 Dask DataFrame 转换为 Pandas DataFrame
        else:
            data_temp = dd.read_csv(os.path.join(path, file), sep=args.split_flag, header=None).astype(float)
            data_temp = data_temp.compute()  # 将 Dask DataFrame 转换为 Pandas DataFrame
            data = pd.concat([data, data_temp], axis=0)
    # 数据的行数与列数
    data = data.values
    n, m = data.shape
    row_indices = np.arange(n).reshape(n, 1)
    data = np.concatenate([data, row_indices], axis=1)
    data = pd.DataFrame(data)
    # 加一列key
    data.columns = [str(i) for i in range(m)] + ['keywords']
    print("开始执行psm分析过程!".format(datetime.datetime.now()))
    psm = PsmPy(data, treatment=str(m - 2), indx='keywords', exclude=[str(m - 1)])
    psm.logistic_ps(balance=True)
    print("开始进行匹配:{}".format(datetime.datetime.now()))
    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=0.1)
    # psm.knn_matched_12n(matcher='propensity_logit', how_many=1)
    effect_size = cohenD(data, str(m - 2), str(m - 1))

    print("effect_size(Cohen's d)={}".format(effect_size))
    print("小效应‌：当Cohen's d值约为0.2时，表示两组之间的差异很小 中等效应‌：当Cohen's d值约为0.5时，表示两组之间的差异中等 当Cohen's d值约为0.8时，表示两组之间的差异很大")
    # 进行 t 检验
    _, p_value = stats.ttest_ind(data[data[str(m - 2)] == 1][str(m - 1)], data[data[str(m - 2)] == 0][str(m - 1)])
    print("P-value: {}".format(p_value))

    if data.loc[psm.matched_ids.loc[0]['keywords']][str(m-2)] == 0:
        treatment0 = data.loc[psm.matched_ids['keywords']][str(m - 1)].mean()
        treatment1 = data.loc[psm.matched_ids['largerclass_0group']][str(m - 1)].mean()
    else:
        treatment1 = data.loc[psm.matched_ids['keywords']][str(m - 1)].mean()
        treatment0 = data.loc[psm.matched_ids['largerclass_0group']][str(m - 1)].mean()

    print("treatment0={}   treatment1={}".format(treatment0, treatment1))
