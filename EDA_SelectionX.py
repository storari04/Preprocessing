import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from sklearn import model_selection, metrics
import BorutaPy
from sklearn.ensemble import RandomForestRegressor

# delete descriptors with high rate of the same values
threshold_of_r = 0.95  # variable whose absolute correlation coefficnent with other variables is higher than threshold_of_r is searched
threshold_of_rate_of_same_value = 1

rate_of_same_value = list()
num = 0
for X_variable_name in train_x.columns:
    num += 1
    #print('{0} / {1}'.format(num, train_x.shape[1]))
    same_value_number = train_x[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / train_x.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
if len(deleting_variable_numbers[0]) != 0:
    train_x = train_x.drop(train_x.columns[deleting_variable_numbers], axis = 1)
    test_x = test_x.drop(train_x.columns[deleting_variable_numbers], axis = 1)
    print('Same X-variable: {0}'.format(deleting_variable_numbers[0] + 1))

#偏差0のものは消去
deleting_variable_numbers = np.where(train_x.var() == 0)
if len(deleting_variable_numbers[0]) != 0:
    train_x = train_x.drop(train_x.columns[deleting_variable_numbers], axis = 1)
    test_x = test_x.drop(train_x.columns[deleting_variable_numbers], axis = 1)
    print('Variable numbers zero variance: {0}'.format(deleting_variable_numbers[0] + 1))

#相関係数の高いものを削除
threshold = 1.00

feat_corr = set()
corr_matrix = train_x.corr()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) >= threshold:
            feat_name = corr_matrix.columns[i]
            feat_corr.add(feat_name)

train_x.drop(labels=feat_corr, axis='columns', inplace=True)
test_x.drop(labels=feat_corr, axis='columns', inplace=True)

print(feat_corr)

#boruta 特徴量選択
rf = RandomForestRegressor(max_depth = 7)
feat_selector = BorutaPy(rf, n_estimators = 'auto', two_step = False, verbose = 2, random_state = 42, max_iter = 20)
feat_selector.fit(train_x.values, train_y.values)

train_x = train_x.iloc[:,feat_selector.support_]
test_x = test_x.iloc[:,feat_selector.support_]
