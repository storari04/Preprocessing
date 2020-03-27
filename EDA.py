import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from sklearn import model_selection, metrics, datasets
import BorutaPy
from sklearn.ensemble import RandomForestRegressor

# load boston dataset
boston = datasets.load_boston()
x = boston.data
y = boston.target

# Divide samples into training samples and test samples
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

#ターゲットエンコーディング
train_x['X_target'] = train_x.groupby('X')['Y'].transform('mean')
test_x['X_target'] = test_x['X'].map(train_x.groupby('X')['Y'].mean())

#各列の欠損値を確認
def missing_data(data):
    #欠損値を含む行の合計数
    total = data.isnull().sum()
    #欠損値を含む行の合計(%表示)
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    #各列の型を確認
    tt['Types'] = types
    #行列を入れ替える
print(missing_data(raw_data_test))

#欠損値同定
l = []
for index, value in train_x.isnull().sum().iteritems():
    if value > 0:
        l.append(index)

print(l)

#trainとtestデータの各行の平均値を比較
plt.figure(figsize=(16,6))
features = train_x.columns.values[2:3805]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train_x[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test_x[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()

#各列の相関係数を求め、絶対値を取り、行列を1列に直し、相関係数の大きさで並べ替え、indexを再度付与
features = train_x.columns.values[2:3805]
correlations = train_x[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
#同じ列同士の行は消去
correlations = correlations[correlations['level_0'] != correlations['level_1']]
print(correlations.tail(100))

#各列の相関係数を求め、絶対値を取り、行列を1列に直し、相関係数の大きさで並べ替え、indexを再度付与
features_test = test_x.columns.values[2:3770]
correlations_test = test_x[features_test].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
#同じ列同士の行は消去
correlations_test = correlations_test[correlations_test['level_0'] != correlations_test['level_1']]
print(correlations_test.tail(100))

#select high correlation
train = pd.concat([train_x,train_y], axis = 1)
corr = train.corr()
rel_vars = corr.Score[(corr.Score > 0.21)]
rel_cols = list(rel_vars.index.values)

print(rel_cols)

corr2 = train[rel_cols].corr()
plt.figure(figsize = (8,8))
hm = sns.heatmap(corr2, annot = True, annot_kws = {'size':10})
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
plt.show()


#Kennard-Stone (KS)アルゴリズム
from dcekit.sampling import kennard_stone

number_of_selected_samples = 20

# standardize x
autocalculated_train_x = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0, ddof=1)

# select samples using Kennard-Stone algorithm
selected_sample_numbers, remaining_sample_numbers = kennard_stone(autocalculated_train_x, number_of_selected_samples)
print('selected sample numbers')
print(selected_sample_numbers)
print('---')
print('remaining sample numbers')
print(remaining_sample_numbers)

selected_train_x = train_x[selected_sample_numbers, :]

# plot samples
plt.rcParams['font.size'] = 18
plt.figure()
plt.scatter(autocalculated_train_x[:, 0], autocalculated_train_x[:, 1], label='all samples')
plt.scatter(autocalculated_train_x[selected_sample_numbers, 0], autocalculated_train_x[selected_sample_numbers, 1], marker='*',
            label='selected samples')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper right')
plt.show()


# 外れ値自動除去
from dcekit.learning import ensemble_outlier_sample_detection

number_of_submodels = 100
max_iteration_number = 30
fold_number = 2
max_pls_component_number = 30

# PLS
pls_components = np.arange(1, min(max_pls_component_number, x.shape[1]) + 1)
cv_regressor = GridSearchCV(PLSRegression(), {'n_components': pls_components}, cv=fold_number)
outlier_sample_flags = ensemble_outlier_sample_detection(cv_regressor, x, y, cv_flag=True,
                                                         n_estimators=number_of_submodels,
                                                         iteration=max_iteration_number, autoscaling_flag=True,
                                                         random_state=0)

outlier_sample_flags = pd.DataFrame(outlier_sample_flags)
outlier_sample_flags.columns = ['TRUE if outlier samples']
