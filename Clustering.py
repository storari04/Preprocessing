# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster  # SciPy の中の階層的クラスタリングを実行したり樹形図を作成したりするためのライブラリをインポート
from sklearn.decomposition import PCA

number_of_clusters = 3  # クラスターの数

autocalculated_train_x = (train_x - train_x.mean()) / train_x.std()  # オートスケーリング

# 階層的クラスタリング
clustering_results = linkage(autocalculated_train_x, metric='euclidean', method='ward')
# 　metric, method を下のように変えることで、それぞれ色々な距離、手法で階層的クラスタリングを実行可能
#
# metric の種類
# - euclidean : ユークリッド距離
# - cityblock : マンハッタン距離(シティブロック距離)
# など
#
# method の種類
# - single : 最近隣法
# - complete : 最遠隣法
# - weighted : 重心法
# - average : 平均距離法
# - ward : ウォード法
# など

# デンドログラムの作成
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
dendrogram(clustering_results, labels=x.index, color_threshold=0,
           orientation='right')  # デンドログラムの作成。labels=x.index でサンプル名を入れています
plt.xlabel('distance')  # 横軸の名前
plt.show()

# クラスター番号の保存
cluster_numbers = fcluster(clustering_results, number_of_clusters, criterion='maxclust')  # クラスターの数で分割し、クラスター番号を出力
cluster_numbers = pd.DataFrame(cluster_numbers, index=x.index,
                               columns=['cluster_numbers'])  # DataFrame 型に変換。行の名前・列の名前も設定
cluster_numbers.to_csv('cluster_numbers.csv')
