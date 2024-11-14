from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np


feature_data = pd.read_csv('./feature/normalized_combine_all.csv',sep=',')
target_data = pd.read_csv('./feature/label.csv', sep='\t')


X = feature_data.drop(columns=['ID'])
y = target_data['label']


np.random.seed(42)

feature_mu= mutual_info_classif(X.values, y.values)
