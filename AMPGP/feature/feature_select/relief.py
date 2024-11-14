from skrebate import ReliefF
import pandas as pd


feature_data = pd.read_csv('./feature/normalized_combine_all.csv',sep=',')
target_data = pd.read_csv('./feature/label.csv', sep='\t')


X = feature_data.drop(columns=['ID'])
y = target_data['label']


relief = ReliefF(n_features_to_select=10, n_neighbors=100)

relief.fit(X.values, y.values)

feature_importances = relief.feature_importances_
