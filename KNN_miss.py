import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np


file_path = 'ESSOREI_LV_24.csv'
data = pd.read_csv(file_path)
knn_imputer = KNNImputer(n_neighbors=2)
data_imputed = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)


# Just in case you want integer values.
# data_imputed = data_imputed.astype(int)


# if file_path == 'EI_2024_FOR_ANALYSIS_KNN.csv':
#     data_imputed.to_csv('EI_MICE_2024_imputed.csv', index=False)
if file_path == 'EI 2020 Data.csv':
    data_imputed.to_csv('EI 2020 Data_KNN.csv', index=False)

if file_path == 'ESSOREI_LV_24.csv':
    data_imputed.to_csv('ESSOREI_LV_24_KNN_Imputed.csv', index=False)