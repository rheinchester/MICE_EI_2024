import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBRegressor
import warnings

from causalml.inference.meta import LRSRegressor, xlearner
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
from causalml.propensity import ElasticNetPropensityModel
from causalml.dataset import *
from causalml.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# warnings.filterwarnings('ignore')
# plt.style.use('fivethirtyeight')


# # import importlib
# # print(importlib.metadata.version('causalml'))






# # Example data preparation
# # Assuming `data` is your DataFrame and has 'gender', 'features', and 'outcome'
# data = pd.read_csv('ESSOREI_LV_24.csv')  # Load your dataset

# # Define your feature columns and outcome
# features = ['Year-F']  # replace with your actual feature columns
# X = data[features].values
# T = data['ESS'].values  # binary treatment: 0 for male, 1 for female
# gender = data['gender'].values
# y = data['OR'].values

# # Split the data into training and testing sets
# X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.3, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Initialize the XLearner with RandomForest as base learners
# learner = xlearner(models=RandomForestRegressor())

# # Train the model
# learner.fit(X_train, T_train, y_train)

# # Estimate the treatment effects
# te_train = learner.predict(X_train)
# te_test = learner.predict(X_test)

# # Summarize the results
# ate_train = np.mean(te_train)
# ate_test = np.mean(te_test)

# print(f'Estimated Average Treatment Effect (ATE) on training data: {ate_train}')
# print(f'Estimated Average Treatment Effect (ATE) on testing data: {ate_test}')
