# pip install econml
"""
Draw indirect effect, pls-sem
Verify it's robustness with do-why


vary with moderators using econml.
    Customer Segmentation
    Estimate individualized responses to incentives
"""

import pandas as pd
import matplotlib.pyplot as plt
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

# Generic ML imports
from xgboost import XGBRegressor, XGBClassifier

# EconML imports
from econml.dr import LinearDRLearner

import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline



data = pd.read_csv("ESSOREI_LV_24_KNN_Imputed.csv")
print(data.head())

inputs = data['ESS']

Y = data['EI']
X = data['Gender-F']
W = data.drop(
    columns=['ESS', 'EI', 'Gender-F']
)

group_by_ESS = data[["ESS", "EI", "Gender-F"]].groupby(
    by=["ESS"], as_index=False
).mean().astype(int)
print(group_by_ESS)


















# X = data[['Year-F', 'Status-F']]  # Covariates
# T = data['ESS']  # Treatment variable
# Y = data['OR6']  # Outcome variable
# gender = data['Gender-F']  # Gender variable
# print(X.shape)

# model = LinearDML(
#     model_y=RandomForestRegressor(),
#     model_t=Lasso(),
#     featurizer=None,
#     discrete_treatment=False
# )

# model.fit(Y, T, X=X, W=gender)


# # Average treatment effect for males
# ate_male = model.ate(X[X['gender'] == 'male'])
# print("Average Treatment Effect for Males:", ate_male)

# # Average treatment effect for females
# ate_female = model.ate(X[X['gender'] == 'female'])
# print("Average Treatment Effect for Females:", ate_female)


# cate_male = model.effect(X[X['gender'] == 'male'])
# cate_female = model.effect(X[X['gender'] == 'female'])


# plt.hist(cate_male, bins=30, alpha=0.5, label='Males')
# plt.hist(cate_female, bins=30, alpha=0.5, label='Females')
# plt.title("Distribution of CATE by Gender")
# plt.xlabel("CATE")
# plt.ylabel("Frequency")
# plt.legend()
# plt.show()
