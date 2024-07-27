import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

file_path = 'EI 2020 Data.csv'
data = pd.read_csv(file_path)

# imputer = IterativeImputer(max_iter=10, random_state=0)
# # Apply the imputer
# data_imputed = imputer.fit_transform(data)
# # Convert the imputed data back to a DataFrame
# data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
# knn_imputer = KNNImputer(n_neighbors=5)
# data_imputed = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)


from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomKNNMICEImputer(BaseEstimator, TransformerMixin):
    def __init__(self, k=5, max_iter=10, tol=1e-3, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def fit(self, X, y=None):
        self.X_ = X.copy()
        self.n_features_ = X.shape[1]
        self.imputers_ = [KNNImputer(n_neighbors=self.k) for _ in range(self.n_features_)]
        return self
    
    def transform(self, X):
        X_filled = X.copy()
        missing_mask = np.isnan(X_filled)
        
        for i in range(self.max_iter):
            X_old = X_filled.copy()
            
            for feature in range(self.n_features_):
                missing_indices = missing_mask[:, feature]
                if np.any(missing_indices):
                    X_temp = X_filled.copy()
                    X_temp[:, feature] = np.nan  # Mask the current feature
                    
                    imputer = self.imputers_[feature]
                    X_filled[missing_indices, feature] = imputer.fit_transform(X_temp)[:, feature][missing_indices]
            
            # Check for convergence
            if np.linalg.norm(X_filled - X_old, ord='fro') < self.tol:
                break
        
        return X_filled

# Create the custom imputer object
custom_imputer = CustomKNNMICEImputer(k=5, max_iter=10, random_state=0)
imputed_data = custom_imputer.fit_transform(data)

# Convert the imputed data back to a DataFrame
imputed_data = pd.DataFrame(imputed_data, columns=data.columns)



if file_path == 'EI 2020 Data.csv':
    imputed_data.to_csv('EI 2020 Data_KNN+MICE.csv', index=False)