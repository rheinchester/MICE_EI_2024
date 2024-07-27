import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load the dataset
file_path = 'EI 2020 Data.csv'
data = pd.read_csv(file_path)

# Inspect the dataset
print(data.head())
# print(data.info())
if 'blank' in data.columns:
    data = data.drop(columns=['blank'])
if 'OR6' in data.columns:
    data = data.drop(columns=['OR6'])

# Create the imputer object
imputer = IterativeImputer(max_iter=10, random_state=0)

# Apply the imputer
imputed_data = imputer.fit_transform(data)


# Convert the imputed data back to a DataFrame
imputed_data = pd.DataFrame(imputed_data, columns=data.columns)
print(data.head())


# Inspect the imputed dataset
print(imputed_data.head())
print(imputed_data.info())

if file_path == 'EI_2024_FOR_ANALYSIS.csv':
    imputed_data.to_csv('EI_MICE_2024_imputed.csv', index=False)
if file_path == 'EI 2020 Data.csv':
    imputed_data.to_csv('EI_2020_imputed.csv', index=False)