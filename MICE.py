import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load the dataset
file_path = 'EI_2024_Full_data_for_analysis.csv'
file_name = file_path[:-4]
# file_path = 'ESSOREI_LV_24.csv'

data = pd.read_csv(file_path)

# Inspect the dataset
print(data.head())
print(data.info())
if 'blank' in data.columns:
    data = data.drop(columns=['blank'])
# if 'OR6' in data.columns:
#     data = data.drop(columns=['OR6'])

# Create the imputer object
imputer = IterativeImputer(max_iter=10, random_state=0)

# Apply the imputer
imputed_data = imputer.fit_transform(data)


# Convert the imputed data back to a DataFrame
imputed_data = pd.DataFrame(imputed_data, columns=data.columns)
# print(data.head())


# Inspect the imputed dataset
print(imputed_data.head())
print(imputed_data.info())

# imputed_data.to_csv(file_name+'_MICE.csv', index=False)
rounded_MICE = imputed_data.round()
rounded_MICE.to_csv(file_name+'_ROUNDED_MICE.csv', index=False)

