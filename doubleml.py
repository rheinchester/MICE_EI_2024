import matplotlib.pyplot as plt
# import dowhy
# from dowhy import CausalModel
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import doubleml as dml
import numpy as np

# dowhy.__version__
causal_graph = """digraph{
                ESS->OR;
                ESS->PBC;
                ESS->ATB;
                ESS->SSN;
                OR->PBC;
                SSN->PBC;
                SSN->ATB;
                OR->PBC;
                OR->SSN;
                OR->ATB;
                SSN->EI;
                OR->EI;
                PBC->EI
                OR->EI;
                ATB->EI
            }

"""
df_model = pd.DataFrame(columns=['ESS', 'OR', 'SSN', 'ATB', 'PBC', 'EI'])
df_model.head()
covariates = ['ESS']
dml_data = dml.DoubleMLData(df_model, 'y', 'd')