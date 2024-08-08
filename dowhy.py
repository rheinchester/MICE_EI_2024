import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(0)
data = pd.DataFrame({
    'X': np.random.randint(0, 2, 1000),
    'Z': np.random.randint(0, 2, 1000),
    'Y': np.random.randint(0, 2, 1000)
})

# Define a causal model
model = CausalModel(
    data=data,
    treatment='X',
    outcome='Y',
    graph='''graph[
    Z -> X;
    Z -> Y;
    X -> Y;
    ]'''
)

# Identify the causal effect
identified_estimand = model.identify_effect()

# Estimate the causal effect
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression")

# Print the estimate
print(estimate)
