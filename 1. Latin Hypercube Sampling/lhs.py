import numpy as np
import pandas as pd
from scipy.stats import qmc

# 8000 Samples per Weave pattern
num_samples = 8000
num_dimensions = 3

sampler = qmc.LatinHypercube(d=num_dimensions)
lhs_samples = sampler.random(n=num_samples)

# Continuous inputs: Vf, Wy/Sy, Ty/Sy
param_lows = np.array([0, 0, 0])
param_highs = np.array([1, 1, 1])
scaled_samples = qmc.scale(lhs_samples, param_lows, param_highs)

# Create DataFrame
df = pd.DataFrame(scaled_samples, columns=['Vf', 'Width to Spacing', 'Thickness to Spacing'])

# Save to CSV
df.to_csv("Vf_data_updated.csv", index=False)
print("File saved as Vf_data_updated.csv")
print(df.head())
