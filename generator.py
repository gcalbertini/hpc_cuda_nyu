import numpy as np
import pandas as pd
import os  

N = 1000 #change dimension
M = 10
coeff = np.random.rand(N)*M  # rand 0 to M
b = 123 # intercept
X = np.random.rand(N, N)*M

mu, sigma = 0, 5  # mean and standard deviation for noise
delta = np.random.normal(mu, sigma, N)
y = X @ coeff + b + delta

data = np.c_[y,X]
df = pd.DataFrame(data)

os.makedirs('generated_data', exist_ok=True)  
df.to_csv('generated_data/data.csv', index=False)  