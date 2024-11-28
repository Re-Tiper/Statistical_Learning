import os                         # handle your path
import pandas as pd               # basic functions for dataset import, preprocessing etc.
import matplotlib.pyplot as plt   # plots
import numpy as np                # basic mathematical functions
from sklearn.model_selection import train_test_split  # for splitting data to train and test
from sklearn.preprocessing import StandardScaler      # scale input x 

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# %% Import data

df = pd.read_csv('Housing_Price_Data.csv') # import data 
y = df['price'].values.reshape(-1,1)  # set y (output) the price of real-estate
x = df['area'].values.reshape(-1,1)   # set x (input) the area of real-estate 

# plot x,y 
plt.figure()
plt.plot(x,y, '.')
plt.ylabel('Price in dollar')
plt.xlabel('Area in square feet')

# %% pre-Processing

scaler = StandardScaler()   # (x - mean(x)) / std(x) 
x = scaler.fit_transform(x) # scale input data x

# split to training and testing
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2, random_state=42) 

# %% Train Model

# parameters
learning_rate = 1e-2 # a = 0.01 
epochs = 250         # number of iteration

# initialize weights
b0 = 0
b1 = 0
for epoch in range(epochs):
    y_pred = b0 + b1 * x_tr  # predict y based on current weights
    
    # calculate gradients
    D_b0 = np.mean(y_pred - y_tr)
    D_b1 = np.mean((y_pred - y_tr) * x_tr)
    
    # update weights with gradient descnet 
    b0 = b0 - learning_rate * D_b0
    b1 = b1 - learning_rate * D_b1
    
    # print cost, b0 and b1
    cost = np.mean((y_tr - y_pred) ** 2)
    print(f"Epoch {epoch}: Cost {cost}, b0 {b0}, b1 {b1}")
  
# %% 
    
# caclulate predictions for x_ts
y_pred = b0 + b1 * x_ts
rmse = np.mean((y_ts-y_pred)**2)**(1/2) 
 
# Plot y = b0 + b1 * x line (red) and original (or scaled) data
x_range = np.linspace(min(x), max(x), 100)
y_range = b0 + b1 * x_range
plt.figure()
plt.plot(x, y, '.', label='Data points')
plt.plot(x_range, y_range, '-', color='red', label='Regression line')
plt.ylabel('Price in dollar')
plt.xlabel('Area in square feet')
plt.legend()




