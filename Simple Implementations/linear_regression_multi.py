import os
import pandas as pd               
import matplotlib.pyplot as plt   
import numpy as np                
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# %% Import data

df = pd.read_csv('Housing_Price_Data.csv') # import data 

# convert string values to integer
label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object': # if column is string
         df[col] = label_encoder.fit_transform(df[col]) # convert it to integer

y = df['price'].values.reshape(-1,1)  # set y (output) the price of real-estate
x = df.drop(columns=['price']).values # set x (input) all features but 'price'=y

# %% pre-Processing

scaler = StandardScaler()   # (x - mean(x)) / std(x) 
x = scaler.fit_transform(x) # scale input data x

# add column with 1 in matrix x
one_columm = np.ones((x.shape[0],1))
x = np.hstack((one_columm, x))

# split to training and testing
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2, random_state=42) 

# %% Train Model

# parameters
a = 1e-1 # a = 0.01
epochs = 100 # number of iteration

# initialize weights
theta = np.zeros((x_tr.shape[1],1))
J = [] # save here the cost J for each epoch
for epoch in range(epochs):
    y_pred = np.dot(x_tr, theta)  # predict y based on current weights
    error  = y_pred - y_tr
    
    # calculate the gradient
    gradient = (1/x_tr.shape[0]) * np.dot(x_tr.T, error)
    
    # Update theta (parameters)
    theta = theta - a * gradient
    
    # print cost, b0 and b1
    cost = (1/(2*x_tr.shape[0])) * np.dot(error.T, error)
    J.append(cost[0][0])
    print(f"Epoch {epoch}: Cost {cost}")
 
#plt.figure()
plt.plot(range(epochs), J, label=f'learning rate {a}')
plt.xlabel('Epoch')
plt.ylabel('Cost function J')
plt.legend()
    
# caclulate predictions for x_ts
y_pred = np.dot(x_ts, theta)
rmse = np.mean((y_ts-y_pred)**2)**(1/2) 
 
