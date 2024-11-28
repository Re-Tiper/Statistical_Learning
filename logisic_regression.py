import os
import pandas as pd               
import numpy as np 
from sklearn.preprocessing import StandardScaler       
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# %% functions

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s 

def regularization(reg_mode, theta, m):
    if reg_mode == 'L1':
        cost_term = np.sum(np.abs(theta))
        updt_term = np.sign(theta)

    elif reg_mode == 'L2':
        cost_term = np.sum(theta**2)/(2*m)
        updt_term = theta/m
    
    return cost_term, updt_term
    
# %% Load data and Pre-process the data

df = pd.read_csv('heart.csv')
df = df.dropna() # remove nan rows
y = df['target'].values.reshape(-1,1)

x = df.drop(columns=['target']).values
scaler = StandardScaler()
x = scaler.fit_transform(x)

# add column with 1 in matrix x
one_columm = np.ones((x.shape[0],1))
x = np.hstack((one_columm, x))

x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, test_size=0.2, random_state = 42)

# %% Train Model

# hyper-parameters
a = 1e-1     # learning rates
epochs = 500 # number of iteration
L = 0.1      # lambda parameter for reg.
reg_mode = 'L1' # select reg. method, 'L1' or 'L2'

# initialize weights
theta = np.zeros((x_tr.shape[1],1))
m = x_tr.shape[0] # training set size
J = [] # save here the cost J for each epoch
for epoch in range(epochs):
    y_pred = sigmoid(np.dot(x_tr, theta))  # predict y = {0,1} based on current weights
    error  = y_pred - y_tr
    
    # caclulate reg. cost_term and updt_term for L1 or L2 (check reg_mode)
    cost_term, updt_term = regularization(reg_mode, theta, m)
    
    # calculate the gradient
    gradient = (1/m) * np.dot(x_tr.T, error) + L*updt_term
    
    # Update theta (parameters)
    theta = theta - a * gradient
    
    cost = -np.mean(y_tr*np.log(y_pred) + (1-y_tr)*np.log(1-y_pred)) + L*cost_term
    J.append(cost)
    print(f"Epoch {epoch}: Cost {cost}")
 
# plt.figure()
plt.plot(range(epochs), J, label=f'learning rate {a} and reg. parameter {L}')
plt.xlabel('Epoch')
plt.ylabel('Cost function J')
plt.legend()
plt.grid()

# Accuracy on testing set
y_pred_prob = sigmoid(np.dot(x_ts, theta))
y_pred = (y_pred_prob >= 0.5).astype(int)
acc = np.sum(y_pred == y_ts)/len(y_ts)

# %%  From sklearn.linear_model library

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_tr,y_tr.ravel()) 
y_pred_2 = model.predict(x_ts)
acc_2 = np.sum(y_pred_2 == y_ts.ravel())/len(y_ts)




