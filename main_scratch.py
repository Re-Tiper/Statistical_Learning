import os
current_dir = os.getcwd()
os.chdir(current_dir)

import pandas as pd               
from sklearn.preprocessing import StandardScaler       
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from dl_models import mlp_scratch # here import your script with dl-model

# %%  Import data

df = pd.read_csv('heart.csv')
df = df.dropna() # remove nan rows
y = df['target'].values.reshape(-1,1)

x = df.drop(columns=['target']).values
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, test_size=0.2, random_state = 42) # get training and testing sets
x_tr, x_vl, y_tr, y_vl = train_test_split(x_tr,y_tr, test_size=0.1, random_state = 42) # get training and validation sets

# %% 

input_size = x_tr.shape[1]  # number of input features
hidden_layers = [50,50]     # number of neurons in hidden layers 
output_size = 1             # output size

# train the model
model = mlp_scratch(input_size, hidden_layers, output_size, 'relu')
model.train(x_tr, y_tr, epochs=100, learning_rate=0.001)

# evaluate the model
y_pred_prob = model.forward(x_ts)[-1]
y_pred = (y_pred_prob > 0.5).astype(int)
f1 = f1_score(y_ts, y_pred)