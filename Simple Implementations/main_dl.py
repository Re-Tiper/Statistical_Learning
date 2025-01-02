import os
current_dir = os.getcwd()
os.chdir(current_dir)

import warnings
warnings.filterwarnings('ignore')

import pandas as pd               
from sklearn.preprocessing import StandardScaler       
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
from dl_models import mlp # here import your script with dl-model

# %%  Import data

df = pd.read_csv('heart.csv')
df = df.dropna() # remove nan rows
y = df['target'].values.reshape(-1,1)

x = df.drop(columns=['target']).values
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, test_size=0.2, random_state = 42) # get training and testing sets
x_tr, x_vl, y_tr, y_vl = train_test_split(x_tr,y_tr, test_size=0.1, random_state = 42) # get training and validation sets

# %% Train DL model

"""
Define your model architecture
"""
shape_inp = x_tr.shape[1] # number of input features
units = 50                # number of neurons per layer
dropout = 0.2             # dropout percent
num_epochs = 200 # number of times where the model will "see" all the training-set
batch_size = 64  # mini-batch size
model = mlp(shape_inp, units, dropout) 

"""
Define the optimizer and loss function 
"""
# "sgd" = gradient descent or "Adam" (usually better convergence)
# loss = "CategoricalCrossentropy"  for k > 2 classes; 
# loss = "MeanSquaredError" for regression prediction
opt = tf.optimizers.SGD(0.01)
model.compile(optimizer=opt, loss= "BinaryCrossentropy") 

"""
Train your model
"""
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=20, 
                                                  restore_best_weights=True)
history  = model.fit(x_tr, y_tr, 
                     epochs=num_epochs, 
                     batch_size=batch_size, 
                     validation_data=(x_vl, y_vl), 
                     callbacks = [early_stopping],
                     verbose=1)

# make prediction for testing set
y_pred_prob = model.predict(x_ts)
y_pred = (y_pred_prob > 0.5).astype(int) # make the probabilities binary 
f1 = f1_score(y_ts,y_pred) # f1_score(true values, predicted values)

# %% plotds of the training and validation loss per epoch

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss per Epoch')
plt.show()

# %% save model

model.save('heart_attack_predictor.keras')

# load the model
from tensorflow.keras.models import load_model
loaded_model = load_model('heart_attack_predictor.keras')






