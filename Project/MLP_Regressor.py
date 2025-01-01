# General libraries
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
# Suppresses warnings to keep the output clean
import warnings
warnings.filterwarnings('ignore')
# Basics from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
# TensorFlow for MLP
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2, l1
from imblearn.over_sampling import SMOTE # for resampling

def smoter(X, y, n_samples=None, k_neighbors=5):
    """
    Custom implementation of SMOTER.
    Args:
        X: Features (array-like, shape [n_samples, n_features])
        y: Target (array-like, shape [n_samples])
        n_samples: Number of synthetic samples to generate.
        k_neighbors: Number of neighbors to consider for interpolation.
    Returns:
        X_resampled: Resampled feature set.
        y_resampled: Resampled target values.
    """
    # Ensure y is a NumPy array for consistent indexing
    y = np.array(y)

    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X)
    indices = np.random.choice(len(X), n_samples, replace=True)
    X_synthetic, y_synthetic = [], []

    for idx in indices:
        neighbors = nn.kneighbors([X[idx]], return_distance=False).flatten()
        neighbor_idx = np.random.choice(neighbors)
        lambda_ = np.random.rand()

        X_new = X[idx] + lambda_ * (X[neighbor_idx] - X[idx])
        y_new = y[idx] + lambda_ * (y[neighbor_idx] - y[idx])

        X_synthetic.append(X_new)
        y_synthetic.append(y_new)

    return np.vstack([X, X_synthetic]), np.hstack([y, y_synthetic])


"""Import dataset"""
# Original dataset
#df = pd.read_csv('winequality-white.csv', delimiter=';')

# Dataset with anomalies indication from Anomalies_detection.py
df = pd.read_csv('winequality_anomalies.csv', delimiter=';')
df = df[df['anomaly'] != -1]

# Prepare Features and Target
X = df.drop(columns=['quality', 'anomaly']) # drop column 'anomaly' when using winequality_anomalies.csv
y = df['quality']

#class_counts = y.value_counts()
#print(class_counts)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (no benefit gained)
#pca = PCA(n_components=0.95) # Retain 95% of the variance
#X_scaled = pca.fit_transform(X_scaled)

"""MLP model"""

# Function to define MLP architecture for regression
def build_mlp_regression(input_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),  #kernel_regularizer=l2(0.01)
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  # Single output for continuous prediction
    ])
    return model

# MLP Model Training for Regression
mlp_mse = []  # To store MSE metrics
mlp_mae = []  # To store MAE metrics
mlp_rmse = []  # To store RMSE metrics

for i in range(10):  # Perform 10 different train-test splits
    # Start the timer
    start_time = time.time()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    # Produce synthetic data to better represent
    #X_train, y_train = smoter(X_train, y_train, n_samples=500)

    # Define and compile MLP for regression
    mlp = build_mlp_regression(input_dim=X_train.shape[1])
    opt = tf.optimizers.SGD(learning_rate=0.01, momentum=0.9) # works better than adam, momentum slightly improves convergence.
    mlp.compile(optimizer=opt, loss='mse', metrics=['mse'])

    # Train MLP
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = mlp.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0)

    # Evaluate MLP
    y_test_pred = mlp.predict(X_test).flatten()  # Flatten to make it a 1D array of predictions
    mse = np.mean((y_test - y_test_pred) ** 2)  # Calculate Mean Squared Error
    mae = np.mean(np.abs(y_test - y_test_pred))  # Calculate Mean Absolute Error
    rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2)) # Calculate RMSE
    mlp_mse.append(mse)
    mlp_mae.append(mae)
    mlp_rmse.append(rmse)

    # Stop the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

# MLP Regression Report
mean_mlp_mse = np.mean(mlp_mse)
std_mlp_mse = np.std(mlp_mse)
mean_mlp_mae = np.mean(mlp_mae)
std_mlp_mae = np.std(mlp_mae)
mean_mlp_rmse = np.mean(mlp_rmse)
std_mlp_rmse = np.std(mlp_rmse)

# Display the results
print("MLP Regression:")
print(f"  Mean RMSE: {mean_mlp_rmse:.4f}")
print(f"  Std Dev of RMSE: {std_mlp_rmse:.4f}")
print(f"  Mean MSE: {mean_mlp_mse:.4f}")
print(f"  Std Dev of MSE: {std_mlp_mse:.4f}")
print(f"  Mean MAE: {mean_mlp_mae:.4f}")
print(f"  Std Dev of MAE: {std_mlp_mae:.4f}")
print("-" * 60)

# Save the trained MLP model
#mlp.save('mlp_regression.keras')

# Load the saved MLP model
#loaded_mlp = tf.keras.models.load_model('mlp_regression.keras')

"""
MLP Regression: (no SMOTER)
  Mean RMSE: 0.6686
  Std Dev of RMSE: 0.0139
  Mean MSE: 0.4472
  Std Dev of MSE: 0.0188
  Mean MAE: 0.5216
  Std Dev of MAE: 0.0112
"""

"""Plots"""

# Plot histograms for MLP Model

plt.figure(figsize=(12, 6))
# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (Loss)')
plt.title('MLP Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
# Make predictions on the test set
y_test_pred = loaded_mlp.predict(X_test).flatten()  # Flatten to make it a 1D array of predictions
# Round predictions to nearest integer since wine quality ratings are discrete values
y_test_pred_rounded = np.round(y_test_pred).astype(int)

# Ensure that y_test_pred_rounded and y_test are pandas Series
y_test_pred_rounded_series = pd.Series(y_test_pred_rounded)
y_test_series = pd.Series(y_test)
# Get the count of each class (wine quality rating)
class_counts_pred = y_test_pred_rounded_series.value_counts()
print("Predicted Wine Quality Counts:")
print(class_counts_pred)
class_counts_true = y_test_series.value_counts()
print("True Wine Quality Counts:")
print(class_counts_true)

plt.figure(figsize=(12, 6))
# Round predictions to nearest integer since wine quality ratings are discrete values
y_test_pred_rounded = np.round(y_test_pred).astype(int)
# Histogram for true values and predicted values from the MLP model
plt.subplot(1, 2, 1)
plt.hist(y_test, bins=np.arange(2.5, 10.5, 1), alpha=1, label='True Values', edgecolor='black', color='darkgreen')
plt.xlabel('Wine Quality Rating')
plt.ylabel('Frequency')
plt.title('True values')
plt.legend()
plt.subplot(1, 2, 2)
plt.hist(y_test_pred_rounded, bins=np.arange(2.5, 10.5, 1), alpha=0.5, label='Predicted Values', edgecolor='black', color='darkgreen')
plt.xlabel('Wine Quality Rating')
plt.ylabel('Frequency')
plt.title('Predicted values')
plt.legend()
plt.tight_layout()
plt.show()

# Residual plot to see the differences
plt.figure(figsize=(8, 6))
residuals = y_test - y_test_pred
plt.hist(residuals, bins=np.arange(np.min(residuals.unique()) - 0.5, np.max(residuals.unique()) + 0.5, 1),
         alpha=1, label='Residuals', edgecolor='black', color='darkgreen')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('MLP Model: Residuals Histogram')
plt.legend()
plt.show()
