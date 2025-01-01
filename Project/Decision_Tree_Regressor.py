# General libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# Suppresses warnings to keep the output clean
import warnings
warnings.filterwarnings('ignore')
# Basics from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Models from sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

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
X = X_scaled

# Apply PCA (no benefit gained)
#pca = PCA(n_components=0.95) # Retain 95% of the variance
#X_scaled = pca.fit_transform(X_scaled)

"""Decision Tree"""

# Decision Tree Model Training for Regression
dt_mse = []  # To store MSE metrics
dt_mae = []  # To store MAE metrics
dt_rmse = []  # To store RMSE metrics

for i in range(10):  # Perform 10 different train-test splits

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Produce synthetic data to better represent
    #X_train, y_train = smoter(X_train, y_train, n_samples=500)

    # Check distribution of resampled target variable
    #y_train_rounded = np.round(y_test).astype(int)
    #print(pd.Series(y_train_rounded).value_counts())

    # Define and train Decision Tree model for regression
    dt = DecisionTreeRegressor(max_depth=None, min_samples_leaf=20)
    dt.fit(X_train, y_train)

    # Predict on the test set
    y_test_pred = dt.predict(X_test)

    # Calculate RMSE for train data to check for overfitting
    y_train_pred = dt.predict(X_train)
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    # Print the RMSE for the training data
    #print(f"Train RMSE: {train_rmse:.4f}")

    # Calculate metrics
    mse = np.mean((y_test - y_test_pred) ** 2)  # Mean Squared Error
    mae = np.mean(np.abs(y_test - y_test_pred))  # Mean Absolute Error
    rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))  # Root Mean Squared Error

    # Append metrics
    dt_mse.append(mse)
    dt_mae.append(mae)
    dt_rmse.append(rmse)

# Decision Tree Regression Report
mean_dt_mse = np.mean(dt_mse)
std_dt_mse = np.std(dt_mse)
mean_dt_mae = np.mean(dt_mae)
std_dt_mae = np.std(dt_mae)
mean_dt_rmse = np.mean(dt_rmse)
std_dt_rmse = np.std(dt_rmse)

# Display the results
print("Decision Tree Regression:")
print(f"  Mean RMSE: {mean_dt_rmse:.4f}")
print(f"  Std Dev of RMSE: {std_dt_rmse:.4f}")
print(f"  Mean MSE: {mean_dt_mse:.4f}")
print(f"  Std Dev of MSE: {std_dt_mse:.4f}")
print(f"  Mean MAE: {mean_dt_mae:.4f}")
print(f"  Std Dev of MAE: {std_dt_mae:.4f}")
print("-" * 30)


plt.figure(figsize=(12, 6))
# Round predictions to nearest integer since wine quality ratings are discrete values
# (we round the values if we use a resampling method otherwise the tree returns integer values)
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
residuals = y_test - y_test_pred_rounded
plt.hist(residuals, bins=np.arange(np.min(residuals.unique()) - 0.5, np.max(residuals.unique()) + 0.5, 1),
         alpha=1, label='Residuals', edgecolor='black', color='darkgreen')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Decision Tree Model: Residuals Histogram')
plt.legend()
plt.show()