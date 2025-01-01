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
# Models from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


def regularization(reg_mode, theta, m):
    if reg_mode == 'L1':
        cost_term = np.sum(np.abs(theta))
        updt_term = np.sign(theta)

    elif reg_mode == 'L2':
        cost_term = np.sum(theta ** 2) / (2 * m)
        updt_term = theta / m

    return cost_term, updt_term

"""Import dataset"""
# Original dataset
df = pd.read_csv('winequality-white.csv', delimiter=';')

# Dataset with anomalies indication from Anomalies_detection.py
#df = pd.read_csv('winequality_anomalies.csv', delimiter=';')
#df = df[df['anomaly'] != -1]

# Prepare Features and Target
X = df.drop(columns=['quality']) # drop column 'anomaly' when using winequality_anomalies.csv
y = df['quality']

#class_counts = y.value_counts()
#print(class_counts)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = X_scaled

# Add a bias (ones) column to the feature matrix
one_column = np.ones((X.shape[0], 1))
X_lr = np.hstack((one_column, X))

"""Run PCA to reduce dimensions (does not help with the improvement of the predictions, in this case)"""
'''
# The model
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)
X = X_pca
# PCA components
# The components matrix contains the weights for each original feature in each principal component.
# Each row of the matrix represents a principal component.
# Each column corresponds to the contribution of an original feature to that principal component.
components = pca.components_
#print("PCA Components (eigenvectors):\n", components)

# Variance explained by each component
explained_variance = pca.explained_variance_ratio_
#print("Variance explained by each component:\n", explained_variance)
'''

"""Train and Evaluate Model from scratch"""

# Parameters
learning_rate = 0.5  # Learning rate
epochs = 200  # Number of iterations
L = 0.01  # Regularization parameter
reg_mode = 'L2'  # Regularization method: 'L1' or 'L2'

lm_mse = []  # To store MSE metrics
lm_mae = []  # To store MAE metrics
lm_rmse = []  # To store RMSE metrics

for i in range(10):  # Perform 10 different train-test splits for each model

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_lr, y, test_size=0.2)

    # Initialize weights (theta) and cost history
    theta = np.zeros((X_train.shape[1], 1))  # Weight vector for features
    cost_history = []  # To save cost (J) at each epoch for each iteration

    # Reshape y_train to be a column vector (for calculations)
    y_train = y_train.values.reshape(-1, 1)

    # Gradient descent
    for epoch in range(epochs):
        # Prediction: y_pred = X_train * theta
        y_pred = np.dot(X_train, theta)

        # Error
        error = y_pred - y_train

        # Calculate regularization cost_term and update_term for L1 or L2
        m = X_train.shape[0]  # Training set size
        cost_term, updt_term = regularization(reg_mode, theta, m)

        # Compute cost (Mean Squared Error)
        cost = (1 / (2 * X_train.shape[0])) * np.dot(error.T, error) + L * cost_term
        cost_history.append(cost[0][0])

        # Gradient calculation
        gradient = (1 / X_train.shape[0]) * np.dot(X_train.T, error) + L * updt_term

        # Update weights
        theta -= learning_rate * gradient

        # Print cost for each epoch
        #print(f"Epoch {epoch + 1}: Cost {cost[0][0]:.4f}")

    # %% Visualize Cost Function
    #plt.figure()
    #plt.plot(range(1, epochs + 1), cost_history, label=f'Learning Rate: {learning_rate}')
    #plt.xlabel('Epochs')
    #plt.ylabel('Cost Function J')
    #plt.title('Cost Function vs. Epochs')
    #plt.legend()
    #plt.show()

    # %% Test Model
    # Reshape y_test for consistency
    y_test = y_test.values.reshape(-1, 1)

    # Predict on the test set
    y_test_pred = np.dot(X_test, theta)

    # Calculate RMSE for train data to check for overfitting
    #y_train_pred = np.dot(X_train, theta)
    #train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    # Print the RMSE for the training data
    #print(f"Train RMSE: {train_rmse:.4f}")

    # Calculate metrics
    rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    mse = np.mean((y_test - y_test_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_test_pred))

    lm_mse.append(mse)
    lm_mae.append(mae)
    lm_rmse.append(rmse)

# Linear Regression Report
mean_lm_mse = np.mean(lm_mse)
std_lm_mse = np.std(lm_mse)
mean_lm_mae = np.mean(lm_mae)
std_lm_mae = np.std(lm_mae)
mean_lm_rmse = np.mean(lm_rmse)
std_lm_rmse = np.std(lm_rmse)

print("Scratch linear regression:")
print(f"  Mean RMSE: {mean_lm_rmse:.4f}")
print(f"  Std Dev of RMSE: {std_lm_rmse:.4f}")
print(f"  Mean MSE: {mean_lm_mse:.4f}")
print(f"  Std Dev of MSE: {std_lm_mse:.4f}")
print(f"  Mean MAE: {mean_lm_mae:.4f}")
print(f"  Std Dev of MAE: {std_lm_mae:.4f}")
print("  Learned Weights:")
for i in range(len(theta)):
    print(f"  θ_{i} = {theta[i][0]:.10f}")
print('-' * 30)


"""Plot"""
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
# Residuals as a NumPy array
residuals = np.array(y_test - y_test_pred_rounded).flatten()
# Use np.unique to get unique residual values
unique_residuals = np.unique(residuals)
# Create the histogram
plt.hist(residuals, bins=np.arange(np.min(unique_residuals) - 0.5, np.max(unique_residuals) + 0.5, 1),
         alpha=1, label='Residuals', edgecolor='black', color='darkgreen')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Linear regression Model: Residuals Histogram')
plt.legend()
plt.show()


"""Train and Evaluate Model Using sklearn"""

sklearn_metrics = []
# Evaluate models 10 times
for i in range(10):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_test_pred = model.predict(X_test)

    # Calculate RMSE for train data to check for overfitting
    #y_train_pred = model.predict(X_train)
    #train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    # Print the RMSE for the training data
    #print(f"Train RMSE: {train_rmse:.4f}")

    # Extract the weights (coefficients) and the intercept
    weights = model.coef_  # Coefficients (θ_1, θ_2, ..., θ_n)
    intercept = model.intercept_  # Intercept (θ_0)

    # Calculate metrics
    rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    mse = np.mean((y_test - y_test_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_test_pred))
    # Append metrics to sklearn_metrics
    sklearn_metrics.append((rmse, mse, mae))

# Calculate mean and std deviation for metrics
sklearn_metrics = np.array(sklearn_metrics)

print("Sklearn linear regression:")
print(f"  Mean RMSE: {np.mean(sklearn_metrics[:, 0]):.4f}")
print(f"  Std Dev of RMSE: {np.std(sklearn_metrics[:, 0]):.4f}")
print(f"  Mean MSE: {np.mean(sklearn_metrics[:, 1]):.4f}")
print(f"  Std Dev of MSE: {np.std(sklearn_metrics[:, 1]):.4f}")
print(f"  Mean MAE: {np.mean(sklearn_metrics[:, 2]):.4f}")
print(f"  Std Dev of MAE: {np.std(sklearn_metrics[:, 2]):.4f}")
print("  Learned Weights (using sklearn):")
print(f"  θ_0 = {intercept:.10f}")
for i in range(len(weights)):
    print(f"  θ_{i+1} = {weights[i]:.10f}")
print('-' * 30)


"""Plot"""
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
# Residuals as a NumPy array
residuals = np.array(y_test - y_test_pred_rounded).flatten()
# Use np.unique to get unique residual values
unique_residuals = np.unique(residuals)
# Create the histogram
plt.hist(residuals, bins=np.arange(np.min(unique_residuals) - 0.5, np.max(unique_residuals) + 0.5, 1),
         alpha=1, label='Residuals', edgecolor='black', color='darkgreen')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Linear regression Model: Residuals Histogram')
plt.legend()
plt.show()