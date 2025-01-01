import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def regularization(reg_mode, theta, m):
    if reg_mode == 'L1':
        cost_term = np.sum(np.abs(theta))
        updt_term = np.sign(theta)

    elif reg_mode == 'L2':
        cost_term = np.sum(theta ** 2) / (2 * m)
        updt_term = theta / m

    return cost_term, updt_term

# %% Load data and Pre-process the data

df = pd.read_csv('winequality_anomalies.csv', delimiter=';')
# Remove anomalies
df = df[df['anomaly'] != -1]

# Create Target Classes
def classify_quality(quality):
    if quality <= 4:
        return 0  # Bad
    elif quality <= 6:
        return 1  # Moderate
    else:
        return 2  # Good

df['quality_label'] = df['quality'].apply(classify_quality)

# Prepare Features and Target
X = df.drop(columns=['quality', 'quality_label'])
y = df['quality_label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# add column with 1 in matrix X
#one_columm = np.ones((X.shape[0],1))
#X = np.hstack((one_columm, X))

# Split data into training and test sets
x_tr, x_ts, y_tr, y_ts = train_test_split(X_scaled, y, test_size=0.2, random_state=10)

# %% Train Model using One-vs-All

# Hyperparameters
a = 1e-1 # Learning rate
epochs = 500  # Number of iterations
L = 0.1  # Regularization parameter
reg_mode = 'L1'  # Regularization method: 'L1' or 'L2'

# Number of classes
num_classes = len(np.unique(y))

# Initialize theta (weights) for each class
theta_all = np.zeros((num_classes, x_tr.shape[1]))

# Train one logistic regression classifier per class
for i in range(num_classes):
    #print(f"Training classifier for class {i}")

    # Create the target vector for this class: 1 if the sample belongs to class i, else 0
    y_tr_class = (y_tr == i).astype(int)

    # Initialize weights for this classifier
    theta = np.zeros((x_tr.shape[1], 1))  # Initialize with zeros
    m = x_tr.shape[0]  # Training set size
    J = []  # Save the cost function for each epoch

    # Gradient Descent for binary classifier
    for epoch in range(epochs):
        y_pred = sigmoid(np.dot(x_tr, theta))  # Predict y = {0,1} based on current weights
        error = y_pred - y_tr_class.values.reshape(-1, 1)

        # Calculate regularization cost_term and update_term for L1 or L2
        cost_term, updt_term = regularization(reg_mode, theta, m)

        # Calculate the gradient
        gradient = (1 / m) * np.dot(x_tr.T, error) + L * updt_term

        # Update theta (parameters)
        theta = theta - a * gradient

        # Compute the cost function with regularization
        cost = -np.mean(
            y_tr_class.values.reshape(-1, 1) * np.log(y_pred) + (1 - y_tr_class.values.reshape(-1, 1)) * np.log(
                1 - y_pred)) + L * cost_term
        J.append(cost)

    # Save the learned weights for this class
    theta_all[i, :] = theta.flatten()

    # Plot the cost function convergence for this class
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), J, label=f'Class {i}')
    plt.xlabel('Epochs')
    plt.ylabel('Cost function J')
    plt.title(f'Cost function convergence for class {i}')
    plt.legend()
    plt.grid(True)
    #plt.show()

# Predict the class for each test sample by choosing the classifier with the highest probability
y_pred_prob = np.dot(x_ts, theta_all.T)  # Matrix of probabilities for each class
y_pred = np.argmax(y_pred_prob, axis=1)  # Choose the class with the highest probability
acc = np.sum(y_pred == y_ts.values) / len(y_ts)
print(f"Accuracy from scratch: {acc:.4f}")


"""Train and Evaluate Model Using sklearn"""

# Train using sklearn's LogisticRegression
model = OneVsRestClassifier(LogisticRegression(solver='saga', penalty='l2', max_iter=500))
model.fit(x_tr,y_tr)
y_pred_2 = model.predict(x_ts)
acc_2 = np.sum(y_pred_2 == y_ts)/len(y_ts)
print(f"Accuracy from sklearn: {acc_2:.4f}")

"""
Result:
Accuracy from scratch: 0.6563
Accuracy from sklearn: 0.7830
"""