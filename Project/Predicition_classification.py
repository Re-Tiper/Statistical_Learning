# General libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
# Suppresses warnings to keep the output clean
import warnings
warnings.filterwarnings('ignore')
# Basics from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
# Models from sklearn
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
# Metrics
from sklearn.metrics import f1_score, adjusted_rand_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# TensorFlow for MLP
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2, l1


"""Import dataset"""
# Original dataset
#df = pd.read_csv('winequality-white.csv', delimiter=';')

# Dataset with anomalies indication from Anomalies_detection.py
df = pd.read_csv('winequality_anomalies.csv', delimiter=';')
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

#class_counts = y.value_counts()
#print(class_counts)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (no benefit gained)
#pca = PCA(n_components=0.95) # Retain 95% of the variance
#X_scaled = pca.fit_transform(X_scaled)


"""Train and Evaluate Models"""

models = {
    'SVC': SVC(decision_function_shape='ovo', kernel='rbf', gamma=1.5, C=2),
    'Decision Tree': DecisionTreeClassifier(max_depth=None, min_samples_leaf=10),
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(solver='saga', penalty='l1', max_iter=500))
}

f1s = {model_name: [] for model_name in models.keys()}
aris = {model_name: [] for model_name in models.keys()}
#precisions = {model_name: [] for model_name in models.keys()}

for i in range(1):  # Perform 10 different train-test splits for each model
    for model_name, model in models.items():

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

        # Train the model
        model.fit(X_train, y_train)

        # Test the model
        y_test_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_test_pred, average='micro')
        ari = adjusted_rand_score(y_test, y_test_pred)
        #precision = precision_score(y_test, y_test_pred, average='micro')

        # Save the results from this run
        f1s[model_name].append(f1)
        aris[model_name].append(ari)
        #precisions[model_name].append(precision)

# Store predictions for analysis (last iterations)
predictions = {model_name: model.predict(X_test) for model_name, model in models.items()}

# Calibration Report
for model_name in models.keys():
    # Calculate mean and std of F1 scores
    mean_f1 = np.mean(f1s[model_name])
    std_f1 = np.std(f1s[model_name])
    mean_ari = np.mean(aris[model_name])
    std_ari = np.std(aris[model_name])
    #mean_precision = np.mean(precisions[model_name])
    #std_precision = np.std(precisions[model_name])
    # Display all information
    print(f"{model_name}:")
    print(f"  Mean F1 Score: {mean_f1:.4f}")
    print(f"  Std Dev of F1 Score: {std_f1:.4f}")
    print(f"  Mean Adjusted Rand Index: {mean_ari:.4f}")
    print(f"  Std Dev of Adjusted Rand Index: {std_ari:.4f}")
    #print(f"  Mean Precision Score: {mean_precision:.4f}")
    #print(f"  Std Dev of Precision Score: {std_precision:.4f}")
    print(f"  Classification Report:")
    print(classification_report(y_test, predictions[model_name]))
    print("-" * 60)



"""MLP model"""

# Function to define MLP architecture
def build_mlp(input_dim, num_classes):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'), #kernel_regularizer=l2(0.01)
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model

# MLP Model Training
mlp_f1 = []  # To store F1 metrics
mlp_ari = [] # To store ARI metrics

for i in range(1):  # Perform 10 different train-test splits
    # Start the timer
    start_time = time.time()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    # Define and compile MLP
    num_classes = len(np.unique(y))
    mlp = build_mlp(input_dim=X_train.shape[1], num_classes=num_classes)
    opt = tf.optimizers.SGD(0.01) # use adam for better results
    mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train MLP
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = mlp.fit(X_train, y_train, epochs=500, batch_size=32, callbacks = [early_stopping], verbose=0) # No validation_split=0.1 (better metrics but longer training time)

    # Evaluate MLP
    y_test_pred_prob = mlp.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)  # Convert probabilities to class labels
    f1 = f1_score(y_test, y_test_pred, average='micro')
    mlp_f1.append(f1)
    ari = adjusted_rand_score(y_test, y_test_pred)
    mlp_ari.append(ari)

    # Stop the timer
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time

    print(f"Training time: {elapsed_time:.2f} seconds")

# MLP Report
mean_mlp_f1 = np.mean(mlp_f1)
std_mlp_f1 = np.std(mlp_f1)
mean_mlp_ari = np.mean(mlp_ari)
std_mlp_ari = np.std(mlp_ari)
classification_rep = classification_report(y_test, y_test_pred)
# Display the results
print("MLP:")
print(f"  Mean F1 Score: {mean_mlp_f1:.4f}")
print(f"  Std Dev of F1 Score: {std_mlp_f1:.4f}")
print(f"  Mean Adjusted Rand Index: {mean_mlp_ari:.4f}")
print(f"  Std Dev of Adjusted Rand Index: {std_mlp_ari:.4f}")
print(f"  Classification Report:")
print(classification_rep)
print("-" * 60)

# Save the trained MLP model
#mlp.save('mlp_classification.keras')

# Load the saved MLP model
#loaded_mlp = tf.keras.models.load_model('mlp_classification.keras')


"""Plots"""

# Add the MLP predictions to the predictions dictionary
predictions['MLP'] = y_test_pred

# List of mean F1 scores for each model
mean_f1_scores = {model_name: np.mean(f1s[model_name]) for model_name in models.keys()}
mean_f1_scores['MLP'] = mean_mlp_f1

# Plot the F1 scores
plt.figure(figsize=(8, 6))
plt.bar(mean_f1_scores.keys(), mean_f1_scores.values(), color=['b', 'g', 'r', 'y'])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xlabel('Model')
plt.ylabel('Mean F1 Score')
plt.title('Comparison of Mean F1 Scores for Different Models')
plt.show()

# List of mean ARI scores for each model
mean_ari_scores = {model_name: np.mean(aris[model_name]) for model_name in models.keys()}
mean_ari_scores['MLP'] = mean_mlp_ari

# Plot the ARI scores
plt.figure(figsize=(8, 6))
plt.bar(mean_ari_scores.keys(), mean_ari_scores.values(), color=['b', 'g', 'r', 'y'])
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xlabel('Model')
plt.ylabel('Mean Adjusted Rand Index')
plt.title('Comparison of Mean ARI for Different Models')
plt.show()

# Plot MLP training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss for MLP')
plt.legend()
plt.show()

# Add the MLP model to the models dictionary
models['MLP'] = mlp

# Plot histograms comparing true labels vs predicted labels for each model
for model_name in models.keys():
    plt.figure(figsize=(12, 6))

    # Plot true labels
    plt.subplot(1, 2, 1)
    plt.hist(y_test, bins = np.arange(0, len(np.unique(y)) + 1), alpha=1, label='True Labels', color='darkgreen',
             edgecolor='black')
    plt.xlabel('Wine Quality Class')
    plt.ylabel('Frequency')
    plt.title('True values')
    plt.xticks(np.arange(len(np.unique(y))))
    plt.legend()

    # Plot predicted labels
    plt.subplot(1, 2, 2)
    plt.hist(predictions[model_name], bins = np.arange(0, len(np.unique(y)) + 1), alpha=0.5,
             label='Predicted Labels', color='darkgreen', edgecolor='black')
    plt.xlabel('Wine Quality Class')
    plt.ylabel('Frequency')
    plt.title(f'Predicted values of {model_name}')
    plt.xticks(np.arange(len(np.unique(y))))  # Set the x-axis ticks to be the class labels
    plt.legend()
    plt.show()

for model_name in models.keys():
    plt.figure(figsize=(12, 6))

    # Plot Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, predictions[model_name], labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap='viridis', ax=plt.gca(), colorbar=False)
    plt.title(f'Confusion Matrix of {model_name}')
    plt.grid(False)  # Hide grid to focus on confusion matrix content

    # Plot Misclassification Rates
    plt.subplot(1, 2, 2)
    misclassification_rates = []
    for cls in np.unique(y):
        cls_indices = y_test == cls
        misclassified = np.sum(predictions[model_name][cls_indices] != cls)
        misclassification_rates.append(misclassified / np.sum(cls_indices))

    plt.bar(np.unique(y), misclassification_rates, color='red', edgecolor='black', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Misclassification Rate')
    plt.title(f'Misclassification Rate by Class of {model_name}')
    plt.xticks(np.unique(y))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Show the figure
    plt.tight_layout()
    plt.show()