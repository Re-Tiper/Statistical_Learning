import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import kstest, shapiro, norm # for Normality test

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

"""Import dataset"""
df = pd.read_csv('winequality-white.csv', delimiter=';')

# Separate features and target
X = df.drop(columns=['quality'])  # Exclude 'quality'
y = df['quality']                 # Target variable

# Normality Test
for feature in X.columns:
    data = X[feature]
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Use sample standard deviation (ddof=1)

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = kstest(data, 'norm', args=(mean, std))
    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = shapiro(data)

    print(f"Feature: {feature}")
    print(f"  Distribution tested: Normal (μ = {mean:.4f}, σ = {std:.4f})")
    print(f"  KS Statistic: {ks_stat}")
    print(f"  P-Value: {ks_p}")
    if ks_p < 0.05:
        print("  Result: Not normally distributed (reject null hypothesis)")
    else:
        print("  Result: Normally distributed (fail to reject null hypothesis)")
    print("-" * 50)

# Graphs
for feature in df.columns:
# Histogram: Examine whether the data appears roughly bell-shaped.
    sns.histplot(df[feature], kde=True)
    plt.title(f"Histogram of {feature}")
    plt.show()
# Q-Q Plot: Points should approximately lie on the diagonal line for a normal distribution.
    stats.probplot(df[feature], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {feature}")
    plt.show()

"""Anomaly detection with SVM"""
# We use one class SVM since the normality tests indicate that the features do not follow normal distribution.

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the One-Class SVM
svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')  # nu=0.05 means 5% anomalies expected
svm.fit(X_scaled)

# Predict anomalies
predictions = svm.predict(X_scaled)

# -1 indicates an anomaly, 1 indicates normal
normal_points = X_scaled[predictions == 1]
anomalies = X_scaled[predictions == -1]

# Show anomalies
anomalies_df = df.iloc[predictions == -1]  # Extract the rows where anomalies are detected
print("Anomalies detected:")
print(anomalies_df)

# Add anomaly labels to the DataFrame
df['anomaly'] = predictions
# Update column names to include quotes
df.columns = [f'"{col}"' for col in df.columns]
# Save the DataFrame with the anomaly index to a CSV file
df.to_csv('winequality_anomalies.csv', index=False, header=True, sep=";", quotechar='"', quoting=csv.QUOTE_NONE)

# Plot pairwise scatterplots, coloring anomalies (see figure Anomalies)
sns.pairplot(X, hue='anomaly', palette={1: 'blue', -1: 'red'}, diag_kind='kde') # kde (Kernel Density Estimate) plots the probability density function of each feature,
plt.suptitle("Pairwise Plot with Anomalies Highlighted", y=1.02)
plt.show()
