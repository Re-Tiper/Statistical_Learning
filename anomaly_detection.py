import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# %% Creae synthetic data

np.random.seed(42)
X_normal = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 200) # normal data
X_anomaly = np.random.uniform(low=0, high=10, size=(10, 2))                 # anomaly data   
X = np.vstack([X_normal, X_anomaly])                                        # all together 

# %% Calculate parameter and control with e

mu = np.mean(X, axis=0) # mean value
sigma = np.cov(X.T)     # covariance 
p = multivariate_normal(mean=mu, cov=sigma).pdf(X) # pdf

epsilon = 0.001 # threshold  
anomalies = X[p < epsilon] # detect anomaly data

# %% visualize

plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], label='Normal data', c='blue', alpha=0.6)
plt.scatter(anomalies[:, 0], anomalies[:, 1], label='Anomaly data', c='red', edgecolor='k', s=100)

# plot circle of normal distribution
x, y = np.meshgrid(np.linspace(-3 + np.min(X[:,0]), 3 + np.max(X[:,0]), 100), 
                   np.linspace(-3 + np.min(X[:,1]), 3 + np.max(X[:,1]), 100))
pos = np.dstack((x, y))
z = multivariate_normal(mean=mu, cov=sigma).pdf(pos)
plt.contour(x, y, z, levels=np.logspace(-3, 0, 10), cmap='viridis')

plt.title('Anonaly Detection')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend()
plt.grid()
plt.show()
