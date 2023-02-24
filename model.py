import numpy as np
import pickle
import statsmodels.api as sm

# Generate random values for X and y
n_samples = 100
n_features = 3
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

# Create and fit the model
model = sm.OLS(y, X).fit()

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the saved model from the file
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)
