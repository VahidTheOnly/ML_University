import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Define file path for data
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'q1_dataset.csv')

# Load dataset
df = pd.read_csv(file_path, index_col='Subject')

# Prepare the input (X) and output (y) as column vectors
X = np.reshape(df['Weight'].values, (-1, 1))
y = np.reshape(df['Systolic BP'].values, (-1, 1))

X_norm = (X-np.mean(X))/np.std(X)
y_norm = (y-np.mean(y))/np.std(y)

# Initialize weights and bias
w = np.random.randn(1,1)
b = np.random.randn(1,1)

def forward_prop(X, w, b):
    y_pred = X @ w + b
    return y_pred

# Define the cost function
def cost_func(y, y_pred):
    cost = (1 / 2) * np.mean((y_pred - y) ** 2)
    return cost

# Gradient descent update function
def gradient_descent(w, b, djdw, djdb, learning_rate):
    w -= learning_rate * djdw
    b -= learning_rate * djdb
    return w, b

# Model function to compute predictions, cost, and gradients
def model(w, b, X, y, learning_rate):
    # Forward pass: compute predictions
    y_pred = forward_prop(X, w, b)
    cost = cost_func(y, y_pred)
    
    # Compute gradients
    m = len(y)
    djdw = (1 / m) * (X.T @ (y_pred - y))  # Gradient w.r.t w
    djdb = np.mean(y_pred - y)              # Gradient w.r.t b
    
    # Update weights and bias
    w, b = gradient_descent(w, b, djdw, djdb, learning_rate)
    return w, b, cost

def predict(X_norm, y, w, b):
    y_pred_norm = forward_prop(X_norm, w, b)
    y_pred = y_pred_norm * np.std(y) + np.mean(y)
    return y_pred

# Training loop
epochs = 15
learning_rate = 0.5

for epoch in range(epochs):
    w, b, cost = model(w, b, X_norm, y_norm, learning_rate)
    plt.cla()
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, predict(X_norm, y, w, b), label='Regression Line')
    plt.pause(0.5)
    plt.draw()
    print(f"Epoch {epoch+1}, Cost: {cost}")

print("Final normalized weights and bias:", w, b)

# De-normalize the weights and bias
w_original = w * (np.std(y) / np.std(X))
b_original = b * np.std(y) + np.mean(y) - w_original * np.mean(X)

print("Final weights and bias for original data (unnormalized):", w_original, b_original)

plt.show()
