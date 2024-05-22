import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Load the data from a file
file_path = "C:/Users/andrii/Desktop/MIWczwiczenia/miw04/Dane/dane3.txt"
data = np.loadtxt(file_path)
X = data[:, 0].reshape(-1, 1)  # Reshape for sklearn, which requires a 2D array for features
y = data[:, 1]

# Step 2: Plot the original data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='All Data')
plt.title('All Data Points')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

# Step 3: Split the data into training and testing sets
X_train, X_test, output_train, output_test = train_test_split(X, y, test_size=0.3, random_state=1)
plt.subplot(1, 2, 2)
plt.scatter(X_train, output_train, color='blue', label='Training Data')
plt.scatter(X_test, output_test, color='red', label='Testing Data')
plt.title('Train/Test Split')
plt.xlabel('Input')
plt.legend()
plt.tight_layout()
plt.show()

# Model 1: Linear Regression (y = ax + b)
# design_matrix_linear - (design matrix)
# design_matrix_linear.T - transponowana macierz projektowania
# design_matrix_linear.T @ design_matrix_linear - iloczyn macierzy transponowanej przez macierz projektowania
# @ - macierzowe mnozenia
design_matrix_linear = np.hstack([X_train, np.ones(X_train.shape)])  # Design matrix for linear regression
params_linear = np.linalg.inv(design_matrix_linear.T @ design_matrix_linear) @ design_matrix_linear.T @ output_train # Parameters using the least squares
predictions_train_linear = design_matrix_linear @ params_linear
predictions_test_linear = np.hstack([X_test, np.ones(X_test.shape)]) @ params_linear

# Calculate MSE(Mean Squared Error) for Model 1
train_error_linear = np.mean((output_train - predictions_train_linear) ** 2)
test_error_linear = np.mean((output_test - predictions_test_linear) ** 2)
print("Linear Model - Training Error:", train_error_linear)
print("Linear Model - Testing Error:", test_error_linear)

# Model 2: Quadratic Regression (y = ax^2 + bx + c)
design_matrix_quad = np.hstack([X_train ** 2, X_train, np.ones(X_train.shape)])  # Design matrix for quadratic regression
params_quad = np.linalg.pinv(design_matrix_quad) @ output_train # Parameters using pseudo-inverse
predictions_train_quad = design_matrix_quad @ params_quad
predictions_test_quad = np.hstack([X_test ** 2, X_test, np.ones(X_test.shape)]) @ params_quad

# Calculate MSE(Mean Squared Error) for Model 2
train_error_quad = np.mean((output_train - predictions_train_quad) ** 2)
test_error_quad = np.mean((output_test - predictions_test_quad) ** 2)
print("Quadratic Model - Training Error:", train_error_quad)
print("Quadratic Model - Testing Error:", test_error_quad)

# Correcting the matrix dimension issue for plotting
X_sorted = np.sort(X, axis=0)
X_design = np.hstack([X_sorted, np.ones(X_sorted.shape)])  # Adding a column of ones

# Plotting both models with the data
plt.figure(figsize=(10, 5))
plt.scatter(X_train, output_train, color='blue', label='Training Data')
plt.scatter(X_test, output_test, color='red', label='Testing Data')
plt.plot(X_sorted, X_design @ params_linear, label='Linear Model', color='orange')
plt.plot(X_sorted, np.hstack([X_sorted**2, X_sorted, np.ones(X_sorted.shape)]) @ params_quad, label='Quadratic Model', color='green')
plt.title('Model Comparisons')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
