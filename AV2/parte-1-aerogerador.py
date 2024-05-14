import numpy as np
import matplotlib.pyplot as plt


def fit(X, y, learn_rate=0.5, iterations=10):
    weights = np.zeros(1 + X.shape[1])
    errors = []
    weights_list = []

    for _ in range(iterations):
        errors_count = 0
        for xi, target in zip(X, y):
            update = learn_rate * (target - predict(xi, weights))
            weights[1:] += update * xi
            weights[0] += update
            errors_count += int(update != 0.0)
        errors.append(errors_count)
    weights_list.append(weights.copy())
    return weights, errors, weights_list


def net_input(X, weights):
    return np.dot(X, weights[1:]) + weights[0]


def predict(X, weights):
    return np.where(net_input(X, weights) >= 0.0, 1, 0)


# Load the data from the ".dat" file
data = np.genfromtxt("aerogerador (3).dat")

# Separate the data into features (X) and class labels (y)
X = data[:, 0].reshape(-1, 1)  # Feature
y = data[:, 1]  # Class label

# Plot the data
plt.scatter(X, y)
plt.xlabel("Feature")
plt.ylabel("Class")
plt.title("Data")
plt.show()

# Define the number of repetitions
num_repetitions = 100
accuracies_perceptron = []
weights_combined_perceptron = []

for _ in range(num_repetitions):
    # Shuffle and split the data into training and test sets
    indices = np.random.permutation(len(X))
    X_shuffle = X[indices]
    y_shuffle = y[indices]
    split_index = int(0.7 * len(X))
    X_train, X_test = X_shuffle[:split_index], X_shuffle[split_index:]
    y_train, y_test = y_shuffle[:split_index], y_shuffle[split_index:]

    # Train the Perceptron model
    weights, errors, weights_list = fit(
        X_train, y_train, learn_rate=0.01, iterations=50
    )

    # Calculate the accuracy of the Perceptron
    y_pred_perceptron = predict(X_test, weights)
    accuracy_perceptron = np.mean(y_pred_perceptron == y_test)
    accuracies_perceptron.append(accuracy_perceptron)

    # Store the Perceptron weights
    weights_combined_perceptron.append(weights_list[0])

# Calculate accuracy statistics for the Perceptron
mean_accuracy_perceptron = np.mean(accuracies_perceptron)
min_accuracy_perceptron = np.min(accuracies_perceptron)
max_accuracy_perceptron = np.max(accuracies_perceptron)
median_accuracy_perceptron = np.median(accuracies_perceptron)
std_accuracy_perceptron = np.std(accuracies_perceptron)

# Display the statistics
print("\nPerceptron Statistics:")
print("Accuracy - Mean:", mean_accuracy_perceptron)
print("Accuracy - Min:", min_accuracy_perceptron)
print("Accuracy - Max:", max_accuracy_perceptron)
print("Accuracy - Median:", median_accuracy_perceptron)
print("Accuracy - Std Dev:", std_accuracy_perceptron)

# Display the combined Perceptron weights
weights_combined_perceptron = np.array(weights_combined_perceptron)
print("\nCombined Perceptron Weights:")
print(weights_combined_perceptron)

# Plot the accuracy per round for the Perceptron
plt.bar(range(num_repetitions), accuracies_perceptron)
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Accuracy per Round - Perceptron")
plt.show()

# Plot the decision boundary of the Perceptron
plt.scatter(X_test, y_test, color="black", label="Test Data")
for weights in weights_combined_perceptron:
    slope = -weights[1] / weights[0]  # Slope is the only weight we have
    intercept = -weights[0] / weights[1]
    x_decision = np.linspace(X_test.min(), X_test.max(), 100)
    y_decision = slope * x_decision + intercept
    plt.plot(x_decision, y_decision, linestyle="--", color="green", linewidth=0.5)
plt.xlabel("Feature")
plt.ylabel("Class")
plt.title("Decision Boundary - Perceptron")
plt.legend()
plt.show()
