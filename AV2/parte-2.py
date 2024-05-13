import numpy as np
import matplotlib.pyplot as plt


def fit_perceptron(X, y, learn_rate=0.01, iterations=50):
    weights = np.zeros(1 + X.shape[1])
    errors = []

    for _ in range(iterations):
        errors_count = 0
        for xi, target in zip(X, y):
            update = learn_rate * (target - predict_perceptron(xi, weights))
            weights[1:] += update * xi
            weights[0] += update
            errors_count += int(update != 0.0)
        errors.append(errors_count)
    return weights, errors


def net_input(X, weights):
    return np.dot(X, weights[1:]) + weights[0]


def predict_perceptron(X, weights):
    return np.where(net_input(X, weights) >= 0.0, 1, -1)


def fit_adaline(X, y, learn_rate=0.01, iterations=50):
    weights = np.zeros(1 + X.shape[1])
    errors = []

    for _ in range(iterations):
        errors_count = 0
        for xi, target in zip(X, y):
            update = learn_rate * (target - activation(xi, weights))
            weights[1:] += update * xi
            weights[0] += update
            errors_count += int(update != 0.0)
        errors.append(errors_count)
    return weights, errors


def activation(X, weights):
    return net_input(X, weights)


def predict_adaline(X, weights):
    return np.where(activation(X, weights) >= 0.0, 1, -1)


# Load the data from the URL
glass_data = np.genfromtxt(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
    delimiter=",",
)
# Selecting features 1 and 2
X = glass_data[:, [0, 1]]
# Selecting labels
y = glass_data[:, -1]
# Keeping only the first two classes (1 and 2)
X = X[y <= 2]
y = y[y <= 2]
# Converting labels into binary (-1 and 1)
y = np.where(y == 1, -1, 1)

plt.scatter(
    X[y == -1][:, 0], X[y == -1][:, 1], label="Glass Type 1", marker="o", color="red"
)
plt.scatter(
    X[y == 1][:, 0], X[y == 1][:, 1], label="Glass Type 2", marker="x", color="blue"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Glass Data")
plt.legend()
plt.show()

weights_perceptron, errors_perceptron = fit_perceptron(
    X, y, learn_rate=0.01, iterations=50
)
weights_adaline, errors_adaline = fit_adaline(X, y, learn_rate=0.01, iterations=50)

# Plotting number of misclassifications vs. epochs for Perceptron
plt.plot(
    range(1, len(errors_perceptron) + 1),
    errors_perceptron,
    marker="o",
    label="Perceptron",
)
# Plotting number of misclassifications vs. epochs for Adaline
plt.plot(range(1, len(errors_adaline) + 1), errors_adaline, marker="x", label="Adaline")
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")
plt.title("Perceptron vs Adaline Learning")
plt.legend()
plt.show()


def plot_decision_regions(X, y, predict_func, weights, resolution=0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = colors[: len(np.unique(y))]

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = predict_func(np.array([xx1.ravel(), xx2.ravel()]).T, weights)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, colors=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=cmap[idx],
            marker=markers[idx],
            label=cl,
        )


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plot_decision_regions(X, y, predict_func=predict_perceptron, weights=weights_perceptron)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Regions - Perceptron")

plt.subplot(1, 2, 2)
plot_decision_regions(X, y, predict_func=predict_adaline, weights=weights_adaline)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Regions - Adaline")

plt.tight_layout()
plt.show()
