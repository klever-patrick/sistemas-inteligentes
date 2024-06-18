import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class Perceptron:
    def __init__(self, learn_rate=0.5, iterations=10):
        self.learn_rate = learn_rate
        self.iterations = iterations
        self.errors = []
        self.weights = None

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        for _ in range(self.iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learn_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class Adaline:
    def __init__(self, learn_rate=0.01, iterations=50):
        self.learn_rate = learn_rate
        self.iterations = iterations
        self.errors = []
        self.weights = None

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        for _ in range(self.iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learn_rate * (target - self.activation(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


# Load the dataset
ozone_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.data",
    header=None,
    na_values="?",
)
ozone_data = ozone_data.dropna()

# Selecting features and labels
# Note: Assuming column 0 is the class label and columns 1 and 2 are the features we want to use
X = ozone_data.iloc[:, [1, 2]].values
y = ozone_data.iloc[:, 0].values

# Convert labels to binary (-1 and 1)
y = np.where(y == 1, -1, 1)

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Visualize the dataset
plt.scatter(
    X_std[y == -1][:, 0],
    X_std[y == -1][:, 1],
    label="Class -1",
    marker="o",
    color="red",
)
plt.scatter(
    X_std[y == 1][:, 0], X_std[y == 1][:, 1], label="Class 1", marker="x", color="blue"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Ozone Level Detection Data")
plt.legend()
plt.show()

# Train Perceptron
perceptron = Perceptron(learn_rate=0.01, iterations=50)
perceptron.fit(X_std, y)

# Train Adaline
adaline = Adaline(learn_rate=0.01, iterations=50)
adaline.fit(X_std, y)

# Plotting number of misclassifications vs. epochs for Perceptron
plt.plot(
    range(1, len(perceptron.errors) + 1),
    perceptron.errors,
    marker="o",
    label="Perceptron",
)
# Plotting number of misclassifications vs. epochs for Adaline
plt.plot(range(1, len(adaline.errors) + 1), adaline.errors, marker="x", label="Adaline")
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")
plt.title("Perceptron vs Adaline Learning")
plt.legend()
plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = colors[: len(np.unique(y))]
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
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
plot_decision_regions(X_std, y, classifier=perceptron)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Regions - Perceptron")

plt.subplot(1, 2, 2)
plot_decision_regions(X_std, y, classifier=adaline)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Regions - Adaline")

plt.tight_layout()
plt.show()
