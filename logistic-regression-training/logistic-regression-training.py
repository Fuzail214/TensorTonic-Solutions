import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000
) -> tuple[np.ndarray, float]:

    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):

        # Forward propagation
        z = X @ w + b
        p = sigmoid(z)

        # Gradients
        dw = X.T @ (p - y) / n_samples
        db = np.mean(p - y)

        # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b