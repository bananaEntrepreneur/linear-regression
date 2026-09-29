import numpy as np


class LinearRegression:
    """y = X @ w + b"""

    def __init__(
        self, learning_rate: float = 0.1, n_epochs: int = 300, batch_size: int = 16
    ) -> None:
        self.lr: float = learning_rate
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.w + self.b

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        rng = np.random.default_rng()

        for _ in range(self.n_epochs):
            idx = rng.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            for start in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[start : start + self.batch_size]
                y_batch = y_shuffled[start : start + self.batch_size]
                m = len(y_batch)

                y_pred = X_batch @ self.w + self.b
                error = y_pred - y_batch

                dw = (2 / m) * (X_batch.T @ error)
                db = (2 / m) * np.sum(error)

                self.w -= self.lr * dw
                self.b -= self.lr * db

            self.loss_history.append(self.mse(y, X @ self.w + self.b))

    def save(self, path: str) -> None:
        np.savez(path, w=self.w, b=self.b)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.w = data["w"]
        self.b = float(data["b"])
