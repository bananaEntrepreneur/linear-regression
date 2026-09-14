import numpy as np


class LinearRegression:
    """y = k * x + b"""
    def __init__(self, learning_ratio: float = 0.1, n_iters: int = 1000) -> None:
        self.lr: float = learning_ratio
        self.n: int = n_iters
        self.k: float = 0.0
        self.b: float = 0.0
        self.loss_history: list[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.k * X + self.b

    def _mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return (1/self.n) * sum((y_true - y_pred) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
