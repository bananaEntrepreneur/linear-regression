class LinearRegression:
    """y = k * x + b"""

    def __init__(self, learning_rate: float = 0.1, n_iters: int = 50_000) -> None:
        self.lr: float = learning_rate
        self.n_iters: int = n_iters
        self.k: float = 0.0
        self.b: float = 0.0
        self.loss_history: list[float] = []
        self.k_history: list[float] = []
        self.b_history: list[float] = []

    def predict(self, x: float) -> float:
        return self.k * x + self.b

    @staticmethod
    def mse(y_true: list[float], y_pred: list[float]) -> float:
        n = len(y_true)
        total_error = 0.0
        for i in range(n):
            total_error += (y_true[i] - y_pred[i]) ** 2
        return total_error / n

    @staticmethod
    def mae(y_true: list[float], y_pred: list[float]) -> float:
        n = len(y_true)
        total_error = 0.0
        for i in range(n):
            total_error += abs(y_true[i] - y_pred[i])
        return total_error / n

    def predict_all(self, X: list[float]) -> list[float]:
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.predict(X[i]))
        return y_pred

    def _gradients(
        self, X: list[float], y_true: list[float], y_pred: list[float]
    ) -> tuple[float, float]:
        n = len(X)
        dk_sum = 0.0
        db_sum = 0.0
        for i in range(n):
            error = y_pred[i] - y_true[i]
            dk_sum += X[i] * error
            db_sum += error
        dk = (2 / n) * dk_sum
        db = (2 / n) * db_sum
        return dk, db

    def fit(self, X: list[float], y: list[float]) -> None:
        for _ in range(self.n_iters):
            y_pred = self.predict_all(X)

            self.loss_history.append(self.mse(y, y_pred))
            self.k_history.append(self.k)
            self.b_history.append(self.b)

            dk, db = self._gradients(X, y, y_pred)

            self.k -= self.lr * dk
            self.b -= self.lr * db

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(f"{self.k}\n{self.b}\n")

    def load(self, path: str) -> None:
        with open(path) as f:
            self.k = float(f.readline())
            self.b = float(f.readline())
