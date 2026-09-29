from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linear_regression import LinearRegression

TRAIN_PATH = "data/linear_multi_variant_3_train.csv"
WEIGHTS_PATH = "weights_multi.npz"


def load_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X = df.drop(columns=["y"]).to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return X, y


def plot_loss(loss_history: list[float]) -> None:
    epochs = np.arange(1, len(loss_history) + 1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, loss_history)
    plt.yscale("log")
    plt.title("MSE от эпохи обучения")
    plt.xlabel("эпоха")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_regression(X: np.ndarray, y: np.ndarray, model: LinearRegression) -> None:
    n_features = X.shape[1]
    features = [f"x{i + 1}" for i in range(n_features)]
    mean = X.mean(axis=0)
    pairs = list(combinations(range(n_features), 2))

    fig = plt.figure(figsize=(6 * len(pairs), 5.5))
    for slot, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(1, len(pairs), slot + 1, projection="3d")
        held = next(c for c in range(n_features) if c not in (i, j))

        axis_i = np.linspace(X[:, i].min(), X[:, i].max(), 40)
        axis_j = np.linspace(X[:, j].min(), X[:, j].max(), 40)
        grid_i, grid_j = np.meshgrid(axis_i, axis_j)
        grid = np.tile(mean, grid_i.shape + (1,))
        grid[..., i] = grid_i
        grid[..., j] = grid_j

        ax.scatter(X[:, i], X[:, j], y, c="tab:blue", alpha=0.35, s=12)
        ax.plot_surface(
            grid_i,
            grid_j,
            model.predict(grid),
            color="tab:orange",
            alpha=0.55,
            edgecolor="none",
        )

        ax.set_xlabel(features[i])
        ax.set_ylabel(features[j])
        ax.set_zlabel("y")
        ax.set_title(
            f"{features[i]} / {features[j]}, {features[held]} = {mean[held]:.0f}"
        )

    fig.tight_layout()
    plt.show()


def main() -> None:
    X_train, y_train = load_xy(TRAIN_PATH)

    model = LinearRegression(learning_rate=1e-7, n_epochs=300, batch_size=16)
    model.fit(X_train, y_train)
    model.save(WEIGHTS_PATH)

    print("w =", model.w)
    print("b =", model.b)

    plot_loss(model.loss_history)
    plot_regression(X_train, y_train, model)


if __name__ == "__main__":
    main()
