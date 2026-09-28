import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

TRAIN_PATH = "data/linear_multi_variant_3_train.csv"
WEIGHTS_PATH = "weights_multi.npz"


def load_xy(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["y"]).to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    print(type(X))
    print(type(y))
    return X, y


def main() -> None:
    X_train, y_train = load_xy(TRAIN_PATH)

    model = LinearRegression(learning_rate=1e-7, n_epochs=300, batch_size=16)
    model.fit(X_train, y_train)
    model.save(WEIGHTS_PATH)

    plt.figure()
    plt.plot(range(1, len(model.loss_history) + 1), model.loss_history)
    plt.title("MSE от эпохи обучения")
    plt.xlabel("эпоха")
    plt.ylabel("MSE")

    print("w =", model.w)
    print("b =", model.b)

    plt.show()


if __name__ == "__main__":
    main()
