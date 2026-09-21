import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

TRAIN_PATH = "data/linear-regression-one-var-train.csv"
VAL_PATH = "data/linear-regression-one-var-val.csv"
WEIGHTS_PATH = "weights.txt"


def load_xy(path: str) -> tuple[list[float], list[float]]:
    df = pd.read_csv(path)
    return df["x"].tolist(), df["y"].tolist()


def plot_loss_surface(
    model: LinearRegression, x: list[float], y: list[float], resolution: int = 60
) -> None:
    k_span = max(1.0, abs(model.k) * 2)
    b_span = max(50.0, abs(model.b) * 2)

    k_vals = []
    for i in range(resolution):
        k_vals.append(-k_span + 2 * k_span * i / (resolution - 1))

    b_vals = []
    for i in range(resolution):
        b_vals.append(-b_span + 2 * b_span * i / (resolution - 1))

    z = []
    for b in b_vals:
        row = []
        for k in k_vals:
            y_pred = []
            for xi in x:
                y_pred.append(k * xi + b)
            row.append(LinearRegression.mse(y, y_pred))
        z.append(row)

    plt.figure()
    cs = plt.contour(k_vals, b_vals, z, levels=30)
    plt.clabel(cs, inline=True, fontsize=6)
    plt.plot(
        model.k_history,
        model.b_history,
        color="red",
        linewidth=1,
        label="путь градиентного спуска",
    )
    plt.scatter([model.k], [model.b], color="red", zorder=5)
    plt.title("Изолинии MSE(k, b)")
    plt.xlabel("k")
    plt.ylabel("b")
    plt.legend()


def main() -> None:
    x_train, y_train = load_xy(TRAIN_PATH)
    x_val, y_val = load_xy(VAL_PATH)

    plt.figure()
    plt.scatter(x_train, y_train)
    plt.title("Исходные данные (обучающая выборка)")
    plt.xlabel("x")
    plt.ylabel("y")

    model = LinearRegression(learning_rate=1e-6, n_iters=50_000)
    model.fit(x_train, y_train)
    model.save(WEIGHTS_PATH)

    plt.figure()
    plt.plot(range(1, len(model.loss_history) + 1), model.loss_history)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("MSE от эпохи обучения")
    plt.xlabel("эпоха (лог. шкала)")
    plt.ylabel("MSE (лог. шкала)")

    plot_loss_surface(model, x_train, y_train)

    y_pred_val = []
    for xi in x_val:
        y_pred_val.append(model.predict(xi))

    line_x = [min(x_val), max(x_val)]
    line_y = []
    for xi in line_x:
        line_y.append(model.predict(xi))

    plt.figure()
    plt.scatter(x_val, y_val, label="данные")
    plt.plot(line_x, line_y, color="red", label=f"y = {model.k:.4f}x + {model.b:.4f}")
    plt.title("Линейная регрессия на валидационных данных")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    print(f"k = {model.k:.6f}")
    print(f"b = {model.b:.6f}")
    print(f"MSE (val) = {LinearRegression.mse(y_val, y_pred_val):.4f}")
    print(f"MAE (val) = {LinearRegression.mae(y_val, y_pred_val):.4f}")

    plt.show()


if __name__ == "__main__":
    main()
