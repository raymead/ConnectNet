import matplotlib.pyplot as plt
import pandas


def print_board(board_tensor, title: str = ""):
    plt.figure()
    df = pandas.DataFrame(board_tensor.reshape(6, 7).numpy()) \
        .stack().reset_index() \
        .rename(columns={"level_0": "y", "level_1": "x"}).astype(int)
    df["y"] = 5 - df["y"]
    df["color"] = df[0].map({0: "#FFFFFF", 1: "#FF0000", -1: "#000000"})
    df.plot(kind="scatter", x="x", y="y", c="color", s=200)
    plt.title(title)
