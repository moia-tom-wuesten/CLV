import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcdefaults, rcParams


def read_df(csv_file_path: str) -> pd.DataFrame:
    """
    Read logs csv file.
    :param csv_file_path: path to csv file with logs.
    :return: DataFrame object.
    """
    df = pd.read_csv(csv_file_path, sep=";", index_col=0)
    df = df.where(df.notna(), "-")
    df = df.where(df != "-", 0)
    return df


def build_time_plot(df: pd.DataFrame, name: str) -> None:
    """
    Build time plot.
    :param df: DataFrame with data.
    :param name: plot filename.
    """
    # rcdefaults()
    # rcParams.update({"font.size": 10})
    print(df.columns)
    df = df.sort_values(["Batch size"])
    plots_data = []
    print(df.head())
    dataset = df.Dataset.unique().tolist()

    for arc in dataset:
        sub_df = df.where(df.Dataset == arc).dropna()
        # tf1_data = sub_df.Time.where(sub_df.Framework == "tf1").dropna().tolist()
        # tf2_data = sub_df.Time_min.where(sub_df.Framework == "tf2").dropna().tolist()
        torch_data = (
            sub_df.Time_min.where(sub_df.Framework == "pytorch1").dropna().tolist()
        )
        torch_data2 = (
            sub_df.Time_min.where(sub_df.Framework == "pytorch2").dropna().tolist()
        )
        print(torch_data)
        arrays = [
            # np.array(tf1_data, dtype=np.float64),
            # np.array(tf2_data, dtype=np.float64),
            np.array(torch_data, dtype=np.float64),
            np.array(torch_data2, dtype=np.float64),
        ]

        plot_dict = {
            "title": arc,
            # "tf1": relative_data[:, 0],
            # "tf2": arrays[0],
            "pytorch1": arrays[0],
            "pytorch2": arrays[1],
        }
        plots_data.append(plot_dict)
        print(plot_dict)
    l1, l2, l3 = None, None, None
    plot_parameters = {"linewidth": 2, "markersize": 7.5}
    fig = plt.figure(figsize=(17, 4), dpi=100)
    for i, plot_dict in enumerate(plots_data):
        plt.subplot(1, len(plots_data), i + 1)
        cur_range = ["32", "64", "128"]

        l1 = plt.plot(
            cur_range,
            plot_dict["pytorch2"],
            color="darkorange",
            marker="o",
            **plot_parameters
        )
        # l2 = plt.plot(
        #     cur_range,
        #     plot_dict["tf2"],
        #     color="firebrick",
        #     marker="^",
        #     **plot_parameters
        # )
        l3 = plt.plot(
            cur_range,
            plot_dict["pytorch1"],
            color="navy",
            marker="D",
            **plot_parameters
        )

        plt.grid(color="lightgray", linestyle="--", linewidth=1)

        plt.title(plot_dict["title"], fontsize=14)
    fig.legend(
        [l1, l3],
        # labels=["TensorFlow 1.15.0", "TensorFlow 2.5.0", "PyTorch 1.9.0"],
        labels=["PyTorch 2.0" "PyTorch 1.12.0"],
        loc="center right",
    )
    plt.ylabel("Time in (s)")
    plt.xlabel("Batch size")

    fig.savefig(name, format="png")
    plt.close(fig)


def build_std_plot(df: pd.DataFrame, name: str):
    print(df.columns)
    df = df.sort_values(["Batch size"])
    plots_data = []
    print(df.head())
    dataset = df.Dataset.unique().tolist()

    for arc in dataset:
        sub_df = df.where(df.Dataset == arc).dropna()
        # tf1_data = sub_df.Time.where(sub_df.Framework == "tf1").dropna().tolist()
        # tf2_data = sub_df.Time_std.where(sub_df.Framework == "tf2").dropna().tolist()
        torch_data = (
            sub_df.Time_std.where(sub_df.Framework == "pytorch1").dropna().tolist()
        )
        torch_data2 = (
            sub_df.Time_std.where(sub_df.Framework == "pytorch2").dropna().tolist()
        )
        print(torch_data)
        arrays = [
            # np.array(tf1_data, dtype=np.float64),
            # np.array(tf2_data, dtype=np.float64),
            np.array(torch_data, dtype=np.float64),
            np.array(torch_data2, dtype=np.float64),
        ]

        plot_dict = {
            "title": arc,
            # "tf1": relative_data[:, 0],
            # "tf2": arrays[0],
            "pytorch1": arrays[0],
            "pytorch2": arrays[1],
        }
        plots_data.append(plot_dict)
        print(plot_dict)
    l1, l2, l3 = None, None, None
    plot_parameters = {"linewidth": 2, "markersize": 7.5}
    fig = plt.figure(figsize=(17, 4), dpi=100)
    for i, plot_dict in enumerate(plots_data):
        plt.subplot(1, len(plots_data), i + 1)
        cur_range = ["32", "64", "128"]

        l1 = plt.plot(
            cur_range,
            plot_dict["pytorch2"],
            color="darkorange",
            marker="o",
            **plot_parameters
        )
        # l2 = plt.plot(
        #     cur_range,
        #     plot_dict["tf2"],
        #     color="firebrick",
        #     marker="^",
        #     **plot_parameters
        # )
        l3 = plt.plot(
            cur_range,
            plot_dict["pytorch1"],
            color="navy",
            marker="D",
            **plot_parameters
        )

        plt.grid(color="lightgray", linestyle="--", linewidth=1)

        plt.title(plot_dict["title"], fontsize=14)
    fig.legend(
        [l1, l3],
        # labels=["TensorFlow 1.15.0", "TensorFlow 2.5.0", "PyTorch 1.9.0"],
        labels=["PyTorch 2.0", "PyTorch 1.12.0"],
        loc="center right",
    )
    plt.ylabel("Time in (s)")
    plt.xlabel("Batch size")

    fig.savefig(name, format="png")
    plt.close(fig)


def build_memory_plot(
    df_train: pd.DataFrame, df_inference: pd.DataFrame, name: str
) -> None:
    """
    Build memory plot for pytorch.
    :param df_train: DataFrame with training data.
    :param df_inference: DataFrame with inferencing data.
    :param name: plot filename.
    """
    rcdefaults()
    rcParams.update({"font.size": 10})

    plots_data = []
    architectures = df_train.Architecture.unique().tolist()

    for arc in architectures:
        sub_df_train = df_train.where(df_train.Architecture == arc).dropna()
        sub_df_inference = df_inference.where(df_inference.Architecture == arc).dropna()

        train_data = (
            sub_df_train.Memory.where(sub_df_train.Framework == "pytorch")
            .dropna()
            .tolist()
        )
        infer_data = (
            sub_df_inference.Memory.where(sub_df_inference.Framework == "pytorch")
            .dropna()
            .tolist()
        )

        train_data = np.array(train_data, dtype=np.float64)
        infer_data = np.array(infer_data, dtype=np.float64)
        train_data = np.where(train_data == 0, np.nan, train_data)
        infer_data = np.where(infer_data == 0, np.nan, infer_data)

        plot_dict = {"title": arc, "train": train_data, "inference": infer_data}
        plots_data.append(plot_dict)

    l1, l2 = None, None
    plot_parameters = {"linewidth": 2, "markersize": 7.5}
    fig = plt.figure(figsize=(35, 4), dpi=100)
    for i, plot_dict in enumerate(plots_data):
        plt.subplot(1, len(plots_data), i + 1)

        cur_range = np.arange(0, plot_dict["train"].shape[0])
        cur_range = np.arange(0, plot_dict["tf1"].shape[0])

        l1 = plt.plot(
            cur_range,
            plot_dict["tf1"],
            color="darkorange",
            marker="o",
            **plot_parameters
        )
        l2 = plt.plot(
            cur_range,
            plot_dict["tf2"],
            color="firebrick",
            marker="^",
            **plot_parameters
        )
        l3 = plt.plot(
            cur_range, plot_dict["pytorch"], color="navy", marker="D", **plot_parameters
        )
        l1 = plt.plot(
            cur_range,
            plot_dict["train"],
            color="firebrick",
            marker="o",
            **plot_parameters
        )
        l2 = plt.plot(
            cur_range,
            plot_dict["inference"],
            color="navy",
            marker="D",
            **plot_parameters
        )

        plt.grid(color="lightgray", linestyle="--", linewidth=1)
        axv_line_parameters = {
            "linewidth": 2,
            "color": "gray",
            "linestyle": (0, (4, 1.2)),
        }
        axv_line_x_coordinates = [-0.5, 3.5, 7.5, 11.5]
        for x in axv_line_x_coordinates:
            plt.axvline(x, **axv_line_parameters)

        plt.title(plot_dict["title"], fontsize=14)
        plt.xlim(cur_range[0] - 0.75, cur_range[-1] + 0.75)
        plt.ylim(0, 5934)
        plt.yticks([1000, 2000, 3000, 4000, 5000])
        if i == 0 or i == 4:
            plt.xticks(
                cur_range,
                [
                    "Batch size: 1                  ",
                    8,
                    "16\nInput shape: (128, 128, 3)                           ",
                    32,
                    1,
                    8,
                    "16\n(224, 224, 3)      ",
                    32,
                    1,
                    8,
                    "16\n(320, 320, 3)      ",
                    32,
                ],
            )
            plt.ylabel("Video memory, Mbytes")
        else:
            plt.xticks(
                cur_range,
                [
                    1,
                    8,
                    "16\n(128, 128, 3)      ",
                    32,
                    1,
                    8,
                    "16\n(224, 224, 3)      ",
                    32,
                    1,
                    8,
                    "16\n(320, 320, 3)      ",
                    32,
                ],
            )

    fig.legend(
        [l1, l2],
        labels=["Training (GPU)", "Inferencing (GPU)"],
        loc="center right",
        borderaxespad=1,
    )

    plt.gcf().subplots_adjust(bottom=0.15, left=0.02, right=0.94)

    # plt.show()
    fig.savefig(name, format="png")
    plt.close(fig)

    # Replace last 3 plots downside.
    img = cv2.imread(name)
    row_1 = img[:, :1890, :]
    row_2 = np.zeros_like(row_1) + 255
    img_part = img[:, 1890:, :]
    left_x = row_2.shape[1] // 2 - img_part.shape[1] // 2
    right_x = row_2.shape[1] // 2 - img_part.shape[1] // 2 + img_part.shape[1]
    row_2[:, left_x:right_x, :] = img_part
    new_img = np.concatenate([row_1, row_2], axis=0)
    cv2.imwrite(name, new_img)


def compare_results(df_1, df2):
    df = df_1.merge(df2, how="left", on=["Framework", "Dataset", "Batch size"])

    df["time_difference"] = df.apply(
        lambda x: ((x["Time_y"] - x["Time_x"]) / x["Time_y"]) * 100, axis=1
    )
    df = df.rename(columns={"Time_x": "Time_GPU", "Time_y": "Time_cpu"})
    print(df)
    df.to_csv("comparison_gpu_cpu.csv", index=False)


if __name__ == "__main__":
    # df_1 = read_df("results_gpu.csv")
    # df_2 = read_df("results2.csv")
    # compare_results(df_1, df_2)
    df_3 = read_df("results_inference_cpu.csv")
    build_time_plot(df_3, os.path.join("8_plot_interference_cpu.png"))
    # build_std_plot(df_3, os.path.join("7_plot_interference_cpu_std.png"))
