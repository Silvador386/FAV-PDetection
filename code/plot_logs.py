import json
import os
import matplotlib.pyplot as plt


def load_logs(log_file_path):
    """
    Processes data from log.
    Args:
        log_file_path (str): A path of the json log file.
    Returns:
        train_logs (list): A list of dictionaries with train_logs data.
        val_logs (list): A list of dictionaries with validation data.
    """
    train_logs, val_logs = [], []
    with open(log_file_path, "r") as json_logs:
        for json_line in json_logs:
            log_line = json.loads(json_line.strip())
            if log_line["mode"] == "val_logs":
                val_logs.append(log_line)
            else:
                train_logs.append(log_line)

    return train_logs, val_logs


def create_log_plot(train_logs, val_logs, title, out):
    """
    Plots the data, contains two graphs, first for train loss, second for mAP.
    Args:
        train_logs (list): A list of dictionaries of training data.
        val_logs (list): A list of dictionaries of validation data.
        title (str): A title of the plot.
        out (str): A path where the plot should be stored, may be None.

    """
    num_iter = max(train_log["iter"] for train_log in train_logs)
    num_epochs = max(train_log["epoch"] for train_log in train_logs)
    lr = train_logs[-1]["lr"]

    x_train_loss, y_train_loss = [], []
    for log_dict in train_logs:
        x_train_loss.append(log_dict["iter"] + log_dict["epoch"] * num_iter)
        y_train_loss.append(log_dict["loss"])

    picked_val_names = ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_m", "bbox_mAP_l"]
    x_validation = []
    y_validation_values = {name: [] for name in picked_val_names}
    for log_dict in val_logs:
        x_validation.append(log_dict["epoch"])
        for val_name in picked_val_names:
            y_validation_values[val_name].append(log_dict[val_name])

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    fig.suptitle(f"{title}", fontsize=16)

    axs[0].plot(x_train_loss, y_train_loss, marker="o")
    axs[0].set(xlabel="Iterations")
    axs[0].set(ylabel="Train loss")

    for key in y_validation_values.keys():
        if key == "bbox_mAP":
            axs[1].plot(x_validation, y_validation_values[key], marker="o", linewidth=5)
            continue
        axs[1].plot(x_validation, y_validation_values[key], marker="o")
    axs[1].legend(y_validation_values.keys(), loc="lower right")
    axs[1].set(ylabel="mAP")
    axs[1].set(xlabel="Epochs")

    text = "\n".join((f"Iter/Epoch: {num_iter}",
                      f"Learning r: {lr:.2e}"))
    props = dict(boxstyle='round', facecolor='w', alpha=0.8)
    axs[0].text(0.77, 2.16, text, transform=axs[1].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

    for ax in axs:
        ax.grid()

    if out is None:
        plt.show()

    else:
        print(f"Saving plot to: {out}")
        plt.savefig(out)
    plt.close()


def plot_log(log_file_path, title=None, out_file=None):
    """
    Plots data from the log of the mmdetection format.
    Args:
        log_file_path (str): A name of the log file with data.
        title (str): A title of the plot.
        out_file (str): A path of the plotted data (must end with .png etc.).

    """

    train_logs, val_logs = load_logs(log_file_path)
    if title is None:
        title = log_file_path.split("/")[-1].removesuffix(".json")
    create_log_plot(train_logs, val_logs, title, out_file)


def plot_all_logs_in_dir(work_dir_path, recursive=False):
    """
    Goes trough the whole dir and plots data of every located log file in mmdetection format.

    Args:
        work_dir_path (str): A path of the directory.
        recursive (bool): If any located subdirectory should be searched for logs as well.
    """
    for dir_path, dir_names, file_names in os.walk(work_dir_path):
        for file_name in file_names:
            if file_name.endswith(".json"):
                log_file = dir_path + "/" + file_name
                # Select better title
                if file_name == "None.log.json":
                    title = dir_path.split("\\")[-1]
                else:
                    title = file_name.removesuffix(".json")
                out_file = log_file.removesuffix(".json") + ".png"
                plot_log(log_file, title, out_file)
        if not recursive:
            break
