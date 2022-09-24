import json
import os
import matplotlib.pyplot as plt


def log_types(json_log_path):
    types_present = set()
    with open(json_log_path, "r") as json_logs:
        for json_log in json_logs:
            log = json.loads(json_log.strip())

            if "train" in log.values():
                types_present.add("train_loss")

            if "val" in log.values():
                if "loss" in log.keys():
                    types_present.add("val_loss")
                    continue
                else:
                    types_present.add("val_acc")

    return types_present


def load_logs(json_log_path, types):

    loaded_logs = {type_name: [] for type_name in types}

    with open(json_log_path, "r") as json_logs:
        for json_log in json_logs:
            log = json.loads(json_log.strip())

            if "train" in log.values():
                loaded_logs["train_loss"].append(log)
                continue

            if "val" in log.values():
                if "loss" in log.keys():
                    loaded_logs["val_loss"].append(log)
                else:
                    loaded_logs["val_acc"].append(log)

    return loaded_logs


def select_title(json_log_path):
    log_name = json_log_path.split("/")[-1]
    title = log_name.split(".")[0]
    return title


def plot_loss(loss_logs, type_name, axs, row_idx):
    max_iter = max(loss_log["iter"] for loss_log in loss_logs)
    x_train_loss, y_train_loss = [], []
    for log_dict in loss_logs:
        x_train_loss.append(log_dict["iter"] + log_dict["epoch"] * max_iter)
        y_train_loss.append(log_dict["loss"])

    axs[row_idx].plot(x_train_loss, y_train_loss, marker="o")
    axs[row_idx].set(xlabel="Iterations")
    axs[row_idx].set(ylabel=f"{type_name}")

    text = "\n".join((f"Iter/Epoch: {max_iter}",))

    props = dict(boxstyle='round', facecolor='w', alpha=0.8)
    axs[0].text(0.77, 2.16, text, transform=axs[1].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)


def plot_acc(acc_logs, type_name, axs, row_idx):
    SELECTED_VALUES = ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_m", "bbox_mAP_l"]

    x_val = []
    y_vals = {name: [] for name in SELECTED_VALUES}
    for log in acc_logs:
        x_val.append(log["epoch"])
        for val_name in SELECTED_VALUES:
            y_vals[val_name].append(log[val_name])

    for key in y_vals.keys():
        if key == "bbox_mAP":
            axs[row_idx].plot(x_val, y_vals[key], marker="o", linewidth=5)
            continue
        axs[row_idx].plot(x_val, y_vals[key], marker="o")
    axs[row_idx].legend(y_vals.keys(), loc="lower right")
    axs[row_idx].set(ylabel=f"mAP/{type_name}")
    axs[row_idx].set(xlabel="Epochs")


def plot_log_save(json_log_path, output_path=None):
    type_names = log_types(json_log_path)
    loaded_logs = load_logs(json_log_path, type_names)

    num_types = len(type_names)
    fig, axs = plt.subplots(nrows=num_types, ncols=1, figsize=(12, 8))
    fig.suptitle(f"{select_title(json_log_path)}", fontsize=16)

    row_idx = 0
    for type_name in type_names:
        if "loss" in type_name:
            plot_loss(loaded_logs[type_name], type_name, axs, row_idx)
            row_idx += 1

        if "acc" in type_name:
            plot_acc(loaded_logs[type_name], type_name, axs, row_idx)

    for ax in axs:
        ax.grid()

    if output_path is None:
        output_path = f"{json_log_path.removesuffix('.json')}.png"

    plt.show()
    print(f"Saving plot to: {output_path}")
    plt.savefig(output_path)
    plt.close()


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

    SELECTED_VALUES = ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_m", "bbox_mAP_l"]
    x_validation = []
    y_validation_values = {name: [] for name in SELECTED_VALUES}
    for log_dict in val_logs:
        x_validation.append(log_dict["epoch"])
        for val_name in SELECTED_VALUES:
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


def plot_log_old(json_log_path, title=None, out_file=None):
    """
    Plots data from the log of the mmdetection format.
    Args:
        json_log_path (str): A name of the log file with data.
        title (str): A title of the plot.
        out_file (str): A path of the plotted data (must end with .png etc.).
    """

    train_logs, val_logs = load_logs(json_log_path)
    if title is None:
        title = json_log_path.split("/")[-1].removesuffix(".json")
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
                plot_log_save(log_file, title, out_file)
        if not recursive:
            break
