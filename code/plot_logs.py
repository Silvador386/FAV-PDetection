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


def plot_loss(loss_logs, type_name, subplot):
    max_iter = max(loss_log["iter"] for loss_log in loss_logs)
    x_train_loss, y_train_loss = [], []
    for log_dict in loss_logs:
        x_train_loss.append(log_dict["iter"] + (log_dict["epoch"]-1) * max_iter)
        y_train_loss.append(log_dict["loss"])

    subplot.plot(x_train_loss, y_train_loss, marker="o")
    subplot.set(xlabel="Iterations")
    subplot.set(ylabel=f"{type_name}")
    subplot.set_ylim(bottom=0)
    subplot.set_xlim(left=0)

    text = "\n".join((f"Iter/Epoch: {max_iter}",))

    props = dict(boxstyle='round', facecolor='w', alpha=0.8)
    subplot.text(0.77, 2.16, text, transform=subplot.transAxes, fontsize=14,
                 verticalalignment='top', bbox=props)


def plot_acc(acc_logs, type_name, subplot):
    SELECTED_VALUES = ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_m", "bbox_mAP_l"]

    x_val = []
    y_vals = {name: [] for name in SELECTED_VALUES}
    for log in acc_logs:
        x_val.append(log["epoch"])
        for val_name in SELECTED_VALUES:
            y_vals[val_name].append(log[val_name])

    for key in y_vals.keys():
        if key == "bbox_mAP":
            subplot.plot(x_val, y_vals[key], marker="o", linewidth=5)
            continue
        subplot.plot(x_val, y_vals[key], marker="o")

    subplot.legend(y_vals.keys(), loc="lower right")
    subplot.set(ylabel=f"mAP/{type_name}")
    subplot.set(xlabel="Epochs")
    subplot.set_ylim(bottom=0)
    subplot.set_xlim(left=0)


def plot_log_save(json_log_path, output_path=None):
    type_names = log_types(json_log_path)
    loaded_logs = load_logs(json_log_path, type_names)

    num_types = len(type_names)
    fig, axs = plt.subplots(nrows=num_types, ncols=1, figsize=(12, 8), squeeze=False)
    fig.suptitle(f"{select_title(json_log_path)}", fontsize=16)

    row_idx = 0
    for type_name in sorted(type_names):
        subplot = axs[row_idx, 0]
        subplot.grid()
        if "loss" in type_name:
            plot_loss(loaded_logs[type_name], type_name, subplot)
        if "acc" in type_name:
            plot_acc(loaded_logs[type_name], type_name, subplot)
        row_idx += 1

    if output_path is None:
        output_path = f"{json_log_path.removesuffix('.json')}.png"

    print(f"Saving plot to: {output_path}")
    plt.savefig(output_path)
    plt.show()
    plt.close()


def plot_all_logs_in_dir(work_dir_path, overwrite=False, recursive=False):
    """
    Goes trough the whole dir and plots data of every located log file in mmdetection format.

    Args:
        work_dir_path (str): A path of the directory.
        recursive (bool): If any located subdirectory should be searched for logs as well.
    """
    for dir_path, dir_names, file_names in os.walk(work_dir_path):
        for file_name in file_names:
            if file_name.endswith(".json"):
                json_log_path = f"{dir_path}/{file_name}"
                output_path = f"{dir_path}/{file_name.removesuffix('.json')}.png"
                if os.path.isfile(output_path) and not overwrite:
                    print(f"{output_path} already exists.")
                    continue
                plot_log_save(json_log_path, output_path)
        if not recursive:
            break
