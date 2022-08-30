import json
import matplotlib.pyplot as plt


def load_logs(log_file):
    train, val = [], []
    with open(log_file, "r") as logs:
        for line in logs:
            log_line = json.loads(line.strip())
            if log_line["mode"] == "val":
                val.append(log_line)
            else:
                train.append(log_line)

    return train, val


def plot_log(log_file, out_file):
    train_logs, val_logs = load_logs(log_file)

    num_iter = max(train_log["iter"] for train_log in train_logs)
    num_epochs = max(train_log["epoch"] for train_log in train_logs)

    x, y = [], []
    for log_dict in train_logs:
        x.append(log_dict["iter"] + log_dict["epoch"] * num_iter)
        y.append(log_dict["loss"])

    value_names = ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l"]
    x_val = []
    y_vals = {name: [] for name in value_names}
    for log_dict in val_logs:
        x_val.append(log_dict["epoch"])
        for val_name in value_names:
            y_vals[val_name].append(log_dict[val_name])

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    axs[0].plot(x, y, marker="o")
    axs[0].set(xlabel="Iterations")
    axs[0].set(ylabel="Train loss")

    for key in y_vals.keys():
        if key == "bbox_mAP":
            pass
        axs[1].plot(x_val, y_vals[key], marker="o")
    axs[1].legend(y_vals.keys())
    axs[1].set(ylabel="mAP")
    axs[1].set(xlabel="Epoch")

    for ax in axs:
        ax.grid()

    plt.show()


