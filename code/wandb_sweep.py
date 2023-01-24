import wandb
from train import train


sweep_configuration = {
    "name": "Testing-sweep",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "method": "bayes",
    "parameters": {
        'lr': {'max': 0.005, 'min': 0.0001},
        "wd": {'max': 0.001, 'min': 0.000005},
        "optim": {"values": ["SGD"]}
    }
}


def launch_sweep():
    sweep_id = wandb.sweep(sweep_configuration, project="Test-Sweep")

    wandb.agent(sweep_id, function=train_sweep_wrap, count=2)


def train_sweep_wrap():
    with wandb.init():
        learning_rate = wandb.config.lr
        weight_decay = wandb.config.wd
        optimizer = wandb.config.optim

        train(learning_rate, weight_decay, optimizer)


if __name__ == "__main__":
    launch_sweep()
