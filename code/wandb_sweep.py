import yaml

import wandb
from train_pd import basic_train


sweep_configuration = {
    "name": "Testing-sweep",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "method": "random",
    "parameters": {
        'lr': {'max': 0.005, 'min': 0.00005},
        "wd": {'max': 0.001, 'min': 0.000001},
        "optim": {"values": ["SGD", "Adam"]}
    }
}


def launch_sweep():
    sweep_id = wandb.sweep(sweep_configuration, project="Test-Sweep")

    wandb.agent(sweep_id, function=train_sweep_wrap, count=5)


def train_sweep_wrap():
    wandb.init()

    learning_rate = wandb.config.lr
    weight_decay = wandb.config.wd
    optimizer = wandb.config.optim

    basic_train(learning_rate, weight_decay, optimizer)
