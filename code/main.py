from train import train
from wandb_sweep import launch_sweep
from test import test


def main():
    """ Current pipeline """
    train()
    # launch_sweep()
    # test(config_file="../metacentrum/02/202301031540/work_dirs/main_config_large.py",
    #      checkpoint_file="../metacentrum/02/202301031540/work_dirs/best_bbox_mAP_epoch_16.pth")


if __name__ == "__main__":
    main()

