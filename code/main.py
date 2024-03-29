from train import train
from wandb_sweep import launch_sweep
from predict import predict


def main():
    """ Current pipeline """
    # train()
    # launch_sweep()

    config = "../configs/pdestre/main_config_large.py"
    checkpoint = "../metacentrum/202301261456/work_dirs/best_bbox_mAP_epoch_7.pth"
    output_dir = "../results/station_pdestre"
    img_prefix = "../data/city/images"
    score_thr = "0.3"
    predict([config, checkpoint, output_dir, img_prefix, score_thr])


if __name__ == "__main__":
    main()

