from merge_pdestre2json import select_jsons_to_merge, merge_json_files
from pdestre_conversion import *
from train import TrainManager
from test import test
from wandb_sweep import launch_sweep


def main():
    """ Current pipeline """
    config_path = "../configs/my_config/test_config.py"
    work_dir = "../work_dirs/"
    trainer = TrainManager(config_path, work_dir)
    trainer.train(create_opts=False)

    test(config_file="../metacentrum/02/202301031540/work_dirs/main_config_large.py",
         checkpoint_file="../metacentrum/02/202301031540/work_dirs/best_bbox_mAP_epoch_16.pth")
    # os.system("python ../tools/test.py ../configs/my_config/original_model.py "
    #           "../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth "
    #           "--eval bbox "
    #           "--show"
    #           )


if __name__ == "__main__":
    # main()
    launch_sweep()

