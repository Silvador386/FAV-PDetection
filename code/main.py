from merge_pdestre2json import select_jsons_to_merge, merge_json_files
from pdestre_conversion import *
from settings import *
from train import TrainManager
from test import test
import wandb


def convert_and_merge_pdestre(ann_dir,
                              video_dir,
                              converted_ann_dir,
                              converted_img_dir,
                              merged_dir,
                              frame_rate=20
                              ):
    """
    Converts P-DESTRE dataset by converting videos to images and text annotations to coco formatted .json files.
    Then the annotations are merged into large and small variant of train and test datasets.
    """
    print("\nConverting...\n")
    # Converts every video to images and every annotation to coco format .json file
    convert_pdestre_anns(ann_dir, video_dir,
                         converted_ann_dir, converted_img_dir, frame_rate=frame_rate, override_checks=True)

    print("\nMerging...\n")
    # Create pdestre_large
    train_files, test_files = select_jsons_to_merge(converted_ann_dir, num_files=75, shuffle=True, divide=True)
    merge_json_files(converted_ann_dir, train_files,
                     "large_train", merged_dir, overwrite=True)
    merge_json_files(converted_ann_dir, test_files,
                     "large_test", merged_dir, overwrite=True)

    # Create pdestre_small
    # train_files, test_files = select_jsons_to_merge(converted_ann_dir, num_files=16, shuffle=True, divide=True)
    # merge_json_files(converted_ann_dir, train_files,
    #                  "small_train", merged_dir, overwrite=True)
    # merge_json_files(converted_ann_dir, test_files,
    #                  "small_test", merged_dir, overwrite=True)



sweep_configuration = {
    "name": "Testing-sweep",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "method": "random",
    "parameters": {
        'lr': {'max': 0.005, 'min': 0.00005},
        "wd": {'max': 0.001, 'min': 0.000001}
    }
}


def launch_sweep():
    from train_pd import basic_train

    sweep_id = wandb.sweep(sweep_configuration, project="Test-Sweep")

    wandb.agent(sweep_id, function=basic_train, count=5)



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

