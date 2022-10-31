from merge_pdestre2json import select_jsons_to_merge, merge_json_files
from pdestre_conversion import *
from settings import *
from train import TrainManager
from test import test
import wandb


def convert_and_merge_PDdata():
    """
    Converts P-DESTRE dataset by converting videos to images and text annotations to coco formatted .json files.
    Then the annotations are merged into large and small variant of train and test datasets.
    """
    print("\nConverting...\n")
    # Converts every video to images and every annotation to coco format .json file
    convert_pdestre_anns(NEW_ANNOTATIONS_DIR, NEW_VIDEO_DIR,
                         CONVERTED_ANNOTATIONS_DIR, CONVERTED_IMAGE_DIR, frame_rate=20)

    print("\nMerging...\n")
    # Create pdestre_large
    train_files, test_files = select_jsons_to_merge(CONVERTED_ANNOTATIONS_DIR, num_files=75, shuffle=True, divide=True)
    merge_json_files(CONVERTED_ANNOTATIONS_DIR, train_files,
                     "large_train", "../data/P-DESTRE/coco_format/merged", overwrite=False)
    merge_json_files(CONVERTED_ANNOTATIONS_DIR, test_files,
                     "large_test", "../data/P-DESTRE/coco_format/merged", overwrite=False)

    # Create pdestre_small
    train_files, test_files = select_jsons_to_merge(CONVERTED_ANNOTATIONS_DIR, num_files=16, shuffle=True, divide=True)
    merge_json_files(CONVERTED_ANNOTATIONS_DIR, train_files,
                     "small_train", "../data/P-DESTRE/coco_format/merged", overwrite=True)
    merge_json_files(CONVERTED_ANNOTATIONS_DIR, test_files,
                     "small_test", "../data/P-DESTRE/coco_format/merged", overwrite=True)


def main():
    """ Testing sanity-checks, """
    # sanity_checks.create_mini_dataset("../data/P-DESTRE/coco_format/merged/large_train.json",
    #                                   "../data/P-DESTRE/coco_format/merged", "debug_trial", 2)

    """ Current pipeline """
    config_path = "../configs/my_config/main_config_large.py"
    work_dir = "../work_dirs/"
    trainer = TrainManager(config_path, work_dir)
    trainer.train(create_opts=False)


    # test()
    # os.system("python ../tools/test.py ../configs/my_config/original_model.py "
    #           "../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth "
    #           "--eval bbox "
    #           "--show"
    #           )


if __name__ == "__main__":
    main()
