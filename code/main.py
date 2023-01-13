from merge_pdestre2json import select_jsons_to_merge, merge_json_files
from pdestre_conversion import *
from settings import *
from train import TrainManager
from test import test


def convert_and_merge_PDdata():
    """
    Converts P-DESTRE dataset by converting videos to images and text annotations to coco formatted .json files.
    Then the annotations are merged into large and small variant of train and test datasets.
    """
    print("\nConverting...\n")
    # Converts every video to images and every annotation to coco format .json file
    convert_pdestre_anns(NEW_ANNOTATIONS_DIR, NEW_VIDEO_DIR,
                         CONVERTED_ANNOTATIONS_DIR, CONVERTED_IMAGE_DIR, frame_rate=20, override_checks=True)

    print("\nMerging...\n")
    # Create pdestre_large
    train_files, test_files = select_jsons_to_merge(CONVERTED_ANNOTATIONS_DIR, num_files=75, shuffle=True, divide=True)
    merge_json_files(CONVERTED_ANNOTATIONS_DIR, train_files,
                     "large_train", "../data/P-DESTRE/coco_format/merged", overwrite=True)
    merge_json_files(CONVERTED_ANNOTATIONS_DIR, test_files,
                     "large_test", "../data/P-DESTRE/coco_format/merged", overwrite=True)

    # Create pdestre_small
    train_files, test_files = select_jsons_to_merge(CONVERTED_ANNOTATIONS_DIR, num_files=16, shuffle=True, divide=True)
    merge_json_files(CONVERTED_ANNOTATIONS_DIR, train_files,
                     "small_train", "../data/P-DESTRE/coco_format/merged", overwrite=True)
    merge_json_files(CONVERTED_ANNOTATIONS_DIR, test_files,
                     "small_test", "../data/P-DESTRE/coco_format/merged", overwrite=True)


def main():
    """ Current pipeline """
    config_path = "../configs/my_config/main_config.py"
    work_dir = "../work_dirs/"
    trainer = TrainManager(config_path, work_dir)
    # trainer.train(create_opts=False)

    test(config_file="../metacentrum/02/202212252142/work_dirs/main_config_large.py",
         checkpoint_file="../metacentrum/02/202212252142/work_dirs/latest.pth")
    # os.system("python ../tools/test.py ../configs/my_config/original_model.py "
    #           "../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth "
    #           "--eval bbox "
    #           "--show"
    #           )


if __name__ == "__main__":
    main()

