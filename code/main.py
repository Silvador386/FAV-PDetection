import itertools
import random
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import init_detector, train_detector, inference_detector, show_result_pyplot

import plot_logs
import sanity_checks
from merge_pdestre2json import select_jsons_to_merge, merge_json_files
from pdestre_conversion import *
from settings import *
from train import TrainManager


def convert_and_merge_data():
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


def test():
    """
    Tests current model on the control data (images of pre-selected video and randomly chose images from the dataset).
    The interference is stored in the "out_prefix" folder.

    """

    # Control - original model
    # config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # out_prefix = "../results/control"

    # Test current model
    # config_file = cfg
    config_file = "../configs/my_config/use_latest_config.py"
    checkpoint_file = "../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
    out_prefix = "../results/pdestre"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    test_img_prefix = "../data/city/images"

    image_names = files_in_folder(test_img_prefix)

    # For parts where the tram is present.
    finer_zones = [(200, 300), (1400, 1480)]

    # Test on images (images are each 10th frame of the video)
    for i, image_name in enumerate(image_names):
        image_rate = 20
        if any([zone[0] < i < zone[1] for zone in finer_zones]):
            image_rate = 4
        if i % image_rate == 0:
            img = mmcv.imread(test_img_prefix + "/" + image_name)
            result = inference_detector(model, img)
            model.show_result(img, result, out_file=out_prefix + f"/{i:05}.jpg")

    # Randomly picked images to be shown
    pdestre_examples = [CONVERTED_IMAGE_DIR + "/12-11-2019-2-1_f00160.jpg",
                        CONVERTED_IMAGE_DIR + "/13-11-2019-1-2_f01560.jpg",
                        CONVERTED_IMAGE_DIR + "/10-07-2019-1-1_f00670.jpg",
                        CONVERTED_IMAGE_DIR + "/18-07-2019-1-2_f00280.jpg",
                        CONVERTED_IMAGE_DIR + "/08-11-2019-2-1_f00360.jpg",
                        CONVERTED_IMAGE_DIR + "/08-11-2019-1-2_f01090.jpg"
                        ]

    for e in pdestre_examples:
        img = mmcv.imread(e)
        result = inference_detector(model, img)
        show_result_pyplot(model, img, result)


def main():
    """ Testing sanity-checks, """
    # sanity_checks.create_mini_dataset("../data/P-DESTRE/coco_format/merged/large_train.json",
    #                                   "../data/P-DESTRE/coco_format/merged", "debug_trial", 2)

    """ Current pipeline """
    config_path = "../configs/my_config/main_config.py"
    work_dir = "./work_dirs/main_config"
    trainer = TrainManager(config_path)
    trainer.train(work_dir)

    # test()
    # os.system("python ../tools/test.py ../configs/my_config/use_latest_config.py ../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth --eval bbox --show")


if __name__ == "__main__":
    main()
