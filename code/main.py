import itertools
import random
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import init_detector, train_detector, inference_detector, show_result_pyplot

import plot_logs
from merge_pdestre2json import select_jsons_to_merge, merge_json_files
from pdestre_conversion import *
from settings import *


def convert_data():
    """
    Converts P-DESTRE dataset by converting videos to images and text annotations to coco formatted .json files.
    Then the annotations are merged into large and small variant of train and test datasets.
    """
    print("\nConverting...\n")
    # Converts every video to images and every annotation to coco format .json file
    convert_pdestre_anns(NEW_ANNOTATIONS_FOLDER, NEW_VIDEO_FOLDER,
                         CONVERTED_ANNOTATIONS_FOLDER, CONVERTED_IMAGE_FOLDER, frame_rate=20)

    print("\nMerging...\n")
    # Create pdestre_large
    train_files, test_files = select_jsons_to_merge(CONVERTED_ANNOTATIONS_FOLDER, num_files=75, shuffle=True, divide=True)
    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, train_files,
                     "large_train", "../data/P-DESTRE/coco_format/merged", overwrite=False)
    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, test_files,
                     "large_test", "../data/P-DESTRE/coco_format/merged", overwrite=False)

    # Create pdestre_small
    train_files, test_files = select_jsons_to_merge(CONVERTED_ANNOTATIONS_FOLDER, num_files=16, shuffle=True, divide=True)
    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, train_files,
                     "small_train", "../data/P-DESTRE/coco_format/merged", overwrite=True)
    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, test_files,
                     "small_test", "../data/P-DESTRE/coco_format/merged", overwrite=True)


def train(create_params=False):
    """
    Retrains an pre-existed object detection model and outputs the checkpoints and logs into "train_exports" directory.
    Builds a train dataset from cfg.data.train. Plots the train loss of train epochs and mAP of the validation epochs.

    Args:
        create_params (bool): Creates a combination of multiple params which are used to gradually
                              multiple models.

    Results are stored in train_exports directory.
    """
    from org_config import cfg

    datasets = [build_dataset(cfg.data.train)]  # mmdet/datasets/ utils.py - change __check_head

    learning_rates = [0.00016]
    # weight_decays = [5.8e-3]
    weight_decays = [0]
    if create_params:
        learning_rates = [random.uniform(0.0004, 0.00001) for _ in range(3)]
        weight_decays = [random.uniform(0.01, 0.00001) for _ in range(6)]

    train_combinations = list(itertools.product(learning_rates, weight_decays))

    for i, (learning_rate, weight_decay) in enumerate(train_combinations):
        print(f"Learning rate: {learning_rate}\nWeight decay: {weight_decay}")
        # cfg.load_from = "../checkpoints/results/favorites/s32_e2_lr_1,3e4/latest.pth"
        # cfg.optimizer = dict(type='SGD', lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        # cfg.optimizer_config = dict(grad_clip=None)
        # cfg.lr_config = dict(
        #     policy='step',
        #     warmup='linear',
        #     warmup_iters=500,
        #     warmup_ratio=0.001,
        #     step=[7])
        # cfg.work_dir_path = f"./train_exports/lr{learning_rate:.1e}_wd{weight_decay:.1e}"

        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.CLASSES = ("person", )

        train_detector(model, datasets, cfg, distributed=False, validate=True)
        plot_logs.plot_all_logs_in_dir("train_exports", recursive=False)


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
    config_file = "../configs/my_config/main_config.py"
    checkpoint_file = "../work_dirs/main_config/epoch_7.pth"
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
    pdestre_examples = [CONVERTED_IMAGE_FOLDER + "/12-11-2019-2-1_f00160.jpg",
                        CONVERTED_IMAGE_FOLDER + "/13-11-2019-1-2_f01560.jpg",
                        CONVERTED_IMAGE_FOLDER + "/10-07-2019-1-1_f00670.jpg",
                        CONVERTED_IMAGE_FOLDER + "/18-07-2019-1-2_f00280.jpg",
                        CONVERTED_IMAGE_FOLDER + "/08-11-2019-2-1_f00360.jpg",
                        CONVERTED_IMAGE_FOLDER + "/08-11-2019-1-2_f01090.jpg"
                        ]

    for e in pdestre_examples:
        img = mmcv.imread(e)
        result = inference_detector(model, img)
        show_result_pyplot(model, img, result)


def main():
    # convert_video_to_jpg("video_20220603-1218", "../data/city/video_20220603-1218.mp4", "../Datasets/city/images")
    # prepare_data()
    # train()
    test()


if __name__ == "__main__":
    """ Testing sanity-checks, """
    # sanity_checks.check_formatted_data("../data/P-DESTRE/coco_format/merged/mini_train.json", CONVERTED_IMAGE_FOLDER,
    #                  "../results/test_check")
    # sanity_checks.create_mini_dataset("../data/P-DESTRE/coco_format/merged/large_train.json",
    #                                   "../data/P-DESTRE/coco_format/merged", mini_train, 20)
    # main()
    # sanity_checks.test_image()

    # python tools/train.py configs/my_config/main_config.py
    # checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

    """ Using built-in commands shortcut """
    # os.system("python ../tools/train.py ../configs/my_config/main_config.py")
    os.system("python ../tools/test.py ../configs/my_config/main_config.py work_dirs/main_config/latest.pth --show")
