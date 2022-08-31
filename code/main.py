import itertools
import random
import plot_logs
from mmdet.apis import init_detector, train_detector, inference_detector, show_result_pyplot
from support import *
from settings import *
from main_config import cfg
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def check_pairs(new_annotations_folder, new_video_folder):
    """
    Checks for paired annotation and video names.

    Args:
        new_annotations_folder (str): A path to the folder with annotations.
        new_video_folder (str): A path to folder with videos.

    Returns:
         names_paired: A list of strings (names of files that are paired).
    """

    annotation_names = files_in_folder(new_annotations_folder)
    video_names = files_in_folder(new_video_folder)

    # remove .type suffix
    annotation_type = NEW_ANNOTATION_TYPE
    video_type = NEW_VIDEO_TYPE
    for i, ann in enumerate(annotation_names):
        annotation_names[i] = ann.removesuffix(annotation_type)
    for i, vid in enumerate(video_names):
        video_names[i] = vid.removesuffix(video_type)

    # check paired names
    names_paired = []
    for annotation_name in annotation_names:
        if annotation_name in video_names:
            # remove wrong data pairs:
            if not annotation_name.startswith("._"):
                names_paired.append(annotation_name)

    return names_paired


def convert_pdestre_dataset(new_annotations_folder, new_video_folder, convert_annotations_folder, convert_image_folder,
                            frame_rate, override_checks=False):
    """
    Converts paired videos to jpg images and annotations to coco format json files.

    Args:
        new_annotations_folder (str): A path to the folder with annotations.
        new_video_folder (str): A path to folder with videos.
        convert_annotations_folder (str): A path to the folder for formatted annotations.
        convert_image_folder (str): A path to the folder for images got from the video.
        frame_rate (int): Determines which frames should be converted (each 10th for instance).
        override_checks (bool): Overrides checks if annotations are already present.

    """

    names_paired = check_pairs(new_annotations_folder, new_video_folder)

    # take each paired name, convert video to jpgs, create new COCO-style annotation
    for i, name in enumerate(names_paired):
        video_path = new_video_folder + "/" + name + NEW_VIDEO_TYPE
        ann_path = new_annotations_folder + "/" + name + NEW_ANNOTATION_TYPE

        # test if already converted
        likely_ann_path = convert_annotations_folder + "/" + name + ".json"

        # Checks if the annotation file already exists. If so, skips the conversion.
        if os.path.isfile(likely_ann_path) and not override_checks:
            print(f"{likely_ann_path} already exists.")
            continue

        # Converts an annotation file with corresponding video.
        pdestre_to_coco(ann_path, video_path, name, convert_annotations_folder, convert_image_folder,
                        frame_rate=frame_rate)


def prepare_data():
    """
    Converts P-DESTRE dataset by converting videos to images, text annotations to coco formatted .json files.
    Then the annotations are merged into large and small variant of train and test datasets.
    """
    print("\nConverting...\n")
    # Converts every video to images and every annotation to coco format .json file
    convert_pdestre_dataset(NEW_ANNOTATIONS_FOLDER, NEW_VIDEO_FOLDER,
                            CONVERTED_ANNOTATIONS_FOLDER, CONVERTED_IMAGE_FOLDER, frame_rate=20)

    print("\nMerging...\n")
    # Create pdestre_large
    train_files, test_files = select_json_to_merge(CONVERTED_ANNOTATIONS_FOLDER, 75, shuffle=True, divide=True)
    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, train_files,
                     "large_train", "../Datasets/P-DESTRE/coco_format/merged", overwrite=True)
    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, test_files,
                     "large_test", "../Datasets/P-DESTRE/coco_format/merged", overwrite=True)

    # Create pdestre_small
    train_files, test_files = select_json_to_merge(CONVERTED_ANNOTATIONS_FOLDER, 16, shuffle=True, divide=True)
    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, train_files,
                     "small_train", "../Datasets/P-DESTRE/coco_format/merged", overwrite=True)
    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, test_files,
                     "small_test", "../Datasets/P-DESTRE/coco_format/merged", overwrite=True)


def train(create_params=False):
    """
    Retrains an pre-existed object detection model and outputs the checkpoints and logs into "train_exports" directory.
    Builds a train dataset from cfg.data.train. Plots the train loss of train epochs and mAP of the validation epochs.

    Args:
        create_params (bool): Creates a combination of multiple params which are used to gradually
                              multiple models.

    Results are stored in train_exports directory.
    """

    datasets = [build_dataset(cfg.data.train)]  # mmdet/datasets/ utils.py - change __check_head

    learning_rates = [0.00016]
    weight_decays = [5.8e-3]

    if create_params:
        learning_rates = [random.uniform(0.0004, 0.00001) for _ in range(3)]
        weight_decays = [random.uniform(0.01, 0.00001) for _ in range(6)]

    combinations = list(itertools.product(learning_rates, weight_decays))

    for i, (learning_rate, weight_decay) in enumerate(combinations):
        print(f"Learning rate: {learning_rate}\nWeight decay: {weight_decay}")
        # cfg.load_from = "../checkpoints/results/favorites/s32_e2_lr_1,3e4/latest.pth"
        cfg.optimizer = dict(type='SGD', lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        cfg.optimizer_config = dict(grad_clip=None)
        # cfg.lr_config = dict(
        #     policy='step',
        #     warmup='linear',
        #     warmup_iters=500,
        #     warmup_ratio=0.001,
        #     step=[7])
        # cfg.work_dir = f"./train_exports/lr{learning_rate:.1e}_wd{weight_decay:.1e}"

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
    config_file = cfg
    checkpoint_file = "train_exports/latest.pth"
    out_prefix = "../results/pdestre"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    test_img_prefix = "../Datasets/city/images"

    image_names = files_in_folder(test_img_prefix)

    # For parts where the tram is present.
    finer_zones = [(200, 300), (1400, 1480)]

    # Test on images
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
    # convert_video_to_jpg("video_20220603-1218", "../Datasets/city/video_20220603-1218.mp4", "../Datasets/city/images")
    prepare_data()
    train()
    test()


if __name__ == "__main__":
    main()
