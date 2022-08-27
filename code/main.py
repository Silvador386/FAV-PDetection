import random

from mmdet.apis import init_detector, train_detector, inference_detector, show_result_pyplot
from support import *
from settings import *


# checks for paired annotation and video names
def check_pairs(new_annotations_folder, new_video_folder):
    annotation_names = files_in_folder(NEW_ANNOTATIONS_FOLDER)
    video_names = files_in_folder(NEW_VIDEO_FOLDER)

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


# converts paired videos to jpgs and annotations to json (COCO-style like)
def convert_pdestre_dataset(new_annotations_folder, new_video_folder, convert_annotations_folder, convert_image_folder,
                            frame_rate, override_checks=False):

    names_paired = check_pairs(new_annotations_folder, new_video_folder)

    # take each pair, convert video to jpgs, create new COCO-style annotation
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
        convert_pdestre_dataset(ann_path, video_path, name, convert_annotations_folder, convert_image_folder,
                                frame_rate=frame_rate)


def prepare_data():
    print("\nConverting...\n")
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
    train_files, test_files = select_json_to_merge(CONVERTED_ANNOTATIONS_FOLDER, 32, shuffle=True, divide=True)
    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, train_files,
                     "small_train", "../Datasets/P-DESTRE/coco_format/merged", overwrite=True)
    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, test_files,
                     "small_test", "../Datasets/P-DESTRE/coco_format/merged", overwrite=True)


def train(learning_rates=None):
    from main_config import cfg
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector

    datasets = [build_dataset(cfg.data.train)]  # mmdet/datasets/ utils.py - change __check_head

    if not learning_rates:
        # learning_rates = [random.uniform(0.0004, 0.00001) for _ in range(3)]
        # learning_rates = [0.0004, 0.00005]
        learning_rates = [0.00016]

    for i, learning_rate in enumerate(learning_rates):
        print(f"Learning rate: {learning_rate}")
        # cfg.load_from = "../checkpoints/results/favorites/s32_e2_lr_1,3e4/latest.pth"
        cfg.optimizer = dict(type='SGD', lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        cfg.work_dir = f"./train_exports"

        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.CLASSES = ("person", )

        train_detector(model, datasets, cfg, distributed=False, validate=True)


def test():
    from main_config import cfg
    from mmdet.models import build_detector

    # Control
    # config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # out_prefix = "../results/control"

    # Testing image
    config_file = cfg
    checkpoint_file = "../checkpoints/results/favorites/s32_e2_lr_1,6e4/latest.pth"
    out_prefix = "../results/pdestre"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    test_img_prefix = "../Datasets/city/images"

    image_names = files_in_folder(test_img_prefix)

    finer_zones = [(200, 300), (1400, 1480)]

    for i, image_name in enumerate(image_names):
        image_rate = 20
        if any([zone[0] < i < zone[1] for zone in finer_zones]):
            image_rate = 4
        if i % image_rate == 0:
            img = mmcv.imread(test_img_prefix + "/" + image_name)
            result = inference_detector(model, img)
            # show_result_pyplot(model, img, result)
            model.show_result(img, result, out_file=out_prefix + f"/{i:05}.jpg")

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


if __name__ == "__main__":
    # convert_video_to_jpg("video_20220603-1218", "../Datasets/city/video_20220603-1218.mp4", "../Datasets/city/images")
    # prepare_data()
    # train()
    test()

