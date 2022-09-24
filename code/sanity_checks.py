import json
import cv2
import random

from utils import write_to_json


def load_ann_data(ann_path):
    with open(ann_path, "r") as fp:
        ann_data = json.load(fp)

    return ann_data["images"], ann_data["annotations"], ann_data["categories"]


def check_formatted_data(ann_path, img_folder_path, output_path, num_checked=50):
    images, annotations, _ = load_ann_data(ann_path)

    for _ in range(num_checked):
        img_selected = random.choice(images)
        img_name = img_selected["file_name"]
        img = cv2.imread(img_folder_path + "/" + img_name)
        anns_selected = []
        for ann in annotations:
            if ann["image_id"] == img_selected["id"]:
                anns_selected.append(ann)
                bbox = ann["bbox"]
                bottom_left_corner = int(bbox[0]), int(bbox[1])
                upper_right_corner = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                img = cv2.rectangle(img, bottom_left_corner, upper_right_corner, (0, 0, 255), 2)

        cv2.imwrite(output_path + "/" + img_name, img)


def create_mini_dataset(ann_path, output_path, file_name, num_images):
    images, annotations, categories = load_ann_data(ann_path)

    coco_json = {"images": [],
                 "annotations": [],
                 "categories": categories}

    for _ in range(num_images):
        img_selected = random.choice(images)
        coco_json["images"].append(img_selected)
        for ann in annotations:
            if ann["image_id"] == img_selected["id"]:
                coco_json["annotations"].append(ann)

    write_to_json(coco_json, f"{output_path}/{file_name}.json")


def test_overfit_image():
    import mmcv
    from mmdet.apis import init_detector, inference_detector, show_result_pyplot
    from settings import CONVERTED_IMAGE_FOLDER

    config_file = "../configs/my_config/main_config.py"
    checkpoint_file = "work_dirs/main_config/latest.pth"
    out_prefix = "../results/pdestre"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # Randomly picked images to be shown
    images, annotations, categories = load_ann_data("../data/P-DESTRE/coco_format/merged/micro_train.json")
    pdestre_examples = [CONVERTED_IMAGE_FOLDER + "/" + images[0]["file_name"],
                        CONVERTED_IMAGE_FOLDER + "/" + images[1]["file_name"]
                        ]

    for e in pdestre_examples:
        img = mmcv.imread(e)
        result = inference_detector(model, img)
        show_result_pyplot(model, img, result)


def compare_test_ann_img():
    pass