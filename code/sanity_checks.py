import json
import cv2
import random
import mmcv
from mmcv import Config
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from utils import write_to_json


def load_ann_data(ann_path):
    with open(ann_path, "r") as fp:
        ann_data = json.load(fp)

    return ann_data["images"], ann_data["annotations"], ann_data["categories"]


def create_mini_dataset(ann_path, output_path, file_name, num_images):
    images, annotations, categories = load_ann_data(ann_path)

    coco_json = {"images": [],
                 "annotations": [],
                 "categories": categories}

    selected_images = random.sample(images, num_images)
    for image in selected_images:
        coco_json["images"].append(image)
        for ann in annotations:
            if ann["image_id"] == image["id"]:
                coco_json["annotations"].append(ann)

    write_to_json(coco_json, f"{output_path}/{file_name}.json")


def test_rect_anns(ann_path, img_dir, output_dir, max_num=-1, resize=False):

    images, annotations, _ = load_ann_data(ann_path)
    random.shuffle(images)

    for i, image in enumerate(images):
        img_name = image["file_name"]
        img = cv2.imread(img_dir + "/" + img_name)
        anns_selected = []
        for ann in annotations:
            if ann["image_id"] == image["id"]:
                anns_selected.append(ann)
                bbox = ann["bbox"]
                bottom_left_corner = int(bbox[0]), int(bbox[1])
                upper_right_corner = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                img = cv2.rectangle(img, bottom_left_corner, upper_right_corner, (0, 0, 255), 2)

        if output_dir is not None:
            if resize:
                img = cv2.resize(img, [1333, 800])
                output_path = f"{output_dir}/resized_{img_name}"
            else:
                output_path = f"{output_dir}/rect_{img_name}"
            print(f"Writing to: {output_path}")
            cv2.imwrite(output_path, img)

        if i == max_num:
            break


def resize_imgs(ann_path, img_dir, output_dir, size=(1300, 800)):
    images, annotations, _ = load_ann_data(ann_path)

    for i, image in enumerate(images):
        img_name = image["file_name"]
        img = cv2.imread(img_dir + "/" + img_name)

        resized = cv2.resize(img, size)

        if output_dir is not None:
            output_path = f"{output_dir}/resized_{img_name}"
            print(f"Writing to: {output_path}")
            cv2.imwrite(output_path, resized)


def test_image_inference(config, checkpoint, ann_path, img_folder_path, model, output_dir, max_num=-1):
    if model is None:
        model = init_detector(config, checkpoint, device='cuda:0')

    images, annotations, categories = load_ann_data(ann_path)
    random.shuffle(images)

    for i, image in enumerate(images):
        img_name = image["file_name"]
        img_path = f"{img_folder_path}/{img_name}"
        img = mmcv.imread(img_path)
        result = inference_detector(model, img)

        if output_dir is None:
            show_result_pyplot(model, img, result)
        else:
            output_path = f"{output_dir}/model_{img_name}"
            print(f"Writing to: {output_path}")
            model.show_result(img, result, out_file=output_path)

        if i == max_num:
            break


def test_json_anns(config_path, checkpoint, output_dir=None, model=None, max_num=-1):
    config = Config.fromfile(config_path)
    ann_path = config._ann_file_test
    img_dir = config._img_prefix

    test_rect_anns(ann_path, img_dir, output_dir, max_num)
    test_image_inference(config, checkpoint, ann_path, img_dir, model, output_dir, max_num)
