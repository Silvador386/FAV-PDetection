import json
import cv2
import random


def ann_data(ann_file_path):
    with open(ann_file_path, "r") as fp:
        _ann_data = json.load(fp)

    return _ann_data["images"], _ann_data["annotations"], _ann_data["categories"]


def check_formatted_data(ann_file_path, img_folder_path, output_path):
    images, annotations, _ = ann_data(ann_file_path)

    for _ in range(50):
        img_selected = random.choice(images)
        img_name = img_selected["file_name"]
        img = cv2.imread(img_folder_path + "/" + img_name)
        ann_selected = []
        for ann in annotations:
            if ann["image_id"] == img_selected["id"]:
                ann_selected.append(ann)
                bbox = ann["bbox"]
                start = int(bbox[0]), int(bbox[1])
                end = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                img = cv2.rectangle(img, start, end, (0, 0, 255), 2)

        cv2.imwrite(output_path + "/" + img_name, img)


def create_mini_dataset(ann_file_path, output_path, num_imgs):
    images, annotations, categories = ann_data(ann_file_path)

    final = {"images": [],
             "annotations": [],
             "categories": categories}

    for _ in range(num_imgs):
        img_selected = random.choice(images)
        final["images"].append(img_selected)
        for ann in annotations:
            if ann["image_id"] == img_selected["id"]:
                final["annotations"].append(ann)

    train_name = "mini_train.json"

    with open(output_path + "/" + train_name, "w") as fp:
        json.dump(final, fp)


def test_image():
    from main_config import cfg
    import mmcv
    from mmdet.apis import init_detector, inference_detector, show_result_pyplot
    from settings import CONVERTED_IMAGE_FOLDER

    config_file = cfg
    checkpoint_file = "train_exports/epoch_4.pth"
    out_prefix = "../results/pdestre"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # Randomly picked images to be shown
    images, annotations, categories = ann_data("../data/P-DESTRE/coco_format/merged/mini_train.json")
    pdestre_examples = [CONVERTED_IMAGE_FOLDER + "/" + images[0]["file_name"],

                        ]

    for e in pdestre_examples:
        img = mmcv.imread(e)
        result = inference_detector(model, img)
        show_result_pyplot(model, img, result)




