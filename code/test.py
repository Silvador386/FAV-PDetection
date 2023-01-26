import mmcv
import os.path as osp
from mmdet.apis import init_detector, inference_detector
from utils import files_in_folder


# Parameters for selecting specific images in predict_img_dir methods.
FRAME_RATE = 20
KEY_ZONE_FRAME_RATE = 2
FINER_ZONES = [(200, 300), (1400, 1480)]


def test(config_file="../configs/my_config/main_config_large.py",
         checkpoint_file="./work_dirs/main_config_clc_loss/latest.pth"):

    output_dir = "../results/test"
    img_prefix = "../data/city/images"
    json_path = osp.join(output_dir, "eval.json")

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    score_thr = 0.2

    predict_img_dir_to_json(model, img_prefix, json_path, score_thr)
    predict_img_dir(model, img_prefix, output_dir, score_thr)


def predict_img_dir_to_json(model, img_prefix, json_path, score_thr=0.3):
    image_names = files_in_folder(img_prefix)
    image_names = select_key_frames(image_names, FRAME_RATE, KEY_ZONE_FRAME_RATE, FINER_ZONES)
    result_json_format = {}

    for img_name in image_names:
        img_path = osp.join(img_prefix, img_name)
        img_result = predict_img_to_json(model, img_path, score_thr)
        result_json_format.update(img_result)

    mmcv.dump(result_json_format, json_path)


def predict_img_to_json(model, img_path, score_thr):
    result_dict = {}
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)
    for class_idx, class_result in enumerate(result):
        result_dict[img_path] = []
        mask = class_result[:, 4] > score_thr
        masked_img_results = class_result[mask]
        for results in masked_img_results:
            formatted_result = {"class": class_idx, "bbox": results[:4].tolist(), "score": results[4].tolist()}
            result_dict[img_path] = formatted_result

    return result_dict


def predict_img_dir(model, img_prefix, output_dir, score_thr=0.3):
    image_names = files_in_folder(img_prefix)
    image_names = select_key_frames(image_names, FRAME_RATE, KEY_ZONE_FRAME_RATE, FINER_ZONES)

    for frame_idx, image_name in enumerate(image_names):
        img = mmcv.imread(osp.join(img_prefix, image_name))
        result = inference_detector(model, img)
        model.show_result(img, result, out_file=osp.join(output_dir, f"{frame_idx:05}.jpg"), score_thr=score_thr)


def select_key_frames(image_names, frame_rate, key_zone_frame_rate, key_frame_zones):
    selected_img_names = []

    for frame_idx, image_name in enumerate(image_names):
        if any([zone[0] < frame_idx < zone[1] for zone in key_frame_zones]):
            if frame_idx % key_zone_frame_rate == 0:
                selected_img_names.append(image_name)
        else:
            if frame_idx % frame_rate == 0:
                selected_img_names.append(image_name)

    return selected_img_names
