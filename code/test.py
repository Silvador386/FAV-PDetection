import mmcv
import os.path as osp
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

from utils import files_in_folder


def test(config_file="../configs/my_config/main_config_large.py",
         checkpoint_file="./work_dirs/main_config_clc_loss/latest.pth"):

    output_dir = "../results/test"
    img_prefix = "../data/city/images"
    json_path = osp.join(output_dir, "eval.json")

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    score_thr = 0.2

    predict_img_dir_to_json(model, img_prefix, output_dir, score_thr)
    predict_img_dir(model, img_prefix, json_path, score_thr)


def predict_img_dir_to_json(model, img_prefix, json_path, score_thr=0.3):
    image_names = files_in_folder(img_prefix)
    result_json_format = {}

    for img_name in image_names:
        img_path = osp.join(img_prefix, img_name)
        img_result = predict_img_to_json(model, img_path, score_thr)
        result_json_format.update(img_result)

    print(result_json_format)
    mmcv.dump(result_json_format, json_path)


def predict_img_to_json(model, img_path, score_thr):
    result_dict = {}
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)
    if result:
        img_results = result[0]
        result_dict[img_path] = []
        mask = img_results[:, 4] > score_thr
        masked_img_results = img_results[mask]
        for results in masked_img_results:
            formatted_result = {"bbox": results[:4].tolist(), "score": results[4].tolist()}
            result_dict[img_path] = formatted_result

    return result_dict


def predict_img_dir(model, img_prefix, output_dir, score_thr=0.3, frame_rate=20):
    finer_zones = [(200, 300), (1400, 1480)]  # Frames of interest
    image_names = files_in_folder(img_prefix)

    # Test on images (images are each 10th frame of the video)
    for i, image_name in enumerate(image_names):
        if any([zone[0] < i < zone[1] for zone in finer_zones]):
            frame_rate = 2
        if i % frame_rate == 0:
            img = mmcv.imread(osp.join(img_prefix, image_name))
            result = inference_detector(model, img)
            model.show_result(img, result, out_file=osp.join(output_dir, f"/{i:05}.jpg"), score_thr=score_thr)
