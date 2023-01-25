import mmcv
import os.path as osp
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)



from pdestre_conversion import *
from settings import *


def test(config_file="../configs/my_config/main_config_large.py",
         checkpoint_file="./work_dirs/main_config_clc_loss/latest.pth"):

    output_dir = "../results/test"
    img_prefix = "../data/city/images"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    predict_img_dir_to_json(model, img_prefix, output_dir)
    # test_on_city_imgs(model, img_prefix, output_dir)
    # test_model_on_pdestre_imgs(model)


def predict_img_dir_to_json(model, img_prefix, output_path, threshold=0.3):
    image_names = files_in_folder(img_prefix)
    json_file = osp.join(output_path, f'eval.json')
    result_json_format = {}

    for img_name in image_names[250:255]:
        img_path = osp.join(img_prefix, img_name)
        img_result = predict_img_to_json(model, img_path, threshold)
        result_json_format.update(img_result)

    print(result_json_format)
    mmcv.dump(result_json_format, json_file)


def predict_img_to_json(model, img_path, threshold):
    result_dict = {}
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)
    if result:
        img_results = result[0]
        result_dict[img_path] = []
        mask = img_results[:, 4] > threshold
        masked_img_results = img_results[mask]
        for results in masked_img_results:
            formatted_result = {"bbox": results[:4].tolist(), "score": results[4].tolist()}
            result_dict[img_path] = formatted_result

    return result_dict


def test_on_city_imgs(model, img_prefix, output_dir):
    frame_rate = 20
    finer_zones = [(200, 300), (1400, 1480)]  # Frames of interest

    image_names = files_in_folder(img_prefix)

    # Test on images (images are each 10th frame of the video)
    for i, image_name in enumerate(image_names):
        if any([zone[0] < i < zone[1] for zone in finer_zones]):
            frame_rate = 2
        if i % frame_rate == 0:
            img = mmcv.imread(osp.join(img_prefix, image_name))
            result = inference_detector(model, img)
            model.show_result(img, result, out_file=osp.join(output_dir, f"/{i:05}.jpg"))


def test_model_on_pdestre_img_examples(model):
    # Randomly picked images to be shown on.
    pdestre_examples = [converted_img_dir + "/12-11-2019-2-1_f00160.jpg",
                        converted_img_dir + "/13-11-2019-1-2_f01560.jpg",
                        converted_img_dir + "/10-07-2019-1-1_f00670.jpg",
                        converted_img_dir + "/18-07-2019-1-2_f00280.jpg",
                        converted_img_dir + "/08-11-2019-2-1_f00360.jpg",
                        converted_img_dir + "/08-11-2019-1-2_f01090.jpg"
                        ]

    for e in pdestre_examples:
        img = mmcv.imread(e)
        result = inference_detector(model, img)
        show_result_pyplot(model, img, result)
