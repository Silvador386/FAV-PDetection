import random
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import init_detector, train_detector, inference_detector, show_result_pyplot, set_random_seed
from mmcv import Config

import plot_logs
import sanity_checks
from merge_pdestre2json import select_jsons_to_merge, merge_json_files
from pdestre_conversion import *
from settings import *


def built_in_train(config_path, **optional_args):
    from tools import train as tp
    train_args = [config_path]
    optional = "--cfg-options optimizer.lr='0.007/8'"

    for val in dict_generator(optional_args):
        print(val)
    else:
        pass
        # train_args.append(optional)

    tp.main(train_args)


def train_manager():
    config_paths = ["c:/Programming/Python Projects/FAV_PD/configs/my_config/use_latest_config.py",
                    "../configs/my_config/main_config.py"]
    # user_defined_train(config_paths[0])
    built_in_train(config_paths[1], optimizer=dict(type='SGD', lr=0.007/8, momentum=0.9, weight_decay=0.00000))
    plot_logs.plot_all_logs_in_dir("./work_dirs/main_config")
    sanity_checks.test_json_anns(ann_path="../data/P-DESTRE/coco_format/merged/micro_train.json",
                                 img_dir=CONVERTED_IMAGE_DIR, output_dir="../results/test_json_anns",
                                 model=None
                                 )


def user_defined_train(config_path, **kwargs):
    cfg = Config.fromfile(config_path)
    set_cfg_gpu(cfg)
    # cfg.data.val.pipeline = cfg.train_pipeline  # https://github.com/open-mmlab/mmdetection/issues/1493

    cfg.work_dir = "./work_dirs/main_config_test_new_method/"

    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val)]

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES

    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        # timestamp=timestamp,
        # meta=meta)
        )
    plot_logs.plot_all_logs_in_dir(cfg.work_dir)


def dict_generator(indict, previous=None):
    previous = previous[:] if previous else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, previous + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in dict_generator(v, previous + [key]):
                        yield d
            else:
                yield previous + [key, value]
    else:
        yield previous + [indict]


def set_cfg_gpu(cfg):
    cfg.device = "cuda"
    cfg.data.samples_per_gpu = 3
    cfg.data.workers_per_gpu = 3
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
