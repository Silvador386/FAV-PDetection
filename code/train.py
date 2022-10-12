import random
import os
from itertools import product

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import init_detector, train_detector, inference_detector, show_result_pyplot, set_random_seed
from tools import train as mmdet_train

import plot_logs
import sanity_checks
from settings import *


class TrainManager:
    def __init__(self, config_path, work_dir):
        self.config_path = config_path
        self.work_dir = work_dir
        self.options = []

    def train(self, create_opts=False, **kwargs):
        # train multiple times, with different options
        if create_opts:
            self.create_lr_wd_combs()
            for option in self.options:
                kwargs.update(option)
                self.train_pipeline(**kwargs)
        # train once with base options
        else:
            self.train_pipeline(**kwargs)

    def train_pipeline(self, **kwargs):
        # run train method
        built_in_train(self.config_path, self.work_dir, **kwargs)

        plot_logs.plot_all_logs_in_dir(self.work_dir)

        # creates images with predictions from the model
        # model = init_detector(self.config_path,
        #                       checkpoint="./work_dirs/main_config_clc_loss/latest.pth", device='cuda:0')
        # sanity_checks.test_json_anns(ann_path="../data/P-DESTRE/coco_format/merged/micro_train.json",
        #                              img_dir=CONVERTED_IMAGE_DIR, output_dir="../results/test_json_anns",
        #                              model=model)

    def create_lr_wd_combs(self):
        learning_rates = generate_uniform_values(0.01, 0.0005, 5)
        weight_decays = generate_uniform_values(0.0001, 0.00001, 1)
        combs = list(product(learning_rates, weight_decays))
        for lr, wd in combs:
            self.options.append(dict(optimizer=dict(type='SGD', lr=lr, momentum=0.9, weight_decay=wd)))


def built_in_train(config_path, work_dir, **optional_args):
    train_args = [config_path]
    if work_dir:
        train_args += ["--work-dir", work_dir]
    if optional_args:
        additional_options = ["--cfg-options"]
        for listed_config_option in dict_generator(optional_args):
            print(listed_config_option)
            config_option_to_change = f"{'.'.join(listed_config_option[:-1])}={listed_config_option[-1]}"
            additional_options.append(config_option_to_change)
        train_args += additional_options

    # train if changed build-in method (can accept arguments when called).
    if mmdet_train.main.__code__.co_argcount > 0:
        mmdet_train.main(train_args)
    else:
        # runs build-int train (tools/train.py) from console
        run_arg = f"python ../tools/train.py {' '.join(train_args)}"
        os.system(run_arg)


def dict_generator(indict, previous=None):
    """Recursively generates listed data from the nested data structure."""
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


def generate_uniform_values(max_value, min_value, n) -> list:
    if n <= 1:
        return [min_value]
    values = [random.uniform(min_value, max_value) for _ in range(n)]
    return values


def old_train(create_params=False):
    """
    Retrains an pre-existed object detection model and outputs the checkpoints and logs into "train_exports" directory.
    Builds a train dataset from cfg.data.train. Plots the train loss of train epochs and mAP of the validation epochs.

    Args:
        create_params (bool): Creates a combination of multiple params which are used to gradually
                              multiple models.

    Results are stored in train_exports directory.
    """
    from unused.org_config import cfg
    import itertools

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
        plot_logs.plot_all_logs_in_dir("../workdirs/train_exports", recursive=False)