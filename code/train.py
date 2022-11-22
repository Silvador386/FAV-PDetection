import random
import os
from itertools import product


from mmdet.apis import init_detector
from tools import train as mmdet_train


import sanity_checks


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

        # plot_logs.plot_all_logs_in_dir(self.work_dir)

        # creates images with predictions from the model
        # output_path = "../results/test_json_anns/large"
        # self.test_model_checkpoint_by_img_inference(output_dir=output_path)

    def test_model_checkpoint_by_img_inference(self, output_dir):
        checkpoint_latest = f"{self.work_dir}/latest.pth"
        model = init_detector(self.config_path,
                              checkpoint=checkpoint_latest, device='cuda:0')
        sanity_checks.test_json_anns(config_path=self.config_path, output_dir=output_dir,
                                     model=model, max_num=5)

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
