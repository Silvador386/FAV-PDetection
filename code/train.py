import copy
import time
import mmcv
import os.path as osp

from mmdet.utils import collect_env
from mmdet.utils.logger import get_root_logger
from mmdet.utils.util_distribution import get_device

from mmcv import Config
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def train(learning_rate=None, weight_decay=None, epochs=None, optimizer=None):
    config = "../configs/my_config/test_config.py"
    checkpoint = "../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
    work_dir = "../work_dirs"

    ann_train_file = "../data/P-DESTRE/coco_format/merged/micro.json"
    ann_test_file = "../data/P-DESTRE/coco_format/merged/micro.json"
    img_prefix = "../data/P-DESTRE/coco_format/videos/"

    cfg = Config.fromfile(config)

    cfg.data.train.ann_file = ann_train_file
    cfg.data.train.img_prefix = img_prefix

    cfg.data.test.ann_file = ann_test_file
    cfg.data.test.img_prefix = img_prefix

    cfg.data.val.ann_file = ann_test_file
    cfg.data.val.img_prefix = img_prefix

    cfg.load_from = checkpoint
    cfg.work_dir = work_dir

    cfg.gpu_ids = range(1)

    if learning_rate:
        cfg.optimizer.lr = learning_rate
    if weight_decay:
        cfg.optimizer.weight_decay = weight_decay
    if epochs:
        cfg.runner.max_epochs = epochs
    if optimizer:
        cfg.optimizer.type = optimizer
        if optimizer == "Adam":
            cfg.optimizer.pop("momentum")

    # create work_dir_path
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(device=cfg.device)
    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        validate=True,
        timestamp=timestamp,
        meta=meta)


if __name__ == "__main__":
    train()






