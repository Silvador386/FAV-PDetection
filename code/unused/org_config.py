from mmcv import Config
from mmdet.apis import set_random_seed

"""     Configuration of the train model     """


cfg = Config.fromfile("../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py")

_data_root = '../data/P-DESTRE/coco_format/'
_ann_file_train = "merged/mini_train.json"
_ann_file_test = "merged/mini_train.json"
_img_prefix = "videos/"

# cfg.fp16 = dict(loss_scale_512.)
cfg.device = "cuda"
cfg.classes = ("person", )
cfg.dataset_type = "CocoDataset"

"""    Set data     """
cfg.data.test.type = 'CocoDataset'
cfg.data.test.data_root = _data_root
cfg.data.test.ann_file = _ann_file_train
cfg.data.test.img_prefix = _img_prefix
cfg.data.test.classes = cfg.classes

cfg.data.train.type = 'CocoDataset'
cfg.data.train.data_root = _data_root
cfg.data.train.ann_file = _ann_file_train
cfg.data.train.img_prefix = _img_prefix
cfg.data.train.classes = cfg.classes

cfg.data.val.type = 'CocoDataset'
cfg.data.val.data_root = _data_root
cfg.data.val.ann_file = _ann_file_test
cfg.data.val.img_prefix = _img_prefix
cfg.data.val.classes = cfg.classes

""" Default train settings  """
# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 1

cfg.load_from = "../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
# cfg.load_from = "../checkpoints/results/favorites/s32_e2_lr1,6e4/latest.pth
cfg.work_dir = "../../workdirs/train_exports"

cfg.optimizer = dict(type='SGD', lr=0.0016, momentum=0.9, weight_decay=0.0000)
# cfg.optimizer_config = dict(grad_clip=None)
cfg.lr_config.warmup = None
cfg.workflow = [('train', 1)]

# lr_config = dict(
#     policy='cyclic',
#     target_ratio=(10, 1e-4),
#     cyclic_times=1,
#     step_ratio_up=0.4,
# )
# # learning policy
# cfg.lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[7])

cfg.runner = dict(type='EpochBasedRunner', max_epochs=50)

cfg.evaluation.metric = "bbox"
cfg.evaluation.interval = 4
cfg.checkpoint_config = dict(interval=16)

# """ Logs settigns   """
cfg.log_config.interval = 1

# """ GPU/Seed """
cfg.data.samples_per_gpu = 1
cfg.data.workers_per_gpu = 1
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
