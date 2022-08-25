from mmcv import Config
from mmdet.apis import set_random_seed


cfg = Config.fromfile("../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py")

_data_root = '../Datasets/P-DESTRE/coco_format/'
_ann_file = "large/pdestre_large.json"
_img_prefix = "videos/"

# cfg.fp16 = dict(loss_scale_512.)
cfg.device = "cuda"
cfg.classes = ("person", )
cfg.dataset_type = "CocoDataset"

cfg.data.test.type = 'CocoDataset'
cfg.data.test.data_root = _data_root
cfg.data.test.ann_file = _ann_file
cfg.data.test.img_prefix = _img_prefix
cfg.data.test.classes = cfg.classes


cfg.data.train.type = 'CocoDataset'
cfg.data.train.data_root = _data_root
cfg.data.train.ann_file = _ann_file
cfg.data.train.img_prefix = _img_prefix
cfg.data.train.classes = cfg.classes

cfg.data.val.type = 'CocoDataset'
cfg.data.val.data_root = _data_root
cfg.data.val.ann_file = _ann_file
cfg.data.val.img_prefix = _img_prefix
cfg.data.val.classes = cfg.classes

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 1
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch.
cfg.load_from = "../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
# cfg.load_from = "./train_exports/latest.pth"


# Set up working dir to save files and logs.
cfg.work_dir = "./train_exports"

# The original learning rate is set for 8-GPU training.
# We divide it by 8 since we only use 1 GPU.

# cfg.optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# cfg.optimizer_config = dict(grad_clip=None)
# # learning policy
# cfg.lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[7])

# cfg.optimizer.lr = 0.02 / 8
# cfg.lr_config.warmup = None
# cfg.log_config.interval = 50

cfg.runner = dict(type='EpochBasedRunner', max_epochs=1)
cfg.checkpoint_config = dict(interval=50)

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = "bbox"
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 3

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
