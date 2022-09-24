_base_ = "../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"


# # 1. dataset settings
_ann_file_train = "c:/Programming/Python Projects/FAV_PD/data/P-DESTRE/coco_format/merged/micro_train.json"
_ann_file_test = "c:/Programming/Python Projects/FAV_PD/data/P-DESTRE/coco_format/merged/micro_train.json"
_img_prefix = "c:/Programming/Python Projects/FAV_PD/data/P-DESTRE/coco_format/videos/"
_dataset_type = 'CocoDataset'
_classes = ("person",)
_num_classes = len(_classes)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=_dataset_type,
        # explicitly add your class names to the field `classes`
        classes=_classes,
        ann_file=_ann_file_train,
        img_prefix=_img_prefix),
    val=dict(
        type=_dataset_type,
        # explicitly add your class names to the field `classes`
        classes=_classes,
        ann_file=_ann_file_test,
        img_prefix=_img_prefix),
    test=dict(
        type=_dataset_type,
        # explicitly add your class names to the field `classes`
        classes=_classes,
        ann_file=_ann_file_test,
        img_prefix=_img_prefix))

# 2. model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.000)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=4,
    warmup_ratio=0.1,
    step=[20, 40])

runner = dict(type='EpochBasedRunner', max_epochs=50)

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

workflow = [("train", 1), ("val", 1)]

load_from = "c:/Programming/Python Projects/FAV_PD/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
