_base_ = "../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py"


# # 1. dataset settings
_ann_file_train = "../data/P-DESTRE/coco_format/merged/large_train.json"
_ann_file_test = "../data/P-DESTRE/coco_format/merged/large_test.json"
_img_prefix = "../data/P-DESTRE/coco_format/videos/"
_dataset_type = 'CocoDataset'
_classes = ("person",)
_num_classes = len(_classes)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
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

model = dict(
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=_num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

evaluation = dict(metric="bbox", save_best="auto")

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=1e-04)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.01,
    # step=[20, 30]
    )

runner = dict(type='EpochBasedRunner', max_epochs=3)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={'project': 'FAV_PD'},
             interval=50,
             log_checkpoint=False,
             log_checkpoint_metadata=True,
             num_eval_images=0,
             bbox_score_thr=0.3
             )
    ]
)

workflow = [("train", 1), ("val", 1)]

# load_from = "c:/Programming/Python Projects/FAV_PD/code/work_dirs/main_config/latest.pth"
load_from = "../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
