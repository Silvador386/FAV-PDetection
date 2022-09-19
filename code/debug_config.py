_base_ = "../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"

# # 1. dataset settings
ann_file = "../data/P-DESTRE/coco_format/merged/mini_train.json"
_img_prefix = "../data/P-DESTRE/coco_format/videos/"
_dataset_type = 'CocoDataset'
classes = ("person",)
_num_classes = len(classes)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=_dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=ann_file,
        img_prefix=_img_prefix),
    val=dict(
        type=_dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=ann_file,
        img_prefix=_img_prefix),
    test=dict(
        type=_dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=ann_file,
        img_prefix=_img_prefix))

# 2. model settings
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',

                num_classes=_num_classes),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `_num_classes` field from default 80 to 5.
                num_classes=_num_classes),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `_num_classes` field from default 80 to 5.
                num_classes=_num_classes)],
    # explicitly over-write all the `_num_classes` field from default 80 to 5.
    # https://github.com/open-mmlab/mmdetection/issues/4364
    mask_head=dict(type='FCNMaskHead', num_classes=_num_classes)))


model = dict(
    roi_head=dict(
        bbox_head=dict(type='Shared2FCBBoxHead', num_classes=_num_classes),
        mask_head=dict(type='FCNMaskHead', num_classes=_num_classes)))