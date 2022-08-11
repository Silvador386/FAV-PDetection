_base_ = '../configs/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py'


# 1. dataset settings
ann_file = "../Datasets/P-DESTRE/coco_format/annotations/08-11-2019-1-1.json"
img_prefix = "../Datasets/P-DESTRE/coco_format/videos"
dataset_type = 'CocoDataset'
classes = ("person", )
num_classes = len(classes)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=ann_file,
        img_prefix=img_prefix),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=ann_file,
        img_prefix=img_prefix),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=ann_file,
        img_prefix=img_prefix))

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 5.
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',

                num_classes=num_classes),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=num_classes),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=num_classes)],
    # explicitly over-write all the `num_classes` field from default 80 to 5.
    # https://github.com/open-mmlab/mmdetection/issues/4364
    mask_head=dict(type='FCNMaskHead', num_classes=num_classes)))
