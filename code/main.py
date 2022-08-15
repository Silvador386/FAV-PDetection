from mmdet.apis import init_detector, train_detector, inference_detector, show_result_pyplot
import mmcv
import os
import pdestre
from support import *
from settings import *


def start_demo():
    # Specify the path to model config and checkpoint file
    config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    img = "demo/demo.jpg"  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    # visualize the results in a new window
    model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file='demo/test0.jpg')

    # test a video and show the results
    video = mmcv.VideoReader("demo/demo.mp4")
    for frame in video:
        result = inference_detector(model, frame)
        model.show_result(frame, result, wait_time=1, out_file="demo/test1.jpg")


# checks for paired annotation and video names
def check_pdestre_dataset(new_annotations_folder, new_video_folder):
    annotation_names = files_in_folder(NEW_ANNOTATIONS_FOLDER)
    video_names = files_in_folder(NEW_VIDEO_FOLDER)

    # remove .type suffix
    annotation_type = NEW_ANNOTATION_TYPE
    video_type = NEW_VIDEO_TYPE
    for i, ann in enumerate(annotation_names):
        annotation_names[i] = ann.removesuffix(annotation_type)
    for i, vid in enumerate(video_names):
        video_names[i] = vid.removesuffix(video_type)

    # check paired names
    names_paired = []
    for annotation_name in annotation_names:
        if annotation_name in video_names:
            # remove wrong data pairs:
            if not annotation_name.startswith("._"):
                names_paired.append(annotation_name)

    return names_paired


# converts paired videos to jpgs and annotations to json (COCO-style like)
def convert_pdestre_dataset(new_annotations_folder, new_video_folder, convert_annotations_folder, convert_image_folder):

    names_paired = check_pdestre_dataset(new_annotations_folder, new_video_folder)

    # take each pair, convert video to jpgs, create new COCO-style annotation
    for name in names_paired:
        video_path = new_video_folder + "/" + name + NEW_VIDEO_TYPE
        ann_path = new_annotations_folder + "/" + name + NEW_ANNOTATION_TYPE

        # test if already converted
        likely_image_path = convert_image_folder + "/" + name + "_f0.jpg"
        likely_ann_path = convert_annotations_folder + "/" + name + ".json"

        if not os.path.isfile(likely_image_path):
            convert_video_to_jpg(name, video_path, convert_image_folder)
        if not os.path.isfile(likely_ann_path):
            convert_pdestre_to_coco(ann_path, name, convert_annotations_folder, convert_image_folder)

        # TODO test - one file only
        break


def main():
    convert_pdestre_dataset(NEW_ANNOTATIONS_FOLDER, NEW_VIDEO_FOLDER,
                            CONVERTED_ANNOTATIONS_FOLDER, CONVERTED_IMAGE_FOLDER)

    # Testing image
    config_file = "coco_custom_config.py"
    checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    img = "../Datasets/P-DESTRE/coco_format/videos/08-11-2019-1-1_f10.jpg"

    result = inference_detector(model, img)
    model.show_result(img, result, out_file='../Datasets/P-DESTRE/test.jpg')


def test_other():
    from mid_config import cfg
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    import os.path as osp
    from pdestre import PDESTRE


    convert_pdestre_dataset(NEW_ANNOTATIONS_FOLDER, NEW_VIDEO_FOLDER,
                            CONVERTED_ANNOTATIONS_FOLDER, CONVERTED_IMAGE_FOLDER)

    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = ("person", )

    train_detector(model, datasets, cfg, distributed=False, validate=True, meta=dict())

    # print(cfg.data.train)
    # # Build dataset
    # datasets = [build_dataset(cfg.data.train)]
    #
    # # Build the detector
    # model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"),
    #                         test_cfg=cfg.get("test_cfg"))
    # # Add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES
    #
    # # Create work_dir
    # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # train_detector(model, datasets, cfg, distributed=False, validate=True, meta=dict())

    # Testing video
    # video = mmcv.VideoReader("../Datasets/P-DESTRE/videos/08-11-2019-1-1.MP4")
    # for frame in video:
    #     result = inference_detector(model, frame)
    #     model.show_result(frame, result, wait_time=1, out_file='../demo/test.jpg')
    #     img = cv2.imread('../demo/test.jpg')
    #     cv2.imshow("", img)
    # print(videos)


if __name__ == "__main__":
    main()
    # test_other()
