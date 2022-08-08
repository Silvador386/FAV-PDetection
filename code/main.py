from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import create_dataset
from support import *


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
    # model.show_result(img, result, out_file='demo/test0.jpg')
    #
    # # test a video and show the results
    # video = mmcv.VideoReader("demo/demo.mp4")
    # for frame in video:
    #     result = inference_detector(model, frame)
    #     model.show_result(frame, result, wait_time=1, out_file="demo/test1.jpg")



def main():
    text_path = "../Datasets/P-DESTRE/original/annotation"
    annotations = load_txt_folder(text_path)

    video_path = "../Datasets/P-DESTRE/original/videos"
    out_path = "../Datasets/P-DESTRE/COCO_Format/videos"
    video_names = files_in_folder(video_path)

    pairs = [(name, ann) for name, ann in zip(annotations, video_names)]


    # Testing image
    config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # img = "../Datasets/P-DESTRE/video_jpgs/08-11-2019-1-1_0.jpg"
    #
    # result = inference_detector(model, img)
    # model.show_result(img, result, out_file='../demo/test.jpg')

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