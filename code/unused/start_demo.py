from mmdet.apis import init_detector, train_detector, inference_detector, show_result_pyplot
import mmcv
import torch


def start_demo():
    # Check Pytorch installation

    print(torch.__version__, torch.cuda.is_available())

    # Check MMDetection installation
    import mmdet

    print(mmdet.__version__)

    # Check mmcv installation
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version

    print(get_compiling_cuda_version())
    print(get_compiler_version())

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
