from mmdet.apis import init_detector, train_detector, inference_detector, show_result_pyplot
from support import *
from settings import *


def start_demo():
    # Check Pytorch installation
    import torch

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
def convert_pdestre_dataset(new_annotations_folder, new_video_folder, convert_annotations_folder, convert_image_folder,
                            frame_rate, override_checks=False):

    names_paired = check_pdestre_dataset(new_annotations_folder, new_video_folder)

    # take each pair, convert video to jpgs, create new COCO-style annotation
    for i, name in enumerate(names_paired):
        video_path = new_video_folder + "/" + name + NEW_VIDEO_TYPE
        ann_path = new_annotations_folder + "/" + name + NEW_ANNOTATION_TYPE

        # test if already converted
        likely_image_path = convert_image_folder + "/" + name + "_f00000.jpg"
        likely_ann_path = convert_annotations_folder + "/" + name + ".json"

        # if not os.path.isfile(likely_image_path):
        #     convert_video_to_jpg(name, video_path, convert_image_folder)
        # if not os.path.isfile(likely_ann_path):
        #     convert_pdestre_to_coco(ann_path, name, convert_annotations_folder, convert_image_folder)

        # Checks if the annotation file already exists. If so, skips the conversion.
        if os.path.isfile(likely_ann_path) and not override_checks:
            print(f"{likely_ann_path} already exists.")
            continue
        # Converts an annotation file with corresponding video.
        convert_dataset(ann_path, video_path, name, convert_annotations_folder, convert_image_folder, frame_rate=frame_rate)

        # TODO test - one file only
        if i == 10:
            break


def prepare_data():
    convert_pdestre_dataset(NEW_ANNOTATIONS_FOLDER, NEW_VIDEO_FOLDER,
                            CONVERTED_ANNOTATIONS_FOLDER, CONVERTED_IMAGE_FOLDER, frame_rate=20)

    merge_json_files(CONVERTED_ANNOTATIONS_FOLDER, "pdestre_large", "../Datasets/P-DESTRE/coco_format/large")


def main():
    prepare_data()
    train()
    # test()


def train():
    from mid_config import cfg
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector

    datasets = [build_dataset(cfg.data.train)]  # mmdet/datasets/ utils.py - change __check_head

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = ("person", )

    train_detector(model, datasets, cfg, distributed=False, validate=True)

    img = mmcv.imread("../Datasets/P-DESTRE/coco_format/videos/08-11-2019-1-1_f00010.jpg")

    model.cfg = cfg
    result = inference_detector(model, img)
    show_result_pyplot(model, img, result)


def test():
    from mid_config import cfg
    from mmdet.models import build_detector

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    img = mmcv.imread("../Datasets/P-DESTRE/coco_format/videos/08-11-2019-1-1_f00010.jpg")

    model.cfg = cfg
    result = inference_detector(model, img)
    show_result_pyplot(model, img, result)
    model.show_result(img, result, out_file="../Datasets/P-DESTRE/test0.jpg")

    # Testing image
    config_file = "unused/coco_custom_config.py"
    checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    img = "../Datasets/P-DESTRE/coco_format/videos/08-11-2019-1-1_f00010.jpg"

    result = inference_detector(model, img)
    model.show_result(img, result, out_file='../Datasets/P-DESTRE/test.jpg')


if __name__ == "__main__":

    main()
    # train()
    # test()

