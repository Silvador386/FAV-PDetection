import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

from pdestre_conversion import *
from settings import *


def test(config_file="../configs/my_config/main_config_large.py",
         checkpoint_file="./work_dirs/main_config_clc_loss/latest.pth"):
    """
    Tests current model on the station_control data and some imgs from PDestre dataset.
    The interference is stored in the "out_prefix" folder.

    """

    # Control - original model
    # config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
    # checkpoint_file = "../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
    # out_prefix = "../results/station_BasicFasterRCNN"

    # Test current model
    # checkpoint_file = "../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"


    output_prefix = "../results/station_pdestre"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    test_on_city_data(model, output_prefix)
    # test_model_on_pdestre_imgs(model)


def test_on_city_data(model, output_prefix ):
    img_prefix = "../data/city/images"

    image_names = files_in_folder(img_prefix)

    # For parts where the tram is present.
    finer_zones = [(200, 300), (1400, 1480)]

    # Test on images (images are each 10th frame of the video)
    for i, image_name in enumerate(image_names):
        image_rate = 20
        if any([zone[0] < i < zone[1] for zone in finer_zones]):
            image_rate = 2
        if i % image_rate == 0:
            img = mmcv.imread(img_prefix + "/" + image_name)
            result = inference_detector(model, img)
            model.show_result(img, result, out_file=output_prefix + f"/{i:05}.jpg")


def test_model_on_pdestre_imgs(model):
    # Randomly picked images to be shown on.
    pdestre_examples = [converted_img_dir + "/12-11-2019-2-1_f00160.jpg",
                        converted_img_dir + "/13-11-2019-1-2_f01560.jpg",
                        converted_img_dir + "/10-07-2019-1-1_f00670.jpg",
                        converted_img_dir + "/18-07-2019-1-2_f00280.jpg",
                        converted_img_dir + "/08-11-2019-2-1_f00360.jpg",
                        converted_img_dir + "/08-11-2019-1-2_f01090.jpg"
                        ]

    for e in pdestre_examples:
        img = mmcv.imread(e)
        result = inference_detector(model, img)
        show_result_pyplot(model, img, result)
