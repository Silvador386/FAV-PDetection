import os
import cv2
import json
import mmcv

from settings import *
from utils import files_in_folder


def pdestre_anns_to_coco(ann_path, video_path, file_name, output_folder, image_folder, frame_rate=10):
    """
    Takes annotations at the ann_path, takes video at the video_path. Reads the annotation and for the specific frame
    it creates a jpg image from video which is then stored at the image_folder.

    Args:
        ann_path (str): A path to the annotation.
        video_path (str): A path the the video.
        file_name (str): The name of the file (must be same for the annotation and the video).
        output_folder (str): A path to the folder where the formatted annotation will be stored.
        image_folder (str): A path to the folder where the images will be stored.
        frame_rate (int): Frame rate declares frames should be converted
                          e.g. frame_rate=10 -> each 10. frame will be converted to jpg.
    """

    output_path = output_folder + "/" + file_name + ".json"

    # load video
    print(f"Converting video from: {video_path}")
    vidcap = cv2.VideoCapture(video_path)

    # load annotations as list
    print(f"Converting annotations from: {ann_path}")
    with open(ann_path, "r") as reader:
        ann_list = reader.readlines()

    categories = {"id": 0, "name": "person"}  # only one category - person

    coco_json = {"images": [],
                 "annotations": [],
                 "categories": []
                 }

    coco_json["categories"].append(categories)

    previous_frame_idx = 0  # to keep track of frame of the previous annotation line
    img_idx = 0  #
    for i, ann_line in enumerate(ann_list):
        ann_line_values = ann_line.split(",")
        frame_idx = int(ann_line_values[0])  # current frame

        if frame_idx % frame_rate == 0:
            image_id = int(file_name.replace("-", "").replace("2019", "") + "0" + str(frame_idx))  # generates img id
            ann_id = int(file_name.replace("-", "").replace("2019", "") + str(i))                      # generates ann id

            if frame_idx > previous_frame_idx:  # new frame in annotations -> create new corresponding jpg
                img_name = file_name + f"_f{frame_idx:05}.jpg"
                img_file_name = image_folder + "/" + img_name

                # Checks if the image already exists
                if not os.path.isfile(img_file_name):
                    # Convert the image
                    skip = False  # skips the annotation line if the read was unsuccessful
                    while img_idx < frame_idx:  # iterates util the current frame is found
                        success, image = vidcap.read()
                        if not success:
                            print(f"vidcap.read() not successful!\n Filename: {img_name}")
                            skip = True
                            break
                        img_idx += 1
                        if img_idx == frame_idx:
                            cv2.imwrite(img_file_name, image)  # save frame as JPEG file
                    if skip:
                        continue

                # store the image info
                image = mmcv.imread(img_file_name)
                width, height = image.shape[:2]
                image_info = dict(file_name=img_name, width=width, height=height, id=image_id)
                coco_json["images"].append(image_info)
                previous_frame_idx = frame_idx

            # Store annotation
            bbox = [float(val) for val in ann_line_values[2:6]]
            area = bbox[2] * bbox[3]
            formatted_ann = dict(image_id=image_id, bbox=bbox, category_id=0, id=ann_id,
                                 area=area, iscrowd=0)
            coco_json["annotations"].append(formatted_ann)

    # Convert data to json a store them
    json_output = json.dumps(coco_json)
    with open(output_path, "w") as outfile:
        print(f"Storing annotations to json at: {output_path}")
        outfile.write(json_output)


def check_pairs_in_dataset(new_annotations_folder, new_video_folder):
    """
    Checks for paired annotation and video names.

    Args:
        new_annotations_folder (str): A path to the folder with annotations.
        new_video_folder (str): A path to folder with videos.

    Returns:
         names_paired: A list of strings (names of files that are paired).
    """

    annotation_names = files_in_folder(new_annotations_folder)
    video_names = files_in_folder(new_video_folder)

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


def convert_pdestre_anns(new_annotations_folder, new_video_folder, convert_annotations_folder, convert_image_folder,
                         frame_rate, override_checks=False):
    """
    Converts paired videos to jpg images and annotations to coco format json files.

    Args:
        new_annotations_folder (str): A path to the folder with annotations.
        new_video_folder (str): A path to folder with videos.
        convert_annotations_folder (str): A path to the folder for formatted annotations.
        convert_image_folder (str): A path to the folder for images got from the video.
        frame_rate (int): Determines which frames should be converted (each 10th for instance).
        override_checks (bool): Overrides checks if annotations are already present.

    """

    names_paired = check_pairs_in_dataset(new_annotations_folder, new_video_folder)

    # take each paired name, convert video to jpgs, create new COCO-style annotation
    for i, name in enumerate(names_paired):
        video_path = new_video_folder + "/" + name + NEW_VIDEO_TYPE
        ann_path = new_annotations_folder + "/" + name + NEW_ANNOTATION_TYPE

        # test if already converted
        likely_ann_path = convert_annotations_folder + "/" + name + ".json"

        # Checks if the annotation file already exists. If so, skips the conversion.
        if os.path.isfile(likely_ann_path) and not override_checks:
            print(f"{likely_ann_path} already exists.")
            continue

        # Converts an annotation file with corresponding video.
        pdestre_anns_to_coco(ann_path, video_path, name, convert_annotations_folder, convert_image_folder,
                             frame_rate=frame_rate)
