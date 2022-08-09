import os
import cv2
import json

# Return file names of content in the folder
import mmcv


def files_in_folder(path):
    file_names = []
    for _, __, f_names in os.walk(path):
        for file_name in f_names:
            # filter incompatible files
            if not file_name.startswith("."):
                file_names.append(file_name)
    return file_names


# Loads text files from specified folder
def load_txt_folder(path):
    annotations = {}
    file_names = files_in_folder(path)
    for file_name in file_names:
        with open(path + "/" + file_name) as f:
            annotations[file_name] = [line for line in f.readlines()]

    return annotations


# Converts annotation file to the support structure an saves it into another file.
def convert_pdestre_to_coco(ann_list, out_file, image_prefix):
    image_folder = "../Datasets/P-DESTRE/coco_format/videos/"
    current_name = "08-11-2019-1-1"
    output_folder = "../Datasets/P-DESTRE/coco_format/annotations/"

    category = {"id": 0, "name": "person"}  # only one category - person

    final = {"images": [],
             "annotations": [],
             "categories": []}

    final["categories"].append(category)

    previous = 0  # checks if on a new frame
    for ann_line in ann_list:
        ann_line = ann_line.split(",")

        # select data
        frame_idx = int(ann_line[0])  # current frame
        id = int(ann_line[1])
        bbox = [float(val) for val in ann_line[2:6]]

        # load correct image
        if frame_idx > previous:
            img_file_name = image_folder + current_name + f"_f{frame_idx}.jpg"
            image = mmcv.imread(img_file_name)
            width, height = image.shape[:2]
            image_info = dict(filename=img_file_name, width=width, height=height, id=frame_idx)
            final["images"].append(image_info)
            previous += 1
        # load annotation
        # TODO check if the empty params have any effect on model
        ann_info = dict(image_id=frame_idx, bbox=bbox, category_id=0, id=id,
                        segmentation=[], area=0, iscrowd=0)
        final["annotations"].append(ann_info)

    # convert data to json a store them
    json_out = json.dumps(final)
    with open(output_folder + current_name + ".json", "w") as outfile:
        outfile.write(json_out)


"""
Convert video to jpg
Jpg file convention:
    Datasets/P-DESTRE/coco_format/videos/'video_name'_f%d.jpg'
    Video_name must be same as annotation name
"""


def video_to_jpg(video_path, output_path):
    # filter incompatible files
    video_name = video_path.split("/")[-1]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(output_path + "/" + video_name[:-4] + "_f%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1


# Loads videos from specified folder and converts them to jpgs and stores them into specified folder
def convert_videos_to_jpgs(path, out_path):
    for _, __, video_files in os.walk(path):
        for video_name in video_files:
            # filter incompatible files
            if not video_name.startswith(".") and video_name == "08-11-2019-1-1.MP4":
                video_path = path + "/" + video_name

                vidcap = cv2.VideoCapture(video_path)
                success, image = vidcap.read()
                count = 0
                while success:
                    cv2.imwrite(out_path + "/" + video_name[:-4] + "_f%d.jpg" % count, image)  # save frame as JPEG file
                    success, image = vidcap.read()
                    # print('Read a new frame: ', success)
                    count += 1

