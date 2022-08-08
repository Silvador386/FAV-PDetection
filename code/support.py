import os
import cv2


# Return file names of content in the folder
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
def convert_annotation(annotation_file, image_path, out_path):
    ann = []
    for ann_line in annotation_file:
        data = ann_line.split(",")




"""
Convert video to jpg
Jpg file convention:
    Datasets/P-DESTRE/COCO_Format/videos/'video_name'_%d.jpg'
    Video_name must be same as annotation name
"""
def video_to_jpg(video_name, output_path):
    # filter incompatible files
    if not video_name.startswith(".") and video_name == "08-11-2019-1-1.MP4":
        vidcap = cv2.VideoCapture(video_name)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(output_path + "/" + video_name[:-4] + "_%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1


# Loads videos from specified folder
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
                    cv2.imwrite(out_path + "/" + video_name[:-4] + "_%d.jpg" % count, image)  # save frame as JPEG file
                    success, image = vidcap.read()
                    # print('Read a new frame: ', success)
                    count += 1

