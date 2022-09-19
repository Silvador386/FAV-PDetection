import os
import cv2 as cv2
import json
import mmcv
import random


def files_in_folder(path):
    """
    Return file names of content in the folder.

    Args:
        path (str): A path to the selected folder.

    Returns:
        file_names (list): A list of file names.
    """

    file_names = []
    for _, __, f_names in os.walk(path):
        for file_name in f_names:
            file_names.append(file_name)
        break
    return file_names


def convert_video_to_jpg(video_name, video_path, output_path, frame_rate=10):
    """
    Converts video from video_path to jpg images and stores the in the output_path.
    Jpg file convention:
        output_path/video_name'_f%05d.jpg'

    Args:
        video_name (str): A name of the file (video) to be converted.
        video_path (str): A path to the location where the file (video_name) is located.
        output_path (str): A path to the location where the images will be stored.
        frame_rate (int): Frame rate declares frames should be converted
                          e.g. frame_rate=10 -> each 10. frame will be converted to jpg.
    """

    print(f"Converting video from: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            cv2.imwrite(output_path + "/" + video_name + f"_f{count:05}.jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    print("Converting finished.")


def pdestre_to_coco(ann_path, video_path, current_name, output_folder, image_folder, frame_rate=10):
    """
    Takes annotations at the ann_path, takes video at the video_path. Reads the annotation and for the specific frame
    it creates a jpg image from video which is then stored at the image_folder.

    Args:
        ann_path (str): A path to the annotation.
        video_path (str): A path the the video.
        current_name (str): The name of the file (must be same for the annotation and the video).
        output_folder (str): A path to the folder where the formatted annotation will be stored.
        image_folder (str): A path to the folder where the images will be stored.
        frame_rate (int): Frame rate declares frames should be converted
                          e.g. frame_rate=10 -> each 10. frame will be converted to jpg.
    """

    out_path = output_folder + "/" + current_name + ".json"

    # load video
    print(f"Converting video from: {video_path}")
    vidcap = cv2.VideoCapture(video_path)

    # load annotations as list
    print(f"Converting annotations from: {ann_path}")
    with open(ann_path, "r") as reader:
        ann_list = reader.readlines()

    category = {"id": 0, "name": "person"}  # only one category - person

    final = {"images": [],
             "annotations": [],
             "categories": []}

    final["categories"].append(category)

    previous = 0  # to keep track of frame of the previous annotation line
    img_idx = 0  #
    for i, ann_line in enumerate(ann_list):
        ann_line = ann_line.split(",")
        frame_idx = int(ann_line[0])  # current frame

        if frame_idx % frame_rate == 0:
            image_id = int(current_name.replace("-", "").replace("2019", "") + "0" + str(frame_idx))  # generates img id
            id = int(current_name.replace("-", "").replace("2019", "") + str(i))                      # generates ann id

            if frame_idx > previous:  # new frame in annotations -> create new corresponding jpg
                img_name = current_name + f"_f{frame_idx:05}.jpg"
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
                final["images"].append(image_info)
                previous = frame_idx

            # Store annotation
            bbox = [float(val) for val in ann_line[2:6]]
            area = bbox[2] * bbox[3]
            ann_info = dict(image_id=image_id, bbox=bbox, category_id=0, id=id,
                            area=area, iscrowd=0)
            final["annotations"].append(ann_info)

    # Convert data to json a store them
    json_out = json.dumps(final)
    with open(out_path, "w") as outfile:
        print(f"Storing annotations to json at: {out_path}")
        outfile.write(json_out)


def select_json_to_merge(json_folder, num_files=10, shuffle=False, divide=False):
    """
    Takes in a folder with .json annotations. Selects files to be merged and returns tuple of train and test
    file names in the ration of 10:1.

    Args:
        json_folder (str): A path of the folder.
        num_files (int): The total number of files from which to pick.
        shuffle (bool): If the order of files should be shuffled.
        divide (bool): If false only train files will be picked.

    Return:
        (train_filenames, test_filenames): tuple of lists of file names.
    """

    train_filenames, test_filenames = [], []
    files = files_in_folder(json_folder)

    if 5 > num_files or num_files > len(files):
        print(f"Number of files changed to maximum ({len(files)}).")
        num_files = len(files)

    for i in range(num_files):
        if not shuffle:
            file = files.pop()
        else:
            file = random.choice(files)
            files.remove(file)

        if divide and i < num_files / 10:  # Picks test data
            test_filenames.append(file)
        else:
            train_filenames.append(file)

    return train_filenames, test_filenames


def merge_json_files(json_folder, json_files, name, output_folder, overwrite=False):
    """
    Merges all given json files in json_folder to a new json file that is stored in output_folder under the new name.

    Args:
        json_folder (str): A path of the folder.
        json_files (list): A list of selected files to be merged in to a single .json file.
        name (str): A name of the new file.
        output_folder (str): A path to the folder where the formatted annotation will be stored.
        overwrite (bool): Overwrites any pre-existent merged file.

    """
    out_path = output_folder + "/" + name + ".json"
    result = {}
    # Checks if the file already exists
    if os.path.isfile(out_path) and not overwrite:
        print(f"{out_path} already exists.")
        return

    for file in json_files:
        if file.endswith(".json") and file != name + ".json":
            with open(json_folder + "/" + file, "r") as reader:
                current = json.load(reader)
                if len(list(result.keys())) == 0:
                    for key in list(current.keys()):
                        result[key] = []
                # for unique values (image, annotations)
                for var in list(current.keys())[:-1]:
                    result[var].extend(current[var])
                # for same values (categories)
                result["categories"] = current["categories"]

    with open(out_path, "w") as outfile:
        json.dump(result, outfile)

