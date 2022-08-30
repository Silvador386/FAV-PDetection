import os
import cv2
import json
import mmcv
import random


def files_in_folder(path):
    """
    Return file names of content in the folder
    """
    file_names = []
    for _, __, f_names in os.walk(path):
        for file_name in f_names:
            file_names.append(file_name)
    return file_names


def convert_video_to_jpg(video_name, video_path, output_path, frame_rate=10):
    """
    Converts video from video_path to jpgs
    Jpg file convention:
        output_path/video_name'_f%05d.jpg'

    Frame rate declares frames should be converted e.g. frame_rate=10 -> each 10. frame will be converted to jpg.
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
    1. Take annotations at ann_path, take video at video_path,
    2. Checks if already present if so, skips the step.
    3. Read annotation and for specific frame (each 10th), create specific frame jpg from video and move to the next.

    Frame rate declares frames should be converted aka frame_rate=10 -> each 10. frame will be converted to jpg.
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
                            print("vidcap.read() not successful!")
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
    1. Takes in a folder with .json annotations.
    2. Selects files to be merged.
    3. Returns tuple (train_filenames, test_filenames)
       * divide=False: test_filenames = []
       * divide=True:  test_filenames = [one 10th of num_files, at least 1]
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


def merge_json_files(json_folder, json_files, name, out_folder, overwrite=False):
    """
    Merges all given json files in json_folder to a new json file.
    """
    out_path = out_folder + "/" + name + ".json"
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


