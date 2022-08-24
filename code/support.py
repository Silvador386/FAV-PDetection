import os
import cv2
import json
import mmcv


# Return file names of content in the folder
def files_in_folder(path):
    file_names = []
    for _, __, f_names in os.walk(path):
        for file_name in f_names:
            file_names.append(file_name)
    return file_names


# Converts video to jpg
# Jpg file convention:
#     Datasets/P-DESTRE/coco_format/videos/'video_name'_f%05d.jpg'
#     Video_name must be same as annotation name
def convert_video_to_jpg(video_name, video_path, output_path, frame_rate=10):
    print(f"Converting video from: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            cv2.imwrite(output_path + "/" + video_name + f"_f{count:05}.jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
    print("Converting finished.")


# Converts annotation file to the support structure an saves it into another file.
def convert_pdestre_to_coco(ann_path, current_name, output_folder, image_folder):
    # load annotation as list
    print(f"Converting annotations from: {ann_path}")
    with open(ann_path, "r") as reader:
        ann_list = reader.readlines()

    category = {"id": 0, "name": "person"}  # only one category - person

    final = {"images": [],
             "annotations": [],
             "categories": []}

    final["categories"].append(category)

    previous = int(ann_list[0][0]) - 1  # checks if on a new frame
    for id, ann_line in enumerate(ann_list):
        ann_line = ann_line.split(",")

        # select data
        frame_idx = int(ann_line[0])  # current frame
        # id = int(ann_line[1])
        bbox = [float(val) for val in ann_line[2:6]]
        area = bbox[2] * bbox[3]

        # load correct image
        if frame_idx > previous:
            img_file_name = current_name + f"_f{frame_idx:05}.jpg"
            image = mmcv.imread(image_folder + "/" + img_file_name)
            width, height = image.shape[:2]
            image_info = dict(file_name=img_file_name, width=width, height=height, id=frame_idx)
            final["images"].append(image_info)
            previous += 1

        # load annotation
        # TODO check if the empty params have any effect on model
        ann_info = dict(image_id=frame_idx, bbox=bbox, category_id=0, id=id,
                        area=area, iscrowd=0)
        final["annotations"].append(ann_info)

    # convert data to json a store them
    json_out = json.dumps(final)
    out_path = output_folder + "/" + current_name + ".json"
    with open(out_path, "w") as outfile:
        print(f"Storing annotations to json at: {out_path}")
        outfile.write(json_out)


def convert_dataset(ann_path, video_path, current_name, output_folder, image_folder, frame_rate=10):
    """
    1. Take annotations at ann_path, take video at video_path
    2. Read annotation and for specific frame (each 10th), create specific frame jpg from video and move to the next.
    """
    out_path = output_folder + "/" + current_name + ".json"
    # check if annotation already exists
    if os.path.isfile(out_path):
        return
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

    previous = 0
    img_idx = 0
    for i, ann_line in enumerate(ann_list):
        ann_line = ann_line.split(",")
        frame_idx = int(ann_line[0])  # current frame
        if frame_idx % frame_rate == 0:
            image_id = int(current_name.replace("-", "").replace("2019", "") + "0" + str(frame_idx))
            id = int(current_name.replace("-", "").replace("2019", "") + str(i))
            if frame_idx > previous:  # new frame in annotations -> create corresponding jpg
                img_name = current_name + f"_f{frame_idx:05}.jpg"
                img_file_name = image_folder + "/" + img_name

                # Checks if the image already exists
                if not os.path.isfile(img_file_name):
                    # Convert the image
                    while img_idx < frame_idx:
                        success, image = vidcap.read()
                        if not success:
                            print("vidcap.read() not successful!")
                        img_idx += 1
                        if img_idx == frame_idx:
                            cv2.imwrite(img_file_name, image)  # save frame as JPEG file

                # store the image info
                image = mmcv.imread(img_file_name)
                width, height = image.shape[:2]
                image_info = dict(file_name=img_name, width=width, height=height, id=image_id)
                final["images"].append(image_info)
                previous = frame_idx

            # store annotation
            bbox = [float(val) for val in ann_line[2:6]]
            area = bbox[2] * bbox[3]
            # TODO check if the empty params have any effect on model
            ann_info = dict(image_id=image_id, bbox=bbox, category_id=0, id=id,
                            area=area, iscrowd=0)
            final["annotations"].append(ann_info)

    # convert data to json a store them
    json_out = json.dumps(final)
    with open(out_path, "w") as outfile:
        print(f"Storing annotations to json at: {out_path}")
        outfile.write(json_out)


# Merges all json files in the folder to a new json file.
def merge_json_files(json_folder, name, out_folder):
    files = files_in_folder(json_folder)
    out_path = out_folder + "/" + name + ".json"
    result = {}
    # Checks if the file already exists
    if os.path.isfile(out_path):
        print(f"{out_path} already exists.")
        return

    for file in files:
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


