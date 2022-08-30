import mmcv
import json


def convert_pdestre_to_coco(ann_path, current_name, output_folder, image_folder):
    """
    Converts annotation file to the supported coco structure an saves it into .json file.
    """
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