import json
import os
import random
from pdestre_conversion import convert_pdestre_anns
from utils import files_in_folder, write_to_json


def convert_and_merge_pdestre(ann_dir,
                              video_dir,
                              converted_ann_dir,
                              converted_img_dir,
                              merged_dir,
                              frame_rate=20
                              ):
    """
    Converts P-DESTRE dataset by converting videos to images and text annotations to coco formatted .json files.
    Then the annotations are merged into large and small variant of train and test datasets.
    """
    print("\nConverting...\n")
    # Converts every video to images and every annotation to coco format .json file
    convert_pdestre_anns(ann_dir, video_dir,
                         converted_ann_dir, converted_img_dir, frame_rate=frame_rate, overwrite=True)

    print("\nMerging...\n")
    # Create pdestre_large
    merge_dataset(converted_ann_dir, merged_dir, name="large", num_files=75)


def merge_dataset(converted_ann_dir, merged_dir, name, num_files, shuffle=True, overwrite=True):
    train_anns_name = f"{name}_train"
    test_anns_name = f"{name}_test"
    train_files, test_files = select_jsons_to_merge(converted_ann_dir, num_files=num_files, shuffle=shuffle)
    merge_json_files(converted_ann_dir, train_files,
                     train_anns_name, merged_dir)
    merge_json_files(converted_ann_dir, test_files,
                     test_anns_name, merged_dir)


def select_jsons_to_merge(json_dir, num_files=10, shuffle=False):
    """
    Takes in a folder with .json annotations. Selects files to be merged and returns tuple of train and test
    file names in the ration of 10:1.

    Args:
        json_dir (str): A path of the folder.
        num_files (int): The total number of files from which to pick.
        shuffle (bool): If the order of files should be shuffled.

    Return:
        (train_filenames, test_filenames): tuple of lists of file names.
    """

    train_filenames, test_filenames = [], []
    json_files = files_in_folder(json_dir)

    if 5 > num_files or num_files > len(json_files):
        print(f"Number of files changed to maximum ({len(json_files)}).")
        num_files = len(json_files)

    for i in range(num_files):
        if not shuffle:
            file = json_files.pop()
        else:
            file = random.choice(json_files)
            json_files.remove(file)

        if i < num_files / 10:
            test_filenames.append(file)
        else:
            train_filenames.append(file)

    return train_filenames, test_filenames


def merge_json_files(json_dir, json_files, name, output_dir):
    """
    Merges all given json files in json_dir to a new json file that is stored in output_dir under the new name.

    Args:
        json_dir (str): A path of the folder.
        json_files (list): A list of selected files to be merged in to a single .json file.
        name (str): A name of the new file.
        output_dir (str): A path to the folder where the formatted annotation will be stored.

    """
    output_path = f"{output_dir}/{name}.json"
    coco_json = {}

    for file in json_files:
        if file.endswith(".json") and file != name + ".json":
            with open(json_dir + "/" + file, "r") as reader:
                current = json.load(reader)
                if len(list(coco_json.keys())) == 0:
                    for key in list(current.keys()):
                        coco_json[key] = []
                # for unique values (image, annotations)
                for var in list(current.keys())[:-1]:
                    coco_json[var].extend(current[var])
                # for same values (categories)
                coco_json["categories"] = current["categories"]

    write_to_json(coco_json, output_path)
