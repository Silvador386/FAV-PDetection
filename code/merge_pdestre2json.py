import json
import os
import random

from utils import files_in_folder, write_to_json


def select_jsons_to_merge(json_dir, num_files=10, shuffle=False, divide=False):
    """
    Takes in a folder with .json annotations. Selects files to be merged and returns tuple of train and test
    file names in the ration of 10:1.

    Args:
        json_dir (str): A path of the folder.
        num_files (int): The total number of files from which to pick.
        shuffle (bool): If the order of files should be shuffled.
        divide (bool): If false only train files will be picked.

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

        if divide and i < num_files / 10:  # Picks test data
            test_filenames.append(file)
        else:
            train_filenames.append(file)

    return train_filenames, test_filenames


def merge_json_files(json_dir, json_files, name, output_dir, overwrite=False):
    """
    Merges all given json files in json_dir to a new json file that is stored in output_dir under the new name.

    Args:
        json_dir (str): A path of the folder.
        json_files (list): A list of selected files to be merged in to a single .json file.
        name (str): A name of the new file.
        output_dir (str): A path to the folder where the formatted annotation will be stored.
        overwrite (bool): Overwrites any pre-existent merged file.

    """
    output_path = f"{output_dir}/{name}.json"
    coco_json = {}
    # Checks if the file already exists
    if os.path.isfile(output_path) and not overwrite:
        print(f"{output_path} already exists.")
        return

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
