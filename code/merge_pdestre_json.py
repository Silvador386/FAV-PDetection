import json
import os
import random

from utils import files_in_folder


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
