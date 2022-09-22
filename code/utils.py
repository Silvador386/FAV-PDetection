import os
import cv2 as cv2


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


def convert_video_to_jpgs(video_name, video_path, output_path, frame_rate=10):
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
