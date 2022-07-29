import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
import torchvision as T
import matplotlib.pyplot as plt



def main():
    from mmdet.apis import init_detector, inference_detector

    config_file = 'yolov3_mobilenetv2_320_300e_coco.py'
    checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
    model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
    inference_detector(model, 'demo/cat.jpg')


if __name__ == "__main__":
    main()