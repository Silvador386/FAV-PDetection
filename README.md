# FAV - Pedestrian detection project
* This project aims for better object detection of pedestrians from elevated camara views of public transport stations. 
* This is done via fine-tuning a pre-trained Faster R-CNN neural network on the selected dataset.
  * Pre-trained checkpoint used for fine-tuning: [Faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn)

# Build Status
Project has been finished.
1. ~~Search for appropriate datasets.~~
2. ~~Build train and test pipeline to retrain an existing R-CNN model [MMDetection](https://github.com/open-mmlab/mmdetection).~~
   * ~~Download and prepare customized dataset.~~
   * ~~Choose a model.~~
   * ~~Retrain and fine-tune the model.~~
     * ~~Implement Wandb and Wandb-sweep.~~
     * ~~Use Metacetrum cluster for running the computations.~~
3. ~~Evaluate the results.~~

# Tech/Framework used
* Python version: 3.9.8
* Main framework used for object detection fine-tuning is [MMDetection](https://github.com/open-mmlab/mmdetection).
* It is highly recommended cloning [MMDetection](https://github.com/open-mmlab/mmdetection) repository into the project directory to function properly. 
  * Configs directory is required for pdestre configs to work properly.
  * Tools directory provides many useful functions.
* To assign and monitor trainings, the [Weights & Biases](https://wandb.ai) framework was used.
* To run the training model on the computation cluster, a singularity image and the _sing_FAV_PD_wandb.sh_ bash script was used.

# File structure
 Project directory\
 &nbsp; |-- code\
 &nbsp; |-- configs\
 &nbsp; |&emsp;&emsp;|-- pdestre\
 &nbsp; |\
 &nbsp; |-- checkpoints\
 &nbsp; |-- data\
 &emsp;&emsp; |-- P-DESTRE\
 &emsp;&emsp;&emsp; |-- coco_format\
 &emsp;&emsp;&emsp;&emsp; |-- merged &emsp;_# Annotations_  
 &emsp;&emsp;&emsp;&emsp; |-- videos &emsp;_# Images cut from dataset videos_

## Datasets
Here is a list of datasets with the main focus on pedestrians.
The P-DESTRE dataset has been selected as the most appropriate for this application.

1. **[P-DESTRE](http://p-destre.di.ubi.pt/)** / **[Download](http://p-destre.di.ubi.pt/download.html)**
   * Availability: Downloadable, Requirements: **Only for research purposes**, can not be use commercially, cite. 
   * _(This dataset is freely available, under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. Users can share the dataset only if they: 1) give credit to the citation below; 2) do not use the dataset for any commercial purposes, and 3) distribute any additions, transformations or changes to the dataset under this license)._
   * Bounding box, head, gender, height, etc.
   * Frame size: 3840 x 2160 (40 GB of video, several videos?), 269 subjects. 

2. [Mall Dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)
   * Availability: Downloadable, Requirements: **Only for research purposes**, can not be use commercially, cite.
   * Contains ground truths of 60 000 pedestrians, Ground truth - head position.
   * Video length: 2000 frames, Size 640x480, Frame rate< 2 Hz, Seemingly from one place.

3. [WiderPerson](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/)
   * Availability: Downloadable, Requirements: **Only for research purposes**.
   * 13 382 images of pedestrians in different scenarios, 400k annotations
   * Does not release the bounding box ground truths for test images. Users are required to submit final prediction files.
   
4. [Image VisDrone2019](http://aiskyeye.com/)
   * Availability: Downloadable, Requirements: **Only for research purposes**.
   * Different datasets depending on usage. From drone platforms, in China.
   
### Inappropriate view point locations
5. [PedX](http://pedx.io/)
   * Availability: Downloadable, Requirements: **Commercially available**, Copyright
   * Front (car) view, 3 sequences, 2D/3D labels, [Data structure](https://github.com/umautobots/pedx)
   * Unsuitable points of view.
   
6. [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/)
   * Availability: Downloadable, Requirements: **Only for research purposes**.
   * Top view via drone cam - Unsuitable.
   
7. [Citycam - vehicles](https://www.citycam-cmu.com/)
   * Availability: Downloadable, Requirements: **Commercial**.
   * Vehicles only, CCTV camera images
   
8. [Eurocity dataset](https://eurocity-dataset.tudelft.nl/eval/overview/examples)
   * Availability: Downloadable, Requirements: **Only for research purposes**.
   * Front view - Unsuitable.


### Useful links:
* [Machine learning datasets](https://www.datasetlist.com/)
* [Deeplearning.buzz](https://deeplearning.buzz/deep-learning-datasets/)