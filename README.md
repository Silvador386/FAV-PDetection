# FAV - Pedestrian detection project
* This project aims for better object detection of pedestrians from elevated camara views of public transport stations.
* Python version: 3.9.8
# TODO
1. Search for appropriate datasets.
2. Build train and test pipeline to retrain an existing R-CNN model (MMDetection: https://github.com/open-mmlab/mmdetection).
   * Download and prepare customized dataset.
   * Choose a model.
   * Retrain and fine-tune the model.
3. Evaluate the results.


## Datasets

1. [Mall Dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)
   * Availability: Downloadable, Requirements: **Only for research purposes**, can not be use commercially, cite.
   * Contains ground truths of 60 000 pedestrians, Ground truth - head position.
   * Video length: 2000 frames, Size 640x480, Frame rate< 2 Hz, Seemingly from one place.
2. **[P-DESTRE](http://p-destre.di.ubi.pt/)** / **[Download](http://p-destre.di.ubi.pt/download.html)**
   * Availability: Downloadable, Requirements: **Only for research purposes**, can not be use commercially, cite. 
   * _(This dataset is freely available, under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. Users can share the dataset only if they: 1) give credit to the citation below; 2) do not use the dataset for any commercial purposes, and 3) distribute any additions, transformations or changes to the dataset under this license)._
   * Bounding box, head, gender, height, etc.
   * Frame size: 3840 x 2160 (40 GB of video, several videos?), 269 subjects. 
3. [WiderPerson](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/)
   * Availability: Downloadable, Requirements: **Only for research purposes**.
   * 13 382 images of pedestrians in different scenarios, 400k annotations
   * Does not release the bounding box ground truths for test images. Users required to submit final prediction files.
4. [Image VisDrone2019](http://aiskyeye.com/)
   * Availability: Downloadable, Requirements: **Only for research purposes**.

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
8. [Eurocity dataset](https://eurocity-dataset.tudelft.nl/eval/overview/examples)
   * Availability: Downloadable, Requirements: **Only for research purposes**.
   * Front view - Unsuitable.


### Useful links:
* [Machine learning datasets](https://www.datasetlist.com/)
* [Deeplearning.buzz](https://deeplearning.buzz/deep-learning-datasets/)