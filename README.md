# FAV - Pedestrian detection project
This project aims for better object detection of pedestrians from elevated camara views of public transport stations.

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
   * Availability: Downloadable, Requirements: **Only for research purposes**, can not be use commercially, cite. (This dataset is freely available, under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. Users can share the dataset only if they: 1) give credit to the citation below; 2) do not use the dataset for any commercial purposes, and 3) distribute any additions, transformations or changes to the dataset under this license.)
   * Frame size: 3840 x 2160
3. [TownCentre](https://exposing.ai/oxford_town_centre/)
   * Availability: Unsupported
4. dasf
5. [Long-term Pedestrian Detection Dataset](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/longtermped/)
   * Availability: Downloadable, Requirements: Cite?,
   * Single video without ground truth
   * **Unsuitable**
   * Classifier Grids for Robust Adaptive Object Detection
   Peter M. Roth, Sabine Sternig, Helmut Grabner and Horst Bischof
   In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009 



### Front view
1. https://eurocity-dataset.tudelft.nl/eval/overview/examples
2. 