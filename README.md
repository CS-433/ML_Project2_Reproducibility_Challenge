# EPFL Machine Learning Project 2

## Reproducibility study of "Scaled-YOLOv4: Scaling Cross Stage Partial Network"

Second project for Machine Learning course (CS-443) at EPFL. 
This is the implementation of YOLOv4-CSP in "[Scaled-YOLOv4: Scaling
Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)" 
using PyTorch framework. The report explains the details of reproducibility study, 
and the python scripts are in the root folder.

For YOLOv4-tiny, please directly install and use [Darknet](https://github.com/AlexeyAB/darknet).

* [Getting started](#getting-started)
    * [Data](#data)
    * [Folders and Files](#folders-and-files)
* [Running the code](#running-the-code)
* [Contact us](#contact-us)

## Getting started

### Data

The COCO 2017 dataset can be downloaded from [COCO | 
Common objects in context](https://cocodataset.org/#download),
and the VOC 2007 dataset can be downloaded from [The PASCAL
object visual classes challenges](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/).

Note that COCO and VOC labels should be converted to YOLO format first. 
For VOC 2007 labels, the user can use `VOCtoYOLO.ipynb` for conversion,
and for COCO 2017 labels, the user can use `COCOtoYOLO.py`.
Then, VOC image and label files should be sent to new folders named `voc/image` and `voc/label` 
under the root folder respectively, and COCO image and label files should be sent
to new folders named `coco/image` and `coco/label` under the root folder respectively.


### Folders and Files

`train.py`: the main script for YOLOv4-CSP model training, 

`test.py`: the main script for model testing using the trained weights,

`VOCtoYOLO.ipynb`: converts VOC format label files to YOLO format,

`COCOtoYOLO.py`: converts COCO format label files to YOLO format,

`data/coco.yaml`: contains COCO class names and file path,

`data/VOC07_YOLO.yaml`: contains VOC class names and file path,

`data/hyp.scratch.yaml`: contains hyper-parameters for training,

`utils`: folder containing some util functions,

`models/models.py`: contains the Darknet-CSP backbone network algorithm,

`models/yolov4-csp.cfg`: contains the configuration information of Darknet-CSP,

`report.pdf`: project report explaining our reproducibility study in .pdf format written in Latex,

`project2_description.pdf`: assignment description given by EPFL,

`requirements`: environment used in the project.

## Running the code

For training the VOC dataset, move to the root folder and execute:

    python train.py --device 0 --batch-size 16 --data voc.yaml --cfg yolov4-csp.cfg --weights '' --name yolov4-csp

For testing, 

    python test.py --img 640 --conf 0.001 --iou 0.65 --batch 8 --device 0 --data voc.yaml --cfg models/yolov4-csp.cfg --weights weights/last.pt

Training the COCO dataset has similar procedure, but do not forget to change the data file path.
The current code adopts CIoU loss function and Mish activation function. 
Detailed experiments for the algorithm is documented in `report.pdf`.

## Contact us
Please don't hesitate to contact the authors about any questions about the project, data or algorithms in general:

* Tianyu Gu: tianyu.gu@epfl.ch
* Xinyu Liu: xinyu.liu@epfl.ch

@ 2022 Tianyu Gu