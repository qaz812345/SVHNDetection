# SVHNDetection
This is a task of object detection with [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/). There are 10 number classes in the dataset. We use 30,000 images for training, 3,402 for validation, and 13,068 for testing. Use [YOLOv5](https://github.com/ultralytics/yolov5) model from GitHub and train with pretrained checkpoint. The highest testing mAP can reach 52.115%.

# Reproducing Submission
1. [Installation](#Installation)
2. [File Configuration](#File-Configuration)
3. [Dataset Configuration](#Dataset-Configuration)
4. [Training](#Training)
5. [Inference](#Inference)

# Installation
1. Clone this repository. 
```
git clone https://github.com/qaz812345/SVHNDetection
```

2. We will use [YOLOv5](https://github.com/ultralytics/yolov5) model from GitHub. cd to this repository and clone YOLOv5 repository.
```
cd SVHNDetection
git clone https://github.com/ultralytics/yolov5
```
3. Install the pytyhon packages with requirements.txt.
```
pip install -r requirements.txt
```
# File Configuration
Plase move files to the corresponding destination.

| File Name | File Destination |
|:-------- | :-------------- |
| parse_data.py | yolov5/ |
| detect.py | yolov5/ |
| datasets.py | yolov5/utils/ |
| svhn.yaml | yolov5/data/ |



# Dataset Configuration
To train SVHN dataset on YOLOv5, we need to set up the configuration for it.
1. Download ```train.tar.gz``` and ```test.tar.gz``` from [SVHN dataset site](http://ufldl.stanford.edu/housenumbers/).
2. Unzip the data files and set the data directory structure as:
```
SVHNDetection
|__svhn
|   |__images
|   |   |__train
|   |   |  |__<train_image>
|   |   |  |__ ...
|   |   |  
|   |   |__val
|   |   |  |__<val_image>
|   |   |  |__ ...
|   |   | 
|   |   |__test
|   |      |__<test_image>
|   |      |__ ...
|   |     
|   |__labels
|       |__train
|       |
|       |__val
|
|__yolov5
|   |__ ...
|
|__ parse_data.py
```
3. Pick a number as dividing index (ex.30000) to devide training images into training set and vaildation set. Move the images after dividing index (ex.30001) to ```svhn/images/val``` as vaildation set 
4. Parse label data from ```svhn/images/train/digitStruct.mat```.
```
python parse_data.py -f ../svhn/images/train/digitStruct.mat -d <dividing_index>
```

# Training
### Model Configuration
1. Download yolov5l.pt weights from [here].(https://github.com/ultralytics/yolov5/releases)
2. Edit ```yolov5/models/yolov5l.yaml``` line 2 as:
```
# parameters
nc: 11  # in svhn 0 is labeled as '10' so we have 11 classes '0' to '10'. And class '0' will never appear.
...
...
```
### Data Pre-process and Augmentation
*	Random mosaic
* Resize image (shape=(640, 640))
* Random translation (translate=0.1)
* Random scale (scale=0.5)
* Random horizontal flip

You can edit ```yolov5/data/hyp.scratch.yaml``` for changing augmentation parameters or using the other augmentation methods.

### Hyperparameters
*	Epochs = 100
*	Batch size = 32
*	Optimizer = SGD (learning rate=0.01, momentum=0.937, weight decay=0.0005)
*	Warmup (epochs=3, momentum=0.8, bias learning rate=0.1)
*	Scheduler = cosine learning rate decay (final OneCycleLR learning rate=0.2)
*	Loss function = IOU loss
*	Box loss gain = 0.05
*	Class loss gain = 0.5
*	Object loss gain = 1.0
*	Device = 0,1,2,3

You can edit ```yolov5/data/hyp.scratch.yaml``` for changing some hyperparameters and some need to specify at training time.

### Train
Training command:
```
python train.py --weights weights/yolov5l.pt --cfg models/yolov5l.yaml --data data/svhn.yaml --epochs 100 --batch-size 32 --device 0,1,2,3
```
If your GPUs are out of memory, please decrease batch size or change to smaller model like yolov5m or yolov5s. The way of changing configuration setting is simlar to yolov5l, please check [Model Configuration](#Model-Configuration) section above.

# Inference
To output all testing result to one json file with structure as:
```
[{"bbox": [(y1, x1, y2, x2), (y1, x1, y2, x2)], "score": [conf, conf], "label": [cls, cls]},
{"bbox": [], "score": [], "label": []},
...
...
{"bbox": [(y1, x1, y2, x2)], "score": [conf], "label": [cls]}]
```
You have to change the name of trained parameter file to model you used. For example ```yolov5/runs/train/<exp_no>/weights/last.pt``` to ```yolov5/runs/train/<exp_no>/weights/yolov5l.pt```.
Inference command:
```
python detect.py --source ../svhn/images/test --weights runs/train/<exp_no>/weights/yolov5l.pt --device 0 --json <output_json_file>
```
Increase object confidence threshold to 0.5 and infer with augmentation.
```
python detect.py --source ../svhn/images/test --weights runs/train/<exp_no>/weights/yolov5l.pt --device 0 --augment --conf-thres 0.5 --json <output_json_file>
```

# Reference
*	ultralytics, yolov5, viewed 25 Nov 2020, [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
*	Bartzi, svhn_dataextract_tojson.py, viewed 25 Nov 2020, [https://github.com/Bartzi/stn-ocr/blob/master/datasets/svhn/svhn_dataextract_tojson.py](https://github.com/Bartzi/stn-ocr/blob/master/datasets/svhn/svhn_dataextract_tojson.py)
