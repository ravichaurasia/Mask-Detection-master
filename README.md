# Mask-Detection-master
Mask detection working code with setup information of TFOD 
Extract the object_detection.rar and download the model from
https://drive.google.com/open?id=1c7PrZ8-ov3ek9FgLu-qBmgkTlFdg3XmS
put in model folder.

##### STEPS ################
https://github.com/tensorflow/models/tree/v1.13.0  ###  Download Repo

http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz ### Download

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

https://tzutalin.github.io/labelImg/  # Download latest version of windows format


# Creating virtual env using conda
conda create -n your_env_name python=3.6

conda activate your_env_name

# pypi 
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow-gpu==1.14.0

pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0

########################################################################################################

For protobuff to py conversion
https://github.com/protocolbuffers/protobuf/releases/download/v3.11.0/protoc-3.11.0-win64.zip

cd /d D:\TFOD\models\research

# conda package manager
conda install -c anaconda protobuf

# linux mac
protoc object_detection/protos/*.proto --python_out=.
#windows
protoc object_detection/protos/*.proto --python_out=.

# command to open Jupiter notebook from terminal.
path 
cd..
cd..
cd \d D:\TFOD\models\research
Then write
jupyter notebook
.
and open the "object_detection_tutorial.ipynb"

# command for Object detection program:

Included in notebook file

##############################################################################################################
## Copy all extracted utils file to research folder
## Command Directory should be on research dir only to not get any problem

## Provide labeling to images manually
https://tzutalin.github.io/labelImg/

#copy the file xml_to_csv from Edge elecronics github repo to models repo in reaserch folder

python xml_to_csv.py (TFOD/models/research/object_detection)

#copy the file xml_to_csv from Edge elecronics github repo to models repo in reaserch folder

python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
Change the label in generate_tfrecord.py file

## genarate train and test record file in object_detection or record folder only
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=object_detection/train.record
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=object_detection/test.record


### Copy the file train from legacy folder in object_detection to research folder

## Before running train.py file, go to research/training folder
## open faster_rcnn_inception_v2_coco.config
.
1)num_classes: 2 ## as per considered classes
2)num_steps: 2000 ## set epochs
3)input_path: "object_detection/train.record"
4)label_map_path: "training/labelmap.pbtxt"
5)input_path: "object_detection/test.record"
6)label_map_path: "training/labelmap.pbtxt"
## Check all these parameter 
Change the label labelmap.pbtxt file

## Run from research folder
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config

##Note:- nets error can occur, for that go to research/slim folder, and copy nets and deployment folder
## paste in to research folder and again run the above command 

## Training files are generated in research/training folder

# Copy the file from object_detection to research
## Replace the XXXX with the last generated ckpt file inside the training folder.

### Copy the file of export_inference_graph.py from object_detection to research

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-1309 --output_directory inference_graph

## put some images in test_images folder and run "object_detection_tutorial.ipynb", in research window
## Make the appropriate path changes in a notebook

https://github.com/tzutalin/labelImg
