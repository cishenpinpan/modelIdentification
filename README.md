# Model Recognition using Tensorflow
This is a TensorFlow implementation of the model recognition task. It consists of two separate modules: image classifer, which is implemented and trained by vip-us, and face detector, which is heavily dependent on [FaceNet](https://github.com/davidsandberg/facenet). A more in-depth description of the task can be found in the section below. 

## In-depth Understanding of the task
This recognizer aims at classifying query images into three categories: product, model, and body parts.  
 A product image should include only the product itself. Model image should include a human-being model, speficically a human face. Meanwhile, an image consisting of only human body parts, specifically no human faces, will be considered as the last category. In the implementation, these three categories are numbered as 0->product, 1->model, and 2->body parts.
The nature of this definition leads to such a solution that after a classifier, when an image is classfied as 1->model, a face detector is introduced so as to filter out some false positives. The reason why this is only done on 1->model is because the classifier performed a good recall rate on model images but not body parts, meaning that some body parts are recognized as models.

## Prerequisites
The installations of following libraries are required.
* tensorflow=1.2.x
* scipy
* oepncv-python
* h5py
* matplotlib
* Pillow
* requests
* psutil

## Running demo
1. cd inference
2. python inference.py --image_path [path_to_image] or python inference.py --image_path [path_to_image] --image_label [0, 1, or 2]  
A sample command should look like _python inference.py --image_path ../data/sample.jpg --image_label 1_

## Training data
A total of 3,000 images are used in training and validation process, with 20% of which being the validation set. Among these 3,000 images, three categories are evenly distributed.  

## Performance 
| 			        | class 0 recall | class 1 recall | class 2 recall | precision |
|-------------------|----------------|----------------|----------------|-----------|
|No face detection  |      99.4%     |      99.5%     |      83.0%     |   ~94%    |  
|With face detection|      99.4%     |      99.5%     |      99.5%     |   ~99.5%  | 

## Pre-trained model
Currently, the best performing model is a [vgg-16] model(https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz).
