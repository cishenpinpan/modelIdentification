
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, timeit, argparse
import _init_paths

from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib2
from PIL import Image

from nets import vgg
from preprocessing import vgg_preprocessing

import recognize_face

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

default_checkpoints_dir = '../src/classifier/training_model/'

default_err_log = "../log/incorrect_prediction_log.txt"


# ----------------------Inference------------------------------------------
# 1. An image classifier takes place to classify an image into 
#    3 categories: 0->Product, 1->Model, and 2->Body parts.

# 2. If, which is the only case when step 2 is enforced, an image
#    is classified into category 1->Model, a face detector comes in
#    and detect potential faces so as to filter out false positives.
#    If it fails the face detection, it falls into category 2->body parts.
#--------------------------------------------------------------------------

def main(args) :
    # 1. Image classifier -> predict a label from cnn
    slim = tf.contrib.slim

    # We need default size of image for a particular network.
    # The network was trained on images of that size -- so we
    # resize input image later in the code.
    image_size = vgg.vgg_16.default_image_size

    # record incorrect estimates down on file
    err_log_file = open(args.err_log, 'a')

    clsNames = ['Product', 'Model', 'Body parts']
    with tf.Graph().as_default():
        # Open specified path and load image as a string
        # read in the path from arguments
        url = "file://" + args.image_path
        image_file = urllib2.urlopen(url).read()
        image = tf.image.decode_jpeg(image_file)

        # Resize the input image, preserving the aspect ratio
        # and make a central crop of the resulted image.
        # The crop will be of the size of the default image size of
        # the network.
        processed_image = vgg_preprocessing.preprocess_image(image,
                                                             image_size,
                                                             image_size,
                                                             is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure
        # the batch norm parameters. arg_scope is a very convenient
        # feature of slim library -- you can define default
        # parameters for layers -- like stride, padding etc.
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, _ = vgg.vgg_16(processed_images,
                                   num_classes=3,
                                   is_training=False)

        # In order to get probabilities we apply softmax on the output.
        probabilities = tf.nn.softmax(logits)

        # Create a function that reads the network weights
        # from the checkpoint file that you downloaded.
        # We will run it in session later.
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(args.checkpoints_dir, 'model.ckpt-1980'),
            slim.get_model_variables('vgg_16'))

        with tf.Session() as sess:
            # Load weights
            init_fn(sess)

            # We want to get predictions, image as numpy matrix
            # and resized and cropped piece that is actually
            # being fed to the network.
            np_image, network_input, probabilities = sess.run([image,
                                                               processed_image,
                                                               probabilities])
            probabilities = probabilities[0, 0:]

            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                                key=lambda x: x[1])]

        # Show the image that is actually being fed to the network
        # The image was resized while preserving aspect ratio and then
        # cropped. After that, the mean pixel value was subtracted from
        # each pixel of that crop. We normalize the image to be between [-1, 1]
        # to show the image.

        prediction = sorted_inds[0]

        # Now we print the top-5 predictions that the network gives us with
        # corresponding probabilities. Pay attention that the index with
        # class names is shifted by 1 -- this is because some networks
        # were trained on 1000 classes and others on 1001. VGG-16 was trained
        # on 1000 classes.
        print('Classified as => [%s] with %0.2f confidence.' % (clsNames[prediction], probabilities[prediction]))

        res = slim.get_model_variables()


    # 2. Face detection -> verify model images by looking for faces

    if prediction == 1 : # 2-step verification for model image ONLY
        print("Classified as model. Detecting face...")
        if not recognize_face.has_face(args.image_path) :
            print("Adjusted. Now classified as body parts.")
            prediction = 2
        else :
            print("[Model] confirmed.")

    # 3. record incorrect prediction
    if args.image_label != -1 and prediction != args.image_label :
        err_log_file.write(args.image_path + " " + str(prediction) + " " + str(args.image_label) + "\n")
        plt.figure()
        plt.imshow(np_image.astype(np.uint8))
        plt.suptitle(clsNames[prediction] + "(" + clsNames[args.image_label] + ")",  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()

    err_log_file.close()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoints_dir', type=str,
        help='Path to the checkpoint containing ckpt model files of image classifier.', default=default_checkpoints_dir)
    parser.add_argument('--err_log', type=str,
        help='(Optional) Path to the log file that records incorrect predictions. Valid only if label is provided.', default=default_err_log)
    parser.add_argument('--image_path', type=str, 
        help='Path to the image to be inferenced.')
    parser.add_argument('--image_label', type=int,
        help='(Optional) Correct label(0->Product, 1->Model, or 2->Body parts) of the image to be inferenced.', default=-1)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
            
