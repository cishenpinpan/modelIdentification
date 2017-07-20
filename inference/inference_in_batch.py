from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys


sys.path.insert(0, '/Users/vipus01/Documents/Developer/modelIdentification/')
sys.path.insert(0, '/Users/vipus01/Documents/Developer/modelIdentification/facenet/')
sys.path.insert(0, '/Users/vipus01/Documents/Developer/modelIdentification/facenet/src/')
sys.path.insert(0, '/Users/vipus01/Documents/Developer/modelIdentification/facenet/src/align/')
sys.path.insert(0, '/Users/vipus01/Documents/Developer/modelIdentification/huoguo/third_party/models/slim/')

from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib2
from PIL import Image

from nets import vgg
from preprocessing import vgg_preprocessing

import recognize_face

os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# checkpoints_dir = '/home/ryuan-adm/Dev/huoguo/hotpot_data/train_model/modelIdentification'
checkpoints_dir = '/Users/vipus01/Documents/Developer/modelIdentification/data/train_model/modelIdentification'

# record incorrect estimates down on file
err_log_dir = "/Users/vipus01/Documents/Developer/modelIdentification/log/incorrect_prediction_log.txt"
err_log_file = open(err_log_dir, 'a')

# read in metadata file on test data
metadata_file_path = sys.argv[1]
metadata_file = open(metadata_file_path)
filedir = "/Users/vipus01/Documents/Developer/modelIdentification/data/data/"
lines = metadata_file.readlines()
cnt = len(lines)
idx = 0

# Create the model, use the default arg scope to configure
# the batch norm parameters. arg_scope is a very conveniet
# feature of slim library -- you can define default
# parameters for layers -- like stride, padding etc.

# Create a function that reads the network weights
# from the checkpoint file that you downloaded.
# We will run it in session later.
slim = tf.contrib.slim
init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'model.ckpt-1980'),
    slim.get_model_variables('vgg_16'))

# read from metadata file
for line in lines :
    idx += 1
    strs = line.replace("\n", "").split(" ")
    f = strs[0]
    cls = strs[1]

    # 1. Image classifier -> predict a label from cnn

    # We need default size of image for a particular network.
    # The network was trained on images of that size -- so we
    # resize input image later in the code.
    image_size = vgg.vgg_16.default_image_size

    

    img_path = filedir + f
    prediction = None

    clsNames = ['Product', 'Model', 'Body parts']
    with tf.Graph().as_default():
        # Open specified url and load image as a string
        # read in the url from arguments
        url = "file://" + img_path

        # Open specified url and load image as a string
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
        # Resize the input image, preserving the aspect ratio
        # and make a central crop of the resulted image.
        # The crop will be of the size of the default image size of
        # the network.

        processed_images = tf.expand_dims(processed_image, 0)

        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, _ = vgg.vgg_16(processed_images,
                                   num_classes=3,
                                   is_training=False)

        # In order to get probabilities we apply softmax on the output.
        probabilities = tf.nn.softmax(logits)

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
        print('Probability %0.2f => [%s]' % (probabilities[prediction], clsNames[prediction]))

        res = slim.get_model_variables()


    # 2. Face detection -> verify model images by looking for faces

    if prediction == 1 : # 2-step verification for model image ONLY
        if not recognize_face.has_face(img_path) :
            print("Aha! You are not a model!")
            prediction = 2
        # cfg_path = '/Users/vipus01/Documents/Developer/modelIdentification/vipface/demo_conf/face/syn_cbl_rgb_tcbcn_cbl_single_webcam.json'

        # # initialize and prepare detector
        # detector = Detector.init_from_cfg_file(fpath=cfg_path)
        # detector.debug = True
        # detector.prepare()

        # # detect face
        # image = mio.import_image(filepath=img_path)
        # frame = image
        # frame = image.as_imageio(out_dtype=np.uint8)
        # height, width, channels = frame.shape
        # fx = 1300.0 / height
        # fy = 1100.0 / width
        # frame = cv2.resize(frame, None, fx=fx, fy=fy)
        
        # frame = detector.detect_draw_on_frame(frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('result.jpg', frame)

        # if detector.curr_lmks is None :
        #     prediction = 2

        # print("Prediction: " + str(prediction) + " ground truth: " + str(cls))

    # 3. record incorrect prediction
    if prediction != cls :
        err_log_file.write(img_path + " " + str(prediction) + " " + str(cls) + "\n")
        plt.figure()
        plt.imshow(np_image.astype(np.uint8))
        plt.suptitle(clsNames[prediction] + "(" + clsNames[cls] + ")",  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()

        



# read in metadata file on test data
# filedir = "/Users/vipus01/Documents/Developer/modelIdentification/data/data/"
# lines = metadata_file.readlines()
# cnt = len(lines)
# idx = 0
# for line in lines :
#     idx += 1
#     strs = line.replace("\n", "").split(" ")
#     f = strs[0]
#     cls = strs[1]
#     if idx < 0 :
#         continue
#     print(f)
#     if f.endswith('.jpge') or f.endswith('.jpg') : 
#         subprocess.call('python /Users/vipus01/Documents/Developer/modelIdentification/inference/inference.py ' + filedir + f + " " + cls, shell = True)
#     print(str(idx) + "/" + str(cnt))
    