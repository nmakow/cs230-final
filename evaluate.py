import argparse
import os
import scipy.misc
from scipy.spatial.distance import cityblock
import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# This file is intended to evaluate predictions from the Pix2Pix model. 
#   true_dir is the directory containing the true AB image pairs from the dataset.
#   fake_dir is the directory containing the fake B images predicted from a model.
# It is assumed that the files are aligned such that the images in true_dir
# correspond to the same images in preds_dir.
parser = argparse.ArgumentParser(description='')
parser.add_argument('--true_dir', dest='true_dir', default='./datasets/facades/val')
parser.add_argument('--fake_dir', dest='fake_dir', default='./test/pix2pix')

args = parser.parse_args()

model = VGG16(weights='imagenet', include_top=False)

l1_dists, l2_dists, vgg_sims = [], [], []
for i in range(1, 100):
    true_file = os.path.join(args.true_dir, str(i) + ".jpg")
    if args.fake_dir != 'random':
        fake_file = os.path.join(args.fake_dir, "test_%04d.png" % i)
    
    # Read in the full true image. This is 256x512x3 since the
    # AB images are concatenated horizontally. To get the true B
    # take the first half of the image.
    true_full = image.load_img(true_file)
    true_data = image.img_to_array(true_full)
    true_data = true_data[:, :256, :]
    if args.fake_dir != 'random':
        fake = image.load_img(fake_file)
        fake_data = image.img_to_array(fake)
    else:
        fake_data = np.random.rand(256, 256, 3) * 255
    l1_dist = cityblock(true_data.flatten(), fake_data.flatten())
    l2_dist = np.linalg.norm(true_data - fake_data)
    l1_dists.append(l1_dist)
    l2_dists.append(l2_dist)

    # Next VGG feature vector cosine similarity
    true = image.load_img(true_file, target_size=(224,448))
    true_data = image.img_to_array(true)[:, :224, :]
    true_data = np.expand_dims(true_data, axis=0)
    true_data = preprocess_input(true_data)
    true_feats = model.predict(true_data).flatten() 
    if args.fake_dir != 'random':
    	fake = image.load_img(fake_file, target_size=(224,224))
    	fake_data = image.img_to_array(fake)
    	fake_data = np.expand_dims(fake_data, axis=0)
    	fake_data = preprocess_input(fake_data)
    	fake_feats = model.predict(fake_data).flatten()
    else: 
        fake_feats = np.random.rand(25088)
    cos_sim = np.dot(true_feats, fake_feats) / (np.linalg.norm(true_feats) * np.linalg.norm(fake_feats))
    vgg_sims.append(cos_sim)

print("Average l1: %f, l2: %f, vgg-cos-sim: %f" % (np.mean(l1_dist), np.mean(l2_dist), np.mean(vgg_sims)))
