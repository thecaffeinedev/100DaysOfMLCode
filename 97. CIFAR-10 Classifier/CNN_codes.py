import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
import time
from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from skimage.transform import resize

np.random.seed(42)

def display_cifar10_image(img_flat):
    """
    Resizes an input image to 32x32 and subtracts the mean pixel
    with respect to each channel
    """
    mean_pixel = [103.939, 116.779, 123.68] #mean pixels for VGG16
    img_R = img_flat[0:1024].reshape((32, 32)) - mean_pixel[0]
    img_G = img_flat[1024:2048].reshape((32, 32)) - mean_pixel[1]
    img_B = img_flat[2048:3072].reshape((32, 32)) - mean_pixel[2]
    img = np.dstack((img_R, img_G, img_B))

    return img

def extract_features(model,img,input_size=224,exit_layer = 'fc2'):
    """
    Extracts the bottleneck features of a fully connected layer of a CNN
    """
    x = resize(img,(input_size,input_size))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    model_extractfeatures = Model(input=model.input, output=model.get_layer(exit_layer).output)
    fc2_features = np.squeeze(model_extractfeatures.predict(x))

    return fc2_features

def extract_set(X, model,set_type='training',save=False):
    """
    Extracts features for all images in a given data set
    and saves it as an '.h5' file
    """
    extracted_features = np.zeros((X.shape[0],4096))
    start = time.time()

    for i in range(len(X_train)):
        extracted_features[i] = extract_features(model,img=display_cifar10_image(X[i]))
        if i%100 == 0:
            print("{} %".format(i/100))
            t = time.time() - start
            print("time eclipsed:\nSeconds: {}\nMinutes: {}\nHours: {}".format(t,t/60,t/3600))
    print("TOTAL TIME: {}".format(time.time()-start))

    if save:
        h5f = h5py.File('extracted_tsne'+set_type+'.h5', 'w')
        h5f.create_dataset('extracted_data', data=extracted_features)
        h5f.close()
    else:
        return extracted_features

train = h5py.File('datasets/datatraining.h5','r')
test = h5py.File('datasets/datatest.h5','r')
X_train, y_train = train['data'][:], train['labels'][:]
X_test, y_test = test['data'][:], test['labels'][:]

model = vgg16.VGG16(weights='imagenet', include_top=True)

extract_set(X_train, model,set_type='training',save=True)
extract_set(X_test, model,set_type='test',save=True)
