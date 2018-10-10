import numpy as np
import os
import matplotlib.pyplot as plt
import h5py

def display_cifar10_image(img_flat):
    '''
    Resizes an input image to 32x32 with respect to each channel
    '''
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))

    return img

def plot_images(X, y, classes, number_of_images=10,save=False):
    '''
    plots an NxN matrix of N classes (min = 1, max = 10)
    '''
    inds=[]
    for i in range(number_of_images):
        a = np.where(y==i)[0] #  indicies of each class
        ind = np.random.choice(a,number_of_images,replace=False) #taking a 500 random samples out of 5000
        inds.append(ind)
    inds_plot = np.array(inds).flatten()

    plt.figure(1, figsize=(number_of_images,number_of_images))
    for i in range(number_of_images**2):
        plt.subplot(number_of_images,number_of_images,i+1)
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        xx = X[inds_plot[i]]
        plt.imshow(display_cifar10_image(xx))
        if i%number_of_images == 0:
            class_ = classes[y[inds_plot[i]]]
            plt.ylabel(str(class_),rotation=0,labelpad=30)
    plt.subplots_adjust(wspace=.001, hspace=.001)
    if save:
        plt.savefig("presentation_images/10.png")
    plt.show()

train = h5py.File('datasets/datatraining.h5','r')
test = h5py.File('datasets/datatest.h5','r')
X_train, y_train = train['data'][:], train['labels'][:]
X_test, y_test = test['data'][:], test['labels'][:]

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
plot_images(X_train, y_train, classes,save=True)
