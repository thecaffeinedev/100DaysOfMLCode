import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import h5py


def unpickle(file):
    """
    Unpickles large files into a dict
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def downscale_dataset(X, y, proportion = 0.1, n_classes = 10):
    """
    Downscales dataset based on given proportion

    Args:
    X - X_train or X_test
    y - y_train or y_test
    proportion: 0.1 == 1/10 of dataset
    n_classes: number of classes in the dataset

    Returns:
    X_new -- downscaled X set
    y_new -- downscaled y set

    """
    size = int(X.shape[0]*proportion)
    X_new = np.zeros((size,X.shape[1]),dtype=np.int64)
    y_new = np.zeros((size),dtype=np.int64)
    inds = []

    for i in range(n_classes):
        a = np.where(y==i)[0] #  indicies of each class
        ind = np.random.choice(a,int(size/int(n_classes)),replace=False) #taking a 500 random samples out of 5000
        inds.append(ind)

    inds = np.array(inds).flatten()

    for i in range(len(X_new)):
        X_new[i] = X[inds[i]]
        y_new[i] = y[inds[i]]

    return X_new, y_new

def load_CIFAR10_data(cwd,path):
    '''
    loads CIFAR 10 batch data and returns
    '''
    ##training data
    data_train = [unpickle(batch) for batch in (os.path.join(path,"data_batch_{}".format(i + 1)) for i in range(5))]
    X_train = np.vstack([d[B'data'] for d in data_train])
    y_train = np.hstack([np.asarray(d[B'labels'], np.int8) for d in data_train])

    # test data
    data_test = unpickle(os.path.join(path, 'test_batch'))
    X_test = data_test[b'data']
    y_test = np.asarray(data_test[b'labels'], np.int8)

    return X_train,y_train,X_test,y_test

def save_file(X,y,name='training',proportion=0.1):
    a, b = downscale_dataset(X, y, proportion)
    h5f = h5py.File('datasets/data'+name+'.h5', 'w')
    h5f.create_dataset('data', data=a)
    h5f.create_dataset('labels', data=b)
    h5f.close()


filename = 'datasets/datacifar-10-python.tar.gz'
data_dir = 'datasets/cifar-10-batches-py'
cwd = os.getcwd()
path = os.path.join(cwd, data_dir)

X_train,y_train,X_test,y_test = load_CIFAR10_data(cwd,path)

#training set
save_file(X_train,y_train,name='training',proportion=0.1)

#test set
save_file(X_test,y_test,name='test',proportion=0.1)
