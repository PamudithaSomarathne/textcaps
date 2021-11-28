import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from scipy.io import loadmat
import numpy as np

def loadData(dataset_path):
    dataset = loadmat(dataset_path.numpy().decode('utf-8'))['dataset'][0][0]
    train = dataset[0][0][0]
    x_train = train[0]
    y_train = train[1].reshape(-1)
    test = dataset[1][0][0]
    x_test = test[0]
    y_test = test[1].reshape(-1)
    return x_train.reshape((-1,28,28,1)).astype(np.float32), y_train, x_test.reshape((-1,28,28,1)).astype(np.float32), y_test

def getDataset(path, n_class):
    x_train, y_train, x_test, y_test = tf.py_function(loadData, [path], [tf.float32, tf.int32, tf.float32, tf.int32])
    y_train, y_test = tf.one_hot(y_train, depth=n_class, dtype=tf.float32), tf.one_hot(y_test, depth=n_class, dtype=tf.float32)
    return x_train/255.0, y_train, x_test/255.0, y_test

if __name__=='__main__':
    d = getDataset('D:\TextCaps\datasets\emnist-balanced.mat', 47)
    print(d[0].shape, d[1].shape, d[2].shape, d[3].shape)