import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Conv2DTranspose, Reshape
from tensorflow.python.keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.keras.backend import batch_dot

class CapsNet(tf.keras.models.Model):
    ''' Encoder network'''
    def __init__(self, n_class, routings):
        super(CapsNet, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')
        self.conv2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv2')
        self.conv3 = Conv2D(filters=256, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv3')
        self.primarycaps = PrimaryCaps(dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name='primarycaps')
        self.digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, input_shape=(None, 128, 8), routings=routings, channels=32, name='digitcaps')
        self.outcaps =  Length(n_class=n_class, name='outcaps')

    def call(self, inputs):
        t_features1 = self.conv1(inputs)
        t_features2 = self.conv2(t_features1)
        t_features3 = self.conv3(t_features2)
        t_features4 = self.primarycaps(t_features3)
        t_features5 = self.digitcaps(t_features4)
        t_features6 = self.outcaps(t_features5)
        return t_features5, t_features6

def getDecoder():
    decoder = tf.keras.models.Sequential(name='decoder')
    decoder.add(Dense(units=1568, activation='relu'))
    decoder.add(Reshape((7,7,32)))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Conv2DTranspose(32, 3, 1, 'same', activation='relu'))
    decoder.add(Conv2DTranspose(16, 3, 2, 'same', activation='relu'))
    decoder.add(Conv2DTranspose(8, 3, 2, 'same', activation='relu'))
    decoder.add(Conv2DTranspose(4, 3, 1, 'same', activation='relu'))
    decoder.add(Conv2DTranspose(1, 3, 1, 'same', activation='relu'))
    decoder.add(Reshape((28, 28, 1)))
    return decoder

class TextCaps(tf.keras.models.Model):
    def __init__(self, n_class, routings):
        super(TextCaps, self).__init__()
        self.encoder = CapsNet(n_class, routings)
        self.mask_fn = Mask(n_class)
        self.decoder = getDecoder()
    
    def __call__(self, inputs, y_true=None):
        caps_out, caps_class = self.encoder(inputs)
        if y_true is not None: masked = self.mask_fn(caps_out, y_true)
        else: masked = self.mask_fn(caps_out, caps_class)
        rcn_out = self.decoder(masked)
        return caps_class, rcn_out

class Length(tf.keras.layers.Layer):
    '''
    Compute the length of vectors
    '''
    def __init__(self, n_class, name):
        super(Length, self).__init__(name)
        self.n_class = n_class

    def call(self, inputs):
        lens = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(inputs), -1))
        return tf.one_hot(tf.argmax(lens, axis=-1), depth=self.n_class, dtype=tf.float32)

class Mask(tf.keras.layers.Layer):
    def __init__(self, n_class):
        super(Mask, self).__init__()
        self.n_class = n_class
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, mask):   
        masked = tf.multiply(inputs, tf.expand_dims(mask, -1))
        return self.flatten(masked)

def squash(vectors, axis=-1):
    s_squared_norm = tf.math.reduce_sum(tf.math.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.math.sqrt(s_squared_norm + 1e-7)
    return scale * vectors

class PrimaryCaps(tf.keras.layers.Layer):
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, name):
        super(PrimaryCaps, self).__init__()
        self.conv = Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding, name=name+'_conv')
        self.reshape = Reshape([-1, dim_capsule], name=name+'_reshape')
    
    def call(self, inputs):
        t_features1 = self.conv(inputs)
        t_features2 = self.reshape(t_features1)
        t_features3 = squash(t_features2)
        return t_features3

class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, channels, input_shape, routings=3,
                kernel_initializer='glorot_uniform',
                **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        assert self.routings>0, "The routings should be > 0"
        self.channels = channels
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        assert not(int(self.input_num_capsule)%self.channels)
        self.W = self.add_weight(shape=[self.channels, self.num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                     initializer=self.kernel_initializer,
                                     name='W')
        self.B = self.add_weight(shape=[self.num_capsule, self.dim_capsule],
                                    initializer=self.kernel_initializer,
                                    name='B')
        self.built = True

    def call(self, inputs):
        W2 = tf.repeat(self.W, self.input_num_capsule//self.channels, 0)
        inputs_hat = tf.map_fn(lambda x: batch_dot(x, W2, [1, 3]), elems=inputs)
        b = tf.zeros(shape=(32, 128, 47, 1))
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(tf.reduce_sum(tf.multiply(c, inputs_hat), axis=1, keepdims=True) + self.B)
            if i < self.routings - 1:
                b += tf.reduce_sum(tf.multiply(outputs, inputs_hat), axis=3, keepdims=True)
        return tf.squeeze(outputs)

if __name__ == "__main__":
    import numpy as np
    inputs = np.array([
        [[1, 2, 3], [1, 1, 1], [2, 2, 1]],
        [[1, 1, 1], [1, 2, 3], [1, 1, 2]]
    ], dtype=np.float32)