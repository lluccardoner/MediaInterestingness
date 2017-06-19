"""

Author: Lluc Cardoner

Extract features of each clip of 16 frames with the C3D and save them.

"""

from __future__ import absolute_import
from __future__ import print_function
import multiprocessing
import os
import sys
import time
import traceback

import h5py
import numpy as np
import progressbar
from progressbar import ProgressBar


def C3D_conv_features(summary=False):
    """ Return the Keras model of the network until the fc6 layer where the
    convolutional features can be extracted.
    """
    from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Sequential

    # added by lluc
    from keras import backend as K
    K.set_image_dim_ordering('th')

    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1),
                            input_shape=(3, 16, 112, 112),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropadding'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten(name='flatten'))
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5, name='do2'))
    model.add(Dense(487, activation='softmax', name='fc8'))

    # Load weights
    model.load_weights('/home/lluc/PycharmProjects/activitynet-2016-cvprw/data/models/c3d-sports1M_weights.h5')

    # for _ in range(4):
    #     model.pop_layer()

    for _ in range(4):
        model.pop()

    if summary:
        print(model.summary())
    return model


def load_model():
    # Loading the model
    print('Loading model')
    m = C3D_conv_features(summary=True)
    print('Compiling model')
    m.compile(optimizer='sgd', loss='mse')
    print('Compiling done!')
    return m


def extract_features_clip(clip, model):
    # print('Starting extracting features')
    # print('Loading mean')
    mean_total = np.load('/home/lluc/PycharmProjects/activitynet-2016-cvprw/data/models/c3d-sports1M_mean.npy')
    mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)
    clip = clip - mean

    Y = model.predict(clip, batch_size=1)
    # print('Extracting features done!')
    return Y


def save_clips_features():
    in_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/video_clips.h5py', 'r')
    out_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/features_clips.h5py', 'a')
    my_model = load_model()

    '''
    # devset
    set = 'devset'
    devset = out_file.create_group(set)
    for v in in_file[set]:
        video = devset.create_group(v)
        num_clips = len(in_file[set][v].keys())
        print("{} with {} clips".format(v, num_clips))
        bar = progressbar.ProgressBar(max_value=num_clips)
        for i, c in enumerate(in_file[set][v]):
            X = in_file[set][v][c]
            # print(i, cl.shape)
            assert X.shape == (3, 16, 112, 112), "{} from {} has shape {}".format(c, v, X.shape)
            Y = extract_features_clip(X, my_model)
            video.create_dataset(c, data=Y)
            bar.update(i)
    '''

    # testset
    set = 'testset'
    testset = out_file.create_group(set)
    for v in in_file[set]:
        video = testset.create_group(v)
        num_clips = len(in_file[set][v].keys())
        print("{} with {} clips".format(v, num_clips))
        bar = progressbar.ProgressBar(max_value=num_clips)
        for i, c in enumerate(in_file[set][v]):
            X = in_file[set][v][c]
            # print(i, cl.shape)
            assert X.shape == (3, 16, 112, 112), "{} from {} has shape {}".format(c, v, X.shape)
            Y = extract_features_clip(X, my_model)
            video.create_dataset(c, data=Y)
            bar.update(i)

