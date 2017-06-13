"""

Author: Lluc Cardoner

Training of an LSTM network for predicting video interestingness from C3D video features

"""

from __future__ import print_function
import logging
import traceback

import keras
import numpy as np
import progressbar
from keras.layers import (LSTM, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, Input,
                          MaxPooling3D, TimeDistributed, ZeroPadding3D, Reshape)
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
import time
from telegramBot import bot
from SVM_image_features import load_and_set as l
import h5py
import os
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

# Execution time
t0 = time.time()
# Option to see all the np array on the console
# np.set_printoptions(threshold=np.nan)

############################################
number = 55  # Model number
model_json_file = 'src/LSTM_{}_model.json'.format(number)
model_fig = 'src/model_{}.png'.format(number)
model_checkpoint = 'src/LSTM_{}_weights.hdf5'.format(number)

video_features_dir = '/home/lluc/PycharmProjects/TFG/video/video_features/devset/changed_fps'
video_annotations_file = '/home/lluc/PycharmProjects/TFG/video/data/annotations/devset-video.txt'
###########################################
loss = 'mean_squared_error'

epochs = 100
batch_size = 1

lr = 0.000001
optimizer = RMSprop(lr=lr)


###########################################


def load_features_and_labels(video_num=-1):
    in_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/features_labels_clips.h5py')
    out_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/training.h5py')
    if video_num == -1:

        X = np.empty(shape=(0, 4096))
        Y = []

        bar = progressbar.ProgressBar(max_value=52)

        for i, v in enumerate(in_file['devset_labels_weighted']):
            f_num = len(in_file['devset'][v].items())
            l_num = len(in_file['devset_labels_weighted'][v].items())
            my_set = 'devset_labels_weighted'
            if l_num > f_num:
                my_set = 'devset'
            for c in in_file[my_set][v]:
                f = in_file['devset'][v][c][()]
                l = in_file['devset_labels_weighted'][v][c][()]
                X = np.append(X, f, axis=0)
                Y.append(l)
            bar.update(i)
        Y = np.array(Y)
        print(X.shape)
        print(Y.shape)

        out_file.create_dataset('X', data=X)
        out_file.create_dataset('Y', data=Y)
    else:
        X = np.empty(shape=(0, 4096))
        Y = []

        v = 'video_{}'.format(video_num)
        f_num = len(in_file['devset'][v].items())
        l_num = len(in_file['devset_labels_weighted'][v].items())
        my_set = 'devset_labels_weighted'
        if l_num > f_num:
            my_set = 'devset'
        bar = progressbar.ProgressBar(max_value=min(f_num, l_num))
        for i, c in enumerate(in_file[my_set][v]):
            f = in_file['devset'][v][c][()]
            l = in_file['devset_labels_weighted'][v][c][()]
            X = np.append(X, f, axis=0)
            Y.append(l)
            bar.update(i)
        Y = np.array(Y)
        print(X.shape)
        print(Y.shape)

        vi = out_file.create_group(v)
        vi.create_dataset('X', data=X)
        vi.create_dataset('Y', data=Y)


def temporal_localization_network(summary=False):
    input_features = Input(batch_shape=(batch_size, 1, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(p=.5)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(p=.5)(lstm)
    output = TimeDistributed(Dense(1, activation='sigmoid'), name='fc')(output_dropout)
    model = Model(input=input_features, output=output)

    if summary:
        model.summary()
    return model


def get_train_val_data(video_num=-1, split=True):
    train_data_dir = '/home/lluc/PycharmProjects/TFG/video/data/training.h5py'
    train_data_file = h5py.File(train_data_dir, 'r')
    if video_num == -1:
        # all data set
        # 80% train (multiple of 16), 20% validation
        if split:
            X = train_data_file['X'][:6336]
            Y = train_data_file['Y'][:6336]
            X_val = train_data_file['X'][6336:-7]
            Y_val = train_data_file['Y'][6336:-7]
        else:
            X = train_data_file['X'][()]
            Y = train_data_file['Y'][()]
            X_val = None
            Y_val = None
    else:
        if split:
            v = 'video_{}'.format(video_num)
            l = train_data_file[v]['X'].shape[0]
            l_train = (l * 80 // 100) + 1
            X = train_data_file[v]['X'][:l_train]
            Y = train_data_file[v]['Y'][:l_train]
            X_val = train_data_file[v]['X'][l_train:]
            Y_val = train_data_file[v]['Y'][l_train:]
        else:
            v = 'video_{}'.format(video_num)
            X = train_data_file[v]['X'][()]
            Y = train_data_file[v]['Y'][()]
            X_val = None
            Y_val = None

    # reshape for network input
    X = X.reshape(X.shape[0], 1, X.shape[1])
    Y = Y.reshape(Y.shape[0], 1, 1)
    print("Training data: {}".format(X.shape))  # (6336, 1, 4096) for all dataset
    print("Training labels: {}".format(Y.shape))  # (6336, 1, 1) for all dataset

    if split:
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        Y_val = Y_val.reshape(Y_val.shape[0], 1, 1)
        print("Validation data: {}".format(X_val.shape))  # (1584, 1, 4096) for all dataset
        print("Validation labels: {}".format(Y_val.shape))  # (1584, 1, 1) for all dataset

    if split:
        return X, Y, X_val, Y_val
    else:
        return X, Y


try:
    # bot.send_message('Model ' + str(number))

    print('Compiling model')
    model = temporal_localization_network(True)
    model.compile(optimizer=optimizer, loss=loss)
    print('Model Compiled!')

    print('Fitting model ' + str(number))
    tr_loss = []
    val_loss = []

    '''
    # just train with features of one video
    X, Y, X_val, Y_val = get_train_val_data(video_num=0)
    for i in range(1, epochs + 1):
        print('Epoch {}/{}'.format(i,epochs))
        history = model.fit(X,
                            Y,
                            batch_size=batch_size,
                            validation_data=(X_val, Y_val),
                            verbose=1,
                            nb_epoch=1,
                            shuffle=False)
        tr_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
    '''

    X_val, Y_val = get_train_val_data(video_num=0, split=False)  # get vector features of video 0 as validation for all
    for i in range(1, epochs + 1):
        for v in range(1, 52):
            print('Epoch {}/{}: video_{}'.format(i, epochs, v))
            X, Y = get_train_val_data(video_num=v, split=False)
            history = model.fit(X,
                                Y,
                                batch_size=batch_size,
                                validation_data=(X_val, Y_val),
                                verbose=1,
                                nb_epoch=1,
                                shuffle=False)
            print('Resetting model states')
            model.reset_states()
            print(history.history)
        tr_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])

    # Show plots
    x = np.arange(len(tr_loss))
    fig = plt.figure(1)
    fig.suptitle('TRAINING vs VALIDATION', fontsize=14, fontweight='bold')

    # LOSS: TRAINING vs VALIDATION
    plt.plot(x, tr_loss, '--', linewidth=2, label='tr_loss')
    plt.plot(x, val_loss, label='val_loss')
    plt.legend(loc='upper right')

    print("\n Saving model...")
    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_checkpoint)
    plt.savefig(model_fig)
    execution_time = (time.time() - t0) / 60
    print('Execution time model ' + str(number) + ' (min): ' + str(execution_time))
    bot.send_message('Execution time model ' + str(number) + ' (min): ' + str(execution_time))
    bot.send_image(model_fig)

except Exception:
    logging.error(traceback.format_exc())
    bot.send_message('Exception caught')
