from __future__ import print_function
import logging
import traceback

import numpy as np
import progressbar
from keras.layers import (LSTM, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, Input,
                          MaxPooling3D, TimeDistributed, ZeroPadding3D, Reshape)
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
import time
from telegramBot import bot
from img_features import load_and_set as l
import h5py
import os
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

# Execution time
t0 = time.time()
# Option to see all the np array on the console
# np.set_printoptions(threshold=np.nan)

############################################
number = 51  # Model number
model_json_file = 'src/LSTM_{}_model.json'.format(number)
model_fig = 'src/model_{}.png'.format(number)
model_checkpoint = 'src/LSTM_{}_weights.hdf5'.format(number)

video_features_dir = '/home/lluc/PycharmProjects/TFG/video/video_features/devset/changed_fps'
video_annotations_file = '/home/lluc/PycharmProjects/TFG/video/data/annotations/devset-video.txt'
###########################################
loss = 'mean_squared_error'
optimizer = RMSprop(lr=0.0000001)

epochs = 2
batch_size = 1
###########################################

try:
    # bot.send_message('Model ' + str(number))


    def load_features_and_labels():
        in_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/features_labels_clips.h5py')
        out_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/training.h5py')
        X = np.empty(shape=(0, 4096))
        Y = []

        bar = progressbar.ProgressBar(max_value=52)

        for i, v in enumerate(in_file['devset_labels']):
            f_num = len(in_file['devset'][v].items())
            l_num = len(in_file['devset_labels'][v].items())
            my_set = 'devset_labels'
            if l_num > f_num:
                my_set = 'devset'
            for c in in_file[my_set][v]:
                f = in_file['devset'][v][c][()]
                l = in_file['devset_labels'][v][c][()]
                X = np.append(X, f, axis=0)
                Y.append(l)
            bar.update(i)
        Y = np.array(Y)
        print(X.shape)
        print(Y.shape)

        out_file.create_dataset('X', data=X)
        out_file.create_dataset('Y', data=Y)
        bot.send_message('Features and labels loaded')


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


    train_data_dir = '/home/lluc/PycharmProjects/TFG/video/data/training.h5py'
    train_data_file = h5py.File(train_data_dir, 'r')
    '''
    # 80% train (multiple of 16), 20% validation
    X = train_data_file['X'][:6336]
    Y = train_data_file['Y'][:6336]
    X_val = train_data_file['X'][6336:-8]
    Y_val = train_data_file['Y'][6336:-8]
    '''
    X = train_data_file['X'][:6336]
    Y = train_data_file['Y'][:6336]
    X_val = train_data_file['X'][6336:-8]
    Y_val = train_data_file['Y'][6336:-8]

    # reshape for network input
    X = X.reshape(X.shape[0], 1, X.shape[1])
    Y = Y.reshape(Y.shape[0], 1, 1)
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    Y_val = Y_val.reshape(Y_val.shape[0], 1, 1)

    print("Training data: {}".format(X.shape))  # (6336, 1, 4096)
    print("Training labels: {}".format(Y.shape))  # (6336, 1, 1) or (6336, 1, 2) categorical
    print("Validation data: {}".format(X_val.shape))  # (1592, 1, 4096)
    print("Validation labels: {}".format(Y_val.shape))  # (1592, 1, 1) or (1592, 1, 2) categorical

    print('Compiling model')
    model = temporal_localization_network(True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print('Model Compiled!')

    print('Fitting model ' + str(number))
    tr_loss = []
    val_loss = []
    tr_acc = []
    val_acc = []
    for i in range(1, epochs + 1):
        print('Epoch {}/{}'.format(i, epochs))
        history = model.fit(X,
                            Y,
                            batch_size=batch_size,
                            validation_data=(X_val, Y_val),
                            verbose=1,
                            nb_epoch=1,
                            shuffle=False)
        print('Reseting model states')
        tr_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        tr_acc.extend(history.history['acc'])
        val_acc.extend(history.history['val_acc'])
        model.reset_states()

    # Show plots
    x = np.arange(len(val_loss))
    fig = plt.figure(1)
    fig.suptitle('TRAINNING vs VALIDATION', fontsize=14, fontweight='bold')

    # LOSS: TRAINING vs VALIDATION
    sub_plot1 = fig.add_subplot(211)

    plt.plot(x, tr_loss, '--', linewidth=2, label='tr_loss')
    plt.plot(x, val_loss, label='va_loss')
    plt.legend(loc='upper right')

    # ACCURACY: TRAINING vs VALIDATION
    sub_plot2 = fig.add_subplot(212)

    plt.plot(x, tr_acc, '--', linewidth=2, label='tr_acc')
    plt.plot(x, val_acc, label='val_acc')
    plt.legend(loc='lower right')

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
