import logging
import traceback

import numpy as np
from keras.layers import (LSTM, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, Input,
                          MaxPooling3D, TimeDistributed, ZeroPadding3D)
from keras.models import Model, Sequential
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
np.set_printoptions(threshold=np.nan)

############################################
number = 51  # Model number
model_json_file = 'src/LSTM_{}_model.json'.format(number)
model_fig = 'src/model_{}.png'.format(number)
model_checkpoint = 'src/LSTM_{}_weights.hdf5'.format(number)

video_features_dir = '/home/lluc/PycharmProjects/TFG/video/video_features/devset/changed_fps'
video_annotations_file = '/home/lluc/PycharmProjects/TFG/video/data/annotations/devset-video.txt'
###########################################
loss = 'categorical_crossentropy'
optimizer = 'rmsprop'

epochs = 100
batch_size = 1
###########################################

try:
    bot.send_message('Model ' + str(number))


    def load_features_and_labels(from_file=None, save_to_file=False):
        if from_file == None:
            # features and labels
            annotations = open(video_annotations_file, 'r')
            x = np.empty((0, 4096))  # features
            y = np.array([])  # labels
            count = 0
            for line in annotations:
                line = line.split(",")
                v_num = line[0]
                v_name = line[1]
                v_ann = int(line[2])
                f_name = v_num + "_features_activitynet.hdf5"
                print (v_num, v_name, v_ann, f_name)
                if os.path.isfile(os.path.join(video_features_dir, f_name)):
                    f = h5py.File(os.path.join(video_features_dir, f_name), 'r')
                    try:
                        # feature vector exists
                        item = f[v_name[:-4]][()]
                        if not item.shape[0] == 0:
                            x = np.append(x, item, axis=0)
                        for i in range(item.shape[0]):
                            y = np.append(y, v_ann)
                    except:
                        # feature vector does not exists
                        count += 1
            y.reshape(x.shape[0], 1)

            print(x.shape, y.shape)  # ((7040, 4096), (7040,))
            if save_to_file:
                f = h5py.File("/home/lluc/PycharmProjects/TFG/video/video_features/devset/video_features.h5py", 'w')
                f.create_dataset(name="train", shape=x.shape, data=x)
                f.create_dataset(name="val", shape=y.shape, data=y)
                f.close()
        else:
            f = h5py.File(from_file, 'r')
            x = f['train'][...]
            y = f['val'][...]
            f.close()
        train = (x.shape[0] * 80) // 100
        x_val = x[train:]
        y_val = y[train:]
        x = x[:train]
        y = y[:train]
        return x, y, x_val, y_val


    def temporal_localization_network(summary=False):
        input_features = Input(batch_shape=(1, 1, 4096,), name='features')
        input_normalized = BatchNormalization(name='normalization')(input_features)
        input_dropout = Dropout(p=.5)(input_normalized)
        lstm = LSTM(512, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
        output_dropout = Dropout(p=.5)(lstm)
        output = TimeDistributed(Dense(2, activation='softmax'), name='fc')(output_dropout)
        model = Model(input=input_features, output=output)

        if summary:
            model.summary()
        return model

    X, Y, X_val, Y_val = load_features_and_labels("/home/lluc/PycharmProjects/TFG/video/video_features/devset/video_features.h5py")
    X = X.reshape(X.shape[0], 1, X.shape[1])
    Y = to_categorical(Y)
    Y = Y.reshape(Y.shape[0], 1, Y.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    Y_val = to_categorical(Y_val)
    Y_val = Y_val.reshape(Y_val.shape[0], 1, Y_val.shape[1])
    print ("Training data: {}".format(X.shape))
    print ("Training labels: {}".format(Y.shape))
    print ("Validation data: {}".format(X_val.shape))
    print ("Validation labels: {}".format(Y_val.shape))

    print('Compiling model')
    model = temporal_localization_network(True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print('Model Compiled!')

    print ('Fitting model ' + str(number))
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
        model.reset_states()
        tr_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        tr_acc.extend(history.history['acc'])
        val_acc.extend(history.history['val_acc'])

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

    plt.savefig(model_fig)
    execution_time = (time.time() - t0) / 60
    print('Execution time model ' + str(number) + ' (min): ' + str(execution_time))
    bot.send_message('Execution time model ' + str(number) + ' (min): ' + str(execution_time))
    bot.send_image(model_fig)

except Exception:
    logging.error(traceback.format_exc())
    bot.send_message('Exception caught')
