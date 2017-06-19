"""
Author: Lluc Cardoner

Script for predicting video interestingness with the trained model.

"""

import h5py
import numpy as np
import progressbar
from keras.models import Model
from keras.models import model_from_json

np.set_printoptions(threshold=np.nan)

total_video_num_devtest = 52
total_video_num_testset = 26

###############################################
number = 60

model_json_file = 'src/LSTM_{}_model.json'.format(number)
model_weights_file = 'src/LSTM_{}_weights.hdf5'.format(number)
model_predictions_file = 'src/LSTM_{}_predictions.h5'.format(number)


################################################


def load_features(video_num=-1):
    in_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/features_clips.h5py')
    out_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/testing.h5py')

    if video_num == -1:

        X = np.empty(shape=(0, 4096))

        bar = progressbar.ProgressBar(max_value=26)
        num_videos_testset = len(in_file['testset'].items())
        k = 0
        for i in range(52, 52 + num_videos_testset):
            v = 'video_{}'.format(i)
            for j in range(len(in_file['testset'][v].items())):
                c = 'clip_{}'.format(j)
                f = in_file['testset'][v][c][()]
                X = np.append(X, f, axis=0)
            k += 1
            bar.update(k)
        bar.finish()
        print(X.shape)

        out_file.create_dataset('X', data=X)

    else:
        X = np.empty(shape=(0, 4096))

        v = 'video_{}'.format(video_num)
        f_num = len(in_file['testset'][v].items())
        bar = progressbar.ProgressBar(max_value=f_num)
        for i in range(len(in_file['testset'][v].items())):
            c = 'clip_{}'.format(i)
            f = in_file['testset'][v][c][()]
            X = np.append(X, f, axis=0)
            bar.update(i)
        bar.finish()
        print(X.shape)

        vi = out_file.create_group(v)
        vi.create_dataset('X', data=X)


def get_test_data(video_num=-1):
    test_data_dir = '/home/lluc/PycharmProjects/TFG/video/data/testing.h5py'
    test_data_file = h5py.File(test_data_dir, 'r')
    if video_num == -1:
        # all testset
        X = test_data_file['X'][()]
    else:
        v = 'video_{}'.format(video_num)
        X = test_data_file[v]['X'][()]

    # reshape for network input
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # print("Test data: {}".format(X.shape))

    return X

# load json and create model
print ('Loading model and weights...')
json_file = open(model_json_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_weights_file)

model = Model(input=loaded_model.input, output=loaded_model.output)
model.summary()


# get predictions
print ('Predicting...')
f = h5py.File(model_predictions_file, 'w')
LSTM_output = f.create_group('LSTM_output')
bar = progressbar.ProgressBar(max_value=26)


for i, v in enumerate(range(52, 52+26)):
    loaded_model.reset_states()
    X = get_test_data(v)  # get test features from one video
    nb_instances = X.shape[0]
    Y = loaded_model.predict(X, batch_size=1)
    Y = Y.reshape(nb_instances, 1)
    # save predictions
    vi = LSTM_output.create_group('video_{}'.format(v))
    vi.create_dataset('prediction', data=Y)
    bar.update(i)
bar.finish()
