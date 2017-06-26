"""

Author: Lluc Cardoner

Graphs the training and validation losses for each video individually.

"""
from __future__ import print_function

import h5py

from image.ResNet50.other import path_tools as pt
import numpy as np
import matplotlib.pyplot as plt
import os

model_number = 65
model_losses = '/home/lluc/PycharmProjects/TFG/video/src/LSTM_{}_losses.txt'.format(model_number)
model_fig = '/home/lluc/PycharmProjects/TFG/video/src/model_{}_{}.png'.format(model_number, 0)
predictions = '/home/lluc/PycharmProjects/TFG/video/src/LSTM_{}_predictions.h5'.format(model_number)


def tr_val_loss(video_num=-1):
    """Returns the training and validation loss from all or one video from the loss text file"""
    input_file = open(model_losses, 'r')
    input_file.seek(0)  # offset of 0
    tr_loss = []
    val_loss = []

    for line in input_file:
        line = line.rstrip().split(',')  # epoch/video/tr_loss/val_loss
        if line[1] == str(video_num):
            # print(line)
            tr_loss.append(float(line[2]))
            val_loss.append(float(line[3]))

    # print(len(tr_loss), len(val_loss))
    return tr_loss, val_loss


def labels_training_lstm(video_num=-1):
    """Returns the labels of each clip used for training of all or one video"""
    training_file = '/home/lluc/PycharmProjects/TFG/video/data/training.h5py'
    training = h5py.File(training_file)
    if video_num == -1:
        Y = training['Y'][()]
    else:
        Y = training['video_{}'.format(video_num)]['Y'][()]
    return Y


def labels_ground_truth(video_num=0, set='devset', frame_based=True):
    """Return the labels of each segment from the ground truth"""
    file_dir = '/home/lluc/Documents/ME16IN/{}/annotations/{}-video.txt'.format(set, set)
    input_file = open(file_dir, 'r')
    input_file.seek(0)  # offset of 0
    labels = []

    for line in input_file:
        line = line.rstrip().split(',')
        if line[0].split('_')[1] == str(video_num):
            # print(line)
            if frame_based:
                seg = line[1][:-4].split('-')
                num_frames = int(seg[1]) - int(seg[0])
                for _ in range(num_frames):
                    labels.append(float(line[3]))
            else:
                labels.append(float(line[3]))

    return labels


def predicted_labels(video_num, group, frames):
    """Returns the predicted labels for one video.
    group: 'LSTM_output' or 'back_label_mapping'"""
    assert group in ['LSTM_output', 'back_label_mapping']
    predictions_file = h5py.File(predictions)
    if group == 'LSTM_output':
        Y = predictions_file[group]['video_{}'.format(video_num)]['prediction'][()]
    elif group == 'back_label_mapping':
        Y = []
        g = predictions_file[group]['video_{}'.format(video_num)]
        for i in range(len(g.keys())):
            for it in g.keys():
                num = it.split('_')
                seg = num[1][:-4].split('-')
                num_frames = int(seg[1]) - int(seg[0])
                if int(num[0]) == i:
                    s = g[it][()]
                    if frames:
                        for _ in range(num_frames):
                            Y.append(s)
                    else:
                        Y.append(s)
    return Y


def plots(value_dic, fig_dir, legend=True, show=False, save=True):
    # Plots
    length = 0
    for value in value_dic.values():
        if len(value) > length:
            length = len(value)
    x = np.arange(length)
    print('x_length:', len(x))
    NUM_COLORS = len(value_dic.keys())
    fig = plt.figure(1)
    cm = plt.get_cmap('gist_rainbow')
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    # fig.suptitle('TRAINING and VALIDATION', fontsize=14, fontweight='bold')

    # LOSS: TRAINING and VALIDATION
    for key in value_dic:
        y = value_dic.get(key)
        if len(y) < length:
            z = np.zeros(shape=(length - len(y),))
            y = np.append(y, z)
        assert len(x) == len(y)
        plt.plot(x, y, label=key)

    if legend:
        plt.legend(loc='upper right')
    if show:
        plt.show()

    if save:
        sample = 0
        while os.path.isfile(fig_dir):
            sample += 1
            fig_dir = '/home/lluc/PycharmProjects/TFG/video/src/model_{}_{}.png'.format(model_number, sample)
            print(fig_dir)

        plt.savefig(fig_dir)


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def plot_train_val_losses(video_nums=None, video_val=0, training=True, validation=False):
    """returns dictionary with the training and validation losses"""
    dic = {}
    # general training and validation losses
    if video_nums is None:
        t, v = tr_val_loss()
        if training:
            dic['tr_loss'] = t
        if validation:
            dic['val_loss'] = v

    else:
        # training and validation losses of videos
        assert len(video_nums) > 0
        for i in video_nums:
            if i != 16 and i != video_val:
                t, v = tr_val_loss(video_num=i)
                if training:
                    dic['tr_loss_{}'.format(i)] = t
                if validation:
                    dic['val_loss_video_{}'.format(i)] = v
    return dic


def plot_clip_labels(video_num, frame_based=False):
    """Returns dictionary with the clip labels used for training."""
    dic = {}
    # get the training labels for one video
    tr_labels = labels_training_lstm(video_num)
    if frame_based:
        labels2 = []
        for e in tr_labels:
            for i in range(16):
                labels2.append(e)
        dic['label_clip_{}'.format(video_num)] = labels2
    else:
        dic['label_clip_{}'.format(video_num)] = tr_labels


def plot_segment_labels(video_num, set, frames=True):
    """Returns a dictionary with the ground truth labels of each segment."""
    dic = {}
    # get the ground truth labels for one video
    ann_labels = labels_ground_truth(video_num, set=set, frame_based=frames)
    dic['label_seg_{}'.format(video_num)] = ann_labels
    print(dic)
    return dic


def plot_clip_prediction(video_num):
    """Returns a dictionary with the predictions obtained for each clip/"""
    dic = {}
    # label predictions (for clips)
    pred_labels = predicted_labels(video_num, 'LSTM_output', frames=False)
    dic['pred_clips_{}'.format(video_num)] = pred_labels
    print(dic)
    return dic


def plot_segment_predictions(video_num, frame_based):
    """Returns the predictions obtained for each segment after back label mapping."""
    dic = {}
    pred_labels = predicted_labels(video_num, 'back_label_mapping', frames=frame_based)
    dic['pred_seg_{}'.format(video_num)] = pred_labels
    print(dic)
    return dic


data = merge_dicts(plot_segment_predictions(65, frame_based=True), plot_segment_labels(65, set='testset', frames=True))
# print(dic.keys())
plots(data, model_fig, legend=True, show=True, save=False)
