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

model_number = 60
model_losses = '/home/lluc/PycharmProjects/TFG/video/src/LSTM_{}_losses.txt'.format(model_number)
model_fig = '/home/lluc/PycharmProjects/TFG/video/src/model_{}_{}.png'.format(model_number, 0)


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


def labels_ground_truth(video_num=0, frame_based=True):
    """Return the labels of each segment from the ground truth"""
    file_dir = '/home/lluc/Documents/ME16IN/devset/annotations/devset-video.txt'
    input_file = open(file_dir, 'r')
    input_file.seek(0)  # offset of 0
    labels = []

    for line in input_file:
        line = line.rstrip().split(',')  # epoch/video/tr_loss/val_loss
        if line[0].split('_')[1] == str(video_num):
            # print(line)
            seg = line[1][:-4].split('-')
            num_frames = int(seg[1]) - int(seg[0])
            for _ in range(num_frames):
                labels.append(float(line[3]))

    return labels


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

    fig.suptitle('TRAINING and VALIDATION', fontsize=14, fontweight='bold')

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


dic = {}


# # general training and validation losses
# t, v = tr_val_loss()
# dic['tr_loss'] = t
# dic['val_loss'] = v


# training and validation losses of videos
# video_0 or video_51 are for validation!!!
# for i in range(1, 52):
#     if i != 16:
#         t, v = tr_val_loss(video_num=i)
#         dic['tr_loss_{}'.format(i)] = t
#         # dic['val_loss_video_{}'.format(i)] = v


# get the training labels for one video
vid = 0
# tr_labels = labels_training_lstm(vid)
# labels2 = []
# # for e in tr_labels:
# #     for i in range(16):
# #         labels2.append(e)
# dic['label_clip_{}'.format(vid)] = tr_labels

# get the grountruth labels for one video
# ann_labels = labels_ground_truth(vid, frame_based=True)
# dic['label_seg_{}'.format(vid)] = ann_labels


# print(dic.keys())
plots(dic, model_fig, legend=True, show=True, save=False)
