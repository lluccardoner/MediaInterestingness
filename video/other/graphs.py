"""

Author: Lluc Cardoner

Graphs the training and validation losses for each video individually.

"""
from __future__ import print_function
from image.ResNet50.other import path_tools as pt
import numpy as np
import matplotlib.pyplot as plt
import os

model_number = 56
model_losses = '/home/lluc/PycharmProjects/TFG/video/src/LSTM_{}_losses.txt'.format(model_number)
model_fig = '/home/lluc/PycharmProjects/TFG/video/src/model_{}_{}.png'.format(model_number, 0)

input_file = open(model_losses, 'r')


def tr_val_loss(video_num=-1):
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


def tr_val_plots(loss_dic, fig_dir, legend=True, show=False, save=True):
    # Plots
    key, value = loss_dic.popitem()
    loss_dic[key] = value
    x = np.arange(len(value))
    NUM_COLORS = len(loss_dic.keys())
    fig = plt.figure(1)
    cm = plt.get_cmap('gist_rainbow')
    fig.add_subplot(111).set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    fig.suptitle('TRAINING and VALIDATION', fontsize=14, fontweight='bold')

    # LOSS: TRAINING and VALIDATION
    for key in loss_dic:
        y = loss_dic.get(key)
        assert len(x) == len(y)
        plt.plot(x, loss_dic.get(key), label=key)

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

'''
# general training and validation losses
t, v = tr_val_loss()
dic['tr_loss'] = t
dic['val_loss'] = v
'''

# training and validation losses of videos
# video_0 is for validation!!!
for i in range(1, 51):
    if i != 16:
        t, v = tr_val_loss(video_num=i)
        # dic['tr_loss_{}'.format(i)] = t
        dic['val_loss_video_{}'.format(i)] = v

print(dic.keys())
tr_val_plots(dic, model_fig, legend=False, show=True, save=False)
