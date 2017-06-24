"""
Author: Lluc Cardoner

Script for labeling the test segments with the labels predicted for the video features.

"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import h5py

model_num = 65
annotations = '/home/lluc/Documents/ME16IN/testset/annotations/testset-video.txt'
predictions = '/home/lluc/PycharmProjects/TFG/video/src/LSTM_{}_predictions.h5'.format(model_num)
pred_file = h5py.File(predictions, 'a')
ann_file = open(annotations, 'r')


class Segment:
    """Class that represents a video segment and its properties."""

    def __init__(self, video, name, label, prob, rank):
        self.video = video
        self.name = name
        self.label = label
        self.prediction = -1
        self.prob = prob
        self.rank = rank
        self.start_frame = int(name[:-4].split('-')[0])
        self.end_frame = int(name[:-4].split('-')[1])
        self.clip_nums = []  # list
        self.num_frames = self.end_frame - self.start_frame + 1

    def __str__(self):
        return ('{} - '
                'name: {}, '
                'prediction: {}, '
                'num_frames: {}, '
                'clips: {}').format(self.video, self.name, self.prediction, self.num_frames, self.clip_nums)

    def get_num_frames(self):
        return self.num_frames

    def set_clip_num(self, num):
        self.clip_nums.append(num)

    def get_clip_nums(self):
        return self.clip_nums

    def set_prediction(self, p):
        self.prediction = p

    def get_name(self):
        return self.name


def back_label_mapping():
    segments = []
    for line in ann_file:
        l = line.rstrip().split(',')
        # print(l)
        segments.append(Segment(l[0], l[1], l[2], l[3], l[4]))
    print("Number of segments with annotations: {}".format(len(segments)))

    # get the clip nums for each segment and the number of frames in each clip
    length = 16
    total_videos = 26
    for v in range(52, 52 + total_videos):
        frames = 0
        for s in segments:
            video_num = int(s.video.split('_')[1])
            if v == video_num:
                # print (s.video, s.name)
                num_frames_seg = s.get_num_frames()
                clip_num_1 = frames // length
                frames2 = frames + num_frames_seg
                clip_num_2 = frames2 // length
                if clip_num_1 == clip_num_2:
                    s.set_clip_num((clip_num_1, num_frames_seg))
                    # print(clip_num_1)
                elif clip_num_2 > clip_num_1:
                    j = frames  # counts the frames form [frames to frames2]
                    z = 0  # number of frames in the previous clip for this segment
                    for i in range(clip_num_1, clip_num_2 + 1):
                        while j // 16 == i and j < frames2:
                            j += 1
                        s.set_clip_num((i, j - frames - z))
                        z = j - frames
                        # print(i)
                frames = frames2

    # get label from each segment from the clip (featrue vector) prediction
    new_pred = pred_file.create_group('back_label_mapping')
    for v in range(52, 52 + total_videos):
        vid = new_pred.create_group('video_{}'.format(v))
        x = 0
        for s in segments:
            video_num = int(s.video.split('_')[1])
            if v == video_num:
                a = []  # clip labels
                w = []  # weights
                print(s)
                for e in s.get_clip_nums():
                    clip = e[0]
                    weight = e[1]
                    try:
                        p = pred_file['LSTM_output']['video_{}'.format(v)]['prediction'][clip]
                        a.extend(p)
                        w.append(weight)
                    except ValueError:
                        continue
                # print(a)
                # print(w)
                p = 0
                if sum(w) != 0:
                    p = np.average(a, weights=w)
                    s.set_prediction(p)
                # save the new label prediction of the segment
                name = '{}_{}'.format(x, s.get_name())
                vid.create_dataset(name, data=p)
                x += 1

back_label_mapping()
