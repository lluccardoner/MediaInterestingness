"""
Author: Lluc Cardoner

Script for predicting label the feature vectors extracted from clips of 16 frames.

"""

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import h5py

annotations = '/home/lluc/Documents/ME16IN/devset/annotations/devset-video.txt'

in_file = open(annotations, 'r')
out_file = h5py.File('/home/lluc/PycharmProjects/TFG/video/data/features_labels_clips.h5py', 'a')


class Segment:
    """Class that represents a video segment and its properties."""
    def __init__(self, video, name, label, prob, rank):
        self.video = video
        self.name = name
        self.label = label
        self.prob = prob
        self.rank = rank
        self.start_frame = int(name[:-4].split('-')[0])
        self.end_frame = int(name[:-4].split('-')[1])
        self.clip_nums = []  # list
        self.num_frames = self.end_frame - self.start_frame + 1

    def __str__(self):
        return ('{} - '
                'name: {}, '
                'num_frames: {}, '
                'clips: {}').format(self.video, self.name, self.num_frames, self.clip_nums)

    def get_num_frames(self):
        return self.num_frames

    def set_clip_num(self, num):
        self.clip_nums.append(num)


def label_mapping():
    segments = []
    for line in in_file:
        l = line.split(',')
        segments.append(Segment(l[0], l[1], l[2], l[3], l[4]))
    print("Number of segments with annotations: {}".format(len(segments)))

    # get the clip nums for each segment
    length = 16
    total_videos = 52
    for v in range(total_videos):
        frames = 0
        for s in segments:
            video_num = int(s.video.split('_')[1])
            if v == video_num:
                # print (s.video, s.name)
                clip_num_1 = frames // length
                frames = frames + s.get_num_frames()
                clip_num_2 = frames // length
                if clip_num_1 == clip_num_2:
                    s.set_clip_num(clip_num_1)
                    # print(clip_num_1)
                elif clip_num_2 > clip_num_1:
                    for i in range(clip_num_1, clip_num_2 + 1):
                        s.set_clip_num(i)
                        # print(i)

    # get the probabilities from each segment for a clip
    values = {}
    for v in range(total_videos):
        clips = {}
        for s in segments:
            video_num = int(s.video.split('_')[1])
            if v == video_num:
                c = s.clip_nums
                for n in c:
                    tmp = clips.get(n, [])
                    tmp.append(float(s.prob))
                    clips[n] = tmp
        values['video_{}'.format(v)] = clips

    # calculate the average and save the label
    dev_labels = out_file.create_group('devset_labels')
    for v in values.keys():
        vi = dev_labels.create_group(v)
        cl = values.get(v)
        for c in cl.keys():
            label = np.array(cl.get(c)).mean()
            vi.create_dataset('clip_{}'.format(c), data=label)
            print(c, label)


def label_mapping_with_weights():
    segments = []
    for line in in_file:
        l = line.split(',')
        segments.append(Segment(l[0], l[1], l[2], l[3], l[4]))
    print("Number of segments with annotations: {}".format(len(segments)))

    # get the clip nums for each segment
    length = 16
    total_videos = 52
    for v in range(total_videos):
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

    # get the probabilities from each segment for a clip
    values = {}
    for v in range(total_videos):
        clips = {}
        for s in segments:
            video_num = int(s.video.split('_')[1])
            if v == video_num:
                c = s.clip_nums
                for n in c:  # n is a tupple with (clip_num, num_frames)
                    clip_num = n[0]
                    frms = n[1]
                    tmp = clips.get(clip_num, [])
                    tmp.append((float(s.prob), frms))
                    clips[clip_num] = tmp
        values['video_{}'.format(v)] = clips

    # calculate the average and save the label
    dev_labels = out_file.create_group('devset_labels_weighted')
    for v in values.keys():
        vi = dev_labels.create_group(v)
        cl = values.get(v)
        for c in cl.keys():
            arr = cl.get(c)  # array of values for the clip 'c'
            a = [x[0] for x in arr]  # probabilities
            w = [x[1] for x in arr]  # weights
            if sum(w) != 0:
                label = np.average(a, weights=w)
                vi.create_dataset('clip_{}'.format(c), data=label)
            print('{}: clip {} - label {}'.format(v, c, label))


