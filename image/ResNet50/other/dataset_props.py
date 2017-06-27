"""

Author: Lluc Cardoner

Script for printing all the properties of the dataset.

"""
from __future__ import print_function

import json


def print_dataset_2016(subsets=None, subtasks=None):
    if subsets is None:
        subsets = ['devset', 'testset']

    if subtasks is None:
        subtasks = ['image', 'video']

    dataset = '/home/lluc/PycharmProjects/TFG/video/pre_processing/dataset.json'

    dat_file = open(dataset)
    data = json.load(dat_file)
    total_videos_devset = data['dataset'][0]['devset'][0]['num_videos']
    total_videos_testset = data['dataset'][0]['testset'][0]['num_videos']
    total_segments_devset = data['dataset'][0]['devset'][0]['num_segments']
    total_segments_testset = data['dataset'][0]['testset'][0]['num_segments']

    annotations = '/home/lluc/Documents/ME16IN/{subset}/annotations/{subset}-{subtask}.txt'.format(subset=subsets[0],
                                                                                                   subtask=subtasks[0])
    not_interesting_devset = 0
    interesting_devset = 0
    with open(annotations) as ann_file:
        for i, l in enumerate(ann_file):
            label = l.rstrip().split(',')[2]
            if label == '0':
                not_interesting_devset += 1
            else:
                interesting_devset += 1
    total_segments_devset_ann = i + 1

    annotations = '/home/lluc/Documents/ME16IN/{subset}/annotations/{subset}-{subtask}.txt'.format(subset=subsets[1],
                                                                                                   subtask=subtasks[0])
    not_interesting_testset = 0
    interesting_testset = 0
    with open(annotations) as ann_file:
        for i, l in enumerate(ann_file):
            label = l.rstrip().split(',')[2]
            if label == '0':
                not_interesting_testset += 1
            else:
                interesting_testset += 1
    total_segments_testset_ann = i + 1

    print('Devset videos: {}, seg/img: {}, ann: {}, 0: {}, 1: {}'.format(total_videos_devset, total_segments_devset,
                                                                         total_segments_devset_ann,
                                                                         not_interesting_devset, interesting_devset))
    print('Testset videos: {}, seg/img: {}, ann: {}, 0: {}, 1: {}'.format(total_videos_testset, total_segments_testset,
                                                                          total_segments_testset_ann,
                                                                          not_interesting_testset, interesting_testset))


print_dataset_2016()
