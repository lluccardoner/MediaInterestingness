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
    with open(annotations) as ann_file:
        for i, l in enumerate(ann_file):
            pass
    total_segments_devset_ann = i+1

    annotations = '/home/lluc/Documents/ME16IN/{subset}/annotations/{subset}-{subtask}.txt'.format(subset=subsets[1],
                                                                                              subtask=subtasks[0])
    with open(annotations) as ann_file:
        for i, l in enumerate(ann_file):
            pass
    total_segments_testset_ann = i+1

    print ('Devset videos: {}, seg/img: {}, ann: {}'.format(total_videos_devset, total_segments_devset, total_segments_devset_ann))
    print ('Testset videos: {}, seg/img: {}, ann: {}'.format(total_videos_testset, total_segments_testset, total_segments_testset_ann))

print_dataset_2016()
