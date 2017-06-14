from __future__ import print_function

import logging
import os
import traceback

import h5py
import numpy as np
import progressbar

import video.pre_processing.video_prop as vp
from image.SVM_image_features import load_and_set as l
from telegramBot import bot


def videos_to_h5py():
    devset_num_videos = 52
    testset_num_videos = 26

    less_than_16_d = 0
    less_than_16_t = 0
    less_than_8_d = 0
    less_than_8_t = 0
    min_frames_d = 100
    min_frames_t = 100

    total_continues = 0

    f = h5py.File("videos.h5py", "a")

    # devset videos
    devset = f.create_group('devset')
    for v in range(devset_num_videos):
        v_array = np.empty((3, 1, 112, 112))
        path = '/home/lluc/Documents/ME16IN/devset/videos/video_{}/movies'.format(v)
        dir = l.load_directory(path)
        for name in dir:
            video_path = os.path.join(path, name)
            frames = vp.get_num_frames(video_path)
            if 16 > frames > 8:
                less_than_16_d += 1
            elif frames <= 8:
                less_than_8_d += 1
            if frames < min_frames_d:
                min_frames_d = frames
            new_array = vp.video_to_array(video_path, resize=(112, 112))
            if new_array is None:
                total_continues += 1
                continue
            print(v, name, new_array.shape)
            v_array = np.append(v_array, new_array, axis=1)
        print('-' * 5, v, v_array.shape)
        devset.create_dataset('video_{}'.format(v), data=v_array)

    # testset videos
    devset = f.create_group('testset')
    for v in range(devset_num_videos, devset_num_videos + testset_num_videos):
        v_array = np.empty((3, 1, 112, 112))
        path = '/home/lluc/Documents/ME16IN/testset/videos/video_{}/movies'.format(v)
        dir = l.load_directory(path)
        for name in dir:
            video_path = os.path.join(path, name)
            frames = vp.get_num_frames(video_path)
            if 16 > frames > 8:
                less_than_16_t += 1
            elif frames <= 8:
                less_than_8_t += 1
            if frames < min_frames_t:
                min_frames_t = frames
            new_array = vp.video_to_array(video_path, resize=(112, 112))
            if new_array is None:
                total_continues += 1
                continue
            print(v, name, new_array.shape)
            v_array = np.append(v_array, new_array, axis=1)
        print('-' * 5, v, v_array.shape)
        devset.create_dataset('video_{}'.format(v), data=v_array)

    print("Devset - between 8 and 16: {},   less than 8: {},    min_frames: {}".format(less_than_16_d, less_than_8_d,
                                                                                       min_frames_d))
    print("Testset - between 8 and 16: {},   less than 8: {},    min_frames: {}".format(less_than_16_t, less_than_8_t,
                                                                                        min_frames_t))
    f.close()


def make_clips(length=16):
    try:
        bot.send_message("Making clips")
        out_file = h5py.File("data/video_clips.h5py", "a")
        in_file = h5py.File("data/videos.h5py", "r")

        # devset
        devset = out_file.create_group('devset')
        for v in in_file['devset']:
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
            video = devset.create_group(v)
            m = in_file['devset'][v][()]
            num_clips = m.shape[1] // length
            res = m.shape[1] % length
            print(v, m.shape, num_clips, res)
            if res != 0:
                m = m[:, :-res, :, :]
            clips = np.split(m, num_clips, axis=1)
            for i, c in enumerate(clips):
                assert c.shape == (3, 16, 112, 112), "Clip_{} of {} has shape {}".format(i, v, c.shape)
                video.create_dataset('clip_{}'.format(i), data=c)
                bar.update(i)

        # testset
        testset = out_file.create_group('testset')
        for v in in_file['testset']:
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
            video = testset.create_group(v)
            m = in_file['testset'][v][()]
            num_clips = m.shape[1] // length
            res = m.shape[1] % length
            print(v, m.shape, num_clips)
            if res != 0:
                m = m[:, :-res, :, :]
            clips = np.split(m, num_clips, axis=1)
            for i, c in enumerate(clips):
                assert c.shape == (3, 16, 112, 112), "Clip_{} of {} has shape {}".format(i, v, c.shape)
                video.create_dataset('clip_{}'.format(i), data=c)
                bar.update(i)
        bot.send_message("Clips done")
    except Exception:
        logging.error(traceback.format_exc())
        bot.send_message('Exception caught')


make_clips()
