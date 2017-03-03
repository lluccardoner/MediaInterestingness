from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.io


def load_directory(mypath):
    """Returns the name of all the files in a directory"""
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles


def get_annotations(task, video_num, set_type):
    """Returns the annotations for interestingness of image or video for the given video number"""
    path = '/home/lluc/Documents/ME16IN/' + set_type + '/annotations/' + set_type + '-' + task + '.txt'
    val = []
    with open(path) as inputfile:
        for line in inputfile:
            l = line.strip().split(',')
            if l[0] == 'video_' + str(video_num):
                val.append(int(l[2]))
    return np.array(val)


def get_color_hist(video_num, set_type):
    """Returns the features of all key-frames of the video in a (49,128) array. Could be devset or testset."""
    video_num = str(video_num)
    path = '/home/lluc/Documents/ME16IN/' + set_type + '/features/Features_From_FudanUniversity/Image_Subtask/ColorHist/video_' + video_num + '/images/'
    d = load_directory(path)
    hist = np.empty((0, 128))
    for file_name in d:
        dic = scipy.io.loadmat(file_name=path + file_name)
        if (set_type == 'devset'):
            value = dic.get('hsv')
        else:
            value = dic.get('ColorHist')
        hist = np.append(hist, value, axis=0)
    return hist


def set_results(video_num, res, prob, tofile):
    path = '/home/lluc/Documents/ME16IN/testset/videos/video_' + str(video_num) + '/images'
    d = load_directory(path)

    # overriting what was inside already
    for i in range(len(d)):
        p = prob.item((i, 0))
        if res[i] == 1:
            p = prob.item((i, 1))
        tofile.write('video_' + str(video_num) + ',' + str(d[i]) + ',' + str(res[i]) + ',' + str(p) + '\n')
