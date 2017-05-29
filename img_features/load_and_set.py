from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.io
from keras.preprocessing import image


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
    """Returns the img_features of all key-frames of the video in a (49,128) array. Could be devset or testset."""
    video_num = str(video_num)
    path = '/home/lluc/Documents/ME16IN/' + set_type + '/img_features/Features_From_FudanUniversity/Image_Subtask/ColorHist/video_' + video_num + '/images/'
    d = load_directory(path)
    hist = np.empty((0, 128))
    for file_name in d:
        dic = scipy.io.loadmat(file_name=path + file_name)
        if set_type == 'devset':
            value = dic.get('hsv')
        else:
            value = dic.get('ColorHist')
        hist = np.append(hist, value, axis=0)
    return hist


def get_dense_SIFT(video_num, set_type):
    video_num = str(video_num)
    path = '/home/lluc/Documents/ME16IN/' + set_type + '/img_features/Features_From_FudanUniversity/Image_Subtask/denseSIFT/video_' + video_num + '/images/'
    d = load_directory(path)
    hist = np.empty((0, 300))
    for file_name in d:
        dic = scipy.io.loadmat(file_name=path + file_name)
        v = dic.get('hists').item((0, 0))[0]  # 3 items of shape (300,1), (1200,1), (4800,1)
        v = v.reshape((1, 300))
        hist = np.append(hist, v, axis=0)
    return hist


def get_gist(video_num, set_type):
    video_num = str(video_num)
    path = '/home/lluc/Documents/ME16IN/' + set_type + '/img_features/Features_From_FudanUniversity/Image_Subtask/gist/video_' + video_num + '/images/'
    d = load_directory(path)
    hist = np.empty((0, 512))
    for file_name in d:
        dic = scipy.io.loadmat(file_name=path + file_name)
        v = dic.get('descrs')
        v = v.reshape((1, 512))
        hist = np.append(hist, v, axis=0)
    return hist


def get_hog2x2():
    dic = scipy.io.loadmat(
        '/home/lluc/Documents/ME16IN/devset/img_features/Features_From_FudanUniversity/Image_Subtask/hog2x2/video_0/images/107_102-113.jpg.mat')
    v = dic.get('hists')
    v = v.item(0)[2]  # 3 items of shape (300,1), (1200,1), (4800,1) the same as SIFT
    print type(v)
    print v.shape
    return


def get_fc7(video_num, set_type):
    video_num = str(video_num)
    path = '/home/lluc/Documents/ME16IN/' + set_type + '/img_features/Features_From_FudanUniversity/Image_Subtask/CNN/fc7/video_' + video_num + '/images/'
    d = load_directory(path)
    hist = np.empty((0, 4096))
    for file_name in d:
        dic = scipy.io.loadmat(file_name=path + file_name)
        if set_type == 'devset':
            v = dic.get('AlexNet_fc7')
        else:
            v = dic.get('fc7')
        hist = np.append(hist, v, axis=0)
    return hist


def get_fc7_video(video_num, set_type='devset'):
    video_num = str(video_num)
    path = '/home/lluc/Documents/ME16IN/' + set_type + '/features/Features_From_FudanUniversity/Video_Subtask/CNN/fc7/video_' + video_num + '/movies/'
    d = load_directory(path)
    hist = np.empty((0, 4096))
    for file_name in d:
        dic = scipy.io.loadmat(file_name=path + file_name)
        print(dic)
        if set_type == 'devset':
            v = dic.get('Alex_fc7')
            print('Dic '+file_name, v.shape)
        else:
            v = dic.get('fc7')
        hist = np.append(hist, v, axis=0)
    return hist

def set_results_SVC(video_num, res, prob, tofile, feature):
    """Sets the resuts to the submission file"""
    path = '/home/lluc/Documents/ME16IN/testset/img_features/Features_From_FudanUniversity/Image_Subtask/' + feature + '/video_' + str(
        video_num) + '/images/'
    if feature == 'fc7' or feature == 'prob':
        path = '/home/lluc/Documents/ME16IN/testset/img_features/Features_From_FudanUniversity/Image_Subtask/CNN/' + feature + '/video_' + str(
            video_num) + '/images'
    d = load_directory(path)
    # overriting what was inside already
    for i in range(len(d)):
        p = prob.item((i, 0))
        if res[i] == 1:
            p = prob.item((i, 1))
        s = str(d[i])[:-4]
        if '.jpg' not in s:
            s += '.jpg'
        tofile.write(
            'video_' + str(video_num) + ',' + s + ',' + str(res[i]) + ',' + str(p) + '\n')


def set_results_SVC_ResNet(names, predictions, prob, tofile):
    """Sets the resuts to the submission file"""
    i = 0
    for name in names:
        pred = predictions[i]
        p = prob.item((i, 0))
        if pred == '1' or pred == 1:
            p = 1 - p
        tofile.write(name[0] + ',' + name[1] + ',' + str(pred) + ',' + str(p) + '\n')
        i += 1


def set_results_one_class(video_num, res, tofile, feature):
    """Sets the resuts to the submission file"""
    path = '/home/lluc/Documents/ME16IN/testset/img_features/Features_From_FudanUniversity/Image_Subtask/' + feature + '/video_' + str(
        video_num) + '/images/'
    if feature == 'fc7' or feature == 'prob':
        path = '/home/lluc/Documents/ME16IN/testset/img_features/Features_From_FudanUniversity/Image_Subtask/CNN/' + feature + '/video_' + str(
            video_num) + '/images'
    d = load_directory(path)
    # overriting what was inside already
    for i in range(len(d)):
        if res[i] == -1:
            pred = '0'
        else:
            pred = '1'
        s = str(d[i])[:-4] + '.jpg'
        tofile.write('video_' + str(video_num) + ',' + s + ',' + pred + ',' + '1' + '\n')


def load_images_devset():
    """Load all the images with annotations from devset"""
    total_video_num_devtest = 52
    img_array = []
    for i in range(total_video_num_devtest):
        path = '/home/lluc/Documents/ME16IN/devset/videos/video_' + str(i) + '/images/'
        d = load_directory(path)
        for img_name in d:
            img = image.load_img(path + img_name, target_size=(224, 224))
            img = image.img_to_array(img)
            img_array.append(img)
    return np.array(img_array)


def load_labeled_images_devset():
    """Load just the images from devset"""
    img_array = []
    annotations_path = '/home/lluc/Documents/ME16IN/devset/annotations/devset-image.txt'
    with open(annotations_path) as inputfile:
        for line in inputfile:
            l = line.strip().split(',')
            path = '/home/lluc/Documents/ME16IN/devset/videos/' + l[0] + '/images/' + l[1]
            img = image.load_img(path, target_size=(224, 224))
            img = image.img_to_array(img)
            img_array.append(img)
    return np.array(img_array)


def load_label_and_images_devset():
    """Load just the images (and the labels) with annotations from devset"""
    img_array = []
    label_array = []
    annotations_path = '/home/lluc/Documents/ME16IN/devset/annotations/devset-image.txt'
    with open(annotations_path) as inputfile:
        for line in inputfile:
            l = line.strip().split(',')
            path = '/home/lluc/Documents/ME16IN/devset/videos/' + l[0] + '/images/' + l[1]
            img = image.load_img(path, target_size=(224, 224))
            img = image.img_to_array(img)
            img_array.append(img)
            label_array.append(int(l[2]))
    return np.array(img_array), np.array(label_array)


def load_images_testset():
    """Load all the images from testset"""
    total_video_num_testset = 26
    img_array = []
    for i in range(52, 52 + total_video_num_testset):
        path = '/home/lluc/Documents/ME16IN/testset/videos/video_' + str(i) + '/images/'
        d = load_directory(path)
        for img_name in d:
            img = image.load_img(path + img_name, target_size=(224, 224))
            img = image.img_to_array(img)
            img_array.append(img)
    return np.array(img_array)


def load_labeled_images_testset():
    """Load just the images with annotations from testset"""
    img_array = []
    img_names = []
    annotations_path = '/home/lluc/Documents/ME16IN/testset/annotations/testset-image.txt'
    with open(annotations_path) as inputfile:
        for line in inputfile:
            l = line.strip().split(',')
            path = '/home/lluc/Documents/ME16IN/testset/videos/' + l[0] + '/images/' + l[1]
            img = image.load_img(path, target_size=(224, 224))
            img = image.img_to_array(img)
            img_array.append(img)
            name = [l[0], l[1]]
            img_names.append(name)
        img_names = np.array(img_names)
        img_array = np.array(img_array)
    return img_array, img_names
