"""
Author: Lluc Cardoner

Classifier SVM for color histogram features.
"""


import numpy as np
import time
from sklearn.svm import SVC  # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#
import load_and_set as l

np.set_printoptions(threshold=np.nan)

# x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])

total_video_num_devtest = 52
total_video_num_testset = 26

num_features_color_hist = 128
clf = SVC(probability=True)

x = np.empty((0, num_features_color_hist))
y = np.empty((0, 1), dtype=int)

p = np.empty((0, num_features_color_hist))
q = np.empty((0, 1))

# get the devset data
for i in range(total_video_num_devtest):
    x = np.append(x, l.get_color_hist(i, 'devset'), axis=0)
    y = np.append(y, l.get_annotations('image', i, 'devset'))

# train the classifier
print 'Start training'
t0 = time.time()
clf.fit(x, y)
print("Training time: %s s" % (time.time() - t0))

# get the set data and write to file
open('/home/lluc/Documents/trec_eval.8.1/SVM_results/me16in_wien_image_colorhist.txt', 'w').close()
f = open('/home/lluc/Documents/trec_eval.8.1/SVM_results/me16in_wien_image_colorhist.txt', 'a')

for i in range(total_video_num_devtest, total_video_num_devtest + total_video_num_testset):
    # p = np.append(p, lf.get_color_hist(i, 'testset'), axis=0)
    # q = np.append(q, lf.get_annotations('image', i, 'testset'))
    p = l.get_color_hist(i, 'testset')
    q = l.get_annotations('image', i, 'testset')

    # test the classifier
    res = clf.predict(p)
    prob = clf.predict_proba(p)
    # write results to file
    l.set_results_SVC(i, res, prob, f, 'ColorHist')
