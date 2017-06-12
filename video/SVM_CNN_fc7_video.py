"""

Author: Lluc Cardoner

SVM classifier for CNN fc7 frame based features of videos.
Predicting video interestingness.

"""

import numpy as np
import time
from sklearn.svm import SVC  # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#
import SVM_image_features.load_and_set as l

np.set_printoptions(threshold=np.nan)

total_video_num_devtest = 52
total_video_num_testset = 26

num_features_fc7 = 4096
clf = SVC(probability=True)

x = np.empty((0, num_features_fc7))
y = np.empty((0, 1), dtype=int)

p = np.empty((0, num_features_fc7))
q = np.empty((0, 1))

print ('Get data')
# get the devset data
for i in range(total_video_num_devtest):
    features = l.get_fc7_video(i, 'devset') ### The feature vector are given per frame of each clip
    print (features.shape)
    x = np.append(x, features, axis=0)
    y = np.append(y, l.get_annotations('video', i, 'devset'))

print(x.shape, y.shape)
# train the classifier
print ('Start training')
t0 = time.time()
clf.fit(x, y)
print("Training time: %s s" % (time.time() - t0))

# get the set data and write to file
open('/home/lluc/Documents/trec_eval.8.1/SVM_results/me16in_wien_video_fc7.txt', 'w').close()
f = open('/home/lluc/Documents/trec_eval.8.1/SVM_results/me16in_wien_video_fc7.txt', 'a')

print('Start predicting')
for i in range(total_video_num_devtest, total_video_num_devtest + total_video_num_testset):
    # p = np.append(p, lf.get_color_hist(i, 'testset'), axis=0)
    # q = np.append(q, lf.get_annotations('image', i, 'testset'))
    p = l.get_fc7_video(i, 'testset')
    q = l.get_annotations('video', i, 'testset')

    # test the classifier
    res = clf.predict(p)
    prob = clf.predict_proba(p)

    # write results to file
    l.set_results_SVC(i, res, prob, f, 'fc7')
