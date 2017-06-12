"""
Author: Lluc Cardoner

Classifier OneClassSVM for CNN fc7 features.
"""

import numpy as np
import time
from sklearn.svm import \
    OneClassSVM  # http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
import load_and_set as l

np.set_printoptions(threshold=np.nan)

total_video_num_devtest = 52
total_video_num_testset = 26

num_features_fc7 = 4096

clf = OneClassSVM()

x = np.empty((0, num_features_fc7))
y = np.empty((0, 1), dtype=int)

p = np.empty((0, num_features_fc7))
q = np.empty((0, 1))

print ('Start training')
# get the devset data
for i in range(total_video_num_devtest):
    x = np.append(x, l.get_fc7(i, 'devset'), axis=0)
    y = np.append(y, l.get_annotations('image', i, 'devset'))

# train the classifier
t0 = time.time()
clf.fit(x, y)
print("Training time: %s s" % (time.time() - t0))

# get the set data and write to file
open('/home/lluc/Documents/trec_eval.8.1/SVM_results/me16in_wien_image_fc7oneclass.txt', 'w').close()
f = open('/home/lluc/Documents/trec_eval.8.1/SVM_results/me16in_wien_image_fc7oneclass.txt', 'a')

for i in range(total_video_num_devtest, total_video_num_devtest + total_video_num_testset):
    # p = np.append(p, lf.get_color_hist(i, 'testset'), axis=0)
    # q = np.append(q, lf.get_annotations('image', i, 'testset'))
    p = l.get_fc7(i, 'testset')
    q = l.get_annotations('image', i, 'testset')

    # test the classifier
    res = clf.predict(p)

    # write results to file
    l.set_results_one_class(i, res, f, 'fc7')
