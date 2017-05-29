from keras.applications.resnet50 import ResNet50
from keras.models import Model
from sklearn.svm import SVC

import img_features.load_and_set as l
from sklearn.svm import OneClassSVM
import time
import numpy as np

#############################
number = 47
kernel = 'poly'
class_weight = {0: 0.1899, 1: 1.8054}  # (for the full data set)
degree = 4
file_name = '/home/lluc/Documents/trec_eval.8.1/ResNet_SVM_results/me16in_wien_image_resnetSVM' + str(number) + '.txt'
#############################

t0 = time.time()
# Option to see all the np array on the console
np.set_printoptions(threshold=np.nan)

#################################################################
#       TRAINING                                                #
#################################################################

base_model = ResNet50(weights='imagenet')
base_model = Model(input=base_model.input, output=base_model.get_layer('flatten_1').output)
base_model.summary()

# Load images and labels
print ('Loading images and labels...')
tr_img, labels = l.load_label_and_images_devset()
print('Train images: ', tr_img.shape)  # (5054, 224, 224, 3)
print ('Labels: ', labels.shape)  # (5054,)

# Extract ResNet img_features for all the images
print ('Extracting train img_features...')
tr_features = base_model.predict(tr_img)
print ('Features: ', tr_features.shape)  # (5054, 2048)

# Train the classifier with the img_features
print ('Fit classifier...')
# clf = OneClassSVM(kernel=kernel)
clf = SVC(kernel=kernel, probability=True, degree=degree, class_weight=class_weight)
clf.fit(tr_features, labels)

#################################################################
#       TESTING                                                 #
#################################################################

# Load test images and labels
print ('Loading test images and names...')
test_img, names = l.load_labeled_images_testset()
print('Test images: ', test_img.shape)  # (2342, 224, 224, 3)
print ('Names: ', names.shape)  # (2342, 2)

# Extract ResNet img_features for the testing images
print ('Extracting test img_features...')
test_features = base_model.predict(test_img)
print ('Features: ', test_features.shape)  # (2342, 2048)

# Predict
print ('Predicting...')
pred = clf.predict(test_features)
prob = clf.predict_proba(test_features)
print (pred.shape)  # (2342,)
# print (pred)
# print (prob)

f = open(file_name, 'w')
l.set_results_SVC_ResNet(names, pred, prob, f)
f.close()

t1 = time.time() - t0
print('Execution time: ' + str(t1))
