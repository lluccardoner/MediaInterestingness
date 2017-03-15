import numpy as np
from keras.models import model_from_json
import features.load_and_set as l

np.set_printoptions(threshold=np.nan)

total_video_num_devtest = 52
total_video_num_testset = 26

# Load images
print 'Loading images...'
test_img = l.load_images_testset()

# Load labels
# print 'Loading labels...'
# test_labels = np.empty((0, test_img.shape[0]), dtype=int)
# for i in range(total_video_num_devtest, total_video_num_devtest + total_video_num_testset):
#    x = l.get_annotations('image', i, 'testset')
#    test_labels = np.append(test_labels, x)

# load json and create model
print 'Loading model and weights...'
json_file = open('ResNet50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("resnet50_weights.h5")

loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy')

# get predictions
print 'Predicting...'
score = loaded_model.predict(test_img)
print score
