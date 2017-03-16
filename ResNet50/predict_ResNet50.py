import numpy as np
from keras.models import model_from_json
import features.load_and_set as l
from keras.models import Model
import h5py

np.set_printoptions(threshold=np.nan)

total_video_num_devtest = 52
total_video_num_testset = 26

###############################################
model_json_file = "src/ResNet50_5_model.json"
model_weights_file = 'src/resnet50_5_weights.h5'
model_predictions_file = 'src/resnet50_5_predictions.h5'
################################################

# Load images
print ('Loading images...')
test_img = l.load_images_testset()
#print(test_img.shape)

# Load labels
# print 'Loading labels...'
# test_labels = np.empty((0, test_img.shape[0]), dtype=int)
# for i in range(total_video_num_devtest, total_video_num_devtest + total_video_num_testset):
#    x = l.get_annotations('image', i, 'testset')
#    test_labels = np.append(test_labels, x)

# load json and create model
print ('Loading model and weights...')
json_file = open(model_json_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_weights_file)

model = Model(input=loaded_model.input, output=loaded_model.output)

# get predictions
print ('Predicting...')
score = loaded_model.predict(test_img)
score = np.array(score)
print (score)

# normalization
norm = np.linalg.norm(score)
score = [float(i) / norm for i in score]
print (score)

# save predictions
f = h5py.File(model_predictions_file, 'w')
f.create_dataset(model_predictions_file, data=score)
