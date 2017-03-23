import numpy as np
from keras.models import model_from_json
import features.load_and_set as l
from keras.models import Model
import h5py

np.set_printoptions(threshold=np.nan)

total_video_num_devtest = 52
total_video_num_testset = 26

###############################################
number = '11'
model_json_file = 'src/ResNet50_' + number + '_model.json'
model_weights_file = 'src/resnet50_' + number + '_weights.h5'
model_predictions_file = 'src/resnet50_' + number + '_predictions.h5'
tofile = '/home/lluc/Documents/trec_eval.8.1/ResNet50_results/me16in_wien_image_resnet' + number + '.txt'
################################################

# Load images
print ('Loading images...')
test_img, img_names = l.load_labeled_images_testset()
print('Test', test_img.shape)
print('Names', img_names.shape)

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
model.summary()

# get predictions
print ('Predicting...')
score = loaded_model.predict(test_img)
score = np.array(score)
# score = np.where(score == score.max())[0]
print ('Score', score.shape)

# save predictions
f = h5py.File(model_predictions_file, 'w')
f.create_dataset(model_predictions_file, data=score)

txtfile = open(tofile, 'w')
i = 0
for name in img_names:
    prob = score[i]
    if prob[0] > prob[1]:
        pred = 0
        p = prob[0]
    else:
        pred = 1
        p = prob[1]
    print(pred, p)
    txtfile.write(name[0] + ',' + name[1] + ',' + str(pred) + ',' + str(p) + '\n')
    i += 1
txtfile.close()
