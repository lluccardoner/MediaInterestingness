"""

Author: Lluc Cardoner
Script for predicting interestingness of one image.

"""

from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing import image


def get_prediction(img_path, model_num=31):  # model 37 gives the best MAP results
    """Returns the interestingness prediction of one image using a given model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    model_json_file = 'src/ResNet50_{}_model.json'.format(model_num)
    model_weights_file = 'src/resnet50_{}_weights.hdf5'.format(model_num)

    # Load json and create model
    print('Loading model and weights...')
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # Load weights into new model
    loaded_model.load_weights(model_weights_file)

    model = Model(input=loaded_model.input, output=loaded_model.output)
    # model.summary()

    # Get predictions
    print('Predicting...')
    print(img.shape)
    score = loaded_model.predict(img, verbose=2)
    score = np.array(score)
    # print('Score', score.shape)

    return score

print (get_prediction('/home/lluc/Documents/ME16IN/testset/videos/video_52/images/1144_1124-1164.jpg'))