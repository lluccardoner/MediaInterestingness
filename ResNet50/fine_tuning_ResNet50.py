from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model
import features.load_and_set as l
import numpy as np

np.set_printoptions(threshold=np.nan)

total_video_num_devtest = 52
total_video_num_testset = 26

############################################
model_json_file = "src/ResNet50_5_model.json"
model_weights_file = 'src/resnet50_5_weights.h5'
###########################################

# create the base pre-trained model
base_model = ResNet50(weights='imagenet')  # , include_top=False)
# base_model.summary()

# add layers
x = base_model.output  # returns a Tensor
y = Dense(2, activation='sigmoid')(x)
predictions = Dense(1, activation='sigmoid')(y)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)
model.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

# Load images
print ('Loading images...')
dev_img = l.load_images_devset()

# Load labels
print ('Loading labels...')
dev_labels = np.empty((0, dev_img.shape[0]), dtype=int)
for i in range(total_video_num_devtest):
    x = l.get_annotations('image', i, 'devset')
    dev_labels = np.append(dev_labels, x)

# train the model on the new data for a few epochs
print ('Fitting model...')
history = model.fit(dev_img, dev_labels, batch_size=32, nb_epoch=10, verbose=2)

print("Saving model and weights")
model_json = model.to_json()

with open(model_json_file, "w") as json_file:
    json_file.write(model_json)
print("saving...")
model.save_weights(model_weights_file)
