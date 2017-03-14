from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model
import features.load_and_set as l
import numpy as np

np.set_printoptions(threshold=np.nan)

total_video_num_devtest = 52
total_video_num_testset = 26

# create the base pre-trained model
base_model = ResNet50(weights='imagenet')  # , include_top=False)
# base_model.summary()

# add a 2 neuron layer
x = base_model.output  # returns a Tensor
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)
# model.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

# Load images
print 'Loading images...'
dev_img = l.load_images_devset()
test_img = l.load_images_testset()

# Load labels
print 'Loading labels...'
dev_labels = np.empty((0, dev_img.shape[0]), dtype=int)
test_labels = np.empty((0, test_img.shape[0]), dtype=int)
for i in range(total_video_num_devtest):
    x = l.get_annotations('image', i, 'devset')
    dev_labels = np.append(dev_labels, x)
for i in range(total_video_num_devtest, total_video_num_devtest + total_video_num_testset):
    x = l.get_annotations('image', i, 'testset')
    test_labels = np.append(test_labels, x)

# train the model on the new data for a few epochs
print 'Fitting model...'
history = model.fit(dev_img, dev_labels, batch_size=32, nb_epoch=10, verbose=2)

# get predictions
score = model.predict(test_img, test_labels, verbose=0)
print score
