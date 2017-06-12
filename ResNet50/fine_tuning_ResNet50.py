"""
Author: Lluc Cardoner

First script for fine-tuning the ResNet50 for predicting image interestingness.

"""

import time

import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

import SVM_image_features.load_and_set as l
from telegramBot import bot

t0 = time.time()

np.set_printoptions(threshold=np.nan)

###########################################
total_video_num_devtest = 52
total_video_num_testset = 26
nb_epoch = 100
batch_size = 32
#learning_rate = 0.001
learning_rate = 0.0001
# learning_rate = 0.00001
loss = 'binary_crossentropy'
# optimizer = RMSprop(lr=learning_rate)
optimizer = Adam(lr=learning_rate)
############################################
number = '9'
model_json_file = 'src/ResNet50_' + number + '_model.json'
model_weights_file = 'src/resnet50_' + number + '_weights.h5'
model_fig = 'src/model_' + number + '.png'
###########################################

# create the base pre-trained model
base_model = ResNet50(weights='imagenet')  # , include_top=False)

# add layers
x = base_model.get_layer('flatten_1').output  # returns a Tensor
y = Dense(256, activation='relu')(x)
predictions = Dense(2, activation='softmax')(y)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

model.summary()
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Load images

# Load images
print ('Loading images...')
img = l.load_labeled_images_devset()
dev_img = img[:4044]
val_img = img[4044:]
print (dev_img.shape)
print (val_img.shape)

# Load labels
print ('Loading labels...')
labels = np.empty((0, img.shape[0]), dtype=int)
for i in range(total_video_num_devtest):
    x = l.get_annotations('image', i, 'devset')
    labels = np.append(labels, x)

labels = to_categorical(labels)
labels = labels.astype(int)
dev_labels = labels[:4044]
val_labels = labels[4044:]
print (dev_labels.shape)
print (val_labels.shape)

print ('Fitting model...')
tr_loss = []
val_loss = []
tr_acc = []
val_acc = []
for iteration in range(1, nb_epoch):
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit(dev_img, dev_labels, batch_size=batch_size, nb_epoch=1, verbose=2,
                        validation_data=(val_img, val_labels))
    tr_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])
    tr_acc.extend(history.history['acc'])
    val_acc.extend(history.history['val_acc'])

# Show plots
x = np.arange(iteration)
fig = plt.figure(1)
fig.suptitle('TRAINNING vs VALIDATION', fontsize=14, fontweight='bold')

# LOSS: TRAINING vs VALIDATION
sub_plot1 = fig.add_subplot(211)

plt.plot(x, tr_loss, '--', linewidth=2, label='tr_loss')
plt.plot(x, val_loss, label='va_loss')
plt.legend(loc='upper right')

# ACCURACY: TRAINING vs VALIDATION
sub_plot2 = fig.add_subplot(212)

plt.plot(x, tr_acc, '--', linewidth=2, label='tr_acc')
plt.plot(x, val_acc, label='val_acc')
plt.legend(loc='lower right')

print("\n Saving model and weights")
model_json = model.to_json()

with open(model_json_file, "w") as json_file:
    json_file.write(model_json)
print("saving...")
model.save_weights(model_weights_file)

plt.savefig(model_fig)
execution_time = (time.time() - t0) / 60
print("Execution time (min): " + str(execution_time))
bot.send_message('Execution time (min): ' + str(execution_time))
