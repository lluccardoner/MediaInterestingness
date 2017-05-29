import logging
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import bot

#########################################
# Using data augmentation with Keras    #
# ImageDataGenerator                    #
# Using keras callbacks                 #
#########################################
try:
    # Execution time
    t0 = time.time()
    # Option to see all the np array on the console
    np.set_printoptions(threshold=np.nan)

    ###########################################
    total_video_num_devtest = 52
    total_video_num_testset = 26
    nb_epoch = 100
    batch_size = 32
    learning_rate = 0.0001
    loss = 'binary_crossentropy'
    optimizer = Adam(lr=learning_rate)
    nb_train_samples = 2400  # should be multiple of batch size
    nb_validation_samples = 800  # should be multiple of batch size
    # class_weights = {0: 0.1899, 1: 1.8054}  # model 21 and 26 (for the full data set)
    # class_weights = {0: 0.3469, 1: 1.6531}  # model 29 (augmentation just in the interesting class)
    stop_patience = 20  # Patience for the early stop callback
    ############################################
    number = 49  # Model number
    bot.send_message('Model ' + str(number))
    model_json_file = 'src/ResNet50_' + str(number) + '_model.json'
    # model_weights_file = 'src/resnet50_' + str(number) + '_weights.h5'
    model_fig = 'src/model_' + str(number) + '.png'
    # model_checkpoint = 'src/resnet50_' + str(number) + '_{epoch:02d}-{val_loss:.2f}.hdf5'
    model_checkpoint = 'src/resnet50_' + str(number) + '_weights.hdf5'
    ###########################################

    # create the base pre-trained model
    base_model = ResNet50(weights='imagenet')  # , include_top=False)

    # add layers
    a = base_model.get_layer('flatten_1').output  # returns a Tensor
    # b = Dropout(0.5)(a)
    c = Dense(1024, activation='relu')(a)
    # d = Dropout(0.5)(c)
    e = Dense(256, activation='relu')(c)
    # f = Dropout(0.5)(e)
    predictions = Dense(2, activation='softmax')(e)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional ResNet50 layers
    for layer in base_model.layers:
        layer.trainable = False

    model.summary()
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Load data
    print ('Loading data with ImageDataGenerator')
    # TODO data augmentation
    # train_datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.25, height_shift_range=0.25, width_shift_range=0.125)

    train_datagen = ImageDataGenerator(horizontal_flip=True)

    test_datagen = ImageDataGenerator()

    # TODO change directory
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
        # save_prefix='aug', save_to_dir='data/augmented')

    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    # callbacks
    checkpointer = ModelCheckpoint(filepath=model_checkpoint,
                                   verbose=1,
                                   save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=5,
                                  min_lr=0,
                                  verbose=1)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=stop_patience)

    print ('Fitting model ' + str(number))
    tr_loss = []
    val_loss = []
    tr_acc = []
    val_acc = []
    # TODO change number epochs, class weights
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=10,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        verbose=2)
        # callbacks=[checkpointer, reduce_lr, early_stop])
        # class_weight=class_weights)

    # TODO for model 24 and 25 train last layers of resnet
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:162]:
        layer.trainable = False
    for layer in model.layers[162:]:
        layer.trainable = True
    # we need to recompile the model for these modifications to take effect
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        verbose=2,
        callbacks=[checkpointer, reduce_lr, early_stop])
        # class_weight=class_weights)

    tr_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])
    tr_acc.extend(history.history['acc'])
    val_acc.extend(history.history['val_acc'])

    # Show plots
    x = np.arange(len(val_loss))
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

    print("\n Saving model...")
    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)
    # print("saving...")
    # model.save_weights(model_weights_file) # is done by the check point

    plt.savefig(model_fig)
    execution_time = (time.time() - t0) / 60
    print('Execution time model ' + str(number) + ' (min): ' + str(execution_time))
    bot.send_message('Execution time model ' + str(number) + ' (min): ' + str(execution_time))
    bot.send_image(number)

except Exception:
    logging.error(traceback.format_exc())
    bot.send_message('Exception caught')
