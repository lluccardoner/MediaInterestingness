import logging
import traceback
import matplotlib.pyplot as plt
import time
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from ResNet50.callbacks import EarlyStopping
from telegramBot import bot

#########################################
# Using data augmentation with Keras    #
# ImageDataGenerator                    #
# Using keras callbacks                 #
#########################################
try:
    t0 = time.time()

    np.set_printoptions(threshold=np.nan)

    ###########################################
    total_video_num_devtest = 52
    total_video_num_testset = 26
    nb_epoch = 100
    batch_size = 32
    learning_rate = 0.001
    loss = 'binary_crossentropy'
    # optimizer = RMSprop(lr=learning_rate)
    optimizer = Adam(lr=learning_rate)
    nb_train_samples = 2400  # should be multiple of batch size
    nb_validation_samples = 800  # should be multiple of batch size
    class_weights = [{0: 0.1, 1: 1.}, {0: 0.5, 1: 1.}, {0: 0.7, 1: 1.}, {0: 1., 1: 10.}]
    validation_weights = class_weights
    stop_patience = 100
    stop_cooldown = 10
    ############################################
    number = 19
    model_json_file = 'src/ResNet50_' + str(number) + '_model.json'
    model_weights_file = 'src/resnet50_' + str(number) + '_weights.h5'
    model_fig = 'src/model_' + str(number) + '.png'
    model_checkpoint = 'src/resnet50_' + str(number) + '_{epoch:02d}-{val_loss:.2f}.hdf5'
    ###########################################

    # create the base pre-trained model
    base_model = ResNet50(weights='imagenet')  # , include_top=False)

    # add layers
    a = base_model.get_layer('flatten_1').output  # returns a Tensor
    # b = Dropout(0.5)(a)
    c = Dense(1024, activation='relu')(a)
    # d = Dropout(0.5)(c)
    e = Dense(256, activation='relu')(c)
    predictions = Dense(2, activation='softmax')(e)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional ResNet50 layers
    for layer in base_model.layers:
        layer.trainable = False

    model.summary()
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Load images

    # Load data
    print ('Loading data with ImageDataGenerator')
    train_datagen = ImageDataGenerator(horizontal_flip=True)

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        save_prefix='aug_')  # save_to_dir='data/augmented')

    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    for weights in class_weights:
        model_json_file = 'src/ResNet50_' + str(number) + '_model.json'
        model_weights_file = 'src/resnet50_' + str(number) + '_weights.h5'
        model_fig = 'src/model_' + str(number) + '.png'
        model_checkpoint = 'src/resnet50_' + str(number) + '_{epoch:02d}-{val_loss:.2f}.hdf5'
        # callbacks
        checkpointer = ModelCheckpoint(filepath=model_checkpoint,
                                       verbose=1,
                                       save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.1,
                                      patience=5,
                                      min_lr=0,
                                      verbose=1)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=stop_patience,
                                   cooldown=stop_cooldown)

        print ('Fitting model ' + str(number))
        tr_loss = []
        val_loss = []
        tr_acc = []
        val_acc = []

        history = model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples,
            verbose=2,
            callbacks=[checkpointer, reduce_lr, early_stop],
            class_weight=weights)

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
        # model.save_weights(model_weights_file)

        plt.savefig(model_fig)
        execution_time = (time.time() - t0) / 60
        print('Execution time model ' + str(number) + ' (min): ' + str(execution_time))
        bot.send_message('Execution time model ' + str(number) + ' (min): ' + str(execution_time))
        number += 1

except Exception:
    logging.error(traceback.format_exc())
    bot.send_message('Exception caught: ' + str(Exception.message))
