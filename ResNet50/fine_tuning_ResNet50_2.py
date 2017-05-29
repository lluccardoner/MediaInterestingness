import logging
import traceback

import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.callbacks import *
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from telegramBot import bot

#########################################
# Using data augmentation with Keras    #
# ImageDataGenerator                    #
#########################################
try:
    t0 = time.time()

    np.set_printoptions(threshold=np.nan)

    ###########################################
    total_video_num_devtest = 52
    total_video_num_testset = 26
    nb_epoch = 100
    batch_size = 32
    # learning_rate = 0.001
    # learning_rate = 0.0001
    learning_rate = 0.00001
    # learning_rate = 0.000001
    loss = 'binary_crossentropy'
    # optimizer = RMSprop(lr=learning_rate)
    optimizer = Adam(lr=learning_rate)
    nb_train_samples = 2400  # should be multiple of batch size
    nb_validation_samples = 800  # should be multiple of batch size
    ############################################
    number = '14'
    model_json_file = 'src/ResNet50_' + number + '_model.json'
    model_weights_file = 'src/resnet50_' + number + '_weights.h5'
    model_fig = 'src/model_' + number + '.png'
    ###########################################

    # create the base pre-trained model
    base_model = ResNet50(weights='imagenet')  # , include_top=False)

    # add layers
    a = base_model.get_layer('flatten_1').output  # returns a Tensor
    b = Dense(1024, activation='relu')(a)
    # c = Dense(512, activation='relu')(b)
    d = Dense(256, activation='relu')(b)
    # e = Dense(128, activation='relu')(d)
    # f = Dense(64, activation='relu')(e)
    # g = Dense(32, activation='relu')(f)
    # h = Dense(16, activation='relu')(g)
    # i = Dense(8, activation='relu')(h)
    # j = Dense(4, activation='relu')(i)
    predictions = Dense(2, activation='softmax')(d)

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

    print ('Fitting model...')
    tr_loss = []
    val_loss = []
    tr_acc = []
    val_acc = []
    for iteration in range(1, nb_epoch):
        print('-' * 50)
        print('Iteration', iteration)
        history = model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=1,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples,
            verbose=2)

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

except Exception:
    logging.error(traceback.format_exc())
    bot.send_message('Exception caught: ' + str(Exception.message))
