"""

Author: Lluc Cardoner
Use telegram bot from python-telegram-bot/python-telegram-bot on github
to send and image and return the prediction through telegram

In developement...

"""

from __future__ import print_function
import logging
import telegram
from telegram.error import NetworkError, Unauthorized
from time import sleep

import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing import image


def get_prediction(img_path, model_num=37):  # model 37 gives the best MAP results
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
    model.summary()

    # Get predictions
    print('Predicting...')
    print(img.shape)
    score = loaded_model.predict(img, verbose=2)
    score = np.array(score)
    print('Score', score.shape)

    return score


update_id = None


def main():
    global update_id
    # Telegram Bot Authorization Token
    bot = telegram.Bot('385662610:AAHo7ANi4BRxo7_GOOOda_Cq8VhQicZ3YMY')

    # get the first pending update_id, this is so we can skip over it in case
    # we get an "Unauthorized" exception.
    try:
        update_id = bot.get_updates()[0].update_id
    except IndexError:
        update_id = None

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    while True:
        try:
            photo(bot)
        except NetworkError:
            sleep(1)
        except Unauthorized:
            # The user has removed or blocked the bot.
            update_id += 1


def echo(bot):
    global update_id
    # Request updates after the last update_id
    for update in bot.get_updates(offset=update_id, timeout=10):
        update_id = update.update_id + 1

        if update.message:  # your bot can receive updates without messages
            # Reply to the message
            update.message.reply_text(update.message.text)


def photo(bot):
    global update_id
    # Request updates after the last update_id
    for update in bot.get_updates(offset=update_id, timeout=10):
        update_id = update.update_id + 1

        if update.message:
            print(update.message)
            user = update.message.from_user
            try:
                photo_file = bot.get_file(update.message.photo[-1].file_id)
                photo_file.download('user_photo.jpg')
                update.message.reply_text('Great! I will send the prediction. Wait...')
                p = get_prediction('user_photo.jpg')
                print("Prediction", p)
                update.message.reply_text('No interesting: {}\nInteresting: {}'.format(p[0][0], p[0][1]))
            except:
                continue


if __name__ == '__main__':
    main()
    #p = get_prediction('user_photo.jpg')
    #print (p[0][0], p[0][1])