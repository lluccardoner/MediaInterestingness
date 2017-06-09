#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Simple Bot to reply to Telegram messages. This is built on the API wrapper, see
# echobot2.py to see the same example built on the telegram.ext bot framework.
# This program is dedicated to the public domain under the CC0 license.

import logging
import telegram
from telegram.error import NetworkError, Unauthorized
from time import sleep

import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing import image

logger = logging.getLogger(__name__)


def get_prediction(image, model_num=37):
    total_video_num_devtest = 52
    total_video_num_testset = 26

    model_json_file = 'src/ResNet50_{}_model.json'.format(model_num)
    model_weights_file = 'src/resnet50_{}_weights.hdf5'.format(model_num)

    # Load json and create model
    print ('Loading model and weights...')
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # Load weights into new model
    loaded_model.load_weights(model_weights_file)

    model = Model(input=loaded_model.input, output=loaded_model.output)

    # Get predictions
    print ('Predicting...')
    score = loaded_model.predict(image)
    score = np.array(score)
    print ('Score', score.shape)

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
            print (update.message)
            user = update.message.from_user
            try:
                photo_file = bot.get_file(update.message.photo[-1].file_id)
                photo_file.download('user_photo.jpg')
                # img = image.load_img(path, target_size=(224, 224))
                # img = image.img_to_array(img)
                logger.info("Photo of %s: %s" % (user.first_name, 'user_photo.jpg'))
                update.message.reply_text('Great! I will send the prediction. Wait...')
                # get_prediction(img)
            except:
                continue



if __name__ == '__main__':
    main()
