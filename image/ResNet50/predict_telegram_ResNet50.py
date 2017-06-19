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

from image.ResNet50.predict_image_ResNet50 import get_prediction

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
    # p = get_prediction('user_photo.jpg')
    # print (p[0][0], p[0][1])
