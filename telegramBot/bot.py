import telegram


def send_message(message):
    bot = telegram.Bot(token='358534330:AAE18nzk72-uutKu1gU3qi3bRcevN0rtgsQ')
    updates = bot.getUpdates()
    chat_id = updates[-1].message.chat_id
    bot.sendMessage(chat_id=chat_id, text=message)
