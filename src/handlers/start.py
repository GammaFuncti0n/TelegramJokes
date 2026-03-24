from telegram import Update
from telegram.ext import ContextTypes
import logging

user_logger = logging.getLogger("user_requests")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''
    Welcome message
    '''
    user = update.message.from_user
    text = (
        "Привет! Я бот, который генерирует анекдоты.\n"
        "Напиши /generate и начало анекдота — и я его продолжу.\n"
        "Или добавь меня в чат, я смогу случайно отвечать на сообщения анекдотами\n"
    )
    await update.message.reply_text(text)
    user_logger.info(
        text, 
        extra={
            "user_id": user.id,
            "username": user.username
        })