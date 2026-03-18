from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    CallbackQueryHandler
)
import random
import uuid

from model import load_model, load_tokenizer
from generator import JokeGenerator
from tgtoken import TOKEN
from telegram.request import HTTPXRequest

import logging

# Base logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# User logger
user_logger = logging.getLogger("user_requests")
user_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("user_requests.log", encoding="utf-8")
formatter = logging.Formatter(
    "%(asctime)s | user_id=%(user_id)s | username=%(username)s | prompt=%(prompt)s | response=%(response)s | joke_id=%(joke_id)s"
)

file_handler.setFormatter(formatter)
user_logger.addHandler(file_handler)
user_logger.propagate = False

# Functions for generate
model = load_model("./artifacts/checkpoints/models/LSTM_v01.pt")
tokenizer = load_tokenizer("./artifacts/checkpoints/tokenizers/tokenizer_LSTM_v01.json")
generator = JokeGenerator(model, tokenizer)

def create_feedback_buttons(joke_id: str):
    keyboard = [
        [
            InlineKeyboardButton("👍 Нравится", callback_data=f"like|{joke_id}"),
            InlineKeyboardButton("👎 Не нравится", callback_data=f"dislike|{joke_id}"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    text = (
        "Привет! Я бот, который генерирует анекдоты.\n\n"
        "Напиши \\generate и начало анекдота — и я его продолжу.\n"
    )

    await update.message.reply_text(text)

async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_text = update.message.text[:1000]

    if user_text.strip() == "/generate":
        joke = generator.generate(prompt="", temperature=0.5).strip()
    else:
        joke = generator.generate(prompt=user_text, temperature=0.5).strip()

    joke_id = str(uuid.uuid4())  # уникальный id шутки
    markup = create_feedback_buttons(joke_id)
    await update.message.reply_text(joke, reply_markup=markup)

    user_logger.info(
        "",
        extra={
            "user_id": user.id,
            "username": user.username,
            "prompt": user_text,
            "response": joke,
            "joke_id": joke_id
        }
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user

    if update.message.text:
        user_text = update.message.text[:1000]
    else:
        user_text = ""
    
    p = random.random()
    if p < 0.05:
        joke = "Напомнило анекдот: \n\n" + generator.generate(prompt="", temperature=0.4).strip()
    else:
        # user_logger.info(
        #     "",
        #     extra={
        #         "user_id": user.id,
        #         "username": user.username,
        #         "prompt": user_text,
        #         "response": p,
        #         "joke_id": None
        #     }
        # )
        return None

    logging.info(f"Generated joke: {joke}")
    joke_id = str(uuid.uuid4())  # уникальный id шутки
    markup = create_feedback_buttons(joke_id)
    # Logging user information
    user_logger.info(
        "",
        extra={
            "user_id": user.id,
            "username": user.username,
            "prompt": user_text,
            "response": joke,
            "joke_id": joke_id
        }
    )
    await update.message.reply_text(joke, reply_markup=markup)

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # чтобы убрать "часики" на кнопке

    data = query.data.split("|")
    action = data[0]  # 'like' или 'dislike'
    joke_id = data[1]

    # Логируем реакцию пользователя
    user = query.from_user
    user_logger.info(
        "",
        extra={
            "user_id": user.id,
            "username": user.username,
            "prompt": f"Feedback for joke_id {joke_id}",
            "response": action,
            "joke_id": joke_id
        }
    )

    # Можно обновить сообщение, чтобы кнопки больше не нажимались
    await query.edit_message_reply_markup(reply_markup=None)
    #await query.message.reply_text(f"Вы оценили шутку как: {'👍' if action=='like' else '👎'}")

async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_logger.info(
        "",
        extra={
            "user_id": update.effective_user.id,
            "username": update.effective_user.username,
            "prompt": str(update),
            "response": ""
        }
    )

def main():
    request = HTTPXRequest()

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .request(request)
        .build()
    )

    app.add_handler(CallbackQueryHandler(feedback_callback))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("generate", generate))

    app.add_handler(
        MessageHandler(filters.ALL & ~filters.COMMAND, handle_message)
    )
    #app.add_handler(MessageHandler(filters.ALL, debug))

    app.run_polling()

def test():
    print(generator.generate(prompt=""))

if __name__ == "__main__":
    main()