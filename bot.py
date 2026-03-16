from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

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
    "%(asctime)s | user_id=%(user_id)s | username=%(username)s | prompt=%(prompt)s | response=%(response)s"
)

file_handler.setFormatter(formatter)
user_logger.addHandler(file_handler)
user_logger.propagate = False

# Functions for generate
model = load_model("./artifacts/checkpoints/models/LSTM_v01.pt")
tokenizer = load_tokenizer("./artifacts/checkpoints/tokenizers/tokenizer_LSTM_v01.json")
generator = JokeGenerator(model, tokenizer)


keyboard = ReplyKeyboardMarkup(
    [["Случайный анекдот"]],
    resize_keyboard=True
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    text = (
        "Привет! Я бот, который генерирует анекдоты.\n\n"
        "Напиши начало анекдота — и я его продолжу.\n"
        "Или нажми кнопку «Случайный анекдот»."
    )

    await update.message.reply_text(text, reply_markup=keyboard)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_text = update.message.text

    logging.info(f"Получено сообщение: {user_text}")

    if user_text == "Случайный анекдот":
        joke = generator.generate(prompt="", temperature=0.5)
    else:
        joke = generator.generate(prompt=user_text, temperature=0.4)

    logging.info(f"Отправка шутки: {joke}")

    # Logging user information
    user_logger.info(
        "",
        extra={
            "user_id": user.id,
            "username": user.username,
            "prompt": user_text,
            "response": joke
        }
    )

    await update.message.reply_text(joke)


def main():
    request = HTTPXRequest()

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .request(request)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()

def test():
    print(generator.generate(prompt=""))

if __name__ == "__main__":
    main()