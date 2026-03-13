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

    user_text = update.message.text

    if user_text == "Случайный анекдот":
        joke = generator.generate(prompt="")
    else:
        joke = generator.generate(prompt=user_text)

    await update.message.reply_text(joke)


def main():

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    app.run_polling()

def test():
    print(generator.generate(prompt=""))


if __name__ == "__main__":
    test()