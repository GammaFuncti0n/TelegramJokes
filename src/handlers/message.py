from telegram import Update
from telegram.ext import ContextTypes
from .feedback import create_feedback_buttons
import uuid
import random
import logging

user_logger = logging.getLogger("user_requests")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    generator = context.bot_data["generator"]

    if update.message.text:
        user_text = update.message.text
    else:
        user_text = ""
    
    p = random.random()
    if p < 0.05:
        joke = "Напомнило анекдот: \n\n" + generator.generate(prompt="", temperature=0.5).strip()
    else:
        return None

    joke_id = str(uuid.uuid4())
    markup = create_feedback_buttons(joke_id)
    await update.message.reply_text(joke, reply_markup=markup)
    
    user_logger.info(
        f"prompt: {user_text} | response: {joke} | joke_id: {joke_id}",
        extra={
            "user_id": user.id,
            "username": user.username
        }
    )