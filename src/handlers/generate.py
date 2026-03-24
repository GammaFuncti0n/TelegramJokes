from telegram import Update
from telegram.ext import ContextTypes
from .feedback import create_feedback_buttons
import logging
import uuid

user_logger = logging.getLogger("user_requests")

async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    generator = context.bot_data["generator"]
    config = context.bot_data["config"]
    #user_text = update.message.text[:1000]

    prompt = " ".join(context.args)
    
    joke = generator.generate(prompt=prompt, maxlen=config['generating']['maxlen'], temperature=config['generating']['temperature'], ).strip()
    joke = joke.replace('/n', '')

    joke_id = str(uuid.uuid4())
    markup = create_feedback_buttons(joke_id, '', '')
    await update.message.reply_text(joke, reply_markup=markup)

    user_logger.info(
        f"prompt: {prompt} | response: {joke} | joke_id: {joke_id}",
        extra={
            "user_id": user.id,
            "username": user.username
        }
    )