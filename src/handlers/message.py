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
    rag_joke = context.bot_data["rag_joke"]
    config = context.bot_data["config"]

    if update.message.text:
        user_text = update.message.text
    else:
        user_text = ""
    
    p = random.random()
    if p < config['generating']['probability_for_chat']:
        #joke = "Напомнило анекдот: \n\n" + generator.generate(prompt="", maxlen=config['generating']['maxlen'], temperature=config['generating']['temperature']).strip()
        joke = "Напомнило анекдот: \n\n" + rag_joke.generate(prompt=user_text).strip()
    elif user_text.lower().startswith('олух') or user_text.lower().startswith('костян') or user_text.lower().startswith('костик') or user_text.lower().startswith('влад') or user_text.lower().startswith('макан'):
        #joke = generator.generate(prompt="", maxlen=config['generating']['maxlen'], temperature=config['generating']['temperature']).strip()
        joke = rag_joke.generate(prompt=user_text).strip()
    else:
        return None
    
    joke = joke.replace('/n', '')
    joke_id = str(uuid.uuid4())
    #markup = create_feedback_buttons(joke_id, '', '')
    #await update.message.reply_text(joke, reply_markup=markup)
    
    user_logger.info(
        f"prompt: {user_text} | response: {joke} | joke_id: {joke_id}",
        extra={
            "user_id": user.id,
            "username": user.username
        }
    )