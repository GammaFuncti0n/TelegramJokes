from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import sqlite3
from datetime import datetime, timezone
import logging

user_logger = logging.getLogger("user_requests")

def create_feedback_buttons(joke_id: str):
    keyboard = [
        [
            InlineKeyboardButton("👍 Нравится", callback_data=f"like|{joke_id}"),
            InlineKeyboardButton("👎 Не нравится", callback_data=f"dislike|{joke_id}"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user = query.from_user
    await query.answer()

    data = query.data.split("|")
    action = data[0]
    joke_id = data[1]
    reacted_at = datetime.now(timezone.utc).isoformat()

    conn = context.bot_data["user_votes"]
    try:
        conn.execute(
            "INSERT INTO votes (user_id, joke_id, action, reacted_at) VALUES (?, ?, ?, ?)",
            (user.id, joke_id, action, reacted_at)
        )
        conn.commit()

        user_logger.info(
            f"joke_id: {joke_id} | response: {action}",
            extra={
                "user_id": user.id,
                "username": user.username,
            }
        )

    except sqlite3.IntegrityError:
        pass