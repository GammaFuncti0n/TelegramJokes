from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import sqlite3
from datetime import datetime, timezone
import logging

user_logger = logging.getLogger("user_requests")

def create_feedback_buttons(joke_id: str, likes: str, dislikes: str):
    keyboard = [
        [
            InlineKeyboardButton(f"👍 {likes}", callback_data=f"like|{joke_id}"),
            InlineKeyboardButton(f"👎 {dislikes}", callback_data=f"dislike|{joke_id}"),
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

        likes, dislikes = get_vote_counts(conn, joke_id)
        new_markup = create_feedback_buttons(joke_id, str(likes), str(dislikes))
        await query.edit_message_reply_markup(reply_markup=new_markup)

    except sqlite3.IntegrityError:
        pass


def get_vote_counts(conn, joke_id: str):
    cursor = conn.execute(
        """
        SELECT 
            SUM(CASE WHEN action = 'like' THEN 1 ELSE 0 END) as likes,
            SUM(CASE WHEN action = 'dislike' THEN 1 ELSE 0 END) as dislikes
        FROM votes
        WHERE joke_id = ?
        """,
        (joke_id,)
    )
    row = cursor.fetchone()
    return row[0] or 0, row[1] or 0