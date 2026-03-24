import yaml
from src.utils import check_paths, setup_loggers, init_db
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters
from telegram.request import HTTPXRequest
from src.handlers import start, generate, feedback_callback, handle_message
from src.generator import LSTMGenerator, TransformerGenerator, RAGJoke

# Main config
with open("configs/transformer_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Token
with open(config['token_path'], 'r') as f:
    TOKEN = yaml.safe_load(f)['TOKEN']

# Utils
check_paths(config['paths'])
setup_loggers(config['paths']['log'])

def main():
    app = ApplicationBuilder().token(TOKEN).request(HTTPXRequest()).build()

    app.bot_data["config"] = config
    app.bot_data["generator"] = TransformerGenerator(config) #LSTMGenerator
    app.bot_data['rag_joke'] = RAGJoke(config)
    app.bot_data["user_votes"] = init_db(config['paths']['log'])

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("generate", generate))
    app.add_handler(CallbackQueryHandler(feedback_callback))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))

    app.run_polling()

if __name__ == "__main__":
    main()