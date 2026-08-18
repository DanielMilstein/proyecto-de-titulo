import os, asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
print(f"Using TELEGRAM_BOT_TOKEN: {TELEGRAM_BOT_TOKEN}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Received /start command")
    print(f"Update: {update}")
    await update.message.reply_text('Hello! This is a test bot.')

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(update.message.text)

def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    application.run_polling(close_loop=False)

if __name__ == '__main__':
    asyncio.run(main())