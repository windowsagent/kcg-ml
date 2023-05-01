import os
from create_bot import *
from aiogram import executor
from handlers.bot_handlers import register_bot_handlers,register_callback_query_handlers

register_callback_query_handlers()
register_bot_handlers()
if __name__ == '__main__':
    executor.start_polling(dp,skip_updates=True)
    os.remove("merged_image.jpg")
