from aiogram.types import ReplyKeyboardMarkup,KeyboardButton,InlineKeyboardButton,InlineKeyboardMarkup
bb1 = KeyboardButton("/take_photos")
bb2 = KeyboardButton("/download_all_data")
kb_bot = ReplyKeyboardMarkup(resize_keyboard=True)
kb_bot.add(bb1,bb2)
#-----------------------
cb1 = InlineKeyboardButton(text="1⃣",callback_data="command_first")
cb2 = InlineKeyboardButton(text="2⃣",callback_data="command_second")
cb3 = InlineKeyboardButton(text="Show hashes",callback_data="command_hashes")
kb_choice = InlineKeyboardMarkup(row_width=8).add(cb1,cb2,cb3)
