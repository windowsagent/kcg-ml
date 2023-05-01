import random
import os
from PIL import Image,ImageOps
from all_download import chat_id
from all_download import all_download
from create_bot import *
from aiogram import types
from keyboards.keyboard import kb_bot,kb_choice
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State,StatesGroup
from datetime import datetime
from aiogram.types import CallbackQuery
import imagehash
import asyncio
from configparser import ConfigParser

# Read the values from the config file
config = ConfigParser()
config.read('config.ini')
api_id = config['api']['user_id']
api_hash = config['api']['user_hash']
token = config['api']['bot_token']
chat_id = config['chat']['main_chat_id']
data_chat_id = config['chat']['data_chat_id']
image_path = config['dir']['images_path']

class FSMGetId(StatesGroup):
    hash_first_image = State()
    hash_second_image = State()
async def change_photo(path_to_first_image,path_to_second_image):
    image1 = Image.open(path_to_first_image)
    image2 = Image.open(path_to_second_image)
    width1, height1 = image1.size
    width2, height2 = image2.size
    square_size = max(width1, height1, width2, height2)
    min_width = 640
    min_height = 480
    max_width = 1980
    max_height = 1048
    if square_size < min_width or square_size < min_height:
        square_size = max(min_width, min_height)
    if square_size > max_width or square_size > max_height:
        square_size = min(max_width, max_height)
    image1 = ImageOps.pad(image1.crop((0, 0, width1, height1)), (square_size, square_size))
    image2 = ImageOps.pad(image2.crop((0, 0, width2, height2)), (square_size, square_size))
    new_width = square_size * 2
    new_height = square_size
    if new_width < min_width or new_height < min_height:
        new_width = max(min_width, min_height * 2)
        new_height = max(min_height, int(new_width / 2))
    if new_width > max_width or new_height > max_height:
        new_width = min(max_width, int(max_height * 2))
        new_height = min(max_height, int(new_width / 2))

    new_image = Image.new('RGB', (new_width, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (square_size, 0))
    new_image.save("merged_image.jpg")
    # await FSMGetId.next()

async def command_take_photos(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        used_images = data.get("used_images", [])

        images = [img for img in os.listdir(image_path) if img not in used_images]

        if not images:
            await bot.send_message(message.chat.id, "All images have been used.")
            return

        image_names = random.sample(images, 2)
        used_images.extend(image_names)
        data["used_images"] = used_images

        await change_photo("images/" + image_names[0], "images/" + image_names[1])
        await bot.send_photo(message.chat.id, types.InputFile("merged_image.jpg"), reply_markup=kb_choice)

        with Image.open('images/' + image_names[0]) as first_image:
            hash_first = imagehash.average_hash(first_image)
        with Image.open('images/' + image_names[1]) as second_image:
            hash_second = imagehash.average_hash(second_image)

        data["hash_first_image"] = hash_first
        data["hash_second_image"] = hash_second

        await FSMGetId.next()

async def command_download_all_data(message: types.Message):
    await all_download()

async def command_help(message: types.Message):
    help_text = "This bot will merge two images, send them, and do a poll for the user to choose one. To start, use the /take_photos command. \n\nAvailable commands: \n/take_photos - send photo in the photo channel with polls \ \n /download_all_data - Download all of the data from the JSON data channel \n /help - Consult the help"
    await bot.send_message(message.chat.id, help_text)

@dp.callback_query_handler(text='command_hashes',state=FSMGetId)
async def hashes_callback(callback: CallbackQuery, state: FSMContext):
    message = callback.message
    current_date = datetime.now()
    current_date = str(current_date.year) + "-" + str(current_date.month) + "-" + str(current_date.day) + " " + str(
        current_date.hour) + ":" + str(current_date.minute)
    async with state.proxy() as data:
        message_text = f"Hash of first image: {data['hash_first_image']}\nHash of second image: {data['hash_second_image']}"
    await bot.send_message(chat_id=chat_id, text=message_text)
    await state.finish()

@dp.callback_query_handler(text='command_first',state=FSMGetId)
async def first_photo_callback(callback: CallbackQuery,state: FSMContext):
    message = callback.message
    current_date = datetime.now()
    current_date = str(current_date.year) + "-" + str(current_date.month) + "-" + str(current_date.day) + " " + str(
        current_date.hour) + ":" + str(current_date.minute)
    async with state.proxy() as data:
        message_json = {"message_id": message.message_id, "date": current_date,
                        "platform": "telegram",
                        "first_name": message.from_user.first_name,
                        "lastname": message.from_user.last_name,
                        "hash_first_image": str(data["hash_first_image"]),
                        "hash_second_image": str(data["hash_second_image"]),
                        "hash_selected_message": str(data["hash_first_image"])}
    await bot.send_message(chat_id=data_chat_id, text=message_json)
    #chat_id = Channel for data
    await state.finish()
@dp.callback_query_handler(text='command_second',state=FSMGetId)
async def second_photo_callback(callback: CallbackQuery,state: FSMContext):
    message = callback.message
    current_date = datetime.now()
    current_date = str(current_date.year) + "-" + str(current_date.month) + "-" + str(current_date.day) + " " + str(
        current_date.hour) + ":" + str(current_date.minute)
    async with state.proxy() as data:
        message_json = {"messasge_id": message.message_id,
                        "date": current_date,
                        "platform": "telegram",
                        "first_name": message.from_user.first_name,
                        "lastname": message.from_user.last_name,
                        "hash_first_image": str(data["hash_first_image"]),
                        "hash_second_image": str(data["hash_second_image"]),
                        "hash_selected_message": str(data["hash_second_image"])}
    await bot.send_message(chat_id=data_chat_id, text=message_json)
    # chat_id = Channel for data
    await state.finish()

def register_callback_query_handlers():
    dp.callback_query_handler(first_photo_callback,text="command_first")
    dp.callback_query_handler(second_photo_callback,text="command_second")
    dp.callback_query_handler(hashes_callback,text="command_hashes")
def register_bot_handlers():
    dp.register_message_handler(command_take_photos,commands="take_photos")
    dp.register_message_handler(command_download_all_data,commands="download_all_data")
    dp.register_message_handler(command_help,commands="help")
