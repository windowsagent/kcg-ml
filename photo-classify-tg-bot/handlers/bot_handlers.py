import random
import subprocess
import os
import hashlib
import base64
from PIL import Image,ImageOps
from all_download import all_download
from create_bot import *
from aiogram import types
from keyboards.keyboard import kb_bot,kb_choice
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State,StatesGroup
from datetime import datetime
from aiogram.types import CallbackQuery
from aiogram.types import ContentTypes
import imagehash
import asyncio
import json
from configparser import ConfigParser
from handlers.hash_shortener import HashShortener

# Read the values from the config file
config = ConfigParser()
config.read('config.ini')
api_id = config['api']['user_id']
api_hash = config['api']['user_hash']
token = config['api']['bot_token']
chat_id = config['chat']['main_chat_id']
data_chat_id = config['chat']['data_chat_id']
image_path = config['dir']['images_path']

# Globs
VERSION = "v1.0"

# Calculate image hash

def generate_image_hash(image_path):
    with Image.open(image_path) as img:
        img_bytes = img.tobytes()
        sha256_hash = hashlib.sha256(img_bytes).digest()
        base64_hash = base64.b64encode(sha256_hash).decode('utf-8')
        return base64_hash

def shorten_hash(hash_str, n_character=16):
    """
    Shorten a sha-256 hash string to n_character characters.

    Args:
    hash_str (str): sha-256 hash string to be shortened
    n_character (int): number of characters in the shortened hash (default: 16)

    Returns:
    str: shortened hash string
    """
    # Compute the sha-256 hash of the input string
    hash_obj = hashlib.sha256(hash_str.encode())
    hash_digest = hash_obj.digest()

    # Convert the hash digest to an integer and shorten it
    hash_int = int.from_bytes(hash_digest, byteorder='big')
    hash_shortener = HashShortener(n_character)
    shortened_hash = hash_shortener.encode_id(hash_int)

    return shortened_hash


class FSMGetId(StatesGroup):
    hash_first_image = State()
    hash_second_image = State()

class Form(StatesGroup):
    name = State()

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

async def command_rank(message: types.Message, state: FSMContext):
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

        hash_first = generate_image_hash('images/' + image_names[0])
        hash_second = generate_image_hash('images/' + image_names[1])

        data["hash_first_image"] = hash_first
        data["hash_second_image"] = hash_second
        data["username"] = message.from_user.id

        await FSMGetId.next()

async def command_download_all_data(message: types.Message):
    await all_download()

async def command_help(message: types.Message):
    help_text = ">This bot will merge two images, send them, and do a poll for the user to choose one. To start, use the /rank command. \n \n \
Available commands: \n \
    - /rank Select two random images to do the 'A/B' test \n \
    - /help Display help \n \
    - /set_task Set the task variable for the session \n \
    - /download_all_data Download all data channel messages"
    await bot.send_message(message.chat.id, help_text)

async def command_set_task(message: types.Message, state: FSMContext):
    task_text = "Type the text to set as the current task:"
    await bot.send_message(message.chat.id, task_text)
    task = ""
    await Form.name.set()
    await message.reply("Waiting for your reply...")

@dp.message_handler(state=Form.name)
async def process_name(message: types.Message, state: FSMContext):
    """Write task to global"""

    global task
    task = str(message.text)
    await bot.send_message(message.chat.id, f"Task set as '{task}'")
    await state.finish()

@dp.callback_query_handler(text='command_details', state=FSMGetId)
async def details_callback(callback: CallbackQuery, state: FSMContext):
    message = callback.message
    current_date = datetime.now()
    current_date = str(current_date.year) + "-" + str(current_date.month) + "-" + str(current_date.day) + " " + str(
        current_date.hour) + ":" + str(current_date.minute)
    if 'task' not in globals():
        task = "rank"
    async with state.proxy() as data:
        hash_first_image = shorten_hash(data['hash_first_image'])
        hash_second_image = shorten_hash(data['hash_second_image'])
        message_text = f"image_hash_0: {hash_first_image}\nimage_hash_1: {hash_second_image}\nimage_dir: {image_path}\ntask_type: {task}"
        await bot.send_message(chat_id=chat_id, text=message_text)

@dp.callback_query_handler(text='command_first',state=FSMGetId)
async def first_photo_callback(callback: CallbackQuery,state: FSMContext):
    username = callback.from_user.username
    message = callback.message
    current_date = datetime.now()
    current_date = str(current_date.year) + "-" + str(current_date.month) + "-" + str(current_date.day) + " " + str(
        current_date.hour) + ":" + str(current_date.minute)
    async with state.proxy() as data:
        message_json = {
            "message_id": message.message_id,
            "date": current_date,
            "platform": "telegram",
            "username": username,
            "task"    : task if 'task' in globals() else "rank",
            "api_version": VERSION,
            "hash_first_image": str(data["hash_first_image"]),
            "hash_second_image": str(data["hash_second_image"]),
            "hash_selected_message": str(data["hash_first_image"])
        }
    await bot.send_message(chat_id=data_chat_id, text=message_json)
    #chat_id = Channel for data
    await state.finish()

@dp.callback_query_handler(text='command_second',state=FSMGetId)
async def second_photo_callback(callback: CallbackQuery,state: FSMContext):
    username = callback.from_user.username
    message = callback.message
    current_date = datetime.now()
    current_date = str(current_date.year) + "-" + str(current_date.month) + "-" + str(current_date.day) + " " + str(
        current_date.hour) + ":" + str(current_date.minute)
    async with state.proxy() as data:
        message_json = {
            "message_id": message.message_id,
            "date": current_date,
            "platform": "telegram",
            "username": username,
            "task"    : task if 'task' in globals() else "rank",
            "api_version": VERSION,
            "hash_first_image": str(data["hash_first_image"]),
            "hash_second_image": str(data["hash_second_image"]),
            "hash_selected_message": str(data["hash_second_image"])
        }
    await bot.send_message(chat_id=data_chat_id, text=message_json)
    #chat_id = Channel for data
    await state.finish()

def register_callback_query_handlers():
    dp.callback_query_handler(first_photo_callback,text="command_first")
    dp.callback_query_handler(second_photo_callback,text="command_second")
    dp.callback_query_handler(details_callback,text="command_details")

def register_bot_handlers():
    dp.register_message_handler(command_rank, commands="rank", content_types=ContentTypes.TEXT)
    dp.register_message_handler(command_set_task, commands="set_task")
    dp.register_message_handler(command_download_all_data, commands="download_all_data")
    dp.register_message_handler(command_help, commands="help")
