import aiohttp
from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from configparser import ConfigParser

storage = MemoryStorage()

# Read the values from the config file
config = ConfigParser()
config.read('config.ini')
api_id = config['api']['id']
api_hash = config['api']['hash']
token = config['api']['token']
chat_id = config['chat']['id']
use_proxy = config['proxy'].getboolean('use')
proxy_url = config['proxy']['url']
proxy_login = config['proxy']['login']
proxy_password = config['proxy']['password']

# Set up the bot with the read values
bot = Bot(token=token, proxy=proxy_url if use_proxy else None,
          proxy_auth=aiohttp.BasicAuth(login=proxy_login, password=proxy_password) if use_proxy else None)
dp = Dispatcher(bot, storage=storage)
