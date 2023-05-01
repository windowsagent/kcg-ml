from pyrogram import Client
from configparser import ConfigParser

# Read the values from the config file
config = ConfigParser()
config.read('config.ini')
api_id = config['api']['user_id']
api_hash = config['api']['user_hash']

# Create a Pyrogram client
client = Client('sesion', api_id, api_hash)

# Start the client
client.start()

# Get the chat information
chat = client.get_chat(chat_id='kcgjsondata')

# Print the chat ID
print(chat.id)

# Stop the client
client.stop()
