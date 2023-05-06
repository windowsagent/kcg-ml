import asyncio
import os
from pyrogram import Client
from configparser import ConfigParser

# Read the chat_id and output_path from the config file
config = ConfigParser()
config.read('config.ini')
chat_id = config['chat']['data_chat_id']
api_id = config['api']['user_id']
api_hash = config['api']['user_hash']
output_path = config['dir']['output_json_path']

async def all_download():
    id = chat_id
    #chat_id
    client = Client('sesion', api_id, api_hash)
    await client.start()
    messages = client.get_chat_history(chat_id=id)
    async for mess in messages:
        print(mess.text)
        with open(output_path + "/messages.txt", "a") as file:
            if mess != None:
                file.write(str(mess.text) + "\n")
    await client.stop()

'''
async def main():
    # Check for the presence of ".session" files in the current directory
    session_files = [f for f in os.listdir('.') if f.endswith('.session')]

    await all_download()


if __name__ == '__main__':
    asyncio.run(main())

'''
