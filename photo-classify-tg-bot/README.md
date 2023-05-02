# Setting up the bot

## Telegram
### In telegram, you must create two channels, and use 'Bot Godfather' in order to create the bot.

*One channel will be for the json data of the responses, another one for polling*

First, copy the config.ini template file
```bash
cp config.ini.template config.ini
```
Use your editor of choice.

### Bot setup
+ Search for @BotFather on Telegram, and then click start.
+ Type /newbot
+ Name the bot
+ Choose the username
+ It'll give you the token for using the API

Edit config.ini
```bash
[api]
id =
hash =
token = {token-you-just-got-here}
```
**Write the token without the curly braces {}**

### Setting up user side
We must create an API for your user, as the bot itself can't download messages.
Follow this guide: https://core.telegram.org/api/obtaining_api_id#obtaining-api-id

Now edit the `[api]` section yet again, replacing `id` and `hash` with the ones you just made:
```bash
[api]
id = {your-id-here}
hash = {your-hash-here}
token =
```

### Setting up proxy
Edit the `[proxy]` for use of a proxy, if needed. It's on `false` by default, which you must change to `true` if you want to use a proxy:

```bash
[proxy]
use = false
url =
login =
password =
```

## Running (docker)

Build the image
```bash
DOCKER_BUILDKIT=1 docker build -t kcg-photo-telegram .
```

Run the container (ensure you have docker compose installed)
```bash
docker compose up -d
```
