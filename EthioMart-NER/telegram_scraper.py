from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")
phone = os.getenv("PHONE")

client = TelegramClient('ethiosession', api_id, api_hash)

channel_list = [
    "@aradabrand2",
    "@marakisat2",
    "@belaclassic",
    "@AwasMart",
    "@qnashcom"
]

async def fetch_messages(channel_username, limit=300):
    try:
        entity = await client.get_entity(channel_username)
        history = await client(GetHistoryRequest(
            peer=entity,
            limit=limit,
            offset_date=None,
            offset_id=0,
            max_id=0,
            min_id=0,
            add_offset=0,
            hash=0
        ))

        messages = []
        for msg in history.messages:
            if msg.message:
                messages.append({
                    "date": msg.date,
                    "sender_id": msg.from_id.user_id if msg.from_id else None,
                    "message": msg.message,
                    "channel": channel_username
                })

        df = pd.DataFrame(messages)
        df.to_csv(f"{channel_username.strip('@')}_messages.csv", index=False)
        print(f"✅ Saved {len(messages)} messages from {channel_username}")

    except Exception as e:
        print(f"❌ Error scraping {channel_username}: {e}")

async def run_all():
    await client.start(phone)
    for channel in channel_list:
        await fetch_messages(channel, limit=300)

with client:
    client.loop.run_until_complete(run_all())
