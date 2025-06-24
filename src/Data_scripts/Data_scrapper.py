from telethon.sync import TelegramClient
from telethon.tl.types import PeerChannel
import pandas as pd
import asyncio
import re

api_id = '27286915 '
api_hash = 'f01fde995b174453dcad17d960470643'
phone = '+251965593580'

# List of Telegram channel usernames or URLs
channels = [
    'shageronlinestore',
    'qnashcom',
    'ethiopia_online_market',
    'ZemenExpress',
    'aradabrand2',
    'nevacomputer',
    'meneshayeofficial',
    'Leyueqa',
    'sinayelj',
    'gebeyaadama',
    'MerttEka',
    'AwasMart'
    
]

# Create the client
client = TelegramClient('session_name', api_id, api_hash)

async def scrape_channel(channel_username):
    await client.start()
    if not await client.is_user_authorized():
        await client.send_code_request(phone)
        await client.sign_in(phone, input('Enter the code: '))

    try:
        entity = await client.get_entity(channel_username)
        print(f"Scraping {entity.title}...")
    except Exception as e:
        print(f"Error fetching {channel_username}: {str(e)}")
        return pd.DataFrame()

    messages = []
    async for message in client.iter_messages(entity, limit=100):  # Adjust limit if needed
        if message.text:
            messages.append({
                'text': message.text,
                'date': message.date,
                'views': message.views if hasattr(message, 'views') else None,
                'post_id': message.id,
                'channel': entity.username or entity.title
            })

    df = pd.DataFrame(messages)
    return df

async def scrape_all_channels(channels):
    all_dfs = []
    for channel in channels:
        df = await scrape_channel(channel)
        if not df.empty:
            all_dfs.append(df)
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df

# Run the scraper
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    scraped_data = loop.run_until_complete(scrape_all_channels(channels))
    # Save to CSV
    scraped_data.to_csv('Data/telegram_scraped_data.txt', index=False)
    print("✅ Scraping completed and saved to telegram_scraped_data.csv")
