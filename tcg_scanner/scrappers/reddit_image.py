import praw
import requests
import os

# Setup Reddit API credentials (register at https://www.reddit.com/prefs/apps)
reddit = praw.Reddit(
    client_id='aH3pWs_MBJJIajdh-j5LJw',
    client_secret='gDiYzu9b10dzp38CEehv6Wy3zGUxLg',
    user_agent='pokemon_bot',
    username='IGoonForCards',  # replace with your Reddit username
    password='Test4321'   # replace with your Reddit password
)

subreddit = reddit.subreddit('pokegrading')
posts = subreddit.search('what grade charizard', sort='new', limit=300)
output_dir = 'tcg_scanner/scrappers/reddit_images'
os.makedirs(output_dir, exist_ok=True)

for post in posts:
    if post.url.endswith(('.jpg', '.jpeg', '.png')):
        print(f'Downloading: {post.title}')
        try:
            img_data = requests.get(post.url, headers={
                'User-Agent': 'Mozilla/5.0'
            }).content
            filename = os.path.join(output_dir, post.id + '.jpg')
            with open(filename, 'wb') as f:
                f.write(img_data)
        except Exception as e:
            print(f'Failed to download {post.url}: {e}')
