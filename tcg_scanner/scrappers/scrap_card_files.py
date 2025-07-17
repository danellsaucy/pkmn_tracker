import os
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_SITE = "https://pkmncards.com"
SET_LIST_URL = f"{BASE_SITE}/set/"
SAVE_DIR = "downloaded_cards"

def get_all_set_slugs():
    print("🔎 Scraping available sets from:", SET_LIST_URL)
    resp = requests.get(SET_LIST_URL)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    set_links = soup.find_all("a", href=re.compile(r"^https://pkmncards.com/set/.+?/$"))

    slugs = set()
    for a in set_links:
        match = re.match(r"^https://pkmncards.com/set/([^/]+)/$", a['href'])
        if match:
            slug = match.group(1)
            slugs.add(slug)
    print(f"✅ Found {len(slugs)} sets.")
    return sorted(slugs)

def scrape_set_images(set_slug):
    set_url = f"{BASE_SITE}/set/{set_slug}/"
    print(f"\n📦 Scraping set: {set_slug} → {set_url}")

    try:
        resp = requests.get(set_url)
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to access {set_url}: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    imgs = soup.find_all("img")

    urls = set()
    for img in imgs:
        src = img.get("src")
        if src and (src.lower().endswith(".jpg") or src.lower().endswith(".png")):
            url = src if src.startswith("http") else requests.compat.urljoin(set_url, src)
            urls.add(url)

    print(f"📸 Found {len(urls)} image URLs.")

    # Prepare directory
    set_dir = os.path.join(SAVE_DIR, set_slug)
    os.makedirs(set_dir, exist_ok=True)

    for img_url in tqdm(urls, desc=f"Downloading {set_slug}", unit="img"):
        fname = os.path.basename(img_url.split("?")[0])
        out_path = os.path.join(set_dir, fname)
        if os.path.exists(out_path):
            continue  # Skip existing
        try:
            r = requests.get(img_url, stream=True, timeout=10)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
        except Exception as e:
            print(f"❌ Error downloading {img_url}: {e}")

def main():
    set_slugs = get_all_set_slugs()
    for slug in set_slugs:
        scrape_set_images(slug)

if __name__ == "__main__":
    main()
