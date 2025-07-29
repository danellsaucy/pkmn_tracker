import requests
from bs4 import BeautifulSoup

class PkmnCardFinder:
    def __init__(self, set_name: str, card_img_id: str):
        self.set_name = set_name.strip().lower()
        self.card_img_id = card_img_id.strip()
        self.base_url = f"https://pkmncards.com/set/{self.set_name}/"
        self.headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "max-age=0",
            "priority": "u=0, i",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        }

    def fetch_title(self) -> str | None:
        response = requests.get(self.base_url, headers=self.headers)
        if response.status_code != 200:
            print(f"❌ Failed to fetch page. Status code: {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        img = soup.find("img", src=lambda s: s and self.card_img_id in s)

        if not img:
            print(f"❌ Could not find image with '{self.card_img_id}' in src.")
            return None

        parent_a = img.find_parent("a")
        if parent_a and parent_a.has_attr("title"):
            title = parent_a["title"]
            print(f"✅ Found title: {title}")
            return title
        else:
            print("⚠️ Image found but no <a> parent with title attribute.")
            return None
