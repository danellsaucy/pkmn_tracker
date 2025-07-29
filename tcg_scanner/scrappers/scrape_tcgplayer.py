import requests

class TcgplayerProductFetcher:
    def __init__(self, product_id: int, mpfev: int = 3979):
        self.product_id = product_id
        self.details_url = f"https://mp-search-api.tcgplayer.com/v2/product/{self.product_id}/details?mpfev={mpfev}"
        self.headers = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "origin": "https://www.tcgplayer.com",
            "referer": "https://www.tcgplayer.com/",
            "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        }

    def fetch_details(self) -> dict | None:
        try:
            response = requests.get(self.details_url, headers=self.headers)
            if response.ok:
                return response.json()
            else:
                print(f"❌ Failed to fetch product details. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error fetching product details: {e}")
            return None

    def print_market_price(self):
        data = self.fetch_details()
        if data and "marketPrice" in data:
            print(f"✅ Market Price: ${data['marketPrice']}")
            return data['marketPrice']
        else:
            print("⚠️ Market price not found.")


# Example usage
if __name__ == "__main__":
    fetcher = TcgplayerProductFetcher(product_id=517045)
    fetcher.print_market_price()
