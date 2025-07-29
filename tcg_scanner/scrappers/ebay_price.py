import requests
import base64

EBAY_CLIENT_ID = "DanellFo-PokemonP-PRD-b0bc7982c-f4f4188d"
EBAY_CLIENT_SECRET = "PRD-0bc7982cad25-8ae2-49e9-a7a8-3dc6"

def get_ebay_oauth_token():
    auth = f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}"
    encoded_auth = base64.b64encode(auth.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope/marketplace.insights"
    }

    url = "https://api.ebay.com/identity/v1/oauth2/token"
    resp = requests.post(url, headers=headers, data=data)
    if resp.status_code == 200:
        return resp.json()["access_token"]
    else:
        print("❌ Failed to get token:", resp.text)
        return None


def get_sold_items(token, query="Mew ex PSA 10", limit=5):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    url = f"https://api.ebay.com/buy/marketplace_insights/v1/item_sales/search?q={query}&limit={limit}"

    response = requests.get(url, headers=headers)
    print(f"\n🔍 Request: {url}")
    print(f"🔁 Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        items = data.get("itemSales", [])
        if not items:
            print("⚠️ No sold listings found.")
            return

        prices = []
        for item in items:
            title = item.get("title", "N/A")
            price = item.get("price", {}).get("value")
            currency = item.get("price", {}).get("currency")
            prices.append(float(price))
            print(f"✔️ {title}\n   Sold Price: {price} {currency}\n")

        avg = sum(prices) / len(prices)
        print(f"📊 Average of last {len(prices)} sold: ${avg:.2f}")

    else:
        print(f"❌ Error: {response.text}")


if __name__ == "__main__":
    token = get_ebay_oauth_token()
    if token:
        get_sold_items(token, query="Mew ex PSA 10", limit=5)
