import os
import time
import requests
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# === CONFIGURATION ===
seller_name = "jaivic_87"     # or "pokemon card lot" for raw cards
store = False
max_images = 10000                              # total number of images to download
save_dir = f"ebay_{seller_name}"                     # folder to save images
headless_mode = True                         # set to True to hide browser

os.makedirs(save_dir, exist_ok=True)

# === SETUP SELENIUM ===
chrome_options = Options()
if headless_mode:
    chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# === START SCRAPING ===
count = 0
page = 1
visited_urls = set()

print(f"🕵️ Searching eBay Store for: '{seller_name}'")

while count < max_images:
    print(f"\n📄 Scraping search results page {page}")
    if store:
        search_url = f"https://www.ebay.com/str/{seller_name}?_pgn={page}&_tab=shop"
    else:
        search_url = f"https://www.ebay.com/usr/{seller_name}?_pgn={page}&_tab=shop"
    if count == 0:
        print(f"🔗 Initial search URL: {search_url}")
    driver.get(search_url)
    time.sleep(5)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)

    # Find all product listing links
    articles = driver.find_elements(By.CSS_SELECTOR, "article.str-item-card.StoreFrontItemCard")
    item_urls = [
        article.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
        for article in articles
        if article.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
    ]
    if not item_urls:
        print("⚠️ No more listings found on this page.")
        break

    for url in item_urls:
        print(f"🔗 Found listing: {url}")
        if count >= max_images:
            break
        if url in visited_urls:
            continue
        visited_urls.add(url)

        try:
            driver.get(url)
            time.sleep(5)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)

            # Get all eBay-hosted images from the listing
            img_elements = driver.find_elements(By.CSS_SELECTOR, ".ux-image-carousel-item.image-treatment.active.image img")
            valid_image_found = False

            for img_el in img_elements:
                image_url = img_el.get_attribute("src")
                if not image_url or "ebayimg.com" not in image_url:
                    continue

                try:
                    response = requests.get(image_url, timeout=10)
                    img = Image.open(BytesIO(response.content))
                    width, height = img.size

                    if width >= 600 and height >= 600:
                        file_path = os.path.join(save_dir, f"card_{count:03d}_{seller_name}_raw.jpg")
                        img.save(file_path)
                        print(f"✅ Saved ({width}x{height}): {file_path}")
                        count += 1
                        valid_image_found = True
                        break  # Stop after saving the first valid image
                    else:
                        print(f"⚠️ Skipped ({width}x{height}): too small")

                except Exception as e:
                    print(f"❌ Failed to download/process image: {e}")

            if not valid_image_found:
                print("❌ No valid image found in listing.")

        except Exception as e:
            print(f"❌ Failed to scrape {url}\nError: {e}")

    # Try to go to next page
    # page += 1
    # print(f"➡️ Moving to next page {page}")
    try:
        next_page = driver.find_element(By.CSS_SELECTOR, "a.pagination__next")
        if next_page:
            print("➡️ Moving to next page")
            page += 1
        else:
            print("🚫 No next page found.")
            break
    except Exception:
        print("🚫 No next page button detected.")
        break

driver.quit()
print(f"\n🎉 Done! {count} high-res images saved to '{save_dir}/'")
