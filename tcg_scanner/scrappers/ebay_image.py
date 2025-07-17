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
search_query = "raw pokemon card near mint"     # or "pokemon card lot" for raw cards
max_images = 200                              # total number of images to download
save_dir = "ebay_highres_raw"                     # folder to save images
headless_mode = False                         # set to True to hide browser

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

print(f"🕵️ Searching eBay for: '{search_query}'")

while count < max_images:
    print(f"\n📄 Scraping search results page {page}")
    search_url = f"https://www.ebay.com/sch/i.html?_nkw={search_query.replace(' ', '+')}&_pgn={page}"
    driver.get(search_url)
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    # Find all product listing links
    item_links = driver.find_elements(By.CSS_SELECTOR, "li.s-item a.s-item__link")
    item_urls = [item.get_attribute("href") for item in item_links if item.get_attribute("href")]

    if not item_urls:
        print("⚠️ No more listings found on this page.")
        break

    for url in item_urls:
        if count >= max_images:
            break
        if url in visited_urls:
            continue
        visited_urls.add(url)

        try:
            driver.get(url)
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            # Get all eBay-hosted images from the listing
            img_elements = driver.find_elements(By.CSS_SELECTOR, "img[src*='ebayimg.com']")
            valid_image_found = False

            for img_el in img_elements:
                image_url = img_el.get_attribute("src")
                if not image_url or "ebayimg.com" not in image_url:
                    continue

                try:
                    response = requests.get(image_url, timeout=10)
                    img = Image.open(BytesIO(response.content))
                    width, height = img.size

                    if width >= 400 and height >= 400:
                        file_path = os.path.join(save_dir, f"card_{count:03d}_raw.jpg")
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
    try:
        next_page = driver.find_element(By.CSS_SELECTOR, "a.pagination__next icon-link")
        if next_page and next_page.is_enabled():
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
