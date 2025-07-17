from icrawler.builtin import GoogleImageCrawler
from icrawler.downloader import ImageDownloader
import requests


class CustomDownloader(ImageDownloader):
    def get_response(self, url, **kwargs):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }
        kwargs['headers'] = headers
        try:
            response = requests.get(url, stream=True, timeout=5, **kwargs)
            return response
        except Exception as e:
            self.logger.error(f'Error fetching {url}: {e}')
            return None


# Use the custom downloader in your crawler
google_crawler = GoogleImageCrawler(
    downloader_cls=CustomDownloader,
    storage={'root_dir': 'google/raw_cards'}
)

google_crawler.crawl(keyword='what grade would this pokemon card reddit', max_num=200)
