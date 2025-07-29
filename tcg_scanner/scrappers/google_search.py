from googlesearch import search

class GoogleSearchScraper:
    def __init__(self, query: str, target_site: str, num_results: int = 10):
        self.query = query
        self.target_site = target_site.lower()
        self.num_results = num_results

    def get_first_match(self) -> str | None:
        try:
            for url in search(self.query, num_results=self.num_results):
                if self.target_site in url.lower():
                    print(f"✅ Found match: {url}")
                    return url
            print(f"❌ No results found on site: {self.target_site}")
            return None
        except Exception as e:
            print(f"❌ Google search failed: {e}")
            return None
