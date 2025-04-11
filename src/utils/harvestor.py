import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

WIKI_BASE = "https://en.wikipedia.org"
EXCLUDE_PREFIXES = ["/wiki/Wikipedia", "/wiki/Help", "/wiki/Template", "/wiki/Category", "/wiki/Portal", "/wiki/Special"]

def is_valid_link(href):
    return (
        href.startswith("/wiki/")
        and ":" not in href
        and not any(href.startswith(prefix) for prefix in EXCLUDE_PREFIXES)
    )

def harvest_links(start_url, depth=2, min_links_threshold=50, sleep_time=1.0):
    visited = set()
    harvested_links = set()

    def crawl(url, level):
        if url in visited or level > depth:
            return
        print(f"[Depth {level}] Crawling: {url}")
        visited.add(url)

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            all_links = soup.find_all("a", href=True)

            page_links = [urljoin(WIKI_BASE, a["href"]) for a in all_links if is_valid_link(a["href"])]
            print(f"  Found {len(page_links)} candidate links.")

            if len(page_links) >= min_links_threshold:
                for link in page_links:
                    harvested_links.add(link)
                    crawl(link, level + 1)

            time.sleep(sleep_time)
        except Exception as e:
            print(f"  Error fetching {url}: {e}")

    crawl(start_url, level=1)
    return sorted(harvested_links)

# Example usage:
if __name__ == "__main__":
    seed_url = "https://en.wikipedia.org/wiki/List_of_scientists"
    links = harvest_links(seed_url, depth=3, min_links_threshold=50)
    with open("raw_harvested_urls.txt", "w") as f:
        for link in links:
            f.write(link + "\n")
    print(f"\n✅ Saved {len(links)} links to raw_harvested_urls.txt")
