import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_wikipedia_links(list_url, limit=50):
    """Extract person article links from a list page."""
    response = requests.get(list_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find("div", {"class": "mw-parser-output"})
    
    links = []
    for li in content_div.find_all("li"):
        a_tag = li.find("a")
        print(f"found atag: {a_tag}")
        if a_tag and a_tag.get("href", "").startswith("/wiki/") and ":" not in a_tag["href"]:
            full_url = f"https://en.wikipedia.org{a_tag['href']}"
            links.append(full_url)
        if len(links) >= limit:
            break
    return links

def get_first_paragraph(wiki_url):
    """Extract the first paragraph from a Wikipedia article."""
    response = requests.get(wiki_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.select("div.mw-parser-output > p")
    
    for p in paragraphs:
        text = p.get_text(strip=True)
        if len(text) > 100:
            text = ' '.join(text.split())  # Normalize whitespace
            text = text.replace("\xa0", " ").replace("'", "").replace('"', '')
            return text
    return ""

def crawl_people_list(list_page_url, limit=50, delay=1.0):
    print(f"Fetching up to {limit} bios from: {list_page_url}")
    links = get_wikipedia_links(list_page_url, limit)
    
    records = []
    for i, link in enumerate(links):
        print(f"[{i+1}/{len(links)}] Fetching: {link}")
        paragraph = get_first_paragraph(link)
        if paragraph:
            name = link.split("/wiki/")[-1].replace("_", " ")
            records.append({"person_name": name, "text": paragraph, "source_url": link})
        time.sleep(delay)  # Be polite to Wikipedia

    return pd.DataFrame(records)

# Example usage:
df = crawl_people_list("https://en.wikipedia.org/wiki/List_of_scientists", limit=30)

df.to_csv("wikipedia_bios.csv", index=False)
print("Saved to wikipedia_bios.csv")
