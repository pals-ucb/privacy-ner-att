import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def clean_paragraph(p):
    text = p.get_text().strip()
    text = text.replace("[edit]", "").replace("\n", " ").strip()
    return ' '.join(text.split())

def extract_paragraphs(url, min_len=100):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.select("div.mw-parser-output > p")

        cleaned = [clean_paragraph(p) for p in paragraphs if len(clean_paragraph(p)) > min_len]

        if not cleaned:
            return "", "", "", "Non Person"

        # Pad to always return 3 paragraphs
        while len(cleaned) < 3:
            cleaned.append("")

        return cleaned[0], cleaned[1], cleaned[2], "Valid"
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return "", "", "", "Non Person"

def process_url_list_chunked(input_path, output_prefix="bio_data_chunk", chunk_size=1000):
    with open(input_path, "r") as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    chunk = []
    chunk_idx = 0
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Processing: {url}")
        para1, para2, para3, label = extract_paragraphs(url)
        chunk.append({
            "url": url,
            "label": label,
            "para1": para1,
            "para2": para2,
            "para3": para3
        })

        # Write to file every chunk_size
        if (i + 1) % chunk_size == 0 or (i + 1) == len(urls):
            df_chunk = pd.DataFrame(chunk)
            filename = f"{output_prefix}_{chunk_idx:03}.csv"
            df_chunk.to_csv(filename, index=False)
            print(f"✅ Wrote {len(chunk)} records to {filename}")
            chunk = []
            chunk_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract bios from a list of Wikipedia URLs.")
    parser.add_argument("--harvested_file", type=str, required=True, help="Input file containing list of Wikipedia URLs (one per line)")
    parser.add_argument("--bio_data_file", type=str, required=True, help="Output CSV to save extracted bios")

    args = parser.parse_args()
    process_url_list_chunked(args.harvested_file, args.bio_data_file)
