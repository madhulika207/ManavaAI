import requests
from bs4 import BeautifulSoup
import time
import csv
import json

BASE_URL = "http://quotes.toscrape.com"

def scrape_quotes():
    page_url = "/page/1/"
    all_quotes = []

    while page_url:
        url = BASE_URL + page_url
        print(f"Scraping {url} ...")
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            print(f"Failed to fetch {url} (status {resp.status_code}). Stopping.")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        quotes = soup.find_all("div", class_="quote")

        for q in quotes:
            text = q.find("span", class_="text").get_text(strip=True)
            author = q.find("small", class_="author").get_text(strip=True)
            tags = [tag.get_text(strip=True) for tag in q.find_all("a", class_="tag")]
            all_quotes.append({"quote": text, "author": author, "tags": tags})

        next_button = soup.find("li", class_="next")
        page_url = next_button.find("a")["href"] if next_button else None

        time.sleep(0.5)  # be polite

    return all_quotes

def save_to_csv(quotes, filename="quotes.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["quote", "author", "tags"])
        for q in quotes:
            writer.writerow([q["quote"], q["author"], ", ".join(q["tags"])])

def save_to_json(quotes, filename="quotes.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(quotes, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    data = scrape_quotes()
    print(f"Scraped {len(data)} quotes.")
    # Print first 5 items to verify
    for item in data[:5]:
        print(item)
    
    # Uncomment to save to files
    save_to_csv(data)
    save_to_json(data)


