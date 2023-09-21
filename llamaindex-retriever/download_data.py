import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://paulgraham.com/"
ARTICLES_DIR = "data"


def get_article_links():
    response = requests.get(f"{BASE_URL}articles.html")
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = [link['href'] for link in soup.find_all('a', href=True) if
                link['href'].endswith('.html') and
                link['href'] != 'index.html']
    return articles


def download_article(article_link):
    response = requests.get(f"{BASE_URL}{article_link}")
    soup = BeautifulSoup(response.content, 'html.parser')

    # Assuming the title is in the first <h1> tag
    title = article_link.rstrip('.html')
    content = str(soup)

    with open(os.path.join(ARTICLES_DIR, f"{title}.txt"), "w",
              encoding="utf-8") as file:
        file.write(content)


def main():
    if not os.path.exists(ARTICLES_DIR):
        os.makedirs(ARTICLES_DIR)

    articles = get_article_links()
    for article in articles:
        print(f"Downloading {article}...")
        download_article(article)
    print("Download complete.")


if __name__ == "__main__":
    main()
