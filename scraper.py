
import os, re, requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

URL = "https://medlineplus.gov/temporomandibulardisorders.html"
OUT = "data/temporomandibular-disorders.md"
HEADERS = {"User-Agent": "Mozilla/5.0 (educational project)"}

def main():
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Trying to grab main content; fall back to <main> or body
    main = soup.select_one("#topic-content") or soup.select_one("#topic-summary") \
           or soup.find("main") or soup.body or soup

    # Cleaning obvious noise
    for sel in ["nav","header","footer",".share",".breadcrumbs",".print",".social",".ad",".ads"]:
        for el in main.select(sel):
            el.decompose()

    text = md(str(main), strip=["img","svg","script","style"])
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"# Temporomandibular Disorders (MedlinePlus)\n\nSource: {URL}\n\n{text}\n")

    print(f"Saved -> {OUT}")

if __name__ == "__main__":
    main()
