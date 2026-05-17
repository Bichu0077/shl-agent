"""
Scrape SHL Individual Test Solutions catalog into catalog.json
Run once: python scripts/scrape_catalog.py
"""

import json
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/productcatalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# SHL test type codes
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def get_all_catalog_pages():
    """Paginate through the catalog and collect all individual test product URLs."""
    products = []
    page = 0
    per_page = 12  # SHL uses 12 per page typically

    while True:
        url = f"{CATALOG_URL}?start={page * per_page}&type=1"  # type=1 = Individual Tests
        print(f"  Fetching catalog page {page + 1}: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"  Got {resp.status_code}, stopping pagination.")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find product cards / links
        items = soup.select("div.product-catalogue__item, div[class*='catalogue'] a, .product-card")
        
        # Broader fallback: find all links that look like product pages
        links = soup.find_all("a", href=re.compile(r"/solutions/products/product-catalog/|/en/solutions/products/"))
        product_links = []
        for link in links:
            href = link.get("href", "")
            full_url = urljoin(BASE_URL, href)
            if full_url not in [p["url"] for p in products]:
                name = link.get_text(strip=True)
                if name and len(name) > 2:
                    product_links.append({"name": name, "url": full_url})

        if not product_links:
            print("  No more products found.")
            break

        products.extend(product_links)
        page += 1
        time.sleep(1)

        if len(product_links) < per_page:
            break

    return products


def scrape_product_page(url: str) -> dict:
    """Scrape detail from a single product page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return {}
        soup = BeautifulSoup(resp.text, "html.parser")

        # Title
        title_el = soup.find("h1") or soup.find("h2")
        title = title_el.get_text(strip=True) if title_el else ""

        # Description - try multiple selectors
        desc = ""
        for sel in [".product-description", ".hero__description", ".content-block p", "article p", ".description"]:
            el = soup.select_one(sel)
            if el:
                desc = el.get_text(strip=True)
                break
        if not desc:
            # Fallback: first long paragraph
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 80:
                    desc = text
                    break

        # Test type codes (look for single uppercase letters in labels/tags)
        test_types = []
        type_pattern = re.compile(r'\b([ABCDEKМPS])\b')
        page_text = soup.get_text()
        
        # Look for explicit test type sections
        for label in soup.find_all(["span", "div", "td"], class_=re.compile(r"type|badge|tag|label", re.I)):
            txt = label.get_text(strip=True).upper()
            for code in TEST_TYPE_MAP:
                if txt == code or f"TYPE {code}" in txt:
                    test_types.append(code)

        # Job levels
        job_levels = []
        level_keywords = ["graduate", "manager", "director", "entry", "mid", "senior", "executive", "professional", "operator", "frontline"]
        for kw in level_keywords:
            if kw.lower() in page_text.lower():
                job_levels.append(kw.title())

        # Duration
        duration = ""
        dur_match = re.search(r'(\d+)\s*(min|minute)', page_text, re.I)
        if dur_match:
            duration = f"{dur_match.group(1)} minutes"

        # Remote / proctored
        remote_testing = "remote" in page_text.lower() or "online" in page_text.lower()
        adaptive = "adaptive" in page_text.lower()

        # Languages
        languages = []
        lang_match = re.search(r'(\d+)\s+languages?', page_text, re.I)
        if lang_match:
            languages = [f"Available in {lang_match.group(1)} languages"]

        return {
            "name": title,
            "url": url,
            "description": desc,
            "test_types": test_types,
            "job_levels": job_levels,
            "duration": duration,
            "remote_testing": remote_testing,
            "adaptive": adaptive,
            "languages": languages,
            "raw_text": page_text[:3000],  # Keep first 3000 chars for RAG
        }

    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return {}


def scrape_catalog_table() -> list[dict]:
    """
    Primary scraping strategy: parse the catalog table directly.
    SHL's catalog page has a filterable table of all products.
    """
    print("Fetching main catalog page...")
    
    # Try fetching with type=1 filter for Individual Test Solutions
    url = f"{CATALOG_URL}?type=1&start=0"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")

    products = []

    # Strategy 1: find table rows
    rows = soup.select("table tr, .product-catalogue__item, [data-course-code]")
    print(f"  Found {len(rows)} rows/items via table strategy")

    # Strategy 2: find all product links more broadly
    all_links = soup.find_all("a", href=True)
    product_links = set()
    for a in all_links:
        href = a["href"]
        if "/product-catalog/" in href or "/productcatalog/" in href:
            full = urljoin(BASE_URL, href)
            if full != CATALOG_URL and full != url:
                product_links.add((a.get_text(strip=True), full))

    print(f"  Found {len(product_links)} product links on page")

    # Strategy 3: parse the JS/JSON data if embedded
    scripts = soup.find_all("script")
    for script in scripts:
        content = script.string or ""
        if "productcatalog" in content.lower() or "assessments" in content.lower():
            # Try to extract JSON
            json_match = re.search(r'\[(\{.*?"name".*?\})\]', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(f"[{json_match.group(1)}]")
                    print(f"  Found {len(data)} products in embedded JSON")
                    return data
                except:
                    pass

    for name, link_url in product_links:
        if name and len(name) > 2:
            products.append({"name": name, "url": link_url})

    return products


def build_catalog():
    """Main entry point."""
    print("=" * 60)
    print("SHL Catalog Scraper")
    print("=" * 60)

    # Step 1: get product list from catalog page
    products = scrape_catalog_table()

    if not products:
        print("Table strategy failed, trying pagination...")
        products = get_all_catalog_pages()

    print(f"\nFound {len(products)} products. Scraping detail pages...")

    enriched = []
    seen_urls = set()

    for i, prod in enumerate(products):
        url = prod.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        print(f"  [{i+1}/{len(products)}] {prod.get('name', url)[:60]}")
        detail = scrape_product_page(url)

        if detail:
            # Merge: detail takes precedence, fallback to list name
            merged = {
                "name": detail.get("name") or prod.get("name", ""),
                "url": url,
                "description": detail.get("description", ""),
                "test_types": detail.get("test_types", []),
                "job_levels": detail.get("job_levels", []),
                "duration": detail.get("duration", ""),
                "remote_testing": detail.get("remote_testing", False),
                "adaptive": detail.get("adaptive", False),
                "languages": detail.get("languages", []),
                "raw_text": detail.get("raw_text", ""),
            }
            enriched.append(merged)
        else:
            enriched.append({
                "name": prod.get("name", ""),
                "url": url,
                "description": "",
                "test_types": [],
                "job_levels": [],
                "duration": "",
                "remote_testing": False,
                "adaptive": False,
                "languages": [],
                "raw_text": "",
            })

        time.sleep(0.5)  # polite crawling

    # Save
    out_path = "data/catalog.json"
    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"\nSaved {len(enriched)} products to {out_path}")
    return enriched


if __name__ == "__main__":
    build_catalog()
