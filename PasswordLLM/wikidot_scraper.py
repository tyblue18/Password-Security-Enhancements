import csv
import re
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
from time import sleep
import os
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "https://dnd5e.wikidot.com/"
SITEMAP_URL = "https://dnd5e.wikidot.com/system:page-tags/tag/srd"
OUTPUT_DIR = "wikidot_data"
CSV_HEADERS = ["url", "title", "content", "category", "source_book"]

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_sitemap_links():
    """Get all links from the sitemap page"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        print(f"Fetching sitemap from {SITEMAP_URL}")
        page.goto(SITEMAP_URL, timeout=60000)
        page.wait_for_selector(".page-tags", timeout=30000)
        
        # Extract all links from the sitemap
        links = page.eval_on_selector_all(".page-tags a", """elements => {
            return elements.map(el => {
                return {
                    url: el.href,
                    title: el.innerText.trim()
                }
            })
        }""")
        
        browser.close()
        return links

def scrape_page(url, title):
    """Scrape content from a single page"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            print(f"Scraping: {title} ({url})")
            page.goto(url, timeout=60000)
            page.wait_for_selector("#page-content", timeout=30000)
            
            # Get the main content
            content = page.evaluate("""() => {
                // Remove edit buttons and other noise
                document.querySelectorAll('.btn-group, .printuser').forEach(el => el.remove());
                
                const content = document.querySelector('#page-content');
                return content ? content.innerText : '';
            }""")
            
            # Clean up content
            content = re.sub(r'\n{3,}', '\n\n', content.strip())
            
            # Try to extract category from URL
            category = "unknown"
            if "/spells:" in url:
                category = "spells"
            elif "/classes:" in url:
                category = "classes"
            elif "/races:" in url:
                category = "races"
            elif "/monsters:" in url:
                category = "monsters"
            elif "/items:" in url:
                category = "items"
            
            # Try to identify source book
            source_book = "SRD"
            if "phb" in content.lower():
                source_book = "Player's Handbook"
            elif "dmg" in content.lower():
                source_book = "Dungeon Master's Guide"
            elif "volo" in content.lower():
                source_book = "Volo's Guide to Monsters"
            elif "xanathar" in content.lower():
                source_book = "Xanathar's Guide to Everything"
            
            return {
                "url": url,
                "title": title,
                "content": content,
                "category": category,
                "source_book": source_book
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
        finally:
            browser.close()

def save_to_csv(data, filename):
    """Save scraped data to CSV"""
    path = os.path.join(OUTPUT_DIR, filename)
    file_exists = os.path.isfile(path)
    
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def main():
    ensure_output_dir()
    links = get_sitemap_links()
    
    print(f"Found {len(links)} pages to scrape")
    
    for link in links:
        # Skip external links
        if not link['url'].startswith(BASE_URL):
            continue
            
        page_data = scrape_page(link['url'], link['title'])
        if page_data:
            # Create filename based on category
            filename = f"{page_data['category'] or 'misc'}.csv"
            save_to_csv(page_data, filename)
            
        # Be polite with delay between requests
        sleep(1)

if __name__ == "__main__":
    main()