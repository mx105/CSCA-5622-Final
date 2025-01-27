from playwright.sync_api import sync_playwright
import time

stars = 1
biz = "FoundingFarmers"
url = f"https://www.yelp.com/biz/founding-farmers-washington-washington-3?rr={str(stars)}"

def scrape_yelp_reviews():
    with sync_playwright() as p:
        # Launch the browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Navigate to the Yelp page
        page.goto(url)

        # Wait for reviews to load
        page.wait_for_selector('div[class*="biz-details"]')

        #page.mouse.wheel(0, 500)
        time.sleep(10)
        # Find all review elements
        spans = page.locator("span[class*='raw__']").all()

        reviews = []
        for span in spans:
            # Find the comment text element within each review
            text = span.text_content()
            if text:
                reviews.append(text)

        # Save to file
        with open(f'yelp_reviews_{stars}_{biz}.txt', 'w', encoding='utf-8') as f:
            for idx, review in enumerate(reviews, 1):
                f.write(f"Review {idx}:\n{review}\n\n")
                f.write("="*50 + "\n\n")

        browser.close()

if __name__ == "__main__":
    scrape_yelp_reviews()