
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
STATE_SUBDOMAIN = "wa"
BASE_URL = f"https://{STATE_SUBDOMAIN}.milesplit.com/results"

YEAR = 2015
MONTHS = [8, 9, 10, 11]          # Typical XC season
SEASON = "cross_country"
LEVEL = "hs"

OUTPUT_DIR = "data"
OUTPUT_FILE = f"{OUTPUT_DIR}/wa_hs_xc_meet_urls_2015.csv"

# --------------------------------------------------
# SET UP OUTPUT DIRECTORY
# --------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# SET UP SELENIUM (CHROME)
# --------------------------------------------------
options = Options()
options.add_argument("--headless")          # Run without opening a browser window
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-gpu")

driver = webdriver.Chrome(options=options)

# --------------------------------------------------
# SCRAPING LOGIC
# --------------------------------------------------
meet_data = []

for month in MONTHS:
    page = 1

    while True:
        url = (
            f"{BASE_URL}?"
            f"year={YEAR}"
            f"&month={month}"
            f"&season={SEASON}"
            f"&level={LEVEL}"
            f"&page={page}"
        )

        print(f"Loading: {url}")
        driver.get(url)
        time.sleep(3)  # allow JavaScript to load results

        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

        if not rows:
            break

        found_valid_row = False

        for row in rows:
            try:
                name_link = row.find_element(By.CSS_SELECTOR, "td.name a")
                meet_name = name_link.text.strip()
                meet_url = name_link.get_attribute("href")

                meet_data.append({
                    "meet_name": meet_name,
                    "meet_url": meet_url,
                    "year": YEAR,
                    "month": month,
                    "state": "WA",
                    "level": "HS",
                    "season": "XC"
                })

                found_valid_row = True

            except:
                # Skip ads / spacer rows
                continue

        if not found_valid_row:
            break

        page += 1

# --------------------------------------------------
# CLEAN UP & SAVE
# --------------------------------------------------
driver.quit()

df = pd.DataFrame(meet_data).drop_duplicates()
df.to_csv(OUTPUT_FILE, index=False)

print("\n----------------------------------")
print(f"Saved {len(df)} meet URLs")
print(f"File: {OUTPUT_FILE}")
print("----------------------------------")
