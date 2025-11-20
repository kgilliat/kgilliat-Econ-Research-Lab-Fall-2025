import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import os

INDIVIDUAL_TABLE_HEADERS = ['place', 'video', 'athlete', 'grade', 'team', 'finish', 'point']
TEAM_TABLE_HEADERS = ['place', 'tsTeam', 'point', 'wind', 'heat']


# --------------------------------------------
# Extract race ID
# --------------------------------------------
def extract_race_id(url):
    match = re.search(r'results/(\d+)/', url)
    return match.group(1) if match else None


# --------------------------------------------
# DETECTORS
# --------------------------------------------
REQUIRED_HEADERS_KATIE = {"place", "video", "athlete", "team", "mark", "points"}

def detect_katie(html):
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    event_table = None
    for tbl in soup.find_all("table"):
        cls = tbl.get("class", [])
        if isinstance(cls, str):
            cls = cls.split()
        tokens = [c.strip().lower() for c in cls]
        if any("eventtable" == tok or "eventtable" in tok for tok in tokens):
            event_table = tbl
            break

    if event_table is None:
        return 0.0

    score += 0.6

    th_texts = [
        th.get_text(" ", strip=True).lower()
        for th in event_table.select("thead th")
        if th.get_text(strip=True)
    ]
    headers_found = set(th_texts)
    matched = len(REQUIRED_HEADERS_KATIE.intersection(headers_found))
    if matched:
        score += 0.3 * (matched / len(REQUIRED_HEADERS_KATIE))

    links_in_body = event_table.select("tbody a[href]")
    if len(links_in_body) >= 1:
        score += 0.05
    if len(links_in_body) >= 2:
        score += 0.05

    return min(score, 1.0)


def detect_cole(html):
    REQUIRED_HEADERS_COLE = {"pl", "athlete", "yr", "team", "time"}

    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    results_body = soup.find(id="meetResultsBody") or soup.find(class_="meetResultsBody")
    if not results_body:
        return 0.0

    pre_blocks = results_body.find_all("pre")
    table_blocks = results_body.find_all("table")

    for pre in pre_blocks:
        text = pre.get_text(" ", strip=True).lower()
        if all(h in text for h in REQUIRED_HEADERS_COLE):
            score += 0.6
            break

    if pre_blocks and not table_blocks:
        score += 0.3
    else:
        return 0.0

    found_team_scores = any("team scores" in pre.get_text().lower() for pre in pre_blocks)
    if not found_team_scores:
        score += 0.1

    return float(min(1.0, score))


def detect_max(html):
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    target_table = None
    for tbl in soup.find_all("table"):
        classes = [c.lower() for c in tbl.get("class", [])]
        if any("eventtable" in c for c in classes):
            continue
        target_table = tbl
        break

    if not target_table:
        return 0.0

    score += 0.6

    REQUIRED_HEADERS = {"place", "athlete", "grade", "team", "avg mile", "finish", "points"}
    th_texts = [th.get_text(" ", strip=True).lower() for th in target_table.find_all("th")]
    headers_found = set(th_texts)

    matched = len(REQUIRED_HEADERS.intersection(headers_found))
    if matched:
        score += 0.3 * (matched / len(REQUIRED_HEADERS))

    links_in_body = target_table.select("tbody a[href]")
    if len(links_in_body) == 0:
        score += 0.1

    return min(score, 1.0)


def detect_adam(html):
    return 0.0    # TODO


# --------------------------------------------
# WRANGLERS
# --------------------------------------------
def wrangle_cole(html, race_url=None):
    soup = BeautifulSoup(html, "html.parser")
    results_div = soup.find("div", id="meetResultsBody")
    if not results_div:
        return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)

    pre = results_div.find("pre")
    if not pre:
        return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)

    text = pre.get_text("\n", strip=True)
    lines = text.splitlines()

    data_lines = [ln for ln in lines if re.match(r"^\s*\d+", ln)]

    rows = []
    for ln in data_lines:
        match = re.match(
            r"^\s*(\d+)\s+(.+?)\s+(\d+)?\s+(.+?)\s+(\d{1,2}:\d{2}\.\d)",
            ln
        )
        if not match:
            continue

        rows.append({
            "place": int(match.group(1)),
            "video": None,
            "athlete": match.group(2).strip().title(),
            "grade": int(match.group(3)) if match.group(3) else pd.NA,
            "team": match.group(4).strip(),
            "finish": match.group(5),
            "point": pd.NA
        })

    return pd.DataFrame(rows, columns=INDIVIDUAL_TABLE_HEADERS)


def wrangle_max(html, race_url=None):
    return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS), pd.DataFrame(columns=TEAM_TABLE_HEADERS)


def wrangle_adam(html, race_url=None):
    return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS), pd.DataFrame(columns=TEAM_TABLE_HEADERS)


def wrangle_katie(html, race_url=None):
    return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS), pd.DataFrame(columns=TEAM_TABLE_HEADERS)


# --------------------------------------------
# PROCESS ALL URLS
# --------------------------------------------
def process_all_urls(urls):
    master_individual = []
    master_team = []
    master_metadata = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for url in urls:
            print(f"\n🔍 Processing URL: {url}")

            try:
                page.goto(url, timeout=20000)
                html = page.content()
                race_id = extract_race_id(url)

                cole_score  = detect_cole(html)
                katie_score = detect_katie(html)
                max_score   = detect_max(html)
                adam_score  = detect_adam(html)

                scores = {
                    "cole": cole_score,
                    "katie": katie_score,
                    "max": max_score,
                    "adam": adam_score
                }

                best = max(scores, key=scores.get)
                best_score = scores[best]

                print(f"   👉 Best detector = {best} (score={best_score:.2f})")

                if best_score < 0.7:
                    print("   ❌ Low confidence — skipping")
                    continue

                if best == "cole":
                    indiv_df = wrangle_cole(html, url)
                    team_df  = pd.DataFrame()

                elif best == "katie":
                    indiv_df, team_df = wrangle_katie(html, url)

                elif best == "max":
                    indiv_df, team_df = wrangle_max(html, url)

                elif best == "adam":
                    indiv_df, team_df = wrangle_adam(html, url)

                indiv_df["race_id"] = race_id
                indiv_df["race_url"] = url
                master_individual.append(indiv_df)

                if not team_df.empty:
                    team_df["race_id"] = race_id
                    team_df["race_url"] = url
                    master_team.append(team_df)

                master_metadata.append(pd.DataFrame([{
                    "race_id": race_id,
                    "url": url,
                    "detector": best,
                    "score": best_score,
                    "status": "success"
                }]))

            except Exception as e:
                print(f"   ERROR: {e}")
                master_metadata.append(pd.DataFrame([{
                    "race_id": extract_race_id(url),
                    "url": url,
                    "detector": None,
                    "score": None,
                    "status": f"error: {e}"
                }]))

        browser.close()

    return (
        pd.concat(master_individual, ignore_index=True) if master_individual else pd.DataFrame(),
        pd.concat(master_team, ignore_index=True) if master_team else pd.DataFrame(),
        pd.concat(master_metadata, ignore_index=True)
    )


# --------------------------------------------
# TEST 1 — Run script on FIRST 5 race URLs only
# --------------------------------------------

df = pd.read_csv(
    r"C:\Users\coleg\OneDrive\Documents\Econ Research Lab\Kurtis-Econ-Research-Lab-Fall-2025\race_urls_2016.0.csv"
)

test_urls = df["race_url"].head(5).tolist()

print("\n==============================")
print("  TEST MODE: FIRST 5 URLs")
print("==============================\n")

individual, team, metadata = process_all_urls(test_urls)

print("\n--- Individual Results Preview ---")
print(individual.head())

print("\n--- Team Results Preview ---")
print(team.head())

print("\n--- Metadata Preview ---")
print(metadata.head())

test_output_dir = (
    r"C:\Users\coleg\OneDrive\Documents\Econ Research Lab\Kurtis-Econ-Research-Lab-Fall-2025\output\test_runs"
)
os.makedirs(test_output_dir, exist_ok=True)

individual.to_csv(os.path.join(test_output_dir, "test_individual_first5.csv"), index=False)
team.to_csv(os.path.join(test_output_dir, "test_team_first5.csv"), index=False)
metadata.to_csv(os.path.join(test_output_dir, "test_metadata_first5.csv"), index=False)

print("\n✓ TEST (first 5 URLs) COMPLETE — Files saved in test_runs\n")



# FULL CSV RUN


# print("\n===================================")
# print("  FULL CSV RUN: Processing ALL URLs")
# print("===================================\n")

# all_urls = df["race_url"].tolist()
# individual_all, team_all, metadata_all = process_all_urls(all_urls)

# full_output_dir = (
#     r"C:\Users\coleg\OneDrive\Documents\Econ Research Lab\Kurtis-Econ-Research-Lab-Fall-2025\output"
# )

# os.makedirs(full_output_dir, exist_ok=True)

# individual_all.to_csv(os.path.join(full_output_dir, "all_individual_results.csv"), index=False)
# team_all.to_csv(os.path.join(full_output_dir, "all_team_results.csv"), index=False)
# metadata_all.to_csv(os.path.join(full_output_dir, "all_metadata.csv"), index=False)

# print("\n✓ FULL CSV RUN COMPLETE — Output saved.\n")
