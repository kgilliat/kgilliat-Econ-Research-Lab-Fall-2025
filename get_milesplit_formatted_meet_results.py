import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import os

# ============================================================
# CONSTANTS
# ============================================================

INDIVIDUAL_TABLE_HEADERS = ['place', 'video', 'athlete', 'grade', 'team', 'finish', 'point']
TEAM_TABLE_HEADERS       = ['place', 'tsTeam', 'point', 'wind', 'heat']

TIME_PATTERN = re.compile(
    r"\d+:\d{2}(?:\.\d+)?|\d+:\d+:\d{2}(?:\.\d+)?"
)  # mm:ss(.xx) or h:mm:ss(.xx)

TAG_AFTER_TIME = re.compile(r"^(PR|SR|NR|DNF|DNS|DQ|NT)$", re.IGNORECASE)


# ============================================================
# SHARED: extract_race_id
# ============================================================

def extract_race_id(url: str):
    match = re.search(r'results/(\d+)/', url)
    return match.group(1) if match else None


# ============================================================
# DETECTORS
# Each detector returns a score in [0, 1].
# We will pick the parser with the highest score.
# ============================================================

REQUIRED_HEADERS_KATIE = {"place", "video", "athlete", "grade", "team", "finish", "point"}
REQUIRED_HEADERS_COLE  = {"results", "print", "mile", "run"}  # loose hints
REQUIRED_HEADERS_MAX   = {"fr", "so", "jr", "sr"}             # class codes
REQUIRED_HEADERS_ADAM  = {"place", "athlete", "grade", "school", "time"}


# ============================================================
# IMPROVED DETECTORS
# ============================================================

def detect_cole(html: str) -> float:
    """
    Cole: PRE-based results with NUMERIC grades (6, 7, 8, etc.)
    Format: "   1 Name             7 School              12:46.8"
    """
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    results_body = soup.find(id="meetResultsBody") or soup.find(class_="meetResultsBody")
    if not results_body:
        return 0.0

    pre_blocks = results_body.find_all("pre")
    if not pre_blocks:
        return 0.0

    # Flatten text
    text_all = " ".join(pre.get_text(" ", strip=True) for pre in pre_blocks)
    
    # STRONG INDICATOR: Numeric grades (single digits) with surrounding structure
    # Pattern: place number, then name, then single digit grade, then time
    numeric_grade_pattern = re.compile(
        r'\b\d+\s+[A-Za-z]+\s+[A-Za-z]+\s+(\d)\s+',  # Matches "1 First Last 7 "
        re.MULTILINE
    )
    numeric_grades = numeric_grade_pattern.findall(text_all)
    
    if len(numeric_grades) >= 5:  # Found multiple numeric grades
        score += 0.7
    elif len(numeric_grades) >= 2:
        score += 0.4
    
    # Time tokens
    TIME_PATTERN = re.compile(r"\d+:\d{2}(?:\.\d+)?")
    times = TIME_PATTERN.findall(text_all)
    if len(times) >= 8:
        score += 0.2
    elif len(times) >= 4:
        score += 0.1
    
    # Place markers with leading spaces (Cole format has "   1" not "1.")
    place_markers = re.findall(r'^\s+\d+\s', text_all, re.MULTILINE)
    if len(place_markers) >= 5:
        score += 0.15
    
    # PENALTY: FR/SO/JR/SR indicates Max format, not Cole
    grade_tokens = re.findall(r'\b(FR|SO|JR|SR)\b', text_all)
    if len(grade_tokens) >= 3:
        score *= 0.3  # Strong penalty
    
    return float(min(score, 1.0))


def detect_max(html: str) -> float:
    """
    Max: PRE-based results with FR/SO/JR/SR grade codes.
    Format: "1   Daniel Filipcik         SR   Woodside    5:11    15:18"
    """
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    results_body = soup.find(id="meetResultsBody") or soup.find(class_="meetResultsBody")
    if not results_body:
        return 0.0

    pre_blocks = results_body.find_all("pre")
    if not pre_blocks:
        return 0.0

    text_all = " ".join(pre.get_text(" ", strip=True) for pre in pre_blocks)
    
    # STRONG INDICATOR: FR/SO/JR/SR tokens
    grade_tokens = re.findall(r'\b(FR|SO|JR|SR)\b', text_all)
    if len(grade_tokens) >= 8:  # Many grade codes
        score += 0.7
    elif len(grade_tokens) >= 4:
        score += 0.5
    elif len(grade_tokens) >= 1:
        score += 0.2
    
    # Time tokens
    TIME_PATTERN = re.compile(r"\d+:\d{2}(?:\.\d+)?")
    times = TIME_PATTERN.findall(text_all)
    if len(times) >= 5:
        score += 0.2
    
    # Team scores section (common in Max format)
    if re.search(r'Team\s+Scores', text_all, re.IGNORECASE):
        score += 0.15
    
    # PENALTY: Single digit grades indicate Cole format
    numeric_grade_pattern = re.compile(r'\b\d+\s+[A-Za-z]+\s+[A-Za-z]+\s+(\d)\s+')
    numeric_grades = numeric_grade_pattern.findall(text_all)
    if len(numeric_grades) >= 3:
        score *= 0.4  # Penalty for Cole indicators
    
    return float(min(score, 1.0))


def detect_adam(html: str) -> float:
    """
    Adam: Simple TABLE-based format with <td> cells.
    Format: <table><tr><td>Place</td><td>Athlete</td><td>Grade</td><td>School</td><td>Time</td></tr>
    """
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    container = soup.find(id="meetResultsBody") or soup.find(class_="meetResultsBody")
    if not container:
        return 0.0

    tables = container.find_all("table")
    if not tables:
        return 0.0
    
    # STRONG INDICATOR: Table exists in meetResultsBody
    score += 0.3
    
    # Look for header row patterns
    best_header_match = 0
    expected_headers = {'place', 'athlete', 'grade', 'school', 'time'}
    
    for table in tables:
        # Get first row (likely headers)
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
            
        first_row_cells = rows[0].find_all(["td", "th"])
        if len(first_row_cells) >= 4:  # Has multiple columns
            # Extract text from first row
            header_texts = {cell.get_text(" ", strip=True).lower() 
                          for cell in first_row_cells}
            
            # Check for expected headers
            matches = len(expected_headers.intersection(header_texts))
            best_header_match = max(best_header_match, matches)
    
    # Score based on header matches
    if best_header_match >= 4:
        score += 0.5
    elif best_header_match >= 3:
        score += 0.3
    elif best_header_match >= 2:
        score += 0.2
    
    # Check table structure (many data rows)
    for table in tables:
        rows = table.find_all("tr")
        data_rows = [r for r in rows if len(r.find_all("td")) >= 4]
        
        if len(data_rows) >= 10:  # Has many data rows
            score += 0.2
            break
        elif len(data_rows) >= 5:
            score += 0.1
            break
    
    return float(min(score, 1.0))


def detect_katie(html: str) -> float:
    """
    Katie: Complex table-based pages with class-based cells 
    (e.g., <td class="place">, <td class="athlete">)
    """
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    REQUIRED_HEADERS_KATIE = {"place", "video", "athlete", "grade", "team", "finish", "point"}
    
    tables = soup.find_all("table")
    if not tables:
        return 0.0

    best_hit = 0
    for tbl in tables:
        # Look for class-based cells
        cell_classes = set()
        for cell in tbl.find_all(["td", "th"]):
            cls = cell.get("class", [])
            if isinstance(cls, str):
                cls = cls.split()
            for c in cls:
                cell_classes.add(c.strip().lower())
        
        hits = len(REQUIRED_HEADERS_KATIE.intersection(cell_classes))
        best_hit = max(best_hit, hits)

    # STRONG INDICATOR: Class-based table structure
    if best_hit >= 5:  # Many Katie-specific classes
        score += 0.7
    elif best_hit >= 3:
        score += 0.5
    elif best_hit >= 1:
        score += 0.2

    # Look for 'eventtable' style classes
    has_event_table = False
    for tbl in tables:
        cls = tbl.get("class", [])
        if isinstance(cls, str):
            cls = cls.split()
        if any("eventtable" in c.lower() for c in cls):
            has_event_table = True
            break
    
    if has_event_table:
        score += 0.2

    # Links inside table (athlete/team URLs)
    links = soup.select("table tbody a[href]")
    if len(links) >= 5:
        score += 0.1

    return float(min(score, 1.0))

# ============================================================
# WRANGLERS
# ============================================================

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def wrangle_cole(html: str, race_url: str = None) -> pd.DataFrame:
    """
    Robust PRE parser for Cole-style pages (numeric grades).
    Handles both line-based and '1. 10 Name 23:25 PR Team ...' packed text.
    """
    soup = BeautifulSoup(html, "html.parser")
    results_div = soup.find("div", id="meetResultsBody") or soup.find("div", class_="meetResultsBody")
    if not results_div:
        return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)

    pre = results_div.find("pre")
    if not pre:
        return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)

    text = pre.get_text("\n", strip=True)

    # first try line-based parsing
    rows = []
    for raw_line in text.splitlines():
        line = _normalize_whitespace(raw_line)
        if not re.match(r"^\d+", line):
            continue

        # pattern: place [grade] name time [tag] team
        m = re.match(
            r"^(?P<place>\d+)\.?\s+"
            r"(?:(?P<grade>\d+)\s+)?"
            r"(?P<name>[A-Za-z',.\- ]+?)\s+"
            r"(?P<time>\d+:\d{2}(?:\.\d+)?|\d+:\d+:\d{2}(?:\.\d+)?)"
            r"(?:\s+(?P<tag>[A-Za-z]+))?\s+"
            r"(?P<team>[A-Za-z][A-Za-z .'\-]+)$",
            line
        )
        if not m:
            continue

        g = m.groupdict()
        finish = g["time"]
        # ignore tag (PR, SR, etc.) except we don't want to swallow time
        rows.append({
            "place": int(g["place"]),
            "video": None,
            "athlete": g["name"].strip(),
            "grade": int(g["grade"]) if g["grade"] is not None else pd.NA,
            "team": g["team"].strip(),
            "finish": finish,
            "point": pd.NA
        })

    # if we got enough rows, use them
    if len(rows) >= 3:
        return pd.DataFrame(rows, columns=INDIVIDUAL_TABLE_HEADERS)

    # otherwise, fall back to packed-text parsing:
    flat = _normalize_whitespace(text)

    packed_pattern = re.compile(
        r"(?P<place>\d+)\.\s+"
        r"(?:(?P<grade>\d+)\s+)?"
        r"(?P<name>[A-Za-z',.\- ]+?)\s+"
        r"(?P<time>\d+:\d{2}(?:\.\d+)?|\d+:\d+:\d{2}(?:\.\d+)?)"
        r"(?:\s+(?P<tag>[A-Za-z]+))?\s+"
        r"(?P<team>[A-Za-z][A-Za-z .'\-]+?)"
        r"(?=\s+\d+\.|$)"
    )

    rows = []
    for m in packed_pattern.finditer(flat):
        g = m.groupdict()
        finish = g["time"]
        rows.append({
            "place": int(g["place"]),
            "video": None,
            "athlete": g["name"].strip(),
            "grade": int(g["grade"]) if g["grade"] is not None else pd.NA,
            "team": g["team"].strip(),
            "finish": finish,
            "point": pd.NA
        })

    if not rows:
        return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)

    return pd.DataFrame(rows, columns=INDIVIDUAL_TABLE_HEADERS)


def wrangle_max(html: str, race_url: str = None):
    """
    PRE parser for Max-style pages with FR/SO/JR/SR grades.
    We keep your earlier pattern but with a bit of whitespace normalization.
    """
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", id="meetResultsBody") or soup.find("div", class_="meetResultsBody")
    if not container:
        return (
            pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS),
            pd.DataFrame(columns=TEAM_TABLE_HEADERS)
        )

    pre = container.find("pre")
    if not pre:
        return (
            pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS),
            pd.DataFrame(columns=TEAM_TABLE_HEADERS)
        )

    text = pre.get_text("\n", strip=True)
    text = _normalize_whitespace(text)

    sections = re.split(r'(?=\b[A-Z][A-Za-z/ &-]+ (?:Boys|Girls)\b)', text)

    rows = []

    line_pattern = re.compile(
        r'^(\d+)\s+([A-Za-z\'\-. ]+?)\s+(FR|SO|JR|SR)\s+'
        r'([A-Za-z\'\-. ]+?)\s+\d*:?[\d.]*\s+(\d+:\d+(?:\.\d+)?)\s+(\d+)?$'
    )

    for section in sections:
        section = section.strip()
        if not section:
            continue

        for raw_line in section.splitlines():
            line = _normalize_whitespace(raw_line)
            if not re.match(r'^\d+\s', line):
                continue

            m = line_pattern.match(line)
            if not m:
                continue

            place, athlete, grade, team, finish, point = m.groups()
            rows.append({
                "place": int(place),
                "video": None,
                "athlete": athlete.strip(),
                "grade": grade,
                "team": team.strip(),
                "finish": finish,
                "point": point if point else pd.NA
            })

    indiv_df = pd.DataFrame(rows, columns=INDIVIDUAL_TABLE_HEADERS)
    return indiv_df, pd.DataFrame(columns=TEAM_TABLE_HEADERS)


def wrangle_adam(html: str, race_url: str = None):
    """
    For now, Adam's wrangler simply returns empty; we rely on the
    robust table parser (Katie-style) for table pages.
    We keep this stub to preserve the groupmate structure.
    """
    return (
        pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS),
        pd.DataFrame(columns=TEAM_TABLE_HEADERS)
    )


def wrangle_katie(html: str, race_url: str = None):
    """
    Unused directly; robust table parser below plays Katie's role.
    """
    return (
        pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS),
        pd.DataFrame(columns=TEAM_TABLE_HEADERS)
    )


# ============================================================
# ROBUST TABLE PARSER (Katie-style but more tolerant)
# ============================================================

def extract_table_data(page_content: str, url: str):
    race_id = extract_race_id(url)
    soup    = BeautifulSoup(page_content, 'html.parser')
    tables  = soup.find_all('table')

    if not tables:
        print(f"   No tables found for URL: {url}")
        empty = {"individual": pd.DataFrame(), "team": pd.DataFrame()}
        meta  = pd.DataFrame([{
            "race_id": race_id,
            "url": url,
            "table_index": None,
            "table_type": "no_tables",
            "row_count": 0
        }])
        return empty, meta

    all_data = {"individual": [], "team": []}
    metadata = []

    indiv_headers_set = set(INDIVIDUAL_TABLE_HEADERS)
    team_headers_set  = set(TEAM_TABLE_HEADERS)

    for table_index, table in enumerate(tables, start=1):
        # Collect all classes in this table to decide type
        cell_classes = set()
        for cell in table.find_all(['td', 'th']):
            cls = cell.get('class', [])
            if isinstance(cls, str):
                cls = cls.split()
            for c in cls:
                cell_classes.add(c.strip())

        indiv_hits = indiv_headers_set.intersection(cell_classes)
        team_hits  = team_headers_set.intersection(cell_classes)

        if len(indiv_hits) >= 3 and len(indiv_hits) >= len(team_hits):
            table_type = "individual"
        elif len(team_hits) >= 3:
            table_type = "team"
        else:
            metadata.append({
                "race_id": race_id,
                "url": url,
                "table_index": table_index,
                "table_type": "unknown_headers",
                "row_count": 0
            })
            continue

        tbody = table.find("tbody")
        if tbody:
            rows = tbody.find_all("tr")
        else:
            rows = table.find_all("tr")

        added = 0
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue

            row_data = {
                "race_id": race_id,
                "race_url": url
            }

            for cell in cells:
                cls_list = cell.get("class", [])
                if isinstance(cls_list, str):
                    cls_list = cls_list.split()

                text_val = cell.get_text(" ", strip=True)

                for cls in cls_list:
                    cls = cls.strip()
                    if table_type == "individual" and cls in indiv_headers_set:
                        row_data[cls] = text_val
                        link = cell.find("a")
                        if link and link.get("href"):
                            row_data[f"{cls}_url"] = link.get("href")
                    elif table_type == "team" and cls in team_headers_set:
                        row_data[cls] = text_val
                        link = cell.find("a")
                        if link and link.get("href"):
                            row_data[f"{cls}_url"] = link.get("href")

            if table_type == "individual":
                place_str = str(row_data.get("place", "")).strip()
                if not place_str or not re.match(r"^\d+$", place_str):
                    continue
                if "athlete" not in row_data or "finish" not in row_data:
                    continue
                all_data["individual"].append(row_data)
                added += 1
            else:
                place_str = str(row_data.get("place", "")).strip()
                if not place_str or not re.match(r"^\d+$", place_str):
                    continue
                all_data["team"].append(row_data)
                added += 1

        metadata.append({
            "race_id": race_id,
            "url": url,
            "table_index": table_index,
            "table_type": table_type,
            "row_count": added
        })

    metadata_df = pd.DataFrame(metadata)
    indiv_df    = pd.DataFrame(all_data["individual"])
    team_df     = pd.DataFrame(all_data["team"])

    return {"individual": indiv_df, "team": team_df}, metadata_df


# ============================================================
# WRAPPED PARSER (detectors + wranglers + fallback)
# ============================================================

def extract_table_data_wrapped(page_content: str, url: str):
    race_id = extract_race_id(url)

    cole_score  = detect_cole(page_content)
    katie_score = detect_katie(page_content)
    max_score   = detect_max(page_content)
    adam_score  = detect_adam(page_content)

    scores = {
        "cole": cole_score,
        "katie": katie_score,
        "max": max_score,
        "adam": adam_score
    }

    best  = max(scores, key=scores.get)
    score = scores[best]

    print(f"   Detector scores: {scores}, best = {best} ({score:.2f})")

    try:
        if best == "cole" and score >= 0.70:
            print("   [OUR PARSER] Using COLE pre-parser")
            indiv_df = wrangle_cole(page_content, url)
            team_df  = pd.DataFrame(columns=TEAM_TABLE_HEADERS)
        elif best == "max" and score >= 0.70:
            print("   [OUR PARSER] Using MAX pre-parser")
            indiv_df, team_df = wrangle_max(page_content, url)
        elif best == "adam" and score >= 0.70:
            print("   [OUR PARSER] Using ADAM table parser (via robust fallback)")
            # Adam's wrangler is stub; rely on robust table parser
            data, meta = extract_table_data(page_content, url)
            meta["assigned_parser"] = "adam"
            return data, meta
        else:
            # Katie (or uncertain) -> robust table parser
            print("   [FALLBACK] Using robust table parser (Katie-style)")
            data, meta = extract_table_data(page_content, url)
            meta["assigned_parser"] = "katie_fallback"
            return data, meta

        meta = pd.DataFrame([{
            "race_id": race_id,
            "url": url,
            "assigned_parser": best,
            "table_index": None,
            "table_type": best,
            "row_count": len(indiv_df) + len(team_df),
            "detector_score": score
        }])

        return {"individual": indiv_df, "team": team_df}, meta

    except Exception as e:
        print(f"   ⚠ OUR WRANGLER ERROR ({best}) → falling back to robust table parser. Error: {e}")
        data, meta = extract_table_data(page_content, url)
        meta["assigned_parser"] = "katie_fallback_error"
        return data, meta


# ============================================================
# PROCESS URLS
# ============================================================

import platform
import os

def get_chrome_path():
    system = platform.system()

    if system == "Windows":
        return r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    elif system == "Darwin":  # macOS
        return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    elif system == "Linux":
        return "/usr/bin/google-chrome"

    # fallback: no custom executable path
    return None

def process_urls_and_save_wrapped(urls):
    individual_results = pd.DataFrame()
    team_results       = pd.DataFrame()
    metadata_results   = pd.DataFrame()

    with sync_playwright() as p:
        chrome_path = get_chrome_path()

        if chrome_path and os.path.exists(chrome_path):
            browser = p.chromium.launch(
                headless=True,
                executable_path=chrome_path
            )
        else:
            # Fallback to Playwright's bundled Chromium
            browser = p.chromium.launch(headless=True)
              
        
        page = browser.new_page()

        for url in urls:
            race_id = extract_race_id(url)
            print(f"\n🔍 Processing URL: {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=120000)
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_timeout(3000)

                try:
                    page.wait_for_selector("table", timeout=15000)
                except:
                    print("   ⚠ No table found after 15 seconds — continuing")

                html_content = page.content()

                data, metadata = extract_table_data_wrapped(html_content, url)

                if not data["individual"].empty:
                    individual_results = pd.concat(
                        [individual_results, data["individual"]],
                        ignore_index=True
                    )

                if not data["team"].empty:
                    team_results = pd.concat(
                        [team_results, data["team"]],
                        ignore_index=True
                    )

                if metadata is not None and not metadata.empty:
                    metadata_results = pd.concat(
                        [metadata_results, metadata],
                        ignore_index=True
                    )

            except Exception as e:
                print(f"   ERROR processing URL {url}: {e}")
                error_meta = pd.DataFrame([{
                    "race_id": race_id,
                    "url": url,
                    "assigned_parser": "error",
                    "table_index": '',
                    "table_type": f'error - {e}',
                    "row_count": 0,
                    "detector_score": None
                }])
                metadata_results = pd.concat(
                    [metadata_results, error_meta],
                    ignore_index=True
                )

        browser.close()

    return individual_results, team_results, metadata_results


def test_format_detection():
    """
    Test the improved detectors on the 3 known format examples.
    """
    test_cases = [
        {
            "url": "https://ca.milesplit.com/meets/494231-cvl-meet-3-ace-2022/results/846020/raw",
            "expected": "cole",
            "description": "Cole format - PRE with numeric grades (6, 7, 8)"
        },
        {
            "url": "https://ca.milesplit.com/meets/44115-aragons-center-meet-3-2008/results/80586/raw",
            "expected": "max",
            "description": "Max format - PRE with FR/SO/JR/SR grades"
        },
        {
            "url": "https://ca.milesplit.com/meets/493916-cvaa-preview-2022/results/846055/raw",
            "expected": "adam",
            "description": "Adam format - Simple HTML tables"
        },
    ]
    
    print("=" * 80)
    print("TESTING IMPROVED DETECTORS ON KNOWN FORMATS")
    print("=" * 80)
    print()
    
    results = []
    
    with sync_playwright() as p:
        print("🌐 Launching browser...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n{'=' * 80}")
            print(f"TEST {i}/3: {test['description']}")
            print(f"URL: {test['url']}")
            print(f"Expected format: {test['expected'].upper()}")
            print(f"{'=' * 80}")
            
            try:
                # Fetch the page
                print("📥 Fetching page...")
                page.goto(test['url'], wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(2000)
                html = page.content()
                
                # Run all detectors
                print("🔍 Running detectors...")
                scores = {
                    "cole": detect_cole(html),
                    "max": detect_max(html),
                    "adam": detect_adam(html),
                    "katie": detect_katie(html)
                }
                
                # Determine winner
                best_format = max(scores, key=scores.get)
                best_score = scores[best_format]
                
                # Display results
                print("\n📊 DETECTOR SCORES:")
                for format_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    indicator = "👉" if format_name == best_format else "  "
                    star = "⭐" if format_name == test['expected'] else "  "
                    print(f"  {indicator} {star} {format_name.ljust(8)}: {score:.3f}")
                
                print(f"\n🎯 Best match: {best_format.upper()} (score: {best_score:.3f})")
                print(f"🎯 Threshold check: {'PASS' if best_score >= 0.70 else '❌ FAIL'} (>= 0.70)")
                
                # Check if correct
                is_correct = best_format == test['expected']
                results.append({
                    'test': test['description'],
                    'expected': test['expected'],
                    'detected': best_format,
                    'score': best_score,
                    'correct': is_correct
                })
                
                if is_correct:
                    print(f"\n SUCCESS: Correctly identified as {test['expected'].upper()} format!")
                else:
                    print(f"\nFAILURE: Expected {test['expected'].upper()} but got {best_format.upper()}")
                
            except Exception as e:
                print(f"\n  ERROR: {e}")
                results.append({
                    'test': test['description'],
                    'expected': test['expected'],
                    'detected': 'ERROR',
                    'score': 0.0,
                    'correct': False
                })
        
        browser.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    
    print(f"\n Correct detections: {correct_count}/{total_count}")
    print(f" Success rate: {(correct_count/total_count)*100:.1f}%")
    
    print("\n📋 Detailed Results:")
    print("-" * 80)
    for r in results:
        status = "✅" if r['correct'] else "❌"
        print(f"{status} {r['test']}")
        print(f"   Expected: {r['expected'].upper()} | Detected: {r['detected'].upper()} | Score: {r['score']:.3f}")
    
    print("\n" + "=" * 80)
    
    if correct_count == total_count:
        print("ALL TESTS PASSED! The improved detectors are working correctly!")
    else:
        print("Some tests failed. Review the scores above for details.")
    
    print("=" * 80)
    print()


if __name__ == "__main__":
    test_format_detection()


if __name__ == "__main__":
    test_format_detection()

# ============================================================
# FULL RUN MODE - Process all URLs from CSV
# ============================================================
'''''
if __name__ == "__main__":
    input_csv = "race_urls_2016.0.csv"
    
    print("\n==============================")
    print("  FULL RUN: ALL URLs from 2016")
    print("==============================\n")
    
    # Load the CSV
    df = pd.read_csv(input_csv)
    urls_all = df["race_url"].tolist()
    
    print(f"Processing {len(urls_all)} URLs...")
    
    # Process all URLs
    individual_all, team_all, metadata_all = process_urls_and_save_wrapped(urls_all)
    
    # Save results
    output_dir = "output/full_run_2016"
    os.makedirs(output_dir, exist_ok=True)
    
    individual_all.to_csv(os.path.join(output_dir, "individual.csv"), index=False)
    team_all.to_csv(os.path.join(output_dir, "team.csv"), index=False)
    metadata_all.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    
    print(f"\n✓ FULL RUN COMPLETE")
    print(f"   Results saved to: {output_dir}")
    print(f"   Individual results: {len(individual_all)} rows")
    print(f"   Team results: {len(team_all)} rows")
'''

# ============================================================
# DIAGNOSTIC MODE — SAMPLE SUBSET OF URLS
# ============================================================

if __name__ == "__main__":
    input_csv = r"race_urls_2016.0.csv"

    df   = pd.read_csv(input_csv)
    # adjust n as you like; 80 is a decent compromise
    urls = df["race_url"].sample(n=50, random_state=42).tolist()

    print("\n==============================")
    print("  DIAGNOSTIC MODE: 80 URLs")
    print("==============================\n")

    individual, team, metadata = process_urls_and_save_wrapped(urls)

    # Ensure row_count numeric
    if "row_count" in metadata.columns:
        metadata["row_count"] = pd.to_numeric(metadata["row_count"], errors="coerce").fillna(0)
    else:
        metadata["row_count"] = 0

    print("\n=== PARSER FAILURE SUMMARY (SAMPLED 80 URLS) ===")
    if "assigned_parser" not in metadata.columns:
        metadata["assigned_parser"] = "unknown"

    summary = (
        metadata.groupby("assigned_parser")["row_count"]
        .agg(["count", lambda x: (x == 0).sum()])
        .rename(columns={"count": "urls_assigned", "<lambda_0>": "urls_with_zero_rows"})
    )
    summary["failure_rate"] = summary["urls_with_zero_rows"] / summary["urls_assigned"]
    print(summary)

    # Write diagnostic outputs
    output_dir = r"output/diagnostic"
    os.makedirs(output_dir, exist_ok=True)
    individual.to_csv(os.path.join(output_dir, "diag_individual.csv"), index=False)
    team.to_csv(os.path.join(output_dir, "diag_team.csv"), index=False)
    metadata.to_csv(os.path.join(output_dir, "diag_metadata.csv"), index=False)

    print("\nDiagnostic complete. Files saved in 'output/diagnostic'.\n")

    # ========================================================
    # FULL RUN MODE (COMMENTED OUT FOR NOW)
    # ========================================================
    # If you want to run ALL URLs, comment out the block above
    # and uncomment this block:
    #
    # print("\n==============================")
    # print("  FULL RUN: ALL URLs")
    # print("==============================\n")
    #
    # urls_all = df["race_url"].tolist()
    # individual_all, team_all, metadata_all = process_urls_and_save_wrapped(urls_all)
    #
    # full_output_dir = r"C:\Users\coleg\OneDrive\Documents\Econ Research Lab\Kurtis-Econ-Research-Lab-Fall-2025\output\full_run"
    # os.makedirs(full_output_dir, exist_ok=True)
    # individual_all.to_csv(os.path.join(full_output_dir, "individual.csv"), index=False)
    # team_all.to_csv(os.path.join(full_output_dir, "team.csv"), index=False)
    # metadata_all.to_csv(os.path.join(full_output_dir, "metadata.csv"), index=False)
    #
    # print("\n✓ FULL RUN COMPLETE\n")
