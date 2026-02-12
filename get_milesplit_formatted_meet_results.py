import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import os
import platform
from typing import Optional, Tuple, Dict, Any, List

# ============================================================
# CONSTANTS
# ============================================================

INDIVIDUAL_TABLE_HEADERS = ["place", "video", "athlete", "grade", "team", "finish", "point"]
TEAM_TABLE_HEADERS = ["place", "tsTeam", "point", "wind", "heat"]

# [OPTIMIZATION] avoid re-compiling regex over and over
RE_TIME_TOKEN = re.compile(r"\b\d+:\d+(?:\.\d+)?\b")
RE_PLACE_INT = re.compile(r"^\d+$")
RE_EXTRACT_RACE_ID = re.compile(r"/results/(\d+)(?:/|$)")

# [OPTIMIZATION] detector threshold as a constant (easy to tune)
DETECTOR_THRESHOLD = 0.70


# ============================================================
# UTILS
# ============================================================

def extract_race_id(url: str) -> Optional[int]:
    if not url:
        return None
    m = RE_EXTRACT_RACE_ID.search(url)
    return int(m.group(1)) if m else None


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# [OPTIMIZATION] parse HTML ONCE per URL and reuse everywhere
def _build_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html or "", "html.parser")


# [OPTIMIZATION] cache commonly-used nodes/text for detectors/wranglers
def _get_meet_container(soup: BeautifulSoup):
    return soup.find("div", id="meetResultsBody") or soup.find("div", class_="meetResultsBody")


def _get_pre_text(container) -> Optional[str]:
    if not container:
        return None
    pre = container.find("pre")
    if not pre:
        return None
    return pre.get_text("\n", strip=True)


# ============================================================
# DETECTORS
# ============================================================

def detect_cole(soup: BeautifulSoup, pre_text: Optional[str] = None) -> float:
    """
    Cole pages: PRE text with numeric grades (9–12).
    Supports both:
      1 Name Grade Team Time
      1. Grade Name Time Team
    """

    container = _get_meet_container(soup)
    if pre_text is None:
        pre_text = _get_pre_text(container)
    if not pre_text:
        return 0.0

    text = pre_text
    score = 0.0

    # --- strong signals ---

    # time tokens
    if RE_TIME_TOKEN.search(text):
        score += 0.30

    # numeric grades
    numeric_grades = re.findall(r"\b(9|10|11|12)\b", text)
    if len(numeric_grades) >= 5:
        score += 0.35
    elif len(numeric_grades) >= 2:
        score += 0.20

    # place markers (with optional period)
    place_hits = re.findall(r"^\s*\d+\.?\s", text, re.MULTILINE)
    if len(place_hits) >= 5:
        score += 0.20
    elif len(place_hits) >= 2:
        score += 0.10

    if "mile run" in text.lower():
        score += 0.05

    if "team scores" in text.lower():
        score += 0.05

    # --- penalties ---

    if re.search(r"\b(FR|SO|JR|SR)\b", text):
        score -= 0.15

    return max(0.0, min(1.0, score))


def detect_max(soup: BeautifulSoup, pre_text: Optional[str] = None) -> float:
    """
    Max pages: PRE text with FR/SO/JR/SR.
    """
    container = _get_meet_container(soup)
    if pre_text is None:
        pre_text = _get_pre_text(container)
    if not pre_text:
        return 0.0

    text = pre_text
    score = 0.0

    if RE_TIME_TOKEN.search(text):
        score += 0.25

    if re.search(r"\b(FR|SO|JR|SR)\b", text):
        score += 0.45

    # Penalize numeric grades (more Cole-like)
    if re.search(r"\b(9|10|11|12)\b", text):
        score -= 0.15

    if "Team Scores" in text:
        score += 0.10

    return max(0.0, min(1.0, score))


def has_milesplit_results_header_structure(soup: BeautifulSoup) -> bool:
    article = soup.find("article")
    if not article:
        return False
    header = article.find("header")
    if not header:
        return False
    form = header.find("form", id="frmMeetResultsDetailFilter")
    if not form:
        return False
    select = form.find("select", id="ddResultsPage")
    if not select:
        return False
    if not select.find_all("option"):
        return False
    return True


def detect_adam(soup: BeautifulSoup) -> float:
    """
    Adam pages: structured HTML results pages with the Milesplit header filter form and tables.
    """
    score = 0.0
    if has_milesplit_results_header_structure(soup):
        score += 0.55

    container = _get_meet_container(soup)
    if container and container.find_all("table"):
        score += 0.35

    # Light header overlap (table classnames), if present
    if container:
        classes = set()
        for cell in container.find_all(["td", "th"]):
            cls = cell.get("class", [])
            if isinstance(cls, str):
                cls = cls.split()
            for c in cls:
                classes.add(c.strip())

        target = {"place", "athlete", "grade", "school", "time", "finish", "team"}
        if len(target.intersection(classes)) >= 2:
            score += 0.10

    return max(0.0, min(1.0, score))


def detect_katie(soup: BeautifulSoup) -> float:
    """
    Katie-ish: HTML tables with common cell class names.
    """
    score = 0.0
    tables = soup.find_all("table")
    if not tables:
        return 0.0

    score += 0.25

    classes = set()
    link_count = 0
    cell_count = 0

    for t in tables:
        for cell in t.find_all(["td", "th"]):
            cell_count += 1
            if cell.find("a"):
                link_count += 1

            cls = cell.get("class", [])
            if isinstance(cls, str):
                cls = cls.split()
            for c in cls:
                classes.add(c.strip())

    # header-ish classnames
    likely = {"place", "athlete", "grade", "team", "finish", "time", "school", "tsTeam", "point"}
    hit = len(likely.intersection(classes))
    score += min(0.55, 0.08 * hit)

    # link density is common on Milesplit results
    if cell_count > 0 and (link_count / cell_count) > 0.10:
        score += 0.10

    return max(0.0, min(1.0, score))


# ============================================================
# WRANGLERS
# ============================================================

def wrangle_cole(
    html: str,
    race_url: str = None,
    soup: Optional[BeautifulSoup] = None,
    pre_text: Optional[str] = None
):

    if soup is None:
        soup = _build_soup(html)

    container = _get_meet_container(soup)

    if pre_text is None:
        pre_text = _get_pre_text(container)

    if not pre_text:
        return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)

    text = _normalize_whitespace(pre_text).replace("  ", " ")

    # keep user's section split (non-breaking even if header differs)
    sections = re.split(
        r"(?=\b[A-Z][A-Za-z/ &-]+ (?:Boys|Girls)\b)",
        text
    )

    rows = []

    # ------------------------------------------------------------
    # PATTERN A — original cole layout
    # ------------------------------------------------------------
    pattern_a = re.compile(
        r"^(\d+)\s+"
        r"([A-Za-z'\-. ]+?)\s+"
        r"(9|10|11|12)\s+"
        r"([A-Za-z'\-. ]+?)\s+"
        r".*?"
        r"(\d+:\d+(?:\.\d+)?)"
        r"\s*(\d+)?$"
    )

    # ------------------------------------------------------------
    # PATTERN B — place-grade-name-time-team
    # ------------------------------------------------------------
    pattern_b = re.compile(
        r"^(\d+)\.?\s+"              # place (optional period)
        r"(9|10|11|12)\s+"          # grade
        r"([A-Za-z'\-. ]+?)\s+"     # athlete
        r"(\d+:\d+(?:\.\d+)?)"      # time
        r"(?:\s+(?:PR|SR|NR|DNF|DNS|DQ))?\s+"  # optional tags
        r"([A-Za-z'\-. ]+)$"        # team
    )

    for section in sections:
        section = section.strip()
        if not section:
            continue

        for raw_line in section.splitlines():

            line = _normalize_whitespace(raw_line)

            # must start with place marker
            if not re.match(r"^\d+\.?\s", line):
                continue

            # -------------------------
            # try pattern A
            # -------------------------
            m = pattern_a.match(line)

            if m:
                place, athlete, grade, team, finish, point = m.groups()

                rows.append({
                    "place": int(place),
                    "video": None,
                    "athlete": athlete.strip(),
                    "grade": grade,
                    "team": team.strip(),
                    "finish": finish,
                    "point": point if point else pd.NA,
                })

                continue

            # -------------------------
            # try pattern B fallback
            # -------------------------
            m = pattern_b.match(line)

            if m:
                place, grade, athlete, finish, team = m.groups()

                rows.append({
                    "place": int(place),
                    "video": None,
                    "athlete": athlete.strip(),
                    "grade": grade,
                    "team": team.strip(),
                    "finish": finish,
                    "point": pd.NA,
                })

                continue

    if not rows:
        return pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)

    return pd.DataFrame(rows, columns=INDIVIDUAL_TABLE_HEADERS)


def wrangle_max(html: str, race_url: str = None, soup: Optional[BeautifulSoup] = None, pre_text: Optional[str] = None):
    """
    PRE parser for Max-style pages with FR/SO/JR/SR grades.
    We keep your earlier pattern but with a bit of whitespace normalization.
    """
    if soup is None:
        soup = _build_soup(html)
    container = _get_meet_container(soup)
    if pre_text is None:
        pre_text = _get_pre_text(container)
    if not pre_text:
        return (
            pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS),
            pd.DataFrame(columns=TEAM_TABLE_HEADERS),
        )

    text = _normalize_whitespace(pre_text)
    sections = re.split(r"(?=\b[A-Z][A-Za-z/ &-]+ (?:Boys|Girls)\b)", text)

    rows = []

    line_pattern = re.compile(
        r"^(\d+)\s+([A-Za-z'\-. ]+?)\s+(FR|SO|JR|SR)\s+"
        r"([A-Za-z'\-. ]+?)\s+\d*:?[\d.]*\s+(\d+:\d+(?:\.\d+)?)\s+(\d+)?$"
    )

    for section in sections:
        section = section.strip()
        if not section:
            continue
        for raw_line in section.splitlines():
            line = _normalize_whitespace(raw_line)
            if not re.match(r"^\d+\s", line):
                continue
            m = line_pattern.match(line)
            if not m:
                continue

            place, athlete, grade, team, finish, point = m.groups()
            rows.append(
                {
                    "place": int(place),
                    "video": None,
                    "athlete": athlete.strip(),
                    "grade": grade,
                    "team": team.strip(),
                    "finish": finish,
                    "point": point if point else pd.NA,
                }
            )

    indiv_df = pd.DataFrame(rows, columns=INDIVIDUAL_TABLE_HEADERS)
    return indiv_df, pd.DataFrame(columns=TEAM_TABLE_HEADERS)


def wrangle_adam(html: str, race_url: str = None, soup: BeautifulSoup = None):
    """
    adam wrangler: parse "flattened" results tables where events are represented as
    single-cell header rows (race_name) followed by standard result rows.

    returns: (indiv_df, team_df)
      - indiv_df columns:
        race_id, race_url, race_name, place, video, athlete, athlete_url,
        grade, team, team_url, finish, point
      - team_df: empty (placeholder)
    """
    if race_url is None:
        race_url = ""

    # accept either soup or raw html
    if soup is None:
        soup = _build_soup(html)

    race_id = extract_race_id(race_url)

    # try to focus on meetResultsBody if it exists
    container = soup.find("div", id="meetResultsBody") or soup.find("div", class_="meetResultsBody") or soup

    # adam-style pages sometimes have multiple tables; parse all and concatenate
    all_rows = []

    for table in container.find_all("table"):
        race_name = None

        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue

            # single-cell row => event header
            if len(tds) == 1:
                header = tds[0].get_text(strip=True)
                if header:
                    race_name = header
                continue

            # skip column headers
            if tds and tds[0].get_text(strip=True).lower() == "place":
                continue

            # need an event header to associate rows
            if not race_name:
                continue

            # normal result row: place | athlete | grade | school/team | time/finish | (optional point)
            if len(tds) >= 5:
                place = tds[0].get_text(strip=True)
                athlete = tds[1].get_text(strip=True)
                grade = tds[2].get_text(strip=True)
                team = tds[3].get_text(strip=True)
                finish = tds[4].get_text(strip=True)

                # optional points column (sometimes exists)
                point = tds[5].get_text(strip=True) if len(tds) >= 6 else None
                point = point if point not in ("", None) else None

                # optional hyperlinks for athlete/team
                a_tag = tds[1].find("a")
                t_tag = tds[3].find("a")
                athlete_url = a_tag.get("href", "").strip() if a_tag and a_tag.has_attr("href") else None
                team_url = t_tag.get("href", "").strip() if t_tag and t_tag.has_attr("href") else None

                all_rows.append(
                    {
                        "race_id": race_id,
                        "race_url": race_url,
                        "race_name": race_name,
                        "place": place,
                        "video": None,
                        "athlete": athlete,
                        "athlete_url": athlete_url,
                        "grade": grade,
                        "team": team,
                        "team_url": team_url,
                        "finish": finish,
                        "point": point,
                    }
                )

    indiv_df = pd.DataFrame(
        all_rows,
        columns=[
            "race_id",
            "race_url",
            "race_name",
            "place",
            "video",
            "athlete",
            "athlete_url",
            "grade",
            "team",
            "team_url",
            "finish",
            "point",
        ],
    )

    # light cleanup (don't force-grade if it's FR/SO/JR/SR)
    if not indiv_df.empty:
        indiv_df["place"] = pd.to_numeric(indiv_df["place"], errors="coerce")

        # only coerce grade if it looks numeric
        grade_numeric = indiv_df["grade"].astype(str).str.fullmatch(r"\d+")
        indiv_df.loc[grade_numeric, "grade"] = pd.to_numeric(indiv_df.loc[grade_numeric, "grade"], errors="coerce")

        # points if present
        if "point" in indiv_df.columns:
            indiv_df["point"] = pd.to_numeric(indiv_df["point"], errors="coerce")

    team_df = pd.DataFrame(columns=TEAM_TABLE_HEADERS)
    return indiv_df, team_df



def wrangle_katie(html: str, race_url: str = None):
    """
    Unused directly; robust table parser below plays Katie's role.
    """
    return (
        pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS),
        pd.DataFrame(columns=TEAM_TABLE_HEADERS),
    )


# ============================================================
# ROBUST TABLE PARSER (Katie-style but more tolerant)
# ============================================================

# [OPTIMIZATION] normalize classnames to canonical headers where possible
_CLASS_ALIASES = {
    "time": "finish",
    "school": "team",
}

def extract_table_data(page_content: str, url: str, soup: Optional[BeautifulSoup] = None):
    race_id = extract_race_id(url)

    # [OPTIMIZATION] reuse parsed soup if provided
    if soup is None:
        soup = _build_soup(page_content)

    tables = soup.find_all("table")

    if not tables:
        print(f"   No tables found for URL: {url}")
        empty = {"individual": pd.DataFrame(), "team": pd.DataFrame()}
        meta = pd.DataFrame(
            [
                {
                    "race_id": race_id,
                    "url": url,
                    "table_index": None,
                    "table_type": "no_tables",
                    "row_count": 0,
                }
            ]
        )
        return empty, meta

    all_data = {"individual": [], "team": []}
    metadata = []

    indiv_headers_set = set(INDIVIDUAL_TABLE_HEADERS)
    team_headers_set = set(TEAM_TABLE_HEADERS)

    for table_index, table in enumerate(tables, start=1):
        # Collect all classes in this table to decide type
        cell_classes = set()
        for cell in table.find_all(["td", "th"]):
            cls = cell.get("class", [])
            if isinstance(cls, str):
                cls = cls.split()
            for c in cls:
                # [OPTIMIZATION] alias normalization on classnames
                c = _CLASS_ALIASES.get(c.strip(), c.strip())
                cell_classes.add(c)

        indiv_hits = indiv_headers_set.intersection(cell_classes)
        team_hits = team_headers_set.intersection(cell_classes)

        if len(indiv_hits) >= 3 and len(indiv_hits) >= len(team_hits):
            table_type = "individual"
        elif len(team_hits) >= 3:
            table_type = "team"
        else:
            metadata.append(
                {
                    "race_id": race_id,
                    "url": url,
                    "table_index": table_index,
                    "table_type": "unknown_headers",
                    "row_count": 0,
                }
            )
            continue

        tbody = table.find("tbody")
        rows = tbody.find_all("tr") if tbody else table.find_all("tr")

        added = 0
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue

            row_data = {"race_id": race_id, "race_url": url}

            for cell in cells:
                cls_list = cell.get("class", [])
                if isinstance(cls_list, str):
                    cls_list = cls_list.split()

                text_val = cell.get_text(" ", strip=True)

                link = cell.find("a")
                href = link.get("href") if link and link.get("href") else None

                for cls in cls_list:
                    cls = _CLASS_ALIASES.get(cls.strip(), cls.strip())

                    if table_type == "individual" and cls in indiv_headers_set:
                        row_data[cls] = text_val
                        if href:
                            row_data[f"{cls}_url"] = href

                    elif table_type == "team" and cls in team_headers_set:
                        row_data[cls] = text_val
                        if href:
                            row_data[f"{cls}_url"] = href

            # Basic validation
            place_str = str(row_data.get("place", "")).strip()
            m_place = re.match(r"^(\d+)", place_str)
            if not m_place:
                continue
            row_data["place"] = int(m_place.group(1))

            if table_type == "individual":
                if "athlete" not in row_data or "finish" not in row_data:
                    continue
                all_data["individual"].append(row_data)
            else:
                all_data["team"].append(row_data)

            added += 1

        metadata.append(
            {
                "race_id": race_id,
                "url": url,
                "table_index": table_index,
                "table_type": table_type,
                "row_count": added,
            }
        )

    metadata_df = pd.DataFrame(metadata)
    indiv_df = pd.DataFrame(all_data["individual"])
    team_df = pd.DataFrame(all_data["team"])

    return {"individual": indiv_df, "team": team_df}, metadata_df


# ============================================================
# WRAPPED PARSER (detectors + wranglers + fallback)
# ============================================================

def extract_table_data_wrapped(page_content: str, url: str):
    race_id = extract_race_id(url)

    # [OPTIMIZATION] parse once + compute pre_text once
    soup = _build_soup(page_content)
    container = _get_meet_container(soup)
    pre_text = _get_pre_text(container)

    cole_score = detect_cole(soup, pre_text=pre_text)
    katie_score = detect_katie(soup)
    max_score = detect_max(soup, pre_text=pre_text)
    adam_score = detect_adam(soup)

    scores = {"cole": cole_score, "katie": katie_score, "max": max_score, "adam": adam_score}
    best = max(scores, key=scores.get)
    score = scores[best]

    print(f"   Detector scores: {scores}, best = {best} ({score:.2f})")

    try:
        if best == "cole" and score >= DETECTOR_THRESHOLD:
            print("   [OUR PARSER] Using COLE pre-parser")
            indiv_df = wrangle_cole(page_content, url, soup=soup, pre_text=pre_text)
            team_df = pd.DataFrame(columns=TEAM_TABLE_HEADERS)

        elif best == "max" and score >= DETECTOR_THRESHOLD:
            print("   [OUR PARSER] Using MAX pre-parser")
            indiv_df, team_df = wrangle_max(page_content, url, soup=soup, pre_text=pre_text)

        elif best == "adam" and score >= DETECTOR_THRESHOLD:
            print("   [OUR PARSER] Using Adam pre-parser")

            indiv_df, team_df = wrangle_adam(
                page_content,
                race_url=url,
                soup=soup
            )

            if isinstance(indiv_df, pd.DataFrame) and not indiv_df.empty:
                meta = pd.DataFrame(
                    [
                        {
                            "race_id": extract_race_id(url),
                            "url": url,
                            "assigned_parser": "adam",
                            "table_index": None,
                            "table_type": "adam_flat_table",
                            "row_count": len(indiv_df),
                            "detector_score": score,
                        }
                    ]
                )
                return {"individual": indiv_df, "team": team_df}, meta
            data, meta = extract_table_data(page_content, url, soup=soup)
            meta["assigned_parser"] = "adam_fallback"
            meta["detector_score"] = score
            return data, meta


        else:
            # Katie (or uncertain) -> robust table parser
            print("   [FALLBACK] Using robust table parser (Katie-style)")
            data, meta = extract_table_data(page_content, url, soup=soup)
            meta["assigned_parser"] = "katie_fallback"
            meta["detector_score"] = score
            return data, meta

        meta = pd.DataFrame(
            [
                {
                    "race_id": race_id,
                    "url": url,
                    "assigned_parser": best,
                    "table_index": None,
                    "table_type": best,
                    "row_count": len(indiv_df) + len(team_df),
                    "detector_score": score,
                }
            ]
        )

        return {"individual": indiv_df, "team": team_df}, meta

    except Exception as e:
        print(f"   ⚠ OUR WRANGLER ERROR ({best}) → falling back to robust table parser. Error: {e}")
        data, meta = extract_table_data(page_content, url, soup=soup)
        meta["assigned_parser"] = "katie_fallback_error"
        meta["detector_score"] = score
        return data, meta


# ============================================================
# PROCESS URLS
# ============================================================

def get_chrome_path():
    system = platform.system()

    if system == "Windows":
        return r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    elif system == "Darwin":  # macOS
        return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    elif system == "Linux":
        return "/usr/bin/google-chrome"

    return None


def process_urls_and_save_wrapped(urls):
    # [OPTIMIZATION] avoid repeated pd.concat in the loop (quadratic behavior)
    indiv_chunks: List[pd.DataFrame] = []
    team_chunks: List[pd.DataFrame] = []
    meta_chunks: List[pd.DataFrame] = []

    with sync_playwright() as p:
        chrome_path = get_chrome_path()

        # Keep your "try system Chrome else bundled" behavior
        if chrome_path and os.path.exists(chrome_path):
            browser = p.chromium.launch(headless=True, executable_path=chrome_path)
        else:
            browser = p.chromium.launch(headless=True)

        context = browser.new_context()
        page = context.new_page()

        for i, url in enumerate(urls, start=1):
            print(f"\n[{i}/{len(urls)}] Processing: {url}")

            try:
                # [OPTIMIZATION] remove fixed sleep; use a quick targeted wait
                page.goto(url, wait_until="domcontentloaded", timeout=120000)

                # [OPTIMIZATION] only wait briefly for either a table OR a pre block
                # (many pages are already server-rendered; hard waits slow you down massively)
                try:
                    page.wait_for_selector("#meetResultsBody table, #meetResultsBody pre, table, pre", timeout=6000)
                except Exception:
                    pass

                html_content = page.content()
                data, meta = extract_table_data_wrapped(html_content, url)

                indiv = data.get("individual", pd.DataFrame())
                team = data.get("team", pd.DataFrame())

                if isinstance(indiv, pd.DataFrame) and not indiv.empty:
                    indiv_chunks.append(indiv)
                if isinstance(team, pd.DataFrame) and not team.empty:
                    team_chunks.append(team)
                if isinstance(meta, pd.DataFrame) and not meta.empty:
                    meta_chunks.append(meta)

            except Exception as e:
                print(f"   ⚠ Failed URL: {url} | Error: {e}")
                rid = extract_race_id(url)
                meta_chunks.append(
                    pd.DataFrame(
                        [
                            {
                                "race_id": rid,
                                "url": url,
                                "assigned_parser": "page_error",
                                "table_index": None,
                                "table_type": "page_error",
                                "row_count": 0,
                                "detector_score": pd.NA,
                            }
                        ]
                    )
                )

        context.close()
        browser.close()

    # [OPTIMIZATION] concat once at the end
    individual_results = (
        pd.concat(indiv_chunks, ignore_index=True) if indiv_chunks else pd.DataFrame(columns=INDIVIDUAL_TABLE_HEADERS)
    )
    team_results = pd.concat(team_chunks, ignore_index=True) if team_chunks else pd.DataFrame(columns=TEAM_TABLE_HEADERS)
    metadata_results = pd.concat(meta_chunks, ignore_index=True) if meta_chunks else pd.DataFrame()

    return individual_results, team_results, metadata_results


# ============================================================
# DIAGNOSTIC MODE — SAMPLE SUBSET OF URLS
# ============================================================

if __name__ == "__main__":
    input_csv = r"data/wa_hs_xc_meet_urls_2015_2020.csv"

    df = pd.read_csv(input_csv)
    urls = df["race_url"].sample(n=80).tolist()

    print("\n==============================")
    print(f"  DIAGNOSTIC MODE: {len(urls)} urls")
    print("==============================\n")

    individual, team, metadata = process_urls_and_save_wrapped(urls)

    # Ensure row_count numeric
    if "row_count" in metadata.columns:
        metadata["row_count"] = pd.to_numeric(metadata["row_count"], errors="coerce").fillna(0)
    else:
        metadata["row_count"] = 0

    print("\n=== PARSER FAILURE SUMMARY ===")
    if "assigned_parser" not in metadata.columns:
        metadata["assigned_parser"] = "unknown"

    summary = (
        metadata.groupby("assigned_parser")["row_count"]
        .agg(["count", lambda x: (x == 0).sum()])
        .rename(columns={"count": "urls_assigned", "<lambda_0>": "urls_with_zero_rows"})
    )
    summary["failure_rate"] = summary["urls_with_zero_rows"] / summary["urls_assigned"]
    print(summary)

    output_dir = r"output/diagnostic"
    os.makedirs(output_dir, exist_ok=True)
    individual.to_csv(os.path.join(output_dir, "diag_individual.csv"), index=False)
    team.to_csv(os.path.join(output_dir, "diag_team.csv"), index=False)
    metadata.to_csv(os.path.join(output_dir, "diag_metadata.csv"), index=False)

    print("\nDiagnostic complete. Files saved in 'output/diagnostic'.\n")
