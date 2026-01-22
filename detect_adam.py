from bs4 import BeautifulSoup, Tag
from typing import Iterable, Set
import re

REQUIRED_HEADERS_ADAM = {"place", "athlete", "grade", "school", "time"}

W_STRONG = 0.60
W_HEADERS = 0.30
W_TABLECOUNT = 0.05
W_STRUCTURE = 0.20

SYNONYMS = {
    "mark": "time",
    "result": "time",
    "finish": "time",
    "name": "athlete",
    "competitor": "athlete",
    "team": "school",
    "yr": "grade",
    "year": "grade",
    "pl": "place",
}

TIME_LIKE = re.compile(r"\b\d{1,2}:\d{2}(\.\d+)?\b|\b\d+\.\d+\b")

def _normalize_tokens(tokens: Iterable[str]) -> Set[str]:
    out = set()
    for t in tokens:
        if not t:
            continue
        s = t.strip().lower()
        if not s:
            continue
        s = SYNONYMS.get(s, s)
        out.add(s)
    return out

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
    return bool(select.find_all("option"))

def _find_meetresults_tables(soup: BeautifulSoup) -> list[Tag]:
    container = soup.find(id="meetResultsBody")
    if not container:
        return []
    return container.find_all("table", recursive=True)

def _header_tokens_for_table(tbl: Tag) -> Set[str]:
    # Prefer thead
    thead = tbl.find("thead")
    if thead:
        cells = thead.find_all(["th", "td"])
        return _normalize_tokens(c.get_text(" ", strip=True) for c in cells)

    # Otherwise: examine first few rows, pick the best candidate
    best = set()
    for tr in tbl.find_all("tr", limit=4):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) < 3:
            continue
        toks = _normalize_tokens(c.get_text(" ", strip=True) for c in cells)
        # Candidate row if it contains at least 2 required headers
        if len(toks & REQUIRED_HEADERS_ADAM) >= 2 and len(toks) > len(best):
            best = toks
    return best

def _table_looks_like_results(tbl: Tag) -> bool:
    # Quick heuristic: does it contain any time-like values?
    text = tbl.get_text(" ", strip=True)
    return bool(TIME_LIKE.search(text))

def detect_adam(html: str) -> float:
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    structure_ok = has_milesplit_results_header_structure(soup)
    if structure_ok:
        score += W_STRUCTURE

    tables = _find_meetresults_tables(soup)
    if not tables:
        return score

    # Optional: only consider "results-ish" tables, but fallback if none match
    candidate_tables = [t for t in tables if _table_looks_like_results(t)] or tables

    score += W_STRONG

    best_header_score = 0.0
    for tbl in candidate_tables:
        headers = _header_tokens_for_table(tbl)
        overlap = len(REQUIRED_HEADERS_ADAM & headers)
        if overlap:
            best_header_score = max(best_header_score, overlap / len(REQUIRED_HEADERS_ADAM))
            if best_header_score == 1.0:
                break  # early exit

    score += W_HEADERS * best_header_score

    if len(candidate_tables) >= 2:
        score += W_TABLECOUNT

    score = min(score, 1.0)

    return score