from bs4 import BeautifulSoup, Tag
from typing import Optional, Iterable, Set

# --- Configuration ---
REQUIRED_HEADERS_ADAM = {"place", "athlete", "grade", "school", "time"}

# --- Weights ---
W_STRONG = 0.65
W_HEADERS = 0.3
W_TABLECOUNT = 0.05
W_STRUCTURE = 0.20


def _normalize_tokens(tokens: Iterable[str]) -> Set[str]:
    return {t.strip().lower() for t in tokens if t and t.strip()}


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


def _find_meetresults_tables(soup: BeautifulSoup) -> list[Tag]:
    container = soup.find(id="meetResultsBody") or soup.find("div", id="meetResultsBody")
    if not container:
        return []
    return container.find_all("table")


# --- Main detector ---
def detect_adam(html: str) -> float:
    soup = BeautifulSoup(html, "html.parser")
    score = 0.0

    if has_milesplit_results_header_structure(soup):
        score += W_STRUCTURE

    tables = _find_meetresults_tables(soup)
    if not tables:
        return score  # no tables → return whatever structural score we have

    # Strong match for meetResultsBody structure
    score += W_STRONG

    # Header match (use best-scoring table)
    best_header_score = 0.0
    for tbl in tables:
        headers = _normalize_tokens(
            th.get_text(" ", strip=True)
            for th in tbl.find_all(["th", "td"])
        )
        overlap = len(REQUIRED_HEADERS_ADAM.intersection(headers))
        if overlap:
            best_header_score = max(best_header_score, overlap / len(REQUIRED_HEADERS_ADAM))

    score += W_HEADERS * best_header_score

    if len(tables) >= 2:
        score += W_TABLECOUNT

    return min(score, 1.0)
