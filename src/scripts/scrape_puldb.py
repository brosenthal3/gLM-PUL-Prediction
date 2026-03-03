import html
import re
import time
import urllib.parse
from pathlib import Path

import polars
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from rich.progress import track

BASE_URL = "https://www.cazy.org/PULDB/index.php?pul={}"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PULDB-mapper/1.0)"}

session = requests.Session()
# Retry strategy: retry on connection errors, status codes like 429, 500, 502, 503, 504
retry_strategy = Retry(
    total=5,  # total retries
    backoff_factor=0.5,  # exponential backoff: 0.5, 1, 2, 4... seconds
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)
session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; PULDB-scraper/1.0)"})


def get_url(url, session, min_interval=0.04):
    """
    Performs a GET request to `url` using the robust session.
    Respects a minimum interval between requests (response-aware pacing).
    Returns response or None on failure.
    """
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    time.sleep(min_interval)

    return response


parsed_pul_ids = []
if Path("puldb_data.parquet").is_file():
    df = polars.read_parquet("puldb_data.parquet")
    parsed_pul_ids.extend(df["pul_id"].to_list())

rows = []


def extract_identifier_info(soup):
    """
    Extract:
    - identifier_text
    - accession
    - start_bp
    - end_bp
    """
    rows = soup.find_all("tr")

    identifier_text = None
    accession = None
    start_bp = None
    end_bp = None

    for row in rows:
        cols = row.find_all("td")
        if len(cols) != 2:
            continue

        label = cols[0].get_text(strip=True)

        if "Identifier/JBrowse link" in label:
            value_cell = cols[1]

            # Get visible text (e.g. "Literature-derived PUL 1")
            identifier_text = value_cell.get_text(" ", strip=True)

            # Extract JBrowse link
            link = value_cell.find("a", href=True)
            if link and "loc=" in link["href"]:
                href = html.unescape(link["href"])
                href = urllib.parse.unquote(href)
                match = re.search(r"loc=([^:]+):(\d+)\.\.(\d+)", href)
                if match:
                    accession = match.group(1)
                    start_bp = int(match.group(2))
                    end_bp = int(match.group(3))
            break

    return identifier_text, accession, start_bp, end_bp


MAX_ID = 89000
MAX_EMPTY = 10
empty_count = 0

for pul_id in track(
    range(1, MAX_ID + 1),
    description="Processing PULs...",
    total=MAX_ID,
):
    if pul_id in parsed_pul_ids:
        continue
    url = BASE_URL.format(pul_id)

    r = get_url(url, session)
    if r.status_code != 200 or "PULDB" not in r.text:
        empty_count += 1
        if empty_count >= MAX_EMPTY:
            print("Stopping: too many empty pages.")
            break
        continue
    else:
        empty_count = 0

    soup = BeautifulSoup(r.text, "html.parser")
    identifier_text, accession, start_bp, end_bp = extract_identifier_info(soup)

    if identifier_text is None:
        status = None

    elif "predicted" in identifier_text.lower():
        status = "predicted"

    elif "literature-derived" in identifier_text.lower():
        status = "literature"

    else:
        status = "other"

    rows.append(
        {
            "pul_id": pul_id,
            "status": status,
            "identifier_text": identifier_text,
            "accession": accession,
            "start_bp": start_bp,
            "end_bp": end_bp,
        }
    )

df = polars.DataFrame(rows)
df.write_parquet("puldb_data.parquet")
