"""Weekly literature watch: new arXiv papers on tokamak disruption prediction, sent to Telegram.

Uses arXiv's official API (export.arxiv.org), which is meant for exactly this kind of automated query.
What has been seen is stored privately in MAST_HOME/literature, so each paper is reported once.
"""
import csv
import json
import os
import xml.etree.ElementTree as ET

import requests

from src import config

API = "https://export.arxiv.org/api/query"
QUERY = ('abs:disruption AND (abs:tokamak OR abs:"fusion plasma" OR abs:"spherical tokamak") AND '
         '(abs:prediction OR abs:predictor OR abs:forecasting OR abs:"machine learning" OR abs:"neural network")')
ATOM = {"a": "http://www.w3.org/2005/Atom"}
FOLDER = os.path.join(config.HOME_DIR, "literature")


def parse_feed(xml_text):
    """arXiv Atom XML -> list of dicts with id, title, authors, published, summary, link."""
    papers = []
    for entry in ET.fromstring(xml_text).findall("a:entry", ATOM):
        link = entry.findtext("a:id", default="", namespaces=ATOM).strip()
        papers.append({
            "id": link.rsplit("/", 1)[-1].split("v")[0],  # 2610.06189v1 -> 2610.06189, so new versions are not new papers
            "title": " ".join(entry.findtext("a:title", default="", namespaces=ATOM).split()),
            "authors": [a.findtext("a:name", default="", namespaces=ATOM) for a in entry.findall("a:author", ATOM)],
            "published": entry.findtext("a:published", default="", namespaces=ATOM)[:10],
            "summary": " ".join(entry.findtext("a:summary", default="", namespaces=ATOM).split()),
            "link": link,
        })
    return papers


def search(max_results=config.LITERATURE_MAX_RESULTS):
    """The newest papers matching QUERY, newest first."""
    params = {"search_query": QUERY, "sortBy": "submittedDate", "sortOrder": "descending", "max_results": max_results}
    response = requests.get(API, params=params, timeout=60)
    response.raise_for_status()
    return parse_feed(response.text)


def read_seen(folder=FOLDER):
    """Set of arXiv ids already reported."""
    path = os.path.join(folder, "seen.json")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(json.load(f))


def remember(papers, folder=FOLDER):
    """Mark papers as seen and append them to papers.csv (the private reading list)."""
    os.makedirs(folder, exist_ok=True)
    seen = read_seen(folder) | {p["id"] for p in papers}
    with open(os.path.join(folder, "seen.json"), "w") as f:
        json.dump(sorted(seen), f)
    path = os.path.join(folder, "papers.csv")
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["id", "published", "title", "authors", "link"])
        for p in papers:
            writer.writerow([p["id"], p["published"], p["title"], "; ".join(p["authors"]), p["link"]])


def digest(papers, limit=10):
    """Telegram text for a list of papers: title, first authors, date, link and the start of the abstract."""
    lines = [f"Literature watch: {len(papers)} new paper(s) on disruption prediction"]
    for p in papers[:limit]:
        authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        lines.append(f"\n{p['title']}\n{authors} ({p['published']})\n{p['link']}\n{p['summary'][:280]}...")
    if len(papers) > limit:
        lines.append(f"\n...and {len(papers) - limit} more in literature/papers.csv")
    return "\n".join(lines)


def check():
    """Search, keep only papers not reported before, remember them, and return them (newest first)."""
    papers = search()
    seen = read_seen()
    new = [p for p in papers if p["id"] not in seen]
    if new:
        remember(new)
    return new
