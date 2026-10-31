"""Watch the data sources: is the shot table back, did FAIR-MAST add shots or folders, did DEFUSE become public?

probe() runs wherever STFC is reachable: on this server directly, or on GitHub's machine through the relay.
Only listings and tiny byte ranges are requested, never whole files.
"""
import re

import requests

from src import config

BUCKET_URL = "https://s3.echo.stfc.ac.uk/mast"
DEFUSE_SAMPLE = BUCKET_URL + "/level2/defuse/MASTno26008.h5"  # one listed file; 403 while DEFUSE is private
TIMEOUT_S = 30


def _list(prefix, token=""):
    """One page of an S3 listing under prefix: (folder names, next-page token or '' on the last page).

    The server decides the page size (about 700 folders here), so pages are followed by token, not by count.
    """
    params = {"list-type": 2, "prefix": prefix, "delimiter": "/", "max-keys": 1000}
    if token:
        params["continuation-token"] = token
    text = requests.get(BUCKET_URL, params=params, timeout=TIMEOUT_S).text
    folders = re.findall(r"<Prefix>([^<]+)</Prefix>", text)[1:]  # the first <Prefix> echoes the request
    truncated = "<IsTruncated>true</IsTruncated>" in text
    found = re.findall(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
    return folders, (found[0] if truncated and found else "")


def count_shot_stores():
    """Number of shot folders under level2/shots/ (one per shot), following every listing page."""
    total, token = 0, ""
    while True:
        folders, token = _list("level2/shots/", token)
        total += len(folders)
        if not token:
            return total


def status_code(url, byte_range=False):
    """HTTP status of a GET (only the first 16 bytes when byte_range), or 0 if the server does not answer."""
    headers = {"Range": "bytes=0-15"} if byte_range else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT_S) as response:
            return response.status_code
    except requests.RequestException:
        return 0


def probe():
    """Snapshot of what the data sources offer right now (a small dict, safe to store and compare)."""
    level2_folders, _ = _list("level2/")
    # mastapp.site is retired (UKAEA, 2026-10-27): the table in the bucket is the official one, so only it is probed
    return {"table_status": status_code(config.SHOT_TABLE_URL, byte_range=True),
            "defuse_status": status_code(DEFUSE_SAMPLE, byte_range=True),
            "level2_folders": sorted(level2_folders),
            "shot_stores": count_shot_stores()}


def table_available(snapshot):
    """True if the full shot table in the FAIR-MAST bucket answered. Snapshots from before 2026-10-27 call it
    shot_table_mirror_status."""
    return snapshot.get("table_status", snapshot.get("shot_table_mirror_status")) in (200, 206)


def changes(old, new):
    """Plain-language list of what changed between two probes (empty if nothing worth telling)."""
    if not old:
        return []
    news = []
    if table_available(new) and not table_available(old):
        news.append("the shot table in the FAIR-MAST bucket answers again: the full dataset can be used")
    if not table_available(new) and table_available(old):
        news.append(f"the shot table in the FAIR-MAST bucket stopped answering (HTTP {new.get('table_status')})")
    if new.get("defuse_status") in (200, 206) and old.get("defuse_status") not in (200, 206):
        news.append("the DEFUSE event files are now public: a new label source to inspect (ROADMAP A5)")
    added = sorted(set(new.get("level2_folders", [])) - set(old.get("level2_folders", [])))
    if added:
        news.append(f"new data folders on FAIR-MAST: {', '.join(added)}")
    if new.get("shot_stores", 0) > old.get("shot_stores", 0):
        news.append(f"{new['shot_stores'] - old['shot_stores']} new shot stores on FAIR-MAST "
                    f"({new['shot_stores']} in total)")
    return news
