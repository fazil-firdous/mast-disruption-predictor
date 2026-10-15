"""Tests for the data-source watcher (src/sources.py) and the literature watcher (src/literature.py). No network."""
from src import literature, server, sources

FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2610.06189v2</id>
    <title>Eliminating Tokamak
      Disruptions with Feedback</title>
    <published>2026-10-05T12:07:10Z</published>
    <summary>Tokamak disruptions caused by tearing modes can be prevented.</summary>
    <author><name>A. Author</name></author><author><name>B. Author</name></author>
  </entry>
</feed>"""


def test_parse_feed():
    [paper] = literature.parse_feed(FEED)
    assert paper["id"] == "2610.06189"
    assert paper["title"] == "Eliminating Tokamak Disruptions with Feedback"
    assert paper["authors"] == ["A. Author", "B. Author"] and paper["published"] == "2026-10-05"


def test_seen_papers_are_remembered(tmp_path):
    papers = literature.parse_feed(FEED)
    assert literature.read_seen(str(tmp_path)) == set()
    literature.remember(papers, str(tmp_path))
    assert literature.read_seen(str(tmp_path)) == {"2610.06189"}
    assert "Eliminating" in (tmp_path / "papers.csv").read_text()
    assert "1 new paper" in literature.digest(papers)


def test_source_changes():
    old = {"shot_table_status": 0, "defuse_status": 403, "level2_folders": ["level2/shots/"], "shot_stores": 11573}
    assert sources.changes(None, old) == []
    assert sources.changes(old, dict(old)) == []
    new = {"shot_table_status": 206, "defuse_status": 206, "level2_folders": ["level2/mastu/", "level2/shots/"],
           "shot_stores": 11600}
    news = " | ".join(sources.changes(old, new))
    assert "mastapp.site is back" in news and "DEFUSE" in news and "level2/mastu/" in news and "27 new" in news
    assert "went down" in " ".join(sources.changes(new, old))


def test_due():
    assert server.due(0.0, 24, 1e9)
    assert not server.due(1e9 - 3600, 24, 1e9)
