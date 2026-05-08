#!/usr/bin/env python3
"""
scrape_writeups.py — Fetch external security writeups and reports

Sources:
  1. 0xdf.gitlab.io       — ~450 HackTheBox/CTF machine writeups (HTML → Markdown)
  2. PortSwigger Web Sec  — Web Security Academy lab content (free tier)
  3. HackerOne Hacktivity — Public disclosed vulnerability reports (two-phase approach)
  4. unprotect.it         — Anti-analysis / malware evasion technique database
  5. Project Zero blog    — Google Project Zero in-depth vulnerability research
  6. Phrack magazine      — Classic exploitation technique papers (phrack.org)

Usage:
  pip install requests beautifulsoup4 html2text
  python3 scrape_writeups.py [--sources 0xdf portswigger hackerone unprotect projectzero phrack]

Output:
  data/scraped/0xdf_htb/          *.md
  data/scraped/portswigger_labs/  *.md
  data/scraped/hackerone_reports/ *.json
  data/scraped/unprotect/         *.json
  data/scraped/projectzero/       *.md
  data/scraped/phrack/            *.md
"""

import argparse
import base64
import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    from bs4 import BeautifulSoup
    import html2text
    _HAS_DEPS = True
except ImportError:
    print("[!] Missing deps: pip install requests beautifulsoup4 html2text")
    _HAS_DEPS = False

# ─── Shared helpers ────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Security Research Dataset Builder; not a bot)"
})

_H2T = None
if _HAS_DEPS:
    _H2T = html2text.HTML2Text()
    _H2T.ignore_links = False
    _H2T.ignore_images = True
    _H2T.body_width = 0


def _get(url: str, retries: int = 3, delay: float = 2.0) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 10))
                print(f"  Rate limited. Waiting {wait:.0f}s...")
                time.sleep(wait)
            elif r.status_code == 404:
                return None
        except Exception as e:
            print(f"  Request error ({url}): {e}")
        time.sleep(delay * (attempt + 1))
    return None


def _html_to_md(html: str) -> str:
    if not _H2T:
        return html
    return _H2T.handle(html)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — 0xdf HTB writeups
# ══════════════════════════════════════════════════════════════════════════════

FEED_URL = "https://0xdf.gitlab.io/feed.xml"
BASE_URL  = "https://0xdf.gitlab.io"

# HackTheBox and CTF writeup URL pattern
HTB_PATTERN = re.compile(r"https://0xdf\.gitlab\.io/\d{4}/\d{2}/\d{2}/htb-", re.I)
CTF_PATTERN = re.compile(r"https://0xdf\.gitlab\.io/\d{4}/\d{2}/\d{2}/", re.I)


def _parse_0xdf_post(url: str) -> tuple[str, str] | None:
    """Fetch a single 0xdf post and convert to markdown."""
    r = _get(url)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # Title
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]

    # Main content — look for .post-content or article
    content_tag = (
        soup.find("div", class_="post-content")
        or soup.find("article")
        or soup.find("main")
    )
    if not content_tag:
        return None

    # Remove nav/footer elements
    for tag in content_tag.find_all(["nav", "footer", "script", "style", "aside"]):
        tag.decompose()

    md = _html_to_md(str(content_tag))
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    if len(md) < 500:
        return None

    return title, f"# {title}\n\nSource: {url}\n\n{md}"


def scrape_0xdf(output_dir: Path, max_posts: int = 500, htb_only: bool = False) -> int:
    out = output_dir / "0xdf_htb"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[0xdf] Fetching post list from feed...")
    r = _get(FEED_URL)
    if not r:
        print("  [!] Could not fetch feed.")
        return 0

    # Parse RSS feed for post URLs
    urls = re.findall(r"<link>(https://0xdf\.gitlab\.io/[^<]+)</link>", r.text)
    urls = [u for u in urls if "/20" in u and u != BASE_URL + "/"]
    if htb_only:
        urls = [u for u in urls if "htb-" in u.lower()]

    # Also try paginated index if feed is limited
    if len(urls) < 50:
        for page in range(1, 15):
            page_url = f"{BASE_URL}/page{page}/" if page > 1 else f"{BASE_URL}/"
            pr = _get(page_url)
            if not pr:
                break
            page_urls = re.findall(
                r'href="(https://0xdf\.gitlab\.io/\d{4}/\d{2}/\d{2}/[^"]+)"',
                pr.text
            )
            urls.extend(page_urls)
            if not page_urls:
                break
            time.sleep(1)

    urls = list(dict.fromkeys(urls))[:max_posts]  # deduplicate
    print(f"  Found {len(urls)} posts to scrape")

    saved = 0
    for i, url in enumerate(urls):
        slug = re.sub(r"[^\w-]", "-", url.split("/")[-1].strip("/"))
        out_file = out / f"{slug}.md"

        if out_file.exists() and out_file.stat().st_size > 500:
            saved += 1
            continue

        result = _parse_0xdf_post(url)
        if result:
            title, md = result
            out_file.write_text(md, encoding="utf-8")
            saved += 1
            if (i + 1) % 25 == 0:
                print(f"  Saved {saved}/{len(urls)}...")

        time.sleep(1.5)  # polite crawl rate

    print(f"  [0xdf] Done. Saved {saved} writeups → {out}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — PortSwigger Web Security Academy
# ══════════════════════════════════════════════════════════════════════════════

PS_BASE = "https://portswigger.net/web-security"

# Top-level topic slugs from the academy
PS_TOPICS = [
    "sql-injection", "cross-site-scripting", "csrf",
    "clickjacking", "dom-based", "cors", "xxe",
    "ssrf", "request-smuggling", "command-injection",
    "server-side-template-injection", "path-traversal",
    "access-control", "authentication", "business-logic-vulnerabilities",
    "information-disclosure", "file-upload", "race-conditions",
    "oauth", "jwt", "prototype-pollution", "graphql",
    "nosql-injection", "api-testing", "web-cache-poisoning",
    "web-cache-deception", "insecure-deserialization",
]


def _scrape_ps_topic(topic: str, out_dir: Path) -> list[str]:
    """Scrape a PortSwigger topic page and its sub-pages."""
    results = []
    topic_url = f"{PS_BASE}/{topic}"
    r = _get(topic_url)
    if not r:
        return results

    soup = BeautifulSoup(r.text, "html.parser")

    # Find the main content div
    content = (
        soup.find("div", class_="section-wrapper")
        or soup.find("article")
        or soup.find("div", class_="content")
        or soup.find("main")
    )
    if not content:
        return results

    for el in content.find_all(["nav", "aside", "script", "style"]):
        el.decompose()

    md = _html_to_md(str(content))
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    if len(md) > 300:
        out_file = out_dir / f"{topic}.md"
        title = topic.replace("-", " ").title()
        out_file.write_text(f"# {title}\n\nSource: {topic_url}\n\n{md}", encoding="utf-8")
        results.append(out_file.name)

    # Scrape sub-topics (e.g. /web-security/sql-injection/blind)
    sub_links = re.findall(
        rf'href="(/web-security/{re.escape(topic)}/[a-z0-9-]+)"',
        r.text
    )
    for sub in list(dict.fromkeys(sub_links))[:8]:
        sub_slug = sub.split("/")[-1]
        sub_url = f"https://portswigger.net{sub}"
        sr = _get(sub_url)
        if not sr:
            time.sleep(1)
            continue

        ssoup = BeautifulSoup(sr.text, "html.parser")
        scontent = (
            ssoup.find("div", class_="section-wrapper")
            or ssoup.find("article")
            or ssoup.find("main")
        )
        if scontent:
            for el in scontent.find_all(["nav", "aside", "script", "style"]):
                el.decompose()
            smd = _html_to_md(str(scontent))
            smd = re.sub(r"\n{3,}", "\n\n", smd).strip()
            if len(smd) > 300:
                sub_file = out_dir / f"{topic}_{sub_slug}.md"
                sub_title = f"{topic.replace('-',' ').title()} — {sub_slug.replace('-',' ').title()}"
                sub_file.write_text(
                    f"# {sub_title}\n\nSource: {sub_url}\n\n{smd}",
                    encoding="utf-8"
                )
                results.append(sub_file.name)
        time.sleep(1.5)

    return results


def scrape_portswigger(output_dir: Path) -> int:
    out = output_dir / "portswigger_labs"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[PortSwigger] Scraping {len(PS_TOPICS)} topics...")

    saved = 0
    for i, topic in enumerate(PS_TOPICS):
        print(f"  [{i+1}/{len(PS_TOPICS)}] {topic}")
        results = _scrape_ps_topic(topic, out)
        saved += len(results)
        time.sleep(2)

    print(f"  [PortSwigger] Done. Saved {saved} files → {out}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — HackerOne Public Disclosed Reports (two-phase approach)
# ══════════════════════════════════════════════════════════════════════════════
#
# The hacktivity API endpoint returns a mix of undisclosed and disclosed items
# (ratio ~1:50). The `vulnerability_information` field is always null in the
# hacktivity feed. So we use a two-phase approach:
#
#   Phase 1: Page through the hacktivity REST API (Basic auth) to collect
#            IDs of disclosed reports (items where `disclosed: true`).
#            Uses a thread pool to parallelise pagination.
#
#   Phase 2: Fetch https://hackerone.com/reports/{id}.json for each ID.
#            This is a public endpoint (no auth needed) that returns the
#            full report including `vulnerability_information`.
#            Also uses a thread pool for speed.

H1_API_BASE    = "https://api.hackerone.com/v1"
H1_REPORT_BASE = "https://hackerone.com/reports"


def _h1_fetch_page(page_num: int, auth_header: str, per_page: int = 100) -> list[int]:
    """Return a list of disclosed report IDs from a single hacktivity page."""
    url = (
        f"{H1_API_BASE}/hackers/hacktivity"
        f"?page[number]={page_num}"
        f"&page[size]={per_page}"
        f"&sort_type=popular"
    )
    try:
        r = requests.get(
            url,
            headers={"Authorization": auth_header, "Accept": "application/json"},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        items = r.json().get("data", [])
        return [
            int(item["id"])
            for item in items
            if item.get("attributes", {}).get("disclosed_at")
        ]
    except Exception:
        return []


def _h1_fetch_full_report(report_id: int) -> dict | None:
    """
    Fetch a single report from the public JSON endpoint and return a
    normalised dict, or None if not available / not disclosed.
    """
    url = f"{H1_REPORT_BASE}/{report_id}.json"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Security Research Dataset Builder)"},
            timeout=20,
        )
        if r.status_code != 200:
            return None
        data = r.json()

        if not data.get("public") or not data.get("disclosed_at"):
            return None

        title   = data.get("title", "") or ""
        vuln    = data.get("vulnerability_information", "") or ""
        if not title or not vuln:
            return None

        weakness = data.get("weakness") or {}
        reporter = data.get("reporter") or {}
        team     = data.get("team") or {}

        return {
            "id":                        report_id,
            "title":                     title,
            "vulnerability_information": vuln,
            "severity_rating":           data.get("severity_rating", ""),
            "disclosed_at":              data.get("disclosed_at", ""),
            "weakness":                  {"name": weakness.get("name", ""),
                                          "description": weakness.get("description", "")},
            "program":                   team.get("name", ""),
            "reporter":                  reporter.get("username", ""),
            "url":                       f"https://hackerone.com/reports/{report_id}",
            "cve_ids":                   data.get("cve_ids") or [],
        }
    except Exception:
        return None


def scrape_hackerone(output_dir: Path, username: str, api_token: str,
                     max_reports: int = 2000,
                     workers: int = 4) -> int:
    """
    Two-phase HackerOne scraper.

    Phase 1 — collect recently-disclosed report IDs via the authenticated
               hacktivity API.  The hacktivity feed is a real-time sliding
               window of ~30–50 recently disclosed reports; scanning 30 pages
               in parallel (takes ~10 s) captures the full current window.
               More pages just repeat the same IDs.

    Phase 2 — fetch full report details (including vulnerability_information)
               from the public  https://hackerone.com/reports/{id}.json  endpoint.
               Requests are rate-limited to avoid triggering H1's bot detection.

    Auth: Basic base64(username:api_token)
    Get your API token at: https://hackerone.com/settings/api_token/edit

    Note: The hacktivity API exposes only the ~30–50 most recently disclosed
    reports at any time.  Run this script daily/weekly to accumulate more.
    """
    out = output_dir / "hackerone_reports"
    out.mkdir(parents=True, exist_ok=True)

    creds       = base64.b64encode(f"{username}:{api_token}".encode()).decode()
    auth_header = f"Basic {creds}"
    print(f"\n[HackerOne] Using REST API (Basic {creds[:20]}...)")

    # ── Phase 1: collect disclosed IDs from hacktivity feed ──────────────────
    HACKTIVITY_PAGES = 30
    print(f"[HackerOne] Phase 1 — scanning {HACKTIVITY_PAGES} hacktivity pages for disclosed IDs...")

    disclosed_ids: set[int] = set()
    lock = threading.Lock()

    def fetch_page_task(page_num: int):
        ids = _h1_fetch_page(page_num, auth_header, per_page=100)
        with lock:
            disclosed_ids.update(ids)

    with ThreadPoolExecutor(max_workers=min(workers, HACKTIVITY_PAGES)) as ex:
        list(ex.map(fetch_page_task, range(1, HACKTIVITY_PAGES + 1)))

    id_list = list(disclosed_ids)
    print(f"  Phase 1 done. {len(id_list)} unique disclosed IDs found.")

    if not id_list:
        print("[HackerOne] No disclosed IDs found. Check your credentials.")
        return 0

    # ── Phase 2: fetch full reports from public endpoint ─────────────────────
    # Use a small worker count + inter-request delay to avoid rate limiting.
    print(f"[HackerOne] Phase 2 — fetching {len(id_list)} full reports (rate-limited)...")

    all_reports: list[dict] = []
    report_lock = threading.Lock()
    _h1_sem = threading.Semaphore(min(workers, 4))  # max 4 concurrent requests

    def fetch_report_task(rid: int):
        with _h1_sem:
            report = _h1_fetch_full_report(rid)
            time.sleep(0.5)          # polite: 2 req/s per worker
        if report:
            with report_lock:
                all_reports.append(report)

    with ThreadPoolExecutor(max_workers=min(workers, 4)) as ex:
        list(ex.map(fetch_report_task, id_list))

    with report_lock:
        final = all_reports[:max_reports]

    print(f"  Phase 2 done. {len(final)} publicly accessible reports with vulnerability info.")

    if final:
        for i in range(0, len(final), 500):
            batch     = final[i : i + 500]
            batch_num = i // 500
            out_file  = out / f"hacktivity_{batch_num:04d}.json"
            out_file.write_text(json.dumps(batch, indent=2, ensure_ascii=False))
        print(f"[HackerOne] Saved {len(final)} reports → {out}")
        print(f"  Tip: re-run weekly to accumulate more as new reports get disclosed.")
    else:
        print("[HackerOne] No publicly accessible reports found.")
        print("  The hacktivity API only surfaces ~30–50 recently disclosed reports.")
        print("  Check: https://hackerone.com/hacktivity for currently public reports.")

    return len(final)


# ══════════════════════════════════════════════════════════════════════════════
# New scrapers
# ══════════════════════════════════════════════════════════════════════════════

def scrape_unprotect(out_dir: Path, max_techniques: int = 500) -> int:
    """Scrape unprotect.it — malware evasion / anti-analysis technique database."""
    out = out_dir / "unprotect"
    out.mkdir(parents=True, exist_ok=True)

    API_BASE = "https://search.unprotect.it/api"
    headers = {"Accept": "application/json", "User-Agent": "MythosDataset/1.0"}
    saved = 0

    print(f"\n[unprotect.it] Fetching malware evasion techniques...")
    try:
        # Get technique list
        r = requests.get(f"{API_BASE}/techniques/?format=json&limit={max_techniques}",
                         headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        techniques = data.get("results") or data if isinstance(data, list) else []
    except Exception as e:
        print(f"  [unprotect] Error fetching index: {e}")
        return 0

    batch = []
    for tech in techniques:
        if not isinstance(tech, dict):
            continue
        name    = tech.get("name") or tech.get("technique") or ""
        desc    = tech.get("description") or ""
        cat     = tech.get("category") or tech.get("Category") or ""
        tags    = tech.get("tags") or []
        samples = tech.get("code") or tech.get("samples") or []

        if not name or not desc:
            continue

        batch.append({
            "name": name,
            "category": cat,
            "description": desc,
            "tags": tags,
            "code_samples": samples[:3] if isinstance(samples, list) else [],
        })

    if batch:
        out_file = out / "techniques.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)
        saved = len(batch)
        print(f"  [unprotect] Saved {saved} techniques → {out_file}")

    return saved


def scrape_project_zero(out_dir: Path, max_posts: int = 200) -> int:
    """Scrape Google Project Zero blog posts."""
    out = out_dir / "projectzero"
    out.mkdir(parents=True, exist_ok=True)

    if not _HAS_DEPS:
        print("  [projectzero] html2text not available, skipping")
        return 0

    FEED_URL = "https://googleprojectzero.blogspot.com/feeds/posts/default?alt=json&max-results=25"
    headers = {"User-Agent": "MythosDataset/1.0"}
    saved = 0
    start = 0

    print(f"\n[Project Zero] Scraping blog posts (max {max_posts})...")

    while saved < max_posts:
        try:
            url = FEED_URL + f"&start-index={start + 1}"
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            feed = r.json()
        except Exception as e:
            print(f"  [projectzero] Error fetching feed at offset {start}: {e}")
            break

        entries = feed.get("feed", {}).get("entry", [])
        if not entries:
            break

        for entry in entries:
            if saved >= max_posts:
                break
            try:
                title = entry.get("title", {}).get("$t", "").strip()
                content_html = entry.get("content", {}).get("$t", "") or \
                               entry.get("summary", {}).get("$t", "")
                pub_date = entry.get("published", {}).get("$t", "")[:10]

                if not title or not content_html:
                    continue

                # Convert HTML to markdown
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.body_width = 0
                md_content = h.handle(content_html)

                if len(md_content) < 300:
                    continue

                # Save as markdown
                safe_title = re.sub(r"[^\w\s-]", "", title)[:60].strip().replace(" ", "_")
                out_file = out / f"{pub_date}_{safe_title}.md"
                out_file.write_text(f"# {title}\n\n*Published: {pub_date}*\n\n{md_content}",
                                    encoding="utf-8")
                saved += 1
            except Exception:
                continue

        start += len(entries)
        if len(entries) < 25:
            break
        time.sleep(1)

    print(f"  [projectzero] Saved {saved} posts → {out}")
    return saved


def scrape_phrack(out_dir: Path, max_articles: int = 300) -> int:
    """Scrape Phrack magazine articles."""
    out = out_dir / "phrack"
    out.mkdir(parents=True, exist_ok=True)

    if not _HAS_DEPS:
        print("  [phrack] html2text not available, skipping")
        return 0

    BASE = "https://phrack.org"
    headers = {"User-Agent": "MythosDataset/1.0"}
    saved = 0

    print(f"\n[Phrack] Scraping magazine articles (max {max_articles})...")

    # Focus on security-relevant issues (49-70 are the most technical)
    for issue_num in range(49, 71):
        if saved >= max_articles:
            break
        # Phrack articles are at /issues/NUM/ARTICLE.html (1-indexed)
        for art_idx in range(1, 16):
            if saved >= max_articles:
                break
            try:
                art_url = f"{BASE}/issues/{issue_num}/{art_idx}.html"
                ar = requests.get(art_url, headers=headers, timeout=15)
                if ar.status_code == 404:
                    break  # No more articles in this issue
                if ar.status_code != 200:
                    continue

                art_soup = BeautifulSoup(ar.text, "html.parser")
                pre = art_soup.find("pre")
                content = pre.get_text() if pre else art_soup.get_text()
                if len(content) < 500:
                    continue

                lines = content.strip().splitlines()
                title_line = next((l.strip() for l in lines[:20]
                                   if 10 < len(l.strip()) < 100),
                                  f"Issue {issue_num} Article {art_idx}")
                title = title_line[:80]

                safe = re.sub(r"[^\w-]", "_", title)[:50]
                out_file = out / f"phrack_{issue_num}_{art_idx}_{safe}.md"
                out_file.write_text(
                    f"# Phrack Issue {issue_num}: {title}\n\n```\n{content[:30000]}\n```",
                    encoding="utf-8")
                saved += 1
                time.sleep(0.3)
            except Exception:
                continue

    print(f"  [phrack] Saved {saved} articles → {out}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Scrape security writeups for Mythos v3 dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--sources", nargs="+",
                        choices=["0xdf", "portswigger", "hackerone", "unprotect", "projectzero", "phrack"],
                        default=["0xdf", "portswigger", "hackerone", "unprotect", "projectzero", "phrack"],
                        help="Which sources to scrape")
    parser.add_argument("--h1-username", default="",
                        help="HackerOne API username (leave blank for GraphQL fallback)")
    parser.add_argument("--h1-token", default="",
                        help="HackerOne API token (leave blank for GraphQL fallback)")
    parser.add_argument("--h1-max", type=int, default=2000,
                        help="Max HackerOne reports to fetch")
    parser.add_argument("--h1-workers", type=int, default=8,
                        help="Parallel worker threads for HackerOne scraping")
    parser.add_argument("--0xdf-max", type=int, default=500, dest="xdf_max",
                        help="Max 0xdf posts to scrape")
    parser.add_argument("--htb-only", action="store_true",
                        help="Only scrape HTB writeups from 0xdf")
    args = parser.parse_args()

    if not _HAS_DEPS:
        print("Install deps first: pip install requests beautifulsoup4 html2text")
        return

    scraped_dir = args.data_dir / "scraped"
    scraped_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    if "0xdf" in args.sources:
        total += scrape_0xdf(scraped_dir, max_posts=args.xdf_max, htb_only=args.htb_only)

    if "portswigger" in args.sources:
        total += scrape_portswigger(scraped_dir)

    if "hackerone" in args.sources:
        if args.h1_username and args.h1_token:
            total += scrape_hackerone(
                scraped_dir, args.h1_username, args.h1_token,
                max_reports=args.h1_max, workers=args.h1_workers,
            )
        else:
            print("\n[HackerOne] No credentials provided (--h1-username / --h1-token).")
            print("  Get your API token at: https://hackerone.com/settings/api_token/edit")

    if "unprotect" in args.sources:
        total += scrape_unprotect(scraped_dir)

    if "projectzero" in args.sources:
        total += scrape_project_zero(scraped_dir)

    if "phrack" in args.sources:
        total += scrape_phrack(scraped_dir)

    print(f"\n✓ Total scraped items: {total}")
    print(f"  Output: {scraped_dir.resolve()}")
    print(f"\nNext step:\n  python3 build_dataset.py --output training_data_v5")


if __name__ == "__main__":
    main()
