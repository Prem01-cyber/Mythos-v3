#!/usr/bin/env python3
"""
Mythos v3 — Training Data Preparation Script
Transforms raw security research corpus → Qwen chat-format JSONL

Sources processed:
  1. nvd/nvd_cves_all.json          — NVD CVE database (streaming, 2.7 GB)
  2. github_pocs/cve/               — CVE markdown files with descriptions + PoC links
  3. github_pocs/PoC-in-GitHub/     — GitHub PoC repo metadata per CVE
  4. PayloadsAllTheThings/          — Attack technique writeups (markdown)
  5. hacktricks/src/                — Pentesting knowledge base (markdown)
  6. sql-injection-payload-list/    — SQL injection payload files
  7. xss-payload-list/              — XSS payload files
  8. command-injection-payload-list/— Command injection payload files
  9. waf-bypass-payload-list/       — WAF bypass payload files
 10. crlf-injection-payload-list/   — CRLF injection payload files
 11. xxe-injection-payload-list/    — XXE injection payload files
 12. ssti-advanced-payload-list/    — SSTI payload files
 13. server-side-template-injection-payloads/ — More SSTI payloads
 14. http-request-smuggling-payloads/— HTTP request smuggling payloads
 15. open-redirect-payload-list/    — Open redirect payload files
 16. csv-injection-payload-list/    — CSV injection payload files
 17. directory-payload-list/        — Directory traversal payload files
 18. protocol-injection-payload-list/— Protocol injection payloads
 19. web-cache-poisoning-payloads/  — Web cache poisoning payloads
 20. llm-security-payloads/         — LLM security payload files
 21. payload-box/                   — General payload box

Output:
  training_data/train.jsonl
  training_data/val.jsonl
  training_data/dataset_stats.json
"""

import argparse
import ijson
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterator


# ─── Qwen system prompts (rotated for variety) ────────────────────────────────

SYSTEM_PROMPTS = {
    "cve": [
        "You are a security expert specializing in vulnerability analysis and CVE research.",
        "You are a cybersecurity analyst with deep expertise in vulnerability assessment.",
        "You are a penetration tester and vulnerability researcher. Provide technical, accurate information about CVEs.",
        "You are an expert in the National Vulnerability Database (NVD) and CVE tracking. Analyze vulnerabilities precisely.",
    ],
    "exploit": [
        "You are an exploit developer and offensive security researcher. Write complete, functional exploit code.",
        "You are a red team operator specializing in exploit development. Provide working proof-of-concept code.",
        "You are a security researcher who develops exploits for vulnerability validation and penetration testing.",
    ],
    "payload": [
        "You are a penetration tester specializing in web application security.",
        "You are a bug bounty hunter with expertise in web vulnerabilities and payload crafting.",
        "You are a web application security expert. Provide practical, tested payloads for security testing.",
        "You are a red team specialist focused on web attack techniques and payload development.",
    ],
    "technique": [
        "You are a penetration testing expert with comprehensive knowledge of offensive security techniques.",
        "You are a cybersecurity instructor teaching advanced offensive security methodology.",
        "You are a red team lead with expertise in attack techniques, tools, and methodologies.",
        "You are a security researcher specializing in offensive security tactics, techniques, and procedures (TTPs).",
    ],
    "poc": [
        "You are a vulnerability researcher who tracks and analyzes proof-of-concept exploits.",
        "You are a security engineer analyzing CVE exploitability and available public exploits.",
        "You are a threat intelligence analyst tracking exploit availability for CVEs.",
    ],
}


def pick_system(category: str) -> str:
    prompts = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["technique"])
    return random.choice(prompts)


def make_example(system: str, user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ─── Quality filter ────────────────────────────────────────────────────────────

MIN_ASSISTANT_CHARS = 80
MAX_ASSISTANT_CHARS = 32000  # ~8K tokens

def is_quality(example: dict) -> bool:
    msgs = example.get("messages", [])
    if len(msgs) != 3:
        return False
    assistant_content = msgs[2].get("content", "")
    user_content = msgs[1].get("content", "")
    if not assistant_content or not user_content:
        return False
    if len(assistant_content) < MIN_ASSISTANT_CHARS:
        return False
    if len(assistant_content) > MAX_ASSISTANT_CHARS:
        return False
    return True


# ─── 1. NVD CVE Parser ────────────────────────────────────────────────────────

CVE_QUESTION_TEMPLATES = [
    ("What is {cve_id} and how critical is it?", None),
    ("Explain the vulnerability {cve_id}.", None),
    ("Describe {cve_id}: affected systems, severity, and impact.", None),
    ("What is the CVSS score and severity for {cve_id}?", "severity"),
    ("How can {cve_id} be exploited?", "exploit"),
    ("What products are affected by {cve_id}?", "products"),
    ("Provide a technical analysis of {cve_id}.", None),
    ("What are the mitigation steps for {cve_id}?", "mitigation"),
    ("Is {cve_id} being actively exploited in the wild?", "exploited"),
    ("What CWE category does {cve_id} belong to?", "cwe"),
]


def _get_nvd_field(cve: dict, *keys, default=""):
    """Safely traverse nested dict."""
    obj = cve
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k, default)
    return obj if obj is not None else default


def _format_cvss(cve: dict) -> str:
    metrics = cve.get("metrics", {})
    lines = []
    # V3.1 or V3.0
    for key in ("cvssMetricV31", "cvssMetricV30"):
        if key in metrics and metrics[key]:
            m = metrics[key][0]
            data = m.get("cvssData", {})
            score = data.get("baseScore", "N/A")
            severity = m.get("baseSeverity", data.get("baseSeverity", "N/A"))
            vector = data.get("vectorString", "N/A")
            lines.append(f"CVSS v3 Score: {score} ({severity})")
            lines.append(f"Vector: {vector}")
            break
    # V2
    if not lines and "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
        m = metrics["cvssMetricV2"][0]
        data = m.get("cvssData", {})
        score = data.get("baseScore", "N/A")
        severity = m.get("baseSeverity", "N/A")
        lines.append(f"CVSS v2 Score: {score} ({severity})")
    return "\n".join(lines) if lines else "CVSS score: Not available"


def _format_products(cve: dict) -> str:
    configs = cve.get("configurations", [])
    products = []
    for config in configs:
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                criteria = match.get("criteria", "")
                parts = criteria.split(":")
                if len(parts) >= 5:
                    vendor = parts[3].replace("_", " ").title()
                    product = parts[4].replace("_", " ").title()
                    version = parts[5] if len(parts) > 5 else "*"
                    entry = f"{vendor} {product}"
                    if version and version != "*":
                        entry += f" {version}"
                    if entry not in products:
                        products.append(entry)
    return "\n".join(f"- {p}" for p in products[:20]) if products else "Affected products not specified in NVD data."


def _format_weaknesses(cve: dict) -> str:
    weaknesses = cve.get("weaknesses", [])
    cwes = []
    for w in weaknesses:
        for desc in w.get("description", []):
            val = desc.get("value", "")
            if val and val not in cwes and val != "NVD-CWE-Other" and val != "NVD-CWE-noinfo":
                cwes.append(val)
    return ", ".join(cwes) if cwes else "Not classified"


def _build_cve_answer(cve_id: str, description: str, cvss_str: str, products_str: str,
                       weaknesses_str: str, published: str, template_hint: str) -> str:
    if template_hint == "severity":
        return (
            f"{cve_id} Severity Assessment\n\n"
            f"{cvss_str}\n\n"
            f"CWE: {weaknesses_str}\n\n"
            f"Description: {description}"
        )
    elif template_hint == "products":
        return (
            f"Products affected by {cve_id}:\n\n"
            f"{products_str}\n\n"
            f"Description: {description}\n\n"
            f"Published: {published[:10] if published else 'Unknown'}"
        )
    elif template_hint == "cwe":
        return (
            f"{cve_id} belongs to the following weakness category:\n\n"
            f"CWE: {weaknesses_str}\n\n"
            f"This classification reflects the root cause of the vulnerability.\n\n"
            f"Description: {description}"
        )
    elif template_hint == "mitigation":
        return (
            f"Mitigation for {cve_id}:\n\n"
            f"1. Apply vendor-provided patches immediately\n"
            f"2. Monitor vendor security advisories for updates\n"
            f"3. Implement network-level controls to limit exposure\n"
            f"4. Review and restrict access to affected components\n\n"
            f"Vulnerability details:\n{description}\n\n"
            f"{cvss_str}"
        )
    elif template_hint == "exploited":
        return (
            f"Exploitability analysis for {cve_id}:\n\n"
            f"{cvss_str}\n\n"
            f"Vulnerability: {description}\n\n"
            f"Affected systems:\n{products_str}\n\n"
            f"Note: Check CISA KEV catalog and threat intelligence feeds for active exploitation status."
        )
    elif template_hint == "exploit":
        return (
            f"Exploitation analysis for {cve_id}:\n\n"
            f"Vulnerability: {description}\n\n"
            f"{cvss_str}\n\n"
            f"Exploitation approach depends on the specific attack vector indicated by the CVSS vector string. "
            f"Review vendor advisories and public PoC repositories for technical exploitation details.\n\n"
            f"Affected systems:\n{products_str}"
        )
    else:
        return (
            f"{cve_id}\n\n"
            f"Description:\n{description}\n\n"
            f"{cvss_str}\n\n"
            f"Weakness: {weaknesses_str}\n\n"
            f"Affected Products:\n{products_str}\n\n"
            f"Published: {published[:10] if published else 'Unknown'}"
        )


def parse_nvd(data_dir: Path) -> Iterator[dict]:
    """Stream NVD CVEs from nvd_cves_all.json and yield training examples."""
    nvd_file = data_dir / "nvd" / "nvd_cves_all.json"
    if not nvd_file.exists():
        print(f"  [skip] NVD file not found: {nvd_file}")
        return

    print(f"  Streaming NVD: {nvd_file} ({nvd_file.stat().st_size / 1e9:.1f} GB)")
    count = 0

    with open(nvd_file, "rb") as f:
        for cve_item in ijson.items(f, "vulnerabilities.item"):
            cve = cve_item.get("cve", {})
            cve_id = cve.get("id", "")
            if not cve_id:
                continue

            # Extract description (English)
            description = ""
            for desc in cve.get("descriptions", []):
                if desc.get("lang") == "en":
                    description = desc.get("value", "").strip()
                    break

            if not description or len(description) < 30:
                continue
            if description.startswith("** REJECT"):
                continue

            cvss_str = _format_cvss(cve)
            products_str = _format_products(cve)
            weaknesses_str = _format_weaknesses(cve)
            published = cve.get("published", "")

            # Generate multiple examples per CVE
            # Use a random subset of templates to avoid oversampling
            templates = random.sample(CVE_QUESTION_TEMPLATES, min(3, len(CVE_QUESTION_TEMPLATES)))

            for question_tmpl, hint in templates:
                question = question_tmpl.format(cve_id=cve_id)
                answer = _build_cve_answer(
                    cve_id, description, cvss_str, products_str, weaknesses_str, published, hint
                )
                ex = make_example(pick_system("cve"), question, answer)
                if is_quality(ex):
                    yield ex
                    count += 1

    print(f"  NVD → {count:,} examples")


# ─── 2. CVE Markdown Parser (github_pocs/cve/) ────────────────────────────────

def _parse_cve_md(content: str, cve_id: str) -> dict:
    """Extract description, affected products, PoC links from CVE markdown."""
    result = {"cve_id": cve_id, "description": "", "products": [], "poc_refs": []}

    # Description section
    desc_match = re.search(r"### Description\s*\n+(.*?)(?=\n###|\Z)", content, re.DOTALL)
    if desc_match:
        result["description"] = desc_match.group(1).strip()

    # PoC GitHub links
    github_section = re.search(r"#### Github\s*\n+(.*?)(?=\n###|\n####|\Z)", content, re.DOTALL)
    if github_section:
        links = re.findall(r"https://github\.com/\S+", github_section.group(1))
        result["poc_refs"] = links[:10]

    # Reference links
    ref_section = re.search(r"#### Reference\s*\n+(.*?)(?=\n###|\n####|\Z)", content, re.DOTALL)
    if ref_section:
        refs = re.findall(r"https?://\S+", ref_section.group(1))
        result["poc_refs"].extend(refs[:5])

    return result


CVE_MD_QUESTION_TEMPLATES = [
    "What is {cve_id}? Provide a detailed explanation.",
    "Are there public exploits or PoCs for {cve_id}? What is the vulnerability?",
    "Summarize {cve_id} including the description and available proof-of-concept references.",
    "What do we know about {cve_id} from public security research?",
    "Explain {cve_id} and where researchers have documented it.",
]


def parse_cve_markdowns(data_dir: Path) -> Iterator[dict]:
    """Parse CVE markdown files from github_pocs/cve/YEAR/CVE-ID.md."""
    cve_dir = data_dir / "github_pocs" / "cve"
    if not cve_dir.exists():
        return

    files = list(cve_dir.rglob("CVE-*.md"))
    print(f"  CVE markdowns: {len(files):,} files")
    count = 0

    for md_file in files:
        cve_id = md_file.stem  # e.g. "CVE-2024-0012"
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        parsed = _parse_cve_md(content, cve_id)
        description = parsed["description"]
        if not description or len(description) < 40:
            continue

        poc_refs = parsed["poc_refs"]
        poc_section = ""
        if poc_refs:
            poc_section = "\n\nPublic PoC/References:\n" + "\n".join(f"- {r}" for r in poc_refs[:8])

        # Pick one template
        question_tmpl = random.choice(CVE_MD_QUESTION_TEMPLATES)
        question = question_tmpl.format(cve_id=cve_id)

        if poc_refs:
            answer = (
                f"{cve_id}\n\n"
                f"Description:\n{description}"
                f"{poc_section}"
            )
        else:
            answer = f"{cve_id}\n\nDescription:\n{description}"

        ex = make_example(pick_system("cve"), question, answer)
        if is_quality(ex):
            yield ex
            count += 1

    print(f"  CVE markdowns → {count:,} examples")


# ─── 3. PoC-in-GitHub Parser ──────────────────────────────────────────────────

POC_QUESTION_TEMPLATES = [
    "What public proof-of-concept exploits exist for {cve_id}?",
    "Show me GitHub repositories that have PoC exploits for {cve_id}.",
    "Which researchers have published exploits for {cve_id} on GitHub?",
    "List the available open-source exploits and tools for {cve_id}.",
]


def parse_poc_github(data_dir: Path) -> Iterator[dict]:
    """Parse PoC-in-GitHub JSON files (one per CVE)."""
    poc_dir = data_dir / "github_pocs" / "PoC-in-GitHub"
    if not poc_dir.exists():
        return

    files = list(poc_dir.rglob("CVE-*.json"))
    print(f"  PoC-in-GitHub: {len(files):,} files")
    count = 0

    for json_file in files:
        cve_id = json_file.stem
        try:
            repos = json.loads(json_file.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        if not repos or not isinstance(repos, list):
            continue

        # Build answer with top repos
        repo_lines = []
        for repo in repos[:12]:
            name = repo.get("full_name", "")
            desc = repo.get("description") or ""
            stars = repo.get("stargazers_count", 0)
            url = repo.get("html_url", f"https://github.com/{name}")
            if name:
                line = f"• [{name}]({url})"
                if desc:
                    line += f" — {desc[:120]}"
                line += f" (★{stars})"
                repo_lines.append(line)

        if not repo_lines:
            continue

        answer = (
            f"Public PoC repositories for {cve_id}:\n\n"
            + "\n".join(repo_lines)
            + f"\n\nTotal public PoCs found: {len(repos)}"
        )

        question = random.choice(POC_QUESTION_TEMPLATES).format(cve_id=cve_id)
        ex = make_example(pick_system("poc"), question, answer)
        if is_quality(ex):
            yield ex
            count += 1

    print(f"  PoC-in-GitHub → {count:,} examples")


# ─── 4. PayloadsAllTheThings Parser ───────────────────────────────────────────

PATT_QUESTION_TEMPLATES = [
    "Explain the {topic} attack technique with examples.",
    "What are the key {topic} payloads and techniques?",
    "How do you perform {topic} attacks? Give me a comprehensive guide.",
    "Teach me about {topic}: methodology, payloads, and bypass techniques.",
    "What should I know about {topic} for a penetration test?",
    "Give me a reference guide for {topic} attacks and payloads.",
]

EXCLUDED_MD_NAMES = {"README", "CONTRIBUTING", "DISCLAIMER", "LICENSE", "SUMMARY",
                      "CHANGELOG", "CODE_OF_CONDUCT"}


def _clean_md_content(content: str, max_chars: int = 12000) -> str:
    """Remove image links, shield badges, template directives, and truncate."""
    # Remove mdBook/HackerTricks template includes
    content = re.sub(r"\{\{#include[^}]+\}\}", "", content)
    # Remove image links (markdown + HTML)
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)
    content = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", content)
    content = re.sub(r"<img[^>]+>", "", content, flags=re.IGNORECASE)
    # Remove HTML comments
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    # Remove shield/badge URLs that escaped the above
    content = re.sub(r"https?://img\.shields\.io\S+", "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()[:max_chars]


def _md_to_topic(filename: str) -> str:
    """Convert filename to human-readable topic."""
    name = Path(filename).stem
    return re.sub(r"[-_]", " ", name).title()


def parse_payloads_all_things(data_dir: Path) -> Iterator[dict]:
    """Parse PayloadsAllTheThings markdown files."""
    patt_dir = data_dir / "PayloadsAllTheThings"
    if not patt_dir.exists():
        return

    md_files = [f for f in patt_dir.rglob("*.md")
                if f.stem not in EXCLUDED_MD_NAMES and f.stem.upper() not in EXCLUDED_MD_NAMES]
    print(f"  PayloadsAllTheThings: {len(md_files):,} markdown files")
    count = 0

    for md_file in md_files:
        topic = _md_to_topic(md_file.name)
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        content = _clean_md_content(content)
        if len(content) < 200:
            continue

        question = random.choice(PATT_QUESTION_TEMPLATES).format(topic=topic)
        ex = make_example(pick_system("payload"), question, content)
        if is_quality(ex):
            yield ex
            count += 1

    print(f"  PayloadsAllTheThings → {count:,} examples")


# ─── 5. HackerTricks Parser ───────────────────────────────────────────────────

HACKTRICKS_QUESTION_TEMPLATES = [
    "Explain the {topic} penetration testing technique.",
    "How do you approach {topic} during a penetration test?",
    "What is {topic} and how is it exploited?",
    "Give me a comprehensive guide on {topic} from a hacker's perspective.",
    "Teach me the {topic} methodology used in offensive security.",
    "What are the key concepts and techniques in {topic}?",
]


def parse_hacktricks(data_dir: Path) -> Iterator[dict]:
    """Parse HackerTricks + HackerTricks Cloud markdown files."""
    ht_dirs = []
    for subdir in ("hacktricks/src", "hacktricks-cloud/src", "hacktricks-cloud"):
        d = data_dir / subdir
        if d.exists():
            ht_dirs.append(d)
            break  # prefer most specific match

    # Also include hacktricks-cloud top-level if it has markdown
    cloud_dir = data_dir / "hacktricks-cloud"
    if cloud_dir.exists() and cloud_dir not in ht_dirs:
        ht_dirs.append(cloud_dir)

    md_files = []
    for d in ht_dirs:
        md_files.extend(
            f for f in d.rglob("*.md")
            if f.stem.upper() not in EXCLUDED_MD_NAMES and f.stem.upper() not in {"ADS", "ROBOTS"}
        )

    print(f"  HackerTricks: {len(md_files):,} markdown files (across {len(ht_dirs)} dirs)")
    count = 0

    for md_file in md_files:
        topic = _md_to_topic(md_file.stem)
        topic = topic.replace(".Md", "").strip()

        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        content = _clean_md_content(content, max_chars=14000)
        if len(content) < 200:
            continue

        question = random.choice(HACKTRICKS_QUESTION_TEMPLATES).format(topic=topic)
        ex = make_example(pick_system("technique"), question, content)
        if is_quality(ex):
            yield ex
            count += 1

    print(f"  HackerTricks → {count:,} examples")


# ─── 6. Generic Payload List Parser ──────────────────────────────────────────

PAYLOAD_SOURCES = [
    ("sql-injection-payload-list", "SQL injection", "sqli"),
    ("xss-payload-list", "XSS (Cross-Site Scripting)", "xss"),
    ("command-injection-payload-list", "command injection", "cmd"),
    ("waf-bypass-payload-list", "WAF bypass", "waf"),
    ("crlf-injection-payload-list", "CRLF injection", "crlf"),
    ("xxe-injection-payload-list", "XXE (XML External Entity) injection", "xxe"),
    ("ssti-advanced-payload-list", "SSTI (Server-Side Template Injection)", "ssti"),
    ("server-side-template-injection-payloads", "server-side template injection", "ssti"),
    ("http-request-smuggling-payloads", "HTTP request smuggling", "smuggling"),
    ("open-redirect-payload-list", "open redirect", "redirect"),
    ("csv-injection-payload-list", "CSV injection", "csv"),
    ("directory-payload-list", "directory traversal", "traversal"),
    ("protocol-injection-payload-list", "protocol injection", "protocol"),
    ("web-cache-poisoning-payloads", "web cache poisoning", "cache"),
    ("llm-security-payloads", "LLM security testing", "llm"),
    ("payload-box", "security testing", "general"),
    ("business-logic-exploitation-playbook", "business logic exploitation", "bizlogic"),
]

PAYLOAD_LIST_QUESTIONS = [
    "Give me {attack_type} payloads for security testing.",
    "List common {attack_type} payloads used in penetration testing.",
    "What are effective {attack_type} payloads for testing web applications?",
    "Provide {attack_type} payload examples for a bug bounty assessment.",
    "I need {attack_type} payloads for Burp Intruder. What are the best ones?",
    "Give me a comprehensive {attack_type} payload list.",
    "What {attack_type} payloads should I use when testing for vulnerabilities?",
]


def _read_payload_file(filepath: Path, max_lines: int = 400) -> list[str]:
    """Read payload lines from a text file, filtering noise."""
    try:
        lines = filepath.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip comment lines
        if line.startswith("#") or line.startswith("//") or line.startswith("/*"):
            continue
        result.append(line)
        if len(result) >= max_lines:
            break
    return result


def _build_payload_answer(attack_type: str, payloads: list[str], source_file: str) -> str:
    category_hint = Path(source_file).stem.replace("-", " ").replace("_", " ").title()
    formatted = "\n".join(payloads[:100])
    return (
        f"{attack_type.title()} Payloads — {category_hint}\n\n"
        f"```\n{formatted}\n```\n\n"
        f"Total payloads in this set: {len(payloads)}\n\n"
        f"Usage: Load into Burp Intruder, modify as needed for target context, "
        f"and test systematically across all injection points."
    )


def parse_payload_lists(data_dir: Path) -> Iterator[dict]:
    """Parse all payload list directories for txt/md payload files."""
    total = 0

    for folder_name, attack_type, category in PAYLOAD_SOURCES:
        folder = data_dir / folder_name
        if not folder.exists():
            continue

        # Find all txt and relevant md files (not README/LICENSE)
        txt_files = list(folder.rglob("*.txt"))
        md_files = [f for f in folder.rglob("*.md")
                    if f.stem.upper() not in EXCLUDED_MD_NAMES
                    and "CHEAT" not in f.stem.upper()
                    and "SUMMARY" not in f.stem.upper()]

        all_files = txt_files + md_files
        if not all_files:
            continue

        folder_count = 0

        for filepath in all_files:
            if filepath.stat().st_size < 50:
                continue

            # For markdown files, read as technique guide
            if filepath.suffix == ".md":
                try:
                    content = filepath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                content = _clean_md_content(content)
                if len(content) < 200:
                    continue
                topic = _md_to_topic(filepath.name) or attack_type
                question = random.choice(PATT_QUESTION_TEMPLATES).format(topic=topic)
                ex = make_example(pick_system("payload"), question, content)
                if is_quality(ex):
                    yield ex
                    folder_count += 1
                continue

            # For txt files, read as payload list
            payloads = _read_payload_file(filepath)
            if len(payloads) < 3:
                continue

            question = random.choice(PAYLOAD_LIST_QUESTIONS).format(attack_type=attack_type)
            answer = _build_payload_answer(attack_type, payloads, filepath.name)

            ex = make_example(pick_system("payload"), question, answer)
            if is_quality(ex):
                yield ex
                folder_count += 1

        if folder_count:
            total += folder_count

    print(f"  Payload lists → {total:,} examples (across {len(PAYLOAD_SOURCES)} source folders)")


# ─── Writer / aggregator ──────────────────────────────────────────────────────

def run(data_dir: Path, output_dir: Path, seed: int = 42, val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        limit: int | None = None, per_source_limit: int | None = None, dry_run: bool = False):
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path   = output_dir / "val.jsonl"
    test_path  = output_dir / "test.jsonl"
    stats_path = output_dir / "dataset_stats.json"

    source_stats: dict[str, int] = {}
    total = 0
    filtered = 0
    seen_hashes: set[int] = set()

    def dedup_hash(ex: dict) -> int:
        msgs = ex["messages"]
        return hash(msgs[1]["content"][:200] + msgs[2]["content"][:200])

    # Collect all generators
    generators = [
        ("nvd_cves", parse_nvd(data_dir)),
        ("cve_markdowns", parse_cve_markdowns(data_dir)),
        ("poc_github", parse_poc_github(data_dir)),
        ("payloads_all_things", parse_payloads_all_things(data_dir)),
        ("hacktricks", parse_hacktricks(data_dir)),
        ("payload_lists", parse_payload_lists(data_dir)),
    ]

    print(f"\n[2/5] Generating and streaming examples to output files...")
    t0 = time.time()

    # We stream directly to avoid memory blowup:
    # First pass: write everything to a temp file, tracking which source each came from.
    # Then shuffle and split.

    temp_path = output_dir / "_all_examples.jsonl"

    print(f"  Writing all examples to temp file: {temp_path}")

    with open(temp_path, "w", encoding="utf-8") as tmp_f:
        for source_name, gen in generators:
            print(f"\n  Processing source: {source_name}")
            src_count = 0
            for ex in gen:
                if limit and total >= limit:
                    break
                if per_source_limit and src_count >= per_source_limit:
                    break

                h = dedup_hash(ex)
                if h in seen_hashes:
                    filtered += 1
                    continue
                seen_hashes.add(h)

                tmp_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                total += 1
                src_count += 1

                if total % 50000 == 0:
                    elapsed = time.time() - t0
                    print(f"    ... {total:,} examples ({elapsed:.0f}s elapsed)")

            source_stats[source_name] = src_count
            print(f"    {source_name}: {src_count:,} examples")

            if limit and total >= limit:
                print(f"  [limit reached: {limit:,}]")
                break

    print(f"\n  Total examples (before dedup): {total + filtered:,}")
    print(f"  Duplicates removed: {filtered:,}")
    print(f"  Total unique examples: {total:,}")

    if total == 0:
        print("ERROR: No examples generated. Check data directory.")
        return

    print(f"\n[3/5] Shuffling {total:,} examples...")
    indices = list(range(total))
    random.shuffle(indices)

    print(f"[4/5] Reading temp file and writing train/val/test split (90/5/5)...")
    val_count  = max(1, int(total * val_ratio))
    test_count = max(1, int(total * test_ratio))
    train_count = total - val_count - test_count

    # Read all lines and apply shuffle order
    with open(temp_path, "r", encoding="utf-8") as tmp_f:
        all_lines = tmp_f.readlines()
    shuffled_lines = [all_lines[i] for i in indices]

    with open(train_path, "w", encoding="utf-8") as train_f, \
         open(val_path,   "w", encoding="utf-8") as val_f, \
         open(test_path,  "w", encoding="utf-8") as test_f:
        for i, line in enumerate(shuffled_lines):
            if i < val_count:
                val_f.write(line)
            elif i < val_count + test_count:
                test_f.write(line)
            else:
                train_f.write(line)

    # Clean up temp file
    if not dry_run:
        temp_path.unlink(missing_ok=True)

    elapsed = time.time() - t0

    train_pct = 100 * train_count / total
    val_pct   = 100 * val_count   / total
    test_pct  = 100 * test_count  / total

    stats = {
        "total_examples": total,
        "train_examples": train_count,
        "val_examples":   val_count,
        "test_examples":  test_count,
        "split_ratio": f"{train_pct:.0f}/{val_pct:.0f}/{test_pct:.0f}",
        "duplicates_removed": filtered,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "source_breakdown": source_stats,
    }

    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"\n[5/5] Done!")
    print(f"  train.jsonl : {train_count:,} examples  ({train_pct:.1f}%)")
    print(f"  val.jsonl   : {val_count:,} examples  ({val_pct:.1f}%)")
    print(f"  test.jsonl  : {test_count:,} examples  ({test_pct:.1f}%)")
    print(f"  Total time  : {elapsed/60:.1f} minutes")
    print(f"\n  Source breakdown:")
    for src, cnt in source_stats.items():
        print(f"    {src:30s}: {cnt:>10,}")
    print(f"\n  Output: {output_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Mythos v3 training data for Qwen fine-tuning"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root directory containing raw data sources (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./training_data"),
        help="Output directory for train.jsonl, val.jsonl, dataset_stats.json (default: ./training_data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of data to use for validation (default: 0.05)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.05,
        help="Fraction of data to use for test (default: 0.05)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total examples (for testing)",
    )
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=None,
        help="Limit examples per source (for testing all sources quickly)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Keep temp file for inspection",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Mythos v3 — Training Data Preparation")
    print("=" * 60)
    print(f"  Data dir   : {args.data_dir.resolve()}")
    print(f"  Output dir : {args.output_dir.resolve()}")
    print(f"  Val ratio  : {args.val_ratio}")
    print(f"  Test ratio : {args.test_ratio}")
    print(f"  Seed       : {args.seed}")
    if args.limit:
        print(f"  Limit      : {args.limit:,}")
    if args.per_source_limit:
        print(f"  Per-source : {args.per_source_limit:,}")
    print()

    print("[1/5] Scanning data directory...")
    sources = [d.name for d in args.data_dir.iterdir() if d.is_dir()] if args.data_dir.exists() else []
    print(f"  Found {len(sources)} source directories")
    for s in sorted(sources):
        print(f"    - {s}")

    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        limit=args.limit,
        per_source_limit=args.per_source_limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
