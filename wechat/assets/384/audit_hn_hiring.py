#!/usr/bin/env python3
"""Audit November 'Ask HN: Who is hiring?' threads (2014-2026) via HN Algolia Search API.

Strips HTML tags (preventing hidden href slugs from triggering false positives) and
disambiguates AI Agent / Agentic / MCP terms from non-AI roles (real estate, travel,
customer support, monitoring agents, etc.).
"""
import html
import json
import re
import subprocess
from pathlib import Path

AI_RE = re.compile(
    r"\b(ai|artificial intelligence|llm|llms|gpt|claude|genai|generative ai|agentic|"
    r"ai agent|ai agents|machine learning|deep learning|mcp|model context protocol)\b",
    re.I,
)
NON_AI_AGENT_PREFIX = re.compile(
    r"\b(real estate|travel|insurance|escrow|title|customer|support|call center|"
    r"chat|live|human|sales|booking|monitoring|telemetry|datadog|jenkins|build|ci|user)\s+agents?\b",
    re.I,
)


def strip_html(raw: str) -> str:
    txt = re.sub(
        r"<a\s+[^>]*>.*?</a>",
        lambda m: " " + re.sub(r"<[^>]+>", "", m.group(0)) + " ",
        raw,
        flags=re.I | re.S,
    )
    txt = re.sub(r"<[^>]+>", " ", txt)
    return html.unescape(txt)


def is_ai_agent(visible: str) -> bool:
    cleaned = NON_AI_AGENT_PREFIX.sub(" ", visible)
    if re.search(
        r"\b(agentic|mcp|model context protocol|ai agents?|llm agents?|"
        r"autonomous agents?|coding agents?|voice agents?|multi-agent)\b",
        cleaned,
        re.I,
    ):
        return True
    if re.search(r"\bagents?\b", cleaned, re.I) and AI_RE.search(cleaned):
        return True
    return False


def curl_json(url: str) -> dict:
    for _ in range(4):
        p = subprocess.run(["curl", "-sL", "--retry", "3", url], capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip().startswith("{"):
            return json.loads(p.stdout)
    raise RuntimeError(f"Failed to fetch {url}")


def main() -> None:
    rows = []
    for yr in range(2014, 2027):
        res = curl_json(
            f"https://hn.algolia.com/api/v1/search?tags=story,author_whoishiring&query=November+{yr}"
        )
        iid = next(
            h["objectID"]
            for h in res.get("hits", [])
            if re.search(rf"Who is hiring\?\s*\(November\s+{yr}\)", h.get("title", ""), re.I)
        )
        it = curl_json(f"https://hn.algolia.com/api/v1/items/{iid}")
        tops = [c for c in it.get("children", []) if c.get("text")]
        n = len(tops)
        vis_list = [strip_html(c["text"]) for c in tops]
        ai_cnt = sum(1 for v in vis_list if AI_RE.search(v))
        ag_cnt = sum(1 for v in vis_list if is_ai_agent(v))
        rows.append(
            {
                "year": yr,
                "item_id": iid,
                "top_posts": n,
                "ai_posts": ai_cnt,
                "ai_pct": round(ai_cnt / n * 100, 1),
                "agent_posts": ag_cnt,
                "agent_pct": round(ag_cnt / n * 100, 1),
            }
        )
    out = Path(__file__).with_name("hn_nov_hiring_2014_2026.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} years to {out}")


if __name__ == "__main__":
    main()
