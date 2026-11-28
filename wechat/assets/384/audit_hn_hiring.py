#!/usr/bin/env python3
"""Fetch and audit November 'Ask HN: Who is hiring?' threads (2021-2026) via official Algolia HN API."""
import json
import re
import subprocess

AI_RE = re.compile(
    r"\b(ai|artificial intelligence|llm|llms|gpt|claude|genai|agentic|ai agent|ai agents|machine learning|deep learning|mcp)\b",
    re.I,
)
AGENT_RE = re.compile(r"\b(agent|agents|agentic|mcp)\b", re.I)


def curl_json(url: str) -> dict:
    for _ in range(4):
        p = subprocess.run(["curl", "-sL", "--retry", "3", url], capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip().startswith("{"):
            return json.loads(p.stdout)
    raise RuntimeError(f"Failed to fetch {url}")


def main() -> None:
    rows = []
    for yr in [2021, 2022, 2023, 2024, 2025, 2026]:
        res = curl_json(f"https://hn.algolia.com/api/v1/search?tags=story,author_whoishiring&query=November+{yr}")
        iid = next(
            h["objectID"]
            for h in res.get("hits", [])
            if f"Who is hiring? (November {yr})" in h.get("title", "")
        )
        it = curl_json(f"https://hn.algolia.com/api/v1/items/{iid}")
        tops = [c for c in it.get("children", []) if c.get("text")]
        n = len(tops)
        ai_cnt = sum(1 for c in tops if AI_RE.search(c["text"]))
        agent_cnt = sum(1 for c in tops if AGENT_RE.search(c["text"]))
        rows.append(
            {
                "year": yr,
                "item_id": iid,
                "top_posts": n,
                "ai_posts": ai_cnt,
                "ai_pct": round(ai_cnt / n * 100, 1),
                "agent_posts": agent_cnt,
                "agent_pct": round(agent_cnt / n * 100, 1),
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
