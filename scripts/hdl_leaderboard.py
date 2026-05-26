#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27,<1.0",
# ]
# ///
"""HDL leaderboard generator.

Discovers Verilog / SystemVerilog / VHDL repositories on GitHub, enriches
them with star / fork / commit metrics, and writes CSV + Markdown (and
optionally JSON) per dataset:

  * main top-N by stars across the surveyed languages
  * HFT/quant subset filtered from the main top-N by keyword
  * HFT/quant targeted search (per keyword × language) regardless of stars

Run with `uv` (recommended; auto-installs httpx in an ephemeral env):

    uv run scripts/hdl_leaderboard.py                       # defaults
    uv run scripts/hdl_leaderboard.py --top-n 100           # smaller main
    uv run scripts/hdl_leaderboard.py --no-hft-targeted     # skip live search
    uv run scripts/hdl_leaderboard.py --formats csv,md,json # add JSON output

Authentication priority: --token CLI flag > GH_TOKEN/GITHUB_TOKEN env >
`gh auth token` subprocess. Falls back to clear failure if none works.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx

# --- configuration --------------------------------------------------------

HDL_LANGUAGES = ["Verilog", "SystemVerilog", "VHDL"]

# Default keywords for the targeted HFT/quant search. Multi-word entries
# are auto-wrapped in quotes by `quote_phrase()` so they become GitHub
# phrase queries.
DEFAULT_HFT_KEYWORDS = [
    "trading",
    "high frequency",
    "market data",
    "matching engine",
    "FIX protocol",
    "ITCH",
    "HFT",
    "low latency",
]

# Regex used to flag main-top-N repos as HFT-related from their name/desc.
HFT_FILTER_RE = re.compile(
    r"(?i)(trading|high[- ]?frequency|market[- ]data|matching[- ]engine"
    r"|\bFIX\b|\bITCH\b|\bOUCH\b|\bHFT\b|quant|low[- ]latency|exchange)"
)

# GitHub owner / repo names: alphanumerics plus . _ -
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

API_BASE = "https://api.github.com"
USER_AGENT = "hdl-leaderboard/1.0 (+https://github.com/meefs/fork-repo-sync)"

# Throttling. Search endpoint is hard-capped at 30 req/min for an
# authenticated user; 2.0s spacing keeps us right at that ceiling without
# wasting headroom. GraphQL has its own 5000-points/hr budget, vastly
# more than we need, so a small concurrency limit is plenty.
SEARCH_MIN_INTERVAL_S = 2.0
GRAPHQL_CONCURRENCY = 4
ENRICH_BATCH_SIZE = 50          # repos per GraphQL request (aliased)
HTTP_TIMEOUT_S = 30.0
RETRY_MAX_ATTEMPTS = 4

# Hard cap on the underlying search endpoint (GitHub-side; not ours).
GITHUB_SEARCH_RESULT_CAP = 1000


# --- domain types ---------------------------------------------------------

@dataclass(slots=True)
class Repo:
    """One repository row across all phases of the pipeline."""

    full_name: str
    stars: int
    forks: int
    created_at: str | None
    pushed_at: str | None
    updated_at: str | None
    description: str | None
    language: str | None
    url: str
    commits: int = 0

    @property
    def owner(self) -> str:
        return self.full_name.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.full_name.split("/", 1)[1]

    @classmethod
    def from_search_item(cls, item: dict) -> "Repo":
        return cls(
            full_name=item["full_name"],
            stars=item.get("stargazers_count") or 0,
            forks=item.get("forks_count") or 0,
            created_at=item.get("created_at"),
            pushed_at=item.get("pushed_at"),
            updated_at=item.get("updated_at"),
            description=item.get("description"),
            language=item.get("language"),
            url=item.get("html_url") or f"https://github.com/{item['full_name']}",
        )


# --- auth -----------------------------------------------------------------

def resolve_token(cli_token: str | None) -> str:
    """Return a GitHub token from CLI / env / `gh auth token`, or exit."""
    if cli_token:
        return cli_token
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v
    try:
        out = subprocess.run(
            ["gh", "auth", "token"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        tok = out.stdout.strip()
        if tok:
            return tok
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    sys.exit(
        "error: no GitHub token found. Set GH_TOKEN/GITHUB_TOKEN or run "
        "`gh auth login`, or pass --token."
    )


# --- query helpers --------------------------------------------------------

def quote_phrase(kw: str) -> str:
    """Wrap multi-word keywords in double quotes for GitHub phrase search."""
    kw = kw.strip()
    if " " in kw and not (kw.startswith('"') and kw.endswith('"')):
        return f'"{kw}"'
    return kw


# --- GitHub client --------------------------------------------------------

class GitHubClient:
    """Async REST + GraphQL client with retries and per-endpoint throttling."""

    def __init__(self, token: str, *, log=print) -> None:
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": USER_AGENT,
            },
            timeout=HTTP_TIMEOUT_S,
        )
        self._search_lock = asyncio.Lock()
        self._last_search_at: float = 0.0
        self._gql_sem = asyncio.Semaphore(GRAPHQL_CONCURRENCY)
        self._log = log

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self._client.aclose()

    # ---- low-level request with retries + rate-limit awareness ----------

    async def _request(self, method: str, url: str, **kw) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                resp = await self._client.request(method, url, **kw)
            except httpx.HTTPError as e:
                last_exc = e
                if attempt + 1 == RETRY_MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(2 ** attempt)
                continue

            # Primary rate-limit: respect X-RateLimit-Reset
            if (
                resp.status_code == 403
                and resp.headers.get("X-RateLimit-Remaining") == "0"
            ):
                reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
                wait = max(1, reset - int(time.time()) + 1)
                # Don't sleep ridiculous amounts; cap at 2 minutes
                wait = min(wait, 120)
                self._log(f"  rate-limited; sleeping {wait}s")
                await asyncio.sleep(wait)
                continue

            # Secondary rate-limit: honor Retry-After
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "5"))
                self._log(f"  429 throttled; sleeping {wait}s")
                await asyncio.sleep(min(wait, 60))
                continue

            # Transient 5xx
            if resp.status_code >= 500 and attempt + 1 < RETRY_MAX_ATTEMPTS:
                await asyncio.sleep(2 ** attempt)
                continue

            return resp

        # If we exit the loop without returning, surface the last exception
        if last_exc:
            raise last_exc
        return resp  # type: ignore[return-value]

    # ---- public: search REST --------------------------------------------

    async def search_repos(
        self,
        query: str,
        *,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 100,
        max_results: int = GITHUB_SEARCH_RESULT_CAP,
    ) -> list[Repo]:
        """Run a search; paginate until exhausted or max_results reached."""
        max_results = min(max_results, GITHUB_SEARCH_RESULT_CAP)
        max_pages = (max_results + per_page - 1) // per_page
        out: list[Repo] = []

        for page in range(1, max_pages + 1):
            # Throttle: 30 req/min ceiling, measured start-to-start so
            # slow API calls don't compound the spacing.
            async with self._search_lock:
                gap = time.monotonic() - self._last_search_at
                if gap < SEARCH_MIN_INTERVAL_S:
                    await asyncio.sleep(SEARCH_MIN_INTERVAL_S - gap)
                self._last_search_at = time.monotonic()
                resp = await self._request(
                    "GET", "/search/repositories",
                    params={
                        "q": query, "sort": sort, "order": order,
                        "per_page": per_page, "page": page,
                    },
                )

            if resp.status_code != 200:
                self._log(
                    f"  search HTTP {resp.status_code} q={query!r}: "
                    f"{resp.text[:160]}"
                )
                break

            items = resp.json().get("items", [])
            if not items:
                break
            out.extend(Repo.from_search_item(it) for it in items)
            if len(out) >= max_results or len(items) < per_page:
                break

        return out[:max_results]

    # ---- public: commit-count enrichment via GraphQL --------------------

    async def graphql_commit_counts(self, repos: list[Repo]) -> dict[str, int]:
        """Return {full_name: commit_count} for the given repos.

        Sends `ceil(len(repos) / ENRICH_BATCH_SIZE)` GraphQL requests in
        parallel (bounded by GRAPHQL_CONCURRENCY). Each request bundles up
        to ENRICH_BATCH_SIZE repos via aliased `repository(...)` fields.
        """
        if not repos:
            return {}

        results: dict[str, int] = {}
        batches = [
            repos[i:i + ENRICH_BATCH_SIZE]
            for i in range(0, len(repos), ENRICH_BATCH_SIZE)
        ]

        async def fetch_one(batch: list[Repo]) -> None:
            async with self._gql_sem:
                query = self._build_commit_query(batch)
                if not query:
                    return
                resp = await self._request(
                    "POST", "/graphql", json={"query": query},
                )
                if resp.status_code != 200:
                    self._log(
                        f"  graphql HTTP {resp.status_code}: "
                        f"{resp.text[:160]}"
                    )
                    return
                payload = resp.json()
                if "errors" in payload:
                    for err in payload["errors"]:
                        self._log(f"  graphql error: {err.get('message', '?')}")
                data = payload.get("data") or {}
                for i, repo in enumerate(batch):
                    node = data.get(f"r{i}")
                    total = _extract_commit_count(node)
                    if total is not None:
                        results[repo.full_name] = total

        await asyncio.gather(*(fetch_one(b) for b in batches))
        return results

    @staticmethod
    def _build_commit_query(batch: list[Repo]) -> str:
        """Construct one aliased GraphQL document for a batch of repos.

        Repos whose owner or name fail the safe-name regex are skipped
        (their commit count silently defaults to 0 downstream).
        """
        parts: list[str] = []
        for i, r in enumerate(batch):
            if not (SAFE_NAME_RE.match(r.owner) and SAFE_NAME_RE.match(r.name)):
                continue
            parts.append(
                f'  r{i}: repository(owner: "{r.owner}", name: "{r.name}") {{\n'
                f'    defaultBranchRef {{ target {{ ... on Commit '
                f'{{ history {{ totalCount }} }} }} }}\n'
                f'  }}'
            )
        if not parts:
            return ""
        return "query {\n" + "\n".join(parts) + "\n}"


def _extract_commit_count(node) -> int | None:
    """Walk the GraphQL repository response to a commit count, or None."""
    if not isinstance(node, dict):
        return None
    ref = node.get("defaultBranchRef")
    if not isinstance(ref, dict):
        return None
    target = ref.get("target")
    if not isinstance(target, dict):
        return None
    hist = target.get("history")
    if not isinstance(hist, dict):
        return None
    total = hist.get("totalCount")
    return total if isinstance(total, int) else None


# --- pipeline phases ------------------------------------------------------

async def fetch_main_pool(
    client: GitHubClient, languages: list[str], limit_per_lang: int,
    log,
) -> list[Repo]:
    """Top-N per language, deduped across languages (keep highest stars)."""
    pool: dict[str, Repo] = {}
    for lang in languages:
        log(f"  language:{lang}")
        repos = await client.search_repos(
            f"language:{lang}", max_results=limit_per_lang,
        )
        log(f"    got {len(repos)}")
        for r in repos:
            existing = pool.get(r.full_name)
            if existing is None or r.stars > existing.stars:
                pool[r.full_name] = r
    return sorted(pool.values(), key=lambda r: -r.stars)


async def fetch_keyword_targeted(
    client: GitHubClient, keywords: list[str], languages: list[str],
    log,
) -> list[Repo]:
    """One search per (keyword × language), deduped & ranked by stars."""
    pool: dict[str, Repo] = {}
    for kw_raw in keywords:
        kw = quote_phrase(kw_raw)
        for lang in languages:
            q = f"{kw} language:{lang}"
            log(f"  {q}")
            repos = await client.search_repos(q, max_results=100)
            log(f"    +{len(repos)}")
            for r in repos:
                existing = pool.get(r.full_name)
                if existing is None or r.stars > existing.stars:
                    pool[r.full_name] = r
    return sorted(pool.values(), key=lambda r: -r.stars)


def filter_by_pattern(repos: Iterable[Repo], pattern: re.Pattern) -> list[Repo]:
    return [
        r for r in repos
        if pattern.search((r.description or "") + " " + r.full_name)
    ]


async def enrich_commits(
    client: GitHubClient, repos: list[Repo], log,
) -> None:
    """Populate `.commits` on each Repo using batched GraphQL."""
    if not repos:
        return
    log(f"  enriching {len(repos)} repos "
        f"(batches of {ENRICH_BATCH_SIZE}, concurrency {GRAPHQL_CONCURRENCY})")
    counts = await client.graphql_commit_counts(repos)
    log(f"    got {len(counts)}/{len(repos)} commit counts")
    for r in repos:
        r.commits = counts.get(r.full_name, 0)


# --- output ---------------------------------------------------------------

def _truncate(s: str | None, n: int) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def write_csv(path: Path, repos: list[Repo]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "full_name", "primary_language", "stars", "forks",
            "commits", "created_at", "pushed_at", "updated_at",
            "url", "description",
        ])
        for i, r in enumerate(repos, 1):
            w.writerow([
                i, r.full_name, r.language or "",
                r.stars, r.forks, r.commits,
                (r.created_at or "")[:10],
                (r.pushed_at or "")[:10],
                (r.updated_at or "")[:10],
                r.url,
                (r.description or "").replace("\n", " "),
            ])


def write_markdown(
    path: Path, title: str, description: str, repos: list[Repo],
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {title}",
        "",
        f"Snapshot: {now}  ",
        description,
        "",
        f"**Total: {len(repos)} repos**",
        "",
        "| # | Repo | Lang | ★ | ⑂ | Commits | Created | Pushed | Updated | Description |",
        "|--:|------|------|--:|--:|--------:|---------|--------|---------|-------------|",
    ]
    for i, r in enumerate(repos, 1):
        desc = _truncate(r.description, 80).replace("|", "\\|")
        lines.append(
            f"| {i} | [{r.full_name}]({r.url}) | {r.language or ''} "
            f"| {r.stars:,} | {r.forks:,} | {r.commits:,} "
            f"| {(r.created_at or '')[:10]} "
            f"| {(r.pushed_at or '')[:10]} "
            f"| {(r.updated_at or '')[:10]} | {desc} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, repos: list[Repo]) -> None:
    payload = [
        {
            "full_name": r.full_name, "primary_language": r.language,
            "stars": r.stars, "forks": r.forks, "commits": r.commits,
            "created_at": r.created_at,
            "pushed_at": r.pushed_at,
            "updated_at": r.updated_at,
            "url": r.url,
            "description": r.description,
        }
        for r in repos
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_outputs(
    out_dir: Path, base: str, title: str, description: str,
    repos: list[Repo], formats: list[str], log,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "csv" in formats:
        write_csv(out_dir / f"{base}.csv", repos)
    if "md" in formats:
        write_markdown(out_dir / f"{base}.md", title, description, repos)
    if "json" in formats:
        write_json(out_dir / f"{base}.json", repos)
    log(f"  wrote {base}.{{{','.join(formats)}}}  ({len(repos)} rows)")


# --- CLI ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate FPGA/HDL repo leaderboards from GitHub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--top-n", type=int, default=500,
        help="Number of repos in main leaderboard (after dedup across langs)",
    )
    p.add_argument(
        "--per-language-limit", type=int, default=1000,
        help="Pre-dedup top results per language for the main pool "
             "(GitHub caps at 1000)",
    )
    p.add_argument(
        "--languages", default=",".join(HDL_LANGUAGES),
        help="Comma-separated HDL languages",
    )
    p.add_argument(
        "--hft-keywords", default=",".join(DEFAULT_HFT_KEYWORDS),
        help="Comma-separated keywords for HFT/quant targeted search. "
             "Multi-word keywords are auto-quoted as phrase queries.",
    )
    p.add_argument(
        "--no-main", action="store_true",
        help="Skip the main top-N leaderboard",
    )
    p.add_argument(
        "--no-hft-from-main", action="store_true",
        help="Skip the HFT-filtered-from-top-N subset",
    )
    p.add_argument(
        "--no-hft-targeted", action="store_true",
        help="Skip the targeted HFT keyword search",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("leaderboards"),
        help="Where to write output files",
    )
    p.add_argument(
        "--formats", default="csv,md",
        help="Comma-separated output formats: csv,md,json",
    )
    p.add_argument("--token", default=None, help="GitHub token (overrides env)")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output")
    return p.parse_args()


async def amain() -> None:
    args = parse_args()
    log = (lambda *_a, **_k: None) if args.quiet else print

    token = resolve_token(args.token)
    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    keywords = [s.strip() for s in args.hft_keywords.split(",") if s.strip()]
    formats = [s.strip().lower() for s in args.formats.split(",") if s.strip()]

    skip_main_pool = args.no_main and args.no_hft_from_main
    do_targeted = not args.no_hft_targeted

    t0 = time.monotonic()
    async with GitHubClient(token, log=log) as client:
        main_pool: list[Repo] = []
        main_top: list[Repo] = []
        hft_targeted: list[Repo] = []

        # ---- Phase 1: main pool ---------------------------------------
        if not skip_main_pool:
            log(f"=== main pool: top {args.per_language_limit} per language ===")
            t = time.monotonic()
            main_pool = await fetch_main_pool(
                client, languages, args.per_language_limit, log,
            )
            main_top = main_pool[: args.top_n]
            log(f"  {len(main_pool)} unique → top {len(main_top)} "
                f"({time.monotonic() - t:.1f}s)")

        # ---- Phase 2: targeted HFT search -----------------------------
        if do_targeted:
            log(f"=== targeted HFT search: "
                f"{len(keywords)} kw × {len(languages)} lang ===")
            t = time.monotonic()
            hft_targeted = await fetch_keyword_targeted(
                client, keywords, languages, log,
            )
            log(f"  {len(hft_targeted)} unique repos "
                f"({time.monotonic() - t:.1f}s)")

        # ---- Phase 3: enrich (dedup across datasets) -----------------
        to_enrich: dict[str, Repo] = {}
        for r in main_top:
            to_enrich.setdefault(r.full_name, r)
        for r in hft_targeted:
            to_enrich.setdefault(r.full_name, r)

        if to_enrich:
            log(f"=== enriching {len(to_enrich)} unique repos ===")
            t = time.monotonic()
            await enrich_commits(client, list(to_enrich.values()), log)
            log(f"  ({time.monotonic() - t:.1f}s)")
            # Propagate commit counts to any duplicate Repo instances
            # living in either output list
            for r in main_top:
                r.commits = to_enrich[r.full_name].commits
            for r in hft_targeted:
                r.commits = to_enrich[r.full_name].commits

        # ---- Phase 4: write outputs ----------------------------------
        out = args.output_dir
        if not args.no_main and main_top:
            write_outputs(
                out, "fpga-hdl-top500",
                f"FPGA / HDL Top {args.top_n} by Stars",
                f"Top {args.top_n} across {', '.join(languages)}, "
                f"deduped, ranked by stars.",
                main_top, formats, log,
            )
        if not args.no_hft_from_main and main_top:
            subset = filter_by_pattern(main_top, HFT_FILTER_RE)
            write_outputs(
                out, "fpga-hdl-hft-quant",
                "FPGA / HDL — HFT & Quant Subset (top 500 only)",
                "Repos within the main top-N whose name or description "
                "matches HFT/quant keywords.",
                subset, formats, log,
            )
        if do_targeted and hft_targeted:
            write_outputs(
                out, "fpga-hdl-hft-targeted",
                "FPGA / HDL — Targeted HFT & Quant Search",
                "Direct GitHub search per (keyword × language). All star "
                "counts included.",
                hft_targeted, formats, log,
            )

    log(f"=== done in {time.monotonic() - t0:.1f}s ===")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
