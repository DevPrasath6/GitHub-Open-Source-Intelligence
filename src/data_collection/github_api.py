"""GitHub REST API data collection (public data only).

Run:  python -m src.data_collection.github_api --keyword "machine learning" \
          --language Python --min-stars 100 --limit 500
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import DATA_RAW, RAW_COLUMNS, setup_logging

logger = logging.getLogger(__name__)

BASE_URL = "https://api.github.com"
SEARCH_PAGE_SIZE = 100          # GitHub maximum per page
SEARCH_RESULT_CAP = 1000        # GitHub Search API never returns more than 1000 hits/query
MAX_WAIT_SECONDS = 3600         # never sleep longer than 1 hour for a rate limit


class GitHubClient:
    """Thin wrapper around ``requests`` with auth, retries and rate-limit handling."""

    def __init__(self, token: Optional[str] = None, max_retries: int = 6, timeout: int = 30):
        load_dotenv()  # reads GITHUB_TOKEN from .env if present - never hard-coded
        token = token or os.getenv("GITHUB_TOKEN")
        if token in (None, "", "your_token_here"):
            token = None
        self.has_token = token is not None
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-osint-datascience-educational",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        else:
            logger.warning("No GITHUB_TOKEN found: unauthenticated limit is only 60 requests/hour.")

    @staticmethod
    def _rate_limit_wait(resp: requests.Response) -> Optional[int]:
        """Seconds to wait if the response signals a rate limit, otherwise ``None``."""
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return int(retry_after) + 1
        if resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(resp.headers.get("X-RateLimit-Reset", 0))
            return max(reset - int(time.time()), 0) + 1
        return None

    def get(self, url: str, params: Optional[dict] = None) -> Optional[Any]:
        """GET a JSON resource. Returns ``None`` on permanent failure (never raises)."""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                wait = min(2 ** attempt, 60)
                logger.warning("Request failed (%s). Retry %d/%d in %ds", exc, attempt, self.max_retries, wait)
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                if resp.headers.get("X-RateLimit-Remaining") == "0":  # be polite: pause before next call
                    wait = self._rate_limit_wait(resp) or 0
                    logger.info("Rate limit exhausted - sleeping %ds", wait)
                    time.sleep(min(wait, MAX_WAIT_SECONDS))
                try:
                    return resp.json()
                except ValueError:
                    logger.error("Invalid JSON from %s", url)
                    return None

            if resp.status_code in (403, 429):
                wait = self._rate_limit_wait(resp)
                if wait is None:
                    logger.error("Forbidden (%s): %s", url, resp.json().get("message") if resp.content else "")
                    return None
                logger.warning("Rate limited - sleeping %ds", wait)
                time.sleep(min(wait, MAX_WAIT_SECONDS))
                continue
            if resp.status_code in (404, 410, 422, 451):
                logger.info("Skipping %s (HTTP %d)", url, resp.status_code)
                return None
            if resp.status_code >= 500:
                time.sleep(min(2 ** attempt, 60))
                continue
            logger.error("Unexpected HTTP %d for %s", resp.status_code, url)
            return None
        logger.error("Giving up on %s after %d attempts", url, self.max_retries)
        return None


# --------------------------------------------------------------------------- parsing
def parse_repository(item: dict) -> dict:
    """Convert one GitHub repository JSON object into a flat record.

    Missing fields become ``None`` instead of raising errors.
    NOTE: the Search API's ``watchers_count`` equals the star count, so the real
    watcher number comes from ``subscribers_count`` (only in the detail endpoint).
    """
    owner = item.get("owner") or {}
    lic = item.get("license") or {}
    return {
        "repository_name": item.get("name"),
        "full_name": item.get("full_name"),
        "owner": owner.get("login"),
        "description": item.get("description"),
        "language": item.get("language"),
        "stars": item.get("stargazers_count"),
        "forks": item.get("forks_count"),
        "watchers": item.get("subscribers_count"),
        "open_issues": item.get("open_issues_count"),
        "size": item.get("size"),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "pushed_at": item.get("pushed_at"),
        "license": lic.get("spdx_id") or lic.get("name"),
        "topics": ",".join(item.get("topics") or []),
        "default_branch": item.get("default_branch"),
        "archived": item.get("archived"),
        "is_fork": item.get("fork"),
        "visibility": item.get("visibility", "public"),
        "repository_url": item.get("html_url"),
        "owner_login": owner.get("login"),
        "owner_type": owner.get("type"),
    }


def build_query(keyword: str, language: Optional[str], min_stars: int, max_stars: Optional[int] = None) -> str:
    """Build a GitHub search query string."""
    parts = [keyword] if keyword else []
    if language:
        parts.append(f"language:{language}")
    parts.append(f"stars:>={min_stars}" if max_stars is None else f"stars:{min_stars}..{max_stars}")
    parts.append("is:public")
    return " ".join(parts)


# --------------------------------------------------------------------------- fetchers
def fetch_repositories(client: GitHubClient, keyword: str = "", language: Optional[str] = None,
                       min_stars: int = 100, limit: int = 500) -> list[dict]:
    """Search public repositories (paginated, de-duplicated).

    The Search API caps each query at 1000 results, so for larger ``limit`` values the
    star range is moved down window-by-window (sorted by stars, descending).
    """
    records: dict[str, dict] = {}
    max_stars: Optional[int] = None
    while len(records) < limit:
        query = build_query(keyword, language, min_stars, max_stars)
        new_in_window, lowest_stars, exhausted = 0, None, False
        for page in range(1, SEARCH_RESULT_CAP // SEARCH_PAGE_SIZE + 1):
            data = client.get(f"{BASE_URL}/search/repositories", {
                "q": query, "sort": "stars", "order": "desc",
                "per_page": SEARCH_PAGE_SIZE, "page": page})
            items = (data or {}).get("items") or []
            if not items:
                exhausted = True
                break
            for item in items:
                lowest_stars = item.get("stargazers_count", lowest_stars)
                repo = parse_repository(item)
                if repo["full_name"] and repo["full_name"] not in records:  # duplicate guard
                    records[repo["full_name"]] = repo
                    new_in_window += 1
                if len(records) >= limit:
                    break
            logger.info("Query page %d -> %d repositories so far", page, len(records))
            if len(records) >= limit or len(items) < SEARCH_PAGE_SIZE or page * SEARCH_PAGE_SIZE >= data.get("total_count", 0):
                exhausted = page * SEARCH_PAGE_SIZE >= data.get("total_count", 0) or len(items) < SEARCH_PAGE_SIZE
                break
        if exhausted or new_in_window == 0 or lowest_stars is None or lowest_stars <= min_stars:
            break
        max_stars = lowest_stars  # next window starts at the lowest star count seen
    return list(records.values())[:limit]


def fetch_repository_details(client: GitHubClient, full_name: str) -> dict:
    """Fetch the detail endpoint (adds the true ``watchers`` = subscribers_count)."""
    data = client.get(f"{BASE_URL}/repos/{full_name}")
    return parse_repository(data) if data else {}


def fetch_user_details(client: GitHubClient, login: str, cache: Optional[dict] = None) -> dict:
    """Fetch public profile statistics of a repository owner (cached per login)."""
    cache = cache if cache is not None else {}
    if login in cache:
        return cache[login]
    data = client.get(f"{BASE_URL}/users/{login}") or {}
    result = {
        "owner_type": data.get("type"),
        "followers": data.get("followers"),
        "public_repositories": data.get("public_repos"),
        "account_created_at": data.get("created_at"),
    }
    cache[login] = result
    return result


def collect_dataset(keyword: str = "machine learning", language: Optional[str] = "Python",
                    min_stars: int = 100, limit: int = 500, enrich_details: bool = True,
                    enrich_users: bool = True, token: Optional[str] = None) -> pd.DataFrame:
    """Full collection workflow -> DataFrame with the raw dataset columns."""
    client = GitHubClient(token)
    if not client.has_token and (enrich_details or enrich_users):
        logger.warning("Enrichment disabled (needs a token; otherwise ~60 calls/hour). "
                       "'watchers' will be empty and the model trains without it.")
        enrich_details = enrich_users = False

    repos = fetch_repositories(client, keyword, language, min_stars, limit)
    logger.info("Collected %d repositories from search", len(repos))

    if enrich_details:
        for i, repo in enumerate(repos, 1):
            details = fetch_repository_details(client, repo["full_name"])
            for key in ("watchers", "topics", "license", "description"):
                if details.get(key) not in (None, ""):
                    repo[key] = details[key]
            if i % 25 == 0:
                logger.info("Repository details: %d/%d", i, len(repos))

    if enrich_users:
        cache: dict = {}
        for i, repo in enumerate(repos, 1):
            if repo.get("owner_login"):
                repo.update({k: v for k, v in fetch_user_details(client, repo["owner_login"], cache).items()
                             if v is not None})
            if i % 50 == 0:
                logger.info("Owner details: %d/%d", i, len(repos))

    df = pd.DataFrame(repos)
    df["collected_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return df.reindex(columns=RAW_COLUMNS)


def save_raw_data(df: pd.DataFrame, path: Path = DATA_RAW, source: str = "live") -> Path:
    """Write the raw dataset to CSV (creates folders if needed) and record its source (live/sample)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    path.with_name("data_source.txt").write_text(source)
    logger.info("Saved %d rows x %d columns to %s", len(df), df.shape[1], path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public GitHub repository data.")
    parser.add_argument("--keyword", default="machine learning")
    parser.add_argument("--language", default="Python")
    parser.add_argument("--min-stars", type=int, default=100)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--no-enrich-details", action="store_true", help="skip per-repo detail calls")
    parser.add_argument("--no-enrich-users", action="store_true", help="skip per-owner profile calls")
    parser.add_argument("--sample", action="store_true", help="generate a synthetic dataset instead of calling the API")
    args = parser.parse_args()
    setup_logging()

    if args.sample:
        from src.data_collection.sample_data import generate_sample_dataset
        save_raw_data(generate_sample_dataset(max(args.limit, 300)), source="sample")
        return
    df = collect_dataset(args.keyword, args.language or None, args.min_stars, args.limit,
                         not args.no_enrich_details, not args.no_enrich_users)
    if df.empty:
        logger.error("No data collected. Check your network/token, or run with --sample.")
        raise SystemExit(1)
    save_raw_data(df)


if __name__ == "__main__":
    main()
