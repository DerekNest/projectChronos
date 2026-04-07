"""
agents/research_agent.py  —  PYTHON SCRIPT (async i/o, deterministic scraping)

architecture note: no llm calls, no fuzzy logic. purely deterministic i/o —
fetches and parses articles from rss feeds and reddit, returning a structured
ResearchPayload. the fuzzy sentiment analysis happens downstream in the research
SKILL (orchestrator._call_research_skill), not here.

sources:
  tier 1 (rss, no key required):
    reuters     https://feeds.reuters.com/reuters/topNews
    bbc         http://feeds.bbci.co.uk/news/rss.xml
    ap news     https://rsshub.app/apnews/topics/apf-topnews
    politico    https://www.politico.com/rss/politicopicks.xml
    coindesk    https://www.coindesk.com/arc/outboundfeeds/rss/
    cryptoslate https://cryptoslate.com/feed/

  tier 2 (reddit public json api, no key required):
    r/politics, r/news, r/worldnews, r/CryptoCurrency, r/Bitcoin

  newsapi (optional, key-gated):
    if NEWSAPI_KEY env var is set, also queries newsapi.org for the market
    question directly. higher recall than rss but requires paid key for
    >100 req/day. falls back gracefully if key is absent.

keyword extraction:
  we don't query rss feeds with the raw question text — feeds don't support
  keyword filtering. instead we extract 2-4 salient tokens from the question
  and use them to filter articles whose title or body contains those terms.

  extraction heuristic:
    - strip stop words and question scaffolding ("will", "the", "be", "in", etc.)
    - keep proper nouns (capitalized), numbers, and high-signal verbs
    - prefer shorter queries (2 tokens) to avoid over-filtering
    this is intentionally simple — a 2-token filter on 300+ rss articles
    gives better recall than a 6-token filter that matches nothing.

security:
  article content is returned as-is to the orchestrator, which wraps it in
  <article> tags and labels it UNTRUSTED EXTERNAL DATA before passing to the
  LLM. the prompt injection defense lives in _call_research_skill(), not here.
  gather_research() makes no attempt to sanitize or interpret article content.

concurrency:
  all source fetches run via asyncio.gather — total latency = slowest source,
  not sum. rss feeds typically respond in 200-800ms; reddit ~300ms.
  per-source timeout of SOURCE_TIMEOUT_S prevents one slow source from
  blocking the pipeline.
"""

import asyncio
import html
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from xml.etree import ElementTree as ET

import aiohttp

logger = logging.getLogger(__name__)

# ── parameters ─────────────────────────────────────────────────────────────────

SOURCE_TIMEOUT_S    = 8      # per-source fetch timeout
MAX_ARTICLES_TOTAL  = 30     # cap passed to research skill (token budget)
MAX_BODY_CHARS      = 600    # truncate article body to this length before returning
MIN_KEYWORD_LEN     = 3      # ignore extracted tokens shorter than this
REDDIT_LIMIT        = 25     # posts per subreddit search

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
NEWSAPI_KEY  = os.environ.get("NEWSAPI_KEY", "")

# rss sources with their tier label (used by research skill for source weighting)
RSS_SOURCES = [
    ("https://feeds.reuters.com/reuters/topNews",              "reuters",     "rss"),
    ("http://feeds.bbci.co.uk/news/rss.xml",                   "bbc",         "rss"),
    ("https://www.politico.com/rss/politicopicks.xml",         "politico",    "rss"),
    ("https://feeds.feedburner.com/typepad/alleyinsider/silicon_alley_insider",
                                                               "businessinsider", "rss"),
    ("https://www.coindesk.com/arc/outboundfeeds/rss/",        "coindesk",    "rss"),
    ("https://cryptoslate.com/feed/",                          "cryptoslate", "rss"),
]

REDDIT_SUBS = ["politics", "news", "worldnews", "CryptoCurrency", "Bitcoin"]

# tokens to strip when extracting keywords from question text
_STOP_WORDS = frozenset([
    "will", "the", "be", "is", "are", "was", "were", "a", "an", "in", "on",
    "at", "to", "for", "of", "and", "or", "by", "with", "that", "this",
    "from", "have", "has", "had", "do", "does", "did", "not", "no", "yes",
    "more", "than", "up", "over", "under", "between", "which", "who",
    "when", "where", "how", "what", "why", "before", "after", "during",
    "its", "it", "he", "she", "they", "we", "you", "i", "us", "them",
    "their", "his", "her", "our", "your", "any", "all", "most", "least",
    "new", "old", "would", "could", "should", "may", "might", "can",
    "win", "lose", "get", "make", "take", "into",
    "reach", "sign", "hit", "exceed", "pass", "fall", "rise", "drop",
    "become", "remain", "continue", "return", "stay", "go", "come",
])


# ── data contracts ─────────────────────────────────────────────────────────────

@dataclass
class Article:
    """
    single scraped article. source_domain and source_type are used by the
    research SKILL for credibility weighting (reuters > reddit).
    published_ts enables recency weighting (< 2h old gets 2x weight in SKILL).
    """
    title:        str
    body:         str
    source_domain: str   # e.g. "reuters", "bbc", "reddit/r/politics"
    source_type:  str    # "rss" | "reddit" | "newsapi"
    url:          str
    published_ts: float  # unix timestamp (0.0 if unknown)


@dataclass
class ResearchPayload:
    """
    output of gather_research(). consumed by _call_research_skill() in orchestrator.
    articles are ordered by recency (newest first) so the LLM sees the most
    recent information first within its context window.
    """
    market_id:       str
    market_question: str
    market_yes_price: float
    articles:        list[Article] = field(default_factory=list)
    fetch_ts:        float = field(default_factory=time.time)
    sources_queried: list[str] = field(default_factory=list)


# ── keyword extraction ─────────────────────────────────────────────────────────

def extract_keywords(question: str, max_keywords: int = 4) -> list[str]:
    """
    extracts 2-4 salient search keywords from a market question.

    strategy:
      1. strip punctuation and split on whitespace
      2. remove stop words and short tokens
      3. prefer capitalized tokens (proper nouns) — these are usually the
         most specific and discriminating terms
      4. fall back to any remaining tokens if not enough proper nouns found
      5. return up to max_keywords tokens

    examples:
      "Will Trump sign the tariff bill by June?" -> ["Trump", "tariff", "bill"]
      "Will Bitcoin reach $100k in 2025?"        -> ["Bitcoin", "100k", "2025"]
      "Will the Fed cut rates in Q3?"            -> ["Fed", "cut", "rates"]
    """
    # strip question marks, commas, quotes, parentheses
    clean = re.sub(r"[?!,\"'()\[\]{}]", " ", question)
    tokens = clean.split()

    # separate proper nouns (capitalized, not start of sentence) from others
    proper = []
    common = []
    for i, tok in enumerate(tokens):
        tok_clean = re.sub(r"[^a-zA-Z0-9$%]", "", tok)
        if len(tok_clean) < MIN_KEYWORD_LEN:
            continue
        lower = tok_clean.lower()
        if lower in _STOP_WORDS:
            continue
        # treat as proper noun if capitalized and not the first word
        if i > 0 and tok_clean[0].isupper():
            proper.append(tok_clean)
        else:
            common.append(tok_clean.lower())

    # proper nouns first, then common, deduplicated, capped
    seen = set()
    keywords = []
    for tok in proper + common:
        if tok.lower() not in seen:
            seen.add(tok.lower())
            keywords.append(tok)
        if len(keywords) >= max_keywords:
            break

    logger.debug(f"[RESEARCH] keywords from '{question[:60]}': {keywords}")
    return keywords


def _matches_keywords(text: str, keywords: list[str]) -> bool:
    """
    returns True if the text contains at least one keyword (case-insensitive).
    using OR logic: any keyword match is sufficient to include the article.
    AND logic would over-filter — "Trump tariff bill" would miss articles
    about "Trump signs sweeping trade legislation".
    """
    if not keywords:
        return True
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


# ── rss fetcher ────────────────────────────────────────────────────────────────

def _parse_rss_datetime(date_str: str) -> float:
    """
    parses rfc-2822 and iso-8601 date strings to unix timestamp.
    returns 0.0 on any parse failure — the article is still included,
    just without recency weighting in the research skill.
    """
    if not date_str:
        return 0.0
    # try common formats
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",   # rfc-2822: Mon, 01 Jan 2024 12:00:00 +0000
        "%a, %d %b %Y %H:%M:%S GMT",   # rfc-2822 with literal GMT
        "%Y-%m-%dT%H:%M:%S%z",         # iso-8601
        "%Y-%m-%dT%H:%M:%SZ",          # iso-8601 zulu
    ):
        try:
            from datetime import datetime
            return datetime.strptime(date_str.strip(), fmt).timestamp()
        except ValueError:
            continue
    return 0.0


async def _fetch_rss(
    session: aiohttp.ClientSession,
    url: str,
    source_domain: str,
    source_type: str,
    keywords: list[str],
) -> list[Article]:
    """
    fetches and parses a single rss feed, filtering items by keyword relevance.
    returns [] on any fetch or parse failure — individual source failures are
    non-fatal; the pipeline continues with whatever other sources returned.
    """
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=SOURCE_TIMEOUT_S),
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
        ) as resp:
            if resp.status != 200:
                logger.debug(f"[RESEARCH] rss {source_domain} -> {resp.status}")
                return []
            raw = await resp.text(errors="replace")
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.debug(f"[RESEARCH] rss {source_domain} fetch error: {exc}")
        return []

    articles = []
    try:
        root = ET.fromstring(raw)
        # handle both <rss><channel><item> and <feed><entry> (atom)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item") or root.findall(".//atom:entry", ns)

        for item in items:
            # rss fields
            title_el = item.find("title")
            desc_el  = item.find("description") or item.find("atom:summary", ns)
            link_el  = item.find("link")
            date_el  = item.find("pubDate") or item.find("atom:published", ns)

            title = html.unescape(title_el.text or "") if title_el is not None else ""
            body  = html.unescape(desc_el.text  or "") if desc_el  is not None else ""
            url_  = (link_el.text or "").strip()        if link_el  is not None else ""
            date_ = (date_el.text or "").strip()        if date_el  is not None else ""

            # strip html tags from body (rss descriptions often contain markup)
            body = re.sub(r"<[^>]+>", " ", body).strip()
            body = re.sub(r"\s+", " ", body)[:MAX_BODY_CHARS]

            if not title:
                continue

            if not _matches_keywords(title + " " + body, keywords):
                continue

            articles.append(Article(
                title         = title[:200],
                body          = body,
                source_domain = source_domain,
                source_type   = source_type,
                url           = url_,
                published_ts  = _parse_rss_datetime(date_),
            ))

    except ET.ParseError as exc:
        logger.debug(f"[RESEARCH] rss parse error for {source_domain}: {exc}")

    logger.debug(f"[RESEARCH] {source_domain}: {len(articles)} relevant articles")
    return articles


# ── reddit fetcher ─────────────────────────────────────────────────────────────

async def _fetch_reddit(
    session: aiohttp.ClientSession,
    subreddit: str,
    keywords: list[str],
) -> list[Article]:
    """
    searches a subreddit using reddit's public json search endpoint.
    no api key required for read-only search. rate limit: ~30 req/min.

    uses the `q` param to search post titles and selftext within the subreddit.
    query is the joined keyword list — reddit's search handles OR/AND natively
    when terms are space-separated (it treats spaces as AND by default, which
    is fine since we pass 2-3 high-signal tokens).
    """
    if not keywords:
        return []

    query    = " ".join(keywords[:3])   # reddit search degrades past ~3 terms
    search_url = (
        f"https://www.reddit.com/r/{subreddit}/search.json"
        f"?q={query}&sort=new&limit={REDDIT_LIMIT}&restrict_sr=1&t=day"
    )

    try:
        async with session.get(
            search_url,
            timeout=aiohttp.ClientTimeout(total=SOURCE_TIMEOUT_S),
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
        ) as resp:
            if resp.status != 200:
                logger.debug(f"[RESEARCH] reddit r/{subreddit} -> {resp.status}")
                return []
            data = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as exc:
        logger.debug(f"[RESEARCH] reddit r/{subreddit} error: {exc}")
        return []

    articles = []
    try:
        posts = data.get("data", {}).get("children", [])
        for post in posts:
            p = post.get("data", {})
            title    = p.get("title", "")
            selftext = p.get("selftext", "")[:MAX_BODY_CHARS]
            url_     = p.get("url", "")
            created  = float(p.get("created_utc", 0))
            score    = p.get("score", 0)

            # skip low-engagement posts (likely spam or irrelevant)
            if score < 10:
                continue

            if not title:
                continue

            articles.append(Article(
                title         = title[:200],
                body          = selftext,
                source_domain = f"reddit/r/{subreddit}",
                source_type   = "reddit",
                url           = url_,
                published_ts  = created,
            ))
    except (KeyError, TypeError) as exc:
        logger.debug(f"[RESEARCH] reddit parse error for r/{subreddit}: {exc}")

    logger.debug(f"[RESEARCH] reddit r/{subreddit}: {len(articles)} posts")
    return articles


# ── newsapi fetcher (optional, key-gated) ──────────────────────────────────────

async def _fetch_newsapi(
    session: aiohttp.ClientSession,
    keywords: list[str],
    question: str,
) -> list[Article]:
    """
    queries newsapi.org for the market question. higher recall than rss since
    newsapi indexes hundreds of sources and supports full-text keyword search.
    only called if NEWSAPI_KEY env var is set — falls back gracefully if absent.
    """
    if not NEWSAPI_KEY:
        return []

    query = " ".join(keywords[:4]) if keywords else question[:100]

    try:
        async with session.get(
            NEWSAPI_BASE,
            params={
                "q":        query,
                "sortBy":   "publishedAt",
                "pageSize": "20",
                "language": "en",
                "apiKey":   NEWSAPI_KEY,
            },
            timeout=aiohttp.ClientTimeout(total=SOURCE_TIMEOUT_S),
        ) as resp:
            if resp.status != 200:
                logger.debug(f"[RESEARCH] newsapi -> {resp.status}")
                return []
            data = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.debug(f"[RESEARCH] newsapi error: {exc}")
        return []

    articles = []
    for item in data.get("articles", []):
        title   = item.get("title") or ""
        body    = (item.get("description") or item.get("content") or "")[:MAX_BODY_CHARS]
        url_    = item.get("url", "")
        source  = item.get("source", {}).get("name", "newsapi")
        pub_at  = item.get("publishedAt", "")

        if not title or title == "[Removed]":
            continue

        # parse iso-8601 published date
        ts = 0.0
        if pub_at:
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(pub_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass

        articles.append(Article(
            title         = title[:200],
            body          = body,
            source_domain = source.lower().replace(" ", "_"),
            source_type   = "newsapi",
            url           = url_,
            published_ts  = ts,
        ))

    logger.debug(f"[RESEARCH] newsapi: {len(articles)} articles")
    return articles


# ── assembly ───────────────────────────────────────────────────────────────────

def _deduplicate(articles: list[Article]) -> list[Article]:
    """
    removes near-duplicate articles by title similarity.
    simple approach: strip punctuation/case from title and deduplicate on
    the first 80 characters. this catches syndicated articles (same story
    republished verbatim across AP, reuters, local outlets) that would
    otherwise inflate the research skill's apparent consensus confidence.
    """
    seen: set[str] = set()
    deduped: list[Article] = []
    for a in articles:
        key = re.sub(r"[^a-z0-9 ]", "", a.title.lower())[:80].strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(a)
    return deduped


def _sort_by_recency(articles: list[Article]) -> list[Article]:
    """
    sorts articles newest-first. articles with unknown timestamps (ts=0)
    go to the end — we don't know when they were published so they get
    no recency benefit.
    """
    known   = [a for a in articles if a.published_ts > 0]
    unknown = [a for a in articles if a.published_ts == 0]
    known.sort(key=lambda a: a.published_ts, reverse=True)
    return known + unknown


# ── public entry point ─────────────────────────────────────────────────────────

async def gather_research(
    market_id:        str,
    market_question:  str,
    market_yes_price: float,
) -> ResearchPayload:
    """
    fetches articles from all configured sources in parallel and returns
    a ResearchPayload ready for _call_research_skill().

    concurrency: all rss feeds + all reddit subs + newsapi run simultaneously
    via asyncio.gather. total wall-clock time = slowest single source
    (typically ~1-2s), not the sum across all sources (~10-15s sequential).

    graceful degradation: any individual source failure returns [] and is
    logged at DEBUG level. the pipeline continues with whatever was collected.
    a payload with 0 articles will cause the research skill to return
    confidence=0.0, which triggers the orchestrator's early-exit check
    (skips to next market). this is the correct behavior — don't trade
    on markets with no intelligence signal.

    returns articles sorted newest-first, deduplicated, capped at MAX_ARTICLES_TOTAL.
    """
    keywords = extract_keywords(market_question)
    logger.info(
        f"[RESEARCH] {market_id} — keywords: {keywords} | "
        f"sources: {len(RSS_SOURCES)} rss + {len(REDDIT_SUBS)} reddit"
        + (" + newsapi" if NEWSAPI_KEY else "")
    )

    sources_queried: list[str] = []

    connector = aiohttp.TCPConnector(limit=20, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:

        # build all fetch tasks
        rss_tasks = [
            _fetch_rss(session, url, domain, stype, keywords)
            for url, domain, stype in RSS_SOURCES
        ]
        reddit_tasks = [
            _fetch_reddit(session, sub, keywords)
            for sub in REDDIT_SUBS
        ]
        newsapi_task = [_fetch_newsapi(session, keywords, market_question)]

        all_tasks = rss_tasks + reddit_tasks + newsapi_task

        # fire all concurrently — return_exceptions so one failure doesn't kill others
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

    # flatten results, skip exceptions
    raw_articles: list[Article] = []
    source_labels = (
        [d for _, d, _ in RSS_SOURCES]
        + [f"reddit/r/{s}" for s in REDDIT_SUBS]
        + (["newsapi"] if NEWSAPI_KEY else ["newsapi(no key)"])
    )

    for label, result in zip(source_labels, results):
        if isinstance(result, Exception):
            logger.debug(f"[RESEARCH] {label} exception: {result}")
        elif isinstance(result, list):
            raw_articles.extend(result)
            if result:
                sources_queried.append(label)

    # post-process: deduplicate, sort by recency, cap
    articles = _deduplicate(raw_articles)
    articles = _sort_by_recency(articles)
    articles = articles[:MAX_ARTICLES_TOTAL]

    logger.info(
        f"[RESEARCH] {market_id} — collected {len(raw_articles)} raw, "
        f"{len(articles)} after dedup/cap from {len(sources_queried)} sources"
    )

    return ResearchPayload(
        market_id        = market_id,
        market_question  = market_question,
        market_yes_price = market_yes_price,
        articles         = articles,
        fetch_ts         = time.time(),
        sources_queried  = sources_queried,
    )


# ── self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [RESEARCH] %(message)s",
    )

    async def run_tests():
        print("running research_agent self-test...\n")

        # ── test 1: keyword extraction ──
        cases = [
            ("Will Trump sign the tariff bill by June?",        ["Trump", "tariff", "bill"]),
            ("Will Bitcoin reach $100k in 2025?",               ["Bitcoin", "100k", "2025"]),
            ("Will the Fed cut rates in Q3 2025?",              ["Fed", "cut", "rates"]),
            ("Will Kamala Harris win the 2024 election?",       ["Kamala", "Harris", "win"]),
        ]
        for question, expected_subset in cases:
            kws = extract_keywords(question)
            assert len(kws) >= 1, f"got no keywords from: {question}"
            # check at least one expected token is present
            assert any(e.lower() in [k.lower() for k in kws] for e in expected_subset), \
                f"expected one of {expected_subset} in {kws} for: {question}"
            print(f"  ✓  keywords({question[:45]}...) -> {kws}")

        # ── test 2: keyword matching ──
        assert _matches_keywords("Trump signs new tariff legislation", ["Trump", "tariff"])
        assert _matches_keywords("bitcoin price surges past record", ["Bitcoin"])
        assert not _matches_keywords("local weather update for tuesday", ["Bitcoin", "Fed"])
        print("  ✓  keyword matching: OR logic works correctly")

        # ── test 3: deduplication ──
        a1 = Article("Bitcoin hits new record high", "", "reuters", "rss", "", 1000.0)
        a2 = Article("Bitcoin hits new record high!", "", "bbc",     "rss", "", 999.0)  # near-dup
        a3 = Article("Ethereum struggles amid market uncertainty", "", "coindesk", "rss", "", 998.0)
        deduped = _deduplicate([a1, a2, a3])
        assert len(deduped) == 2, f"expected 2 after dedup, got {len(deduped)}"
        print("  ✓  deduplication: near-duplicate titles removed")

        # ── test 4: recency sort ──
        a_old     = Article("old news", "", "bbc", "rss", "", 1000.0)
        a_new     = Article("new news", "", "bbc", "rss", "", 9999.0)
        a_unknown = Article("unknown age", "", "bbc", "rss", "", 0.0)
        sorted_   = _sort_by_recency([a_old, a_unknown, a_new])
        assert sorted_[0].published_ts == 9999.0, "newest should be first"
        assert sorted_[-1].published_ts == 0.0,   "unknown ts should be last"
        print("  ✓  recency sort: newest first, unknown timestamp last")

        # ── test 5: live gather_research (real network call) ──
        print("\n  running live gather_research (real rss/reddit fetch)...")
        payload = await gather_research(
            market_id        = "TEST-LIVE-001",
            market_question  = "Will the Federal Reserve cut interest rates in 2025?",
            market_yes_price = 0.60,
        )
        assert isinstance(payload, ResearchPayload)
        assert payload.market_question != ""
        assert payload.market_yes_price == 0.60
        print(f"  ✓  gather_research returned {len(payload.articles)} articles "
              f"from {len(payload.sources_queried)} sources")

        if payload.articles:
            a = payload.articles[0]
            print(f"  ✓  top article: [{a.source_domain}] {a.title[:60]}")
            assert a.title
            assert a.source_domain
            assert a.source_type in ("rss", "reddit", "newsapi")

        # confirm payload is ready for _call_research_skill
        assert hasattr(payload, "articles")
        assert hasattr(payload, "market_question")
        assert hasattr(payload, "market_yes_price")
        print("  ✓  ResearchPayload has all fields required by _call_research_skill")

        print("\nall research_agent tests passed ✓")

    asyncio.run(run_tests())