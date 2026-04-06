import html
import logging
from typing import List
import requests
import random
import re
from requests import RequestException
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from config import USER_AGENTS

logger = logging.getLogger("birdwatch")
logger.setLevel(logging.INFO)
session = requests.Session()
retries = Retry(
    total=0, backoff_factor=0, status_forcelist=[], allowed_methods={"HEAD", "GET"}
)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))
session.headers.update({"Accept-Encoding": "gzip, deflate, br"})


def check_url_with_requests(
    url: str, timeout: int = 15, debug: bool = False, user_agent: str = None
) -> bool:
    headers = {
        "User-Agent": user_agent if user_agent else random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    if debug and user_agent:
        print(f"Using UA: {user_agent[:50]}...")
    try:
        resp = session.head(
            url, headers=headers, timeout=timeout, allow_redirects=True, verify=True
        )
        if debug:
            print(f"HEAD response: {resp.status_code}, final URL: {resp.url}")
        if resp.status_code in {200, 301, 302, 303, 307, 308, 429}:
            return True
        elif 500 <= resp.status_code < 600:
            # likely server error, URL is real
            return True
        elif resp.status_code in {401, 403, 405}:
            # likely blocked head, fall back to GET
            pass
        else:
            # other weird 4xx status codes, URL is not real
            if debug:
                print(
                    f"HEAD failed with status {resp.status_code}, not in valid set and not 401/403/405"
                )
            return False
    except RequestException as e:
        if debug:
            print(f"HEAD request exception: {type(e).__name__}: {e}")
        pass  # fall back to GET

    # fallback to GET
    if debug:
        print("Falling back to GET request...")
    try:
        with session.get(
            url, headers=headers, timeout=timeout, allow_redirects=True, stream=True
        ) as resp:
            if debug:
                print(f"GET response: {resp.status_code}, final URL: {resp.url}")
            try:
                preview = (
                    next(resp.iter_content(512))
                    .decode("utf-8", errors="ignore")
                    .lower()
                )
            except:
                preview = ""

            if resp.status_code in {200, 203, 206, 301, 302, 303, 307, 308}:
                if debug:
                    print(f"GET succeeded with status {resp.status_code}")
                return True

            # Paywalls/soft blocks often return 200 or 403 with real content
            if resp.status_code in {403, 405} and any(
                string in preview
                for string in {
                    "paywall",
                    "register",
                    "sign in",
                    "subscribe",
                    "washingtonpost.com",
                }
            ):
                if debug:
                    print(f"GET returned {resp.status_code} but has paywall indicators")
                return True
            if (
                resp.status_code in {401, 403, 405, 429, 451}
                or 500 <= resp.status_code < 600
            ):
                return True
            if debug:
                print(f"GET failed: status {resp.status_code}, no paywall indicators")
            return False
    except RequestException as e:
        if debug:
            print(f"GET request exception: {type(e).__name__}: {e}")
        # Be tolerant: only fail on DNS/network-level errors after retries
        if isinstance(
            e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        ):
            return False
        return False  # Other exceptions (SSL, etc.) → likely bad


def check_url_fn(
    url: str, evidence_urls: List[str], check_with_requests: bool = False
) -> bool:
    if url not in evidence_urls:
        logger.info(f"URL not in evidence URLs: {url}")
        return False
    if check_with_requests:
        return check_url_with_requests(url, debug=False)
    else:
        return True


def check_all_urls_for_note(
    note_text: str, evidence_urls: List[str], check_with_requests: bool = False
) -> bool:
    """
    Check all URLs in the note text to see if they are valid.
    For each URL, there are multiple variants (e.g. with and without trailing punctuation).
    If at least one variant of each URL is valid, return True.
    If there is any URL with no valid variant, return False.

    Args:
        note_text (str): The text of the note to check.
        evidence_urls (List[str]): The list of URLs to check against.
        check_with_requests (bool): Whether to check the URL with requests.
    """
    note_text_unescaped = unescape(note_text)
    urls = _extract_urls(note_text_unescaped)
    if len(urls) == 0:
        logger.info(f"No URLs found in note text: {note_text_unescaped}")
        return False

    for url_variant_list in urls:
        at_least_one_good_url_variant = False
        for url_variant in url_variant_list:
            if check_url_fn(url_variant, evidence_urls, check_with_requests):
                at_least_one_good_url_variant = True
                break
        if not at_least_one_good_url_variant:
            logger.info(
                f"No valid URL found for any variant of {url_variant_list} in note text: {note_text_unescaped}"
            )
            return False
    logger.info(f"All URLs in note text are valid: {note_text_unescaped}")
    return True


def _extract_urls(text: str) -> List[str]:
    # return re.findall(r"https?://[^\s]*", text)
    """
    Return every URL-like substring from *text*.
    Return a List of Lists: each inner List contains multiple possible variants
    of an individual URL.
    """
    pattern = re.compile(
        r"""
            (?:
                https?://               # optional scheme
            )?
            (?:www\.)?                  # optional www
            [\w\-._~%]+                 # subdomain or domain name chars
            \.[a-zA-Z]{2,}              # dot + top level domain (≥2 letters)
            (?:/[^\s]*)?                # optional query/fragment
            """,
        re.VERBOSE,
    )

    raw_matches = pattern.findall(text)

    # Strip common trailing punctuation that often follows URLs in note text
    # Return both variants (with and without trailing punctuation) for each URL.
    strip_trailing = ".,;:!?)]}\"'"
    results = []
    for match in raw_matches:
        # Create a list of variants for each match
        variants = [match]
        stripped_variant = match.rstrip(strip_trailing)
        if stripped_variant != match:
            variants.append(stripped_variant)
        results.append(variants)
    return results


def unescape(text: str) -> str:
    """Remove layers of HTML escaping so the text matches natural language."""
    return html.unescape(html.unescape(text)) if isinstance(text, str) else text
