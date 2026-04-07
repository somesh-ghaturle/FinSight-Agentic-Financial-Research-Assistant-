"""
SEC EDGAR filing fetcher.

Retrieves 10-K, 10-Q, and 8-K filings for a given company ticker or CIK using
the public SEC EDGAR REST API (no API key required).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import requests

from config.settings import (
    SEC_EDGAR_BASE_URL,
    SEC_EDGAR_USER_AGENT,
    SEC_MAX_FILINGS,
)

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": SEC_EDGAR_USER_AGENT}

SUPPORTED_FORMS = {"10-K", "10-Q", "8-K"}


def _get(url: str, params: Optional[Dict] = None) -> dict:
    """Perform a GET request with retry logic."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=_HEADERS, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Rate-limited by SEC EDGAR, retrying in %ss…", wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts")


def ticker_to_cik(ticker: str) -> str:
    """Resolve a ticker symbol to a zero-padded 10-digit CIK string."""
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    ticker_map: dict = requests.get(tickers_url, headers=_HEADERS, timeout=30).json()
    for _idx, entry in ticker_map.items():
        if entry.get("ticker", "").upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR company list")


def get_filings_metadata(
    cik: str,
    form_types: Optional[List[str]] = None,
    max_filings: int = SEC_MAX_FILINGS,
) -> List[Dict]:
    """
    Return metadata dicts for recent filings of *form_types* for *cik*.

    Each dict contains: accessionNumber, filingDate, form, primaryDocument.
    """
    form_types = [f.upper() for f in (form_types or ["10-K"])]
    for ft in form_types:
        if ft not in SUPPORTED_FORMS:
            raise ValueError(f"Unsupported form type: {ft}. Choose from {SUPPORTED_FORMS}")

    url = f"{SEC_EDGAR_BASE_URL}/submissions/CIK{cik}.json"
    data = _get(url)
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    results: List[Dict] = []
    for form, date, accession, primary_doc in zip(forms, dates, accessions, primary_docs):
        if form in form_types:
            results.append(
                {
                    "cik": cik,
                    "form": form,
                    "filingDate": date,
                    "accessionNumber": accession,
                    "primaryDocument": primary_doc,
                }
            )
        if len(results) >= max_filings:
            break
    return results


def fetch_filing_text(cik: str, accession_number: str, primary_document: str) -> str:
    """
    Download the full text of a filing document from SEC EDGAR.

    Returns the raw text content (HTML/XML stripped by the caller).
    """
    accession_clean = accession_number.replace("-", "")
    viewer_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_clean}/{primary_document}"
    )
    resp = requests.get(viewer_url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.text


def fetch_filings_for_ticker(
    ticker: str,
    form_types: Optional[List[str]] = None,
    max_filings: int = SEC_MAX_FILINGS,
) -> List[Dict]:
    """
    High-level helper: resolve ticker → CIK, then fetch filing texts.

    Returns a list of dicts with keys:
        ticker, cik, form, filingDate, accessionNumber, text
    """
    cik = ticker_to_cik(ticker)
    metadata = get_filings_metadata(cik, form_types=form_types, max_filings=max_filings)

    results: List[Dict] = []
    for meta in metadata:
        try:
            text = fetch_filing_text(cik, meta["accessionNumber"], meta["primaryDocument"])
            results.append({**meta, "ticker": ticker, "text": text})
            logger.info("Fetched %s %s (%s)", ticker, meta["form"], meta["filingDate"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch filing %s: %s", meta["accessionNumber"], exc)
    return results
