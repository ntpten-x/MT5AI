from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Mapping, Sequence

import httpx
from cachetools import TTLCache
from loguru import logger


@dataclass(slots=True, frozen=True)
class BeneficialOwnershipEvent:
    filer_name: str
    form: str
    filed_at: datetime | None
    stake_pct: float | None
    shares: float | None
    source_url: str | None = None


@dataclass(slots=True, frozen=True)
class InstitutionalHolderSnapshot:
    manager_name: str
    filed_at: datetime | None
    matched_issuer: str
    value_usd_thousands: float | None
    shares: float | None
    source_url: str | None = None


@dataclass(slots=True, frozen=True)
class OwnershipIntelligence:
    ticker: str
    company_name: str | None
    beneficial_owners: tuple[BeneficialOwnershipEvent, ...]
    institutional_holders: tuple[InstitutionalHolderSnapshot, ...]
    ownership_signal: str | None
    highlights: tuple[str, ...]


class OwnershipIntelligenceClient:
    def __init__(
        self,
        *,
        sec_user_agent: str,
        manager_ciks: Sequence[str] | None = None,
        timeout_seconds: float = 12.0,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self.sec_user_agent = sec_user_agent.strip() or "InvestAdvisorBot/0.2 support@example.com"
        self.manager_ciks = tuple(dict.fromkeys(self._normalize_cik(item) for item in (manager_ciks or ()) if self._normalize_cik(item)))
        self.timeout_seconds = max(2.0, float(timeout_seconds))
        self._http_client: httpx.Client | None = None
        self._lock = RLock()
        self._warning: str | None = None
        self._ticker_mapping_cache: TTLCache[str, dict[str, dict[str, str]]] = TTLCache(maxsize=1, ttl=max(3600, int(cache_ttl_seconds)))
        self._ownership_cache: TTLCache[str, OwnershipIntelligence | None] = TTLCache(maxsize=128, ttl=max(900, int(cache_ttl_seconds)))
        self._text_cache: TTLCache[str, str | None] = TTLCache(maxsize=128, ttl=max(900, int(cache_ttl_seconds)))

    async def aclose(self) -> None:
        with self._lock:
            client = self._http_client
            self._http_client = None
        if client is not None:
            client.close()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": True,
                "configured": bool(self.sec_user_agent),
                "manager_cik_count": len(self.manager_ciks),
                "warning": self._warning,
                "cache_entries": len(self._ownership_cache),
            }

    async def get_company_ownership(self, ticker: str, *, company_name: str | None = None) -> OwnershipIntelligence | None:
        normalized = str(ticker or "").strip().upper()
        if not normalized:
            return None
        with self._lock:
            cached = self._ownership_cache.get(normalized)
        if cached is not None:
            return cached
        result = await asyncio.to_thread(self._get_company_ownership_sync, normalized, company_name)
        with self._lock:
            self._ownership_cache[normalized] = result
        return result

    async def get_company_ownership_batch(
        self,
        tickers: Sequence[str],
        *,
        company_names: Mapping[str, str] | None = None,
    ) -> dict[str, OwnershipIntelligence | None]:
        normalized = [str(item).strip().upper() for item in tickers if str(item).strip()]
        if not normalized:
            return {}
        results = await asyncio.gather(
            *[self.get_company_ownership(ticker, company_name=(company_names or {}).get(ticker)) for ticker in normalized],
            return_exceptions=True,
        )
        return {
            ticker: (None if isinstance(result, Exception) else result)
            for ticker, result in zip(normalized, results, strict=False)
        }

    def _get_company_ownership_sync(self, ticker: str, company_name: str | None = None) -> OwnershipIntelligence | None:
        mapping = self._load_sec_ticker_mapping()
        identity = mapping.get(ticker)
        cik = (identity or {}).get("cik")
        resolved_name = company_name or (identity or {}).get("title")
        if not cik:
            return None
        submissions = self._fetch_sec_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
        beneficial_owners = self._extract_beneficial_owner_events(cik=cik, submissions=submissions)
        institutional_holders = self._extract_institutional_watchlist_holders(ticker=ticker, company_name=resolved_name)
        highlights = self._build_highlights(beneficial_owners=beneficial_owners, institutional_holders=institutional_holders)
        if not beneficial_owners and not institutional_holders:
            return None
        return OwnershipIntelligence(
            ticker=ticker,
            company_name=resolved_name,
            beneficial_owners=tuple(beneficial_owners[:4]),
            institutional_holders=tuple(institutional_holders[:4]),
            ownership_signal=self._classify_signal(beneficial_owners, institutional_holders),
            highlights=tuple(highlights[:6]),
        )

    def _extract_beneficial_owner_events(
        self,
        *,
        cik: str,
        submissions: Mapping[str, Any] | None,
    ) -> list[BeneficialOwnershipEvent]:
        recent = ((submissions or {}).get("filings") or {}).get("recent")
        if not isinstance(recent, Mapping):
            return []
        forms = list(recent.get("form") or [])
        filing_dates = list(recent.get("filingDate") or [])
        accession_numbers = list(recent.get("accessionNumber") or [])
        primary_documents = list(recent.get("primaryDocument") or [])
        items: list[BeneficialOwnershipEvent] = []
        count = min(len(forms), len(filing_dates), len(accession_numbers), len(primary_documents))
        for index in range(count):
            form = str(forms[index] or "").strip().upper()
            if form not in {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}:
                continue
            accession_number = str(accession_numbers[index] or "").strip()
            primary_document = str(primary_documents[index] or "").strip()
            if not accession_number or not primary_document:
                continue
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number.replace('-', '')}/{primary_document}"
            text = self._fetch_text(filing_url)
            if not text:
                continue
            stake_pct = self._extract_pct(text)
            shares = self._extract_shares(text)
            filer_name = self._extract_filer_name(text) or "unknown filer"
            items.append(
                BeneficialOwnershipEvent(
                    filer_name=filer_name,
                    form=form,
                    filed_at=self._parse_datetime(filing_dates[index]),
                    stake_pct=stake_pct,
                    shares=shares,
                    source_url=filing_url,
                )
            )
        items.sort(key=lambda item: item.filed_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return items

    def _extract_institutional_watchlist_holders(
        self,
        *,
        ticker: str,
        company_name: str | None,
    ) -> list[InstitutionalHolderSnapshot]:
        normalized_name = self._normalize_name(company_name or ticker)
        if not normalized_name or not self.manager_ciks:
            return []
        matches: list[InstitutionalHolderSnapshot] = []
        for manager_cik in self.manager_ciks[:12]:
            submissions = self._fetch_sec_json(f"https://data.sec.gov/submissions/CIK{manager_cik}.json")
            recent = ((submissions or {}).get("filings") or {}).get("recent")
            if not isinstance(recent, Mapping):
                continue
            forms = list(recent.get("form") or [])
            filing_dates = list(recent.get("filingDate") or [])
            accession_numbers = list(recent.get("accessionNumber") or [])
            primary_documents = list(recent.get("primaryDocument") or [])
            manager_name = str((submissions or {}).get("name") or manager_cik).strip() or manager_cik
            for index, form in enumerate(forms):
                normalized_form = str(form or "").strip().upper()
                if normalized_form not in {"13F-HR", "13F-HR/A"}:
                    continue
                accession_number = str(accession_numbers[index] or "").strip() if index < len(accession_numbers) else ""
                primary_document = str(primary_documents[index] or "").strip() if index < len(primary_documents) else ""
                if not accession_number or not primary_document:
                    continue
                filing_base = f"https://www.sec.gov/Archives/edgar/data/{int(manager_cik)}/{accession_number.replace('-', '')}/"
                filing_url = f"{filing_base}{primary_document}"
                info_table_text = self._fetch_13f_info_table(filing_url=filing_url, filing_base=filing_base)
                if not info_table_text:
                    continue
                holder = self._extract_matching_13f_holding(
                    info_table_text=info_table_text,
                    manager_name=manager_name,
                    filed_at=self._parse_datetime(filing_dates[index]) if index < len(filing_dates) else None,
                    company_name=normalized_name,
                )
                if holder is not None:
                    matches.append(holder)
                break
        matches.sort(key=lambda item: (item.value_usd_thousands or 0.0), reverse=True)
        return matches

    def _fetch_13f_info_table(self, *, filing_url: str, filing_base: str) -> str | None:
        primary_text = self._fetch_text(filing_url)
        if not primary_text:
            return None
        if "<infoTable" in primary_text:
            return primary_text
        link_match = re.search(r'href="(?P<href>[^"]*infotable[^"]*\.xml)"', primary_text, re.I)
        if link_match:
            candidate = link_match.group("href")
            if not candidate.startswith("http"):
                candidate = filing_base + candidate.lstrip("/")
            return self._fetch_text(candidate)
        return None

    def _extract_matching_13f_holding(
        self,
        *,
        info_table_text: str,
        manager_name: str,
        filed_at: datetime | None,
        company_name: str,
    ) -> InstitutionalHolderSnapshot | None:
        match = re.search(r"(<informationTable\b.*?</informationTable>)", info_table_text, re.I | re.S)
        xml_text = match.group(1) if match else info_table_text
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None
        nodes = root.findall(".//infoTable")
        if not nodes and root.tag.endswith("infoTable"):
            nodes = [root]
        for node in nodes:
            issuer = self._normalize_name(self._extract_xml_text(node, "nameOfIssuer"))
            if not issuer or company_name not in issuer:
                continue
            value = self._coerce_float(self._extract_xml_text(node, "value"))
            shares = self._coerce_float(self._extract_xml_text(node, "shrsOrPrnAmt/sshPrnamt"))
            return InstitutionalHolderSnapshot(
                manager_name=manager_name,
                filed_at=filed_at,
                matched_issuer=self._extract_xml_text(node, "nameOfIssuer") or company_name,
                value_usd_thousands=value,
                shares=shares,
                source_url=None,
            )
        return None

    def _load_sec_ticker_mapping(self) -> dict[str, dict[str, str]]:
        cache_key = "ticker_map"
        with self._lock:
            cached = self._ticker_mapping_cache.get(cache_key)
        if isinstance(cached, dict):
            return dict(cached)
        payload = self._fetch_sec_json("https://www.sec.gov/files/company_tickers.json")
        mapping: dict[str, dict[str, str]] = {}
        if isinstance(payload, Mapping):
            for item in payload.values():
                if not isinstance(item, Mapping):
                    continue
                ticker = str(item.get("ticker") or "").strip().upper()
                cik = self._normalize_cik(item.get("cik_str"))
                title = str(item.get("title") or "").strip()
                if ticker and cik:
                    mapping[ticker] = {"cik": cik, "title": title}
        with self._lock:
            self._ticker_mapping_cache[cache_key] = dict(mapping)
        return mapping

    def _fetch_sec_json(self, url: str) -> dict[str, Any] | None:
        try:
            response = self._get_http_client().get(
                url,
                headers=self._headers(),
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            with self._lock:
                self._warning = f"sec_json_fetch_failed: {exc}"
            logger.warning("SEC JSON fetch failed for {}: {}", url, exc)
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    def _fetch_text(self, url: str) -> str | None:
        with self._lock:
            cached = self._text_cache.get(url)
        if cached is not None:
            return cached
        try:
            response = self._get_http_client().get(
                url,
                headers=self._headers(),
                follow_redirects=True,
            )
            response.raise_for_status()
            text = response.text
        except Exception as exc:
            with self._lock:
                self._warning = f"sec_text_fetch_failed: {exc}"
            logger.warning("SEC text fetch failed for {}: {}", url, exc)
            text = None
        with self._lock:
            self._text_cache[url] = text
        return text

    def _get_http_client(self) -> httpx.Client:
        with self._lock:
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=self.timeout_seconds)
            return self._http_client

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json,text/html,application/xml",
        }

    @staticmethod
    def _extract_pct(text: str) -> float | None:
        patterns = (
            r"percent of class represented by amount in row \(11\)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"percent of class[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)\s*%",
            r"aggregate amount beneficially owned[^%]{0,200}?([0-9]+(?:\.[0-9]+)?)\s*%",
        )
        normalized = re.sub(r"\s+", " ", text or "", flags=re.S | re.I)
        for pattern in patterns:
            match = re.search(pattern, normalized, re.I)
            if match:
                try:
                    return round(float(match.group(1)), 2)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _extract_shares(text: str) -> float | None:
        patterns = (
            r"aggregate amount beneficially owned by each reporting person\s*[:\-]?\s*([0-9,]+)",
            r"amount beneficially owned[^0-9]{0,20}([0-9,]+)",
        )
        normalized = re.sub(r"\s+", " ", text or "", flags=re.S | re.I)
        for pattern in patterns:
            match = re.search(pattern, normalized, re.I)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _extract_filer_name(text: str) -> str | None:
        normalized = re.sub(r"\s+", " ", text or "", flags=re.S)
        match = re.search(r"name of reporting person\s*[:\-]?\s*([A-Z0-9 ,.&'-]{4,80})", normalized, re.I)
        if not match:
            return None
        return match.group(1).strip(" -")

    @staticmethod
    def _build_highlights(
        *,
        beneficial_owners: Sequence[BeneficialOwnershipEvent],
        institutional_holders: Sequence[InstitutionalHolderSnapshot],
    ) -> list[str]:
        highlights: list[str] = []
        if beneficial_owners:
            top_owner = beneficial_owners[0]
            if top_owner.stake_pct is not None:
                highlights.append(f"13D/13G {top_owner.filer_name} stake {top_owner.stake_pct:.2f}%")
        if institutional_holders:
            top_holder = institutional_holders[0]
            if top_holder.value_usd_thousands is not None:
                highlights.append(f"13F watchlist {top_holder.manager_name} value ${top_holder.value_usd_thousands:,.0f}k")
        return highlights

    @staticmethod
    def _classify_signal(
        beneficial_owners: Sequence[BeneficialOwnershipEvent],
        institutional_holders: Sequence[InstitutionalHolderSnapshot],
    ) -> str | None:
        top_stake = max((item.stake_pct or 0.0) for item in beneficial_owners) if beneficial_owners else 0.0
        if top_stake >= 10.0:
            return "activist_or_anchor_holder"
        if top_stake >= 5.0:
            return "meaningful_beneficial_owner"
        if institutional_holders:
            return "watchlist_13f_holder_present"
        return None

    @staticmethod
    def _extract_xml_text(node: ET.Element, path: str) -> str | None:
        element = node.find(path)
        if element is None or element.text is None:
            return None
        text = str(element.text).strip()
        return text or None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _normalize_cik(value: object) -> str | None:
        digits = str(value or "").strip()
        return digits.zfill(10) if digits.isdigit() else None

    @staticmethod
    def _normalize_name(value: str) -> str:
        text = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return None if value is None else float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None
