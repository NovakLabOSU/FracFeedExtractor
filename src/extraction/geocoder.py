"""Nominatim geocoder with disk-based caching and rate limiting."""

import time
from dataclasses import dataclass

import diskcache
import requests

_SENTINEL = object()
_RATE_KEY = "__geocode_last_call__"


@dataclass
class GeocodeResult:
    lat: float
    lon: float
    display_name: str
    confidence: float  # Nominatim importance score, 0–1


class NominatimGeocoder:
    BASE_URL = "https://nominatim.openstreetmap.org/search"

    def __init__(self, user_agent: str, cache_path: str = ".geocode_cache", rate_limit_sec: float = 1.1):
        self._user_agent = user_agent
        self._rate_limit = rate_limit_sec
        self._cache = diskcache.Cache(cache_path)

    def geocode(self, location: str) -> "GeocodeResult | None":
        key = location.lower().strip()

        # Atomic get avoids TOCTOU between membership check and read
        cached = self._cache.get(key, default=_SENTINEL)
        if cached is not _SENTINEL:
            return cached

        # Lock serializes HTTP calls across all processes sharing this cache,
        # enforcing the 1 req/s policy even under ProcessPoolExecutor workers.
        with diskcache.Lock(self._cache, "geocode:http_lock", expire=60):
            # Double-check under lock in case another process just populated it
            cached = self._cache.get(key, default=_SENTINEL)
            if cached is not _SENTINEL:
                return cached

            # Shared rate limit: read/write last-call timestamp from cache
            last_call = self._cache.get(_RATE_KEY, default=0.0)
            elapsed = time.time() - last_call
            if elapsed < self._rate_limit:
                time.sleep(self._rate_limit - elapsed)

            try:
                resp = requests.get(
                    self.BASE_URL,
                    params={"q": location, "format": "json", "limit": 5},
                    headers={"User-Agent": self._user_agent},
                    timeout=10,
                )
                resp.raise_for_status()
                results = resp.json()
            except requests.exceptions.HTTPError as e:
                # 4xx: bad location string — cache None so we don't retry it
                if e.response is not None and 400 <= e.response.status_code < 500:
                    self._cache.set(key, None)
                    return None
                raise
            finally:
                self._cache.set(_RATE_KEY, time.time())

            if not results:
                self._cache.set(key, None)
                return None

            best = max(results, key=lambda r: float(r.get("importance", 0)))
            result = GeocodeResult(
                lat=float(best["lat"]),
                lon=float(best["lon"]),
                display_name=best["display_name"],
                confidence=float(best.get("importance", 0)),
            )
            self._cache.set(key, result)
            return result
