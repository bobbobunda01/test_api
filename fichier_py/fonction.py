#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:59:03 2025

@author: bobunda
"""


import json
from joblib import load
from pydantic import BaseModel
from flask import Flask, jsonify, request
from typing import List
import numpy as np
import pandas as pd
import os
from numpy import floating, integer, ndarray
import datetime
import pathlib
from dateutil import parser
from functools import lru_cache
import requests
import re
from typing import Any, Dict, Optional, Tuple, List
import datetime as dt
from datetime import timedelta
from openai import OpenAI
import threading
##------------------------------- PREDICTION DES EQUIPES WIN LOSS DRAW ------------------------------------------------
#### amélioration de temps de réponse

# Session HTTP réutilisée (keep-alive)
_HTTP = requests.Session()

# Cache mémoire simple avec TTL
_RT_CACHE = {}
_RT_CACHE_LOCK = threading.Lock()

# Mode realtime: "off" | "light" | "full"
REALTIME_MODE = os.getenv("REALTIME_MODE", "light").strip().lower()



REALTIME_TIMEOUT_FIXTURE = 10
REALTIME_TIMEOUT_OPTIONAL = 5
REALTIME_TIMEOUT_STANDINGS = 6
REALTIME_LINEUPS_SOON_MINUTES = 90
REALTIME_MODE = "light"   # ou env var

# lineups uniquement si proche du match (minutes)
REALTIME_LINEUPS_SOON_MINUTES = int(os.getenv("REALTIME_LINEUPS_SOON_MINUTES", "90"))

def _cache_get(key):
    with _RT_CACHE_LOCK:
        item = _RT_CACHE.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at is not None and time.time() > expires_at:
            _RT_CACHE.pop(key, None)
            return None
        return value

def _cache_set(key, value, ttl=60):
    with _RT_CACHE_LOCK:
        expires_at = None if ttl is None or ttl <= 0 else (time.time() + ttl)
        _RT_CACHE[key] = (expires_at, value)

def _cache_key(endpoint: str, params: dict) -> tuple:
    try:
        items = tuple(sorted((params or {}).items()))
    except Exception:
        items = tuple()
    return (endpoint, items)

###---------------------------------------------------------

# log des prédictions utilisateurs

LABEL_HOME = 0
LABEL_DRAW = 1
LABEL_AWAY = 2

REALTIME_API_URL="https://v3.football.api-sports.io"

DEBUG_REALTIME=1

USE_LLM_EXPLANATION = True          

OPENAI_EXPLAIN_ENABLED = True    # True/False
OPENAI_EXPLAIN_MODEL = "gpt-4.1"
OPENAI_EXPLAIN_TEMPERATURE = 0.5
OPENAI_EXPLAIN_MAX_TOKENS = 460
OPENAI_EXPLAIN_TIMEOUT = 20

LLM_DEBUG = False          

# Mapping officiel BetSmart (API-SPORTS)
LEAGUES = {
    "Premier League": 39,
    "Ligue 1": 61,
    "Bundesliga": 78,
    "La Liga": 140,
    "Serie A": 135,
    "Neerdeland": 88,
    "Suisse": 207,
    "Portugais": 94,
    "Turquie": 203,
    "Belgique": 144,
    "Japon": 98,
    "Grece": 197,
    "bresil": 71,
    "ecosse": 179,
    "ecosse_div_1": 180,
    "coree_sud": 292,
    "Argentine_league_1": 128,
    "League_europa": 3,
    "champions_league": 2,
    "egypte": 233,
    "mexique": 262,
    "france_league_2": 62,
    "bundesliga_2": 79,
    "serie_B": 136,
    "Championship": 40,
    "secunda": 141,
    "can": 6
}

REASON_TRANSLATION_FR = {
    # Suspensions
    "yellow cards": "Suspendu (cartons)",
    "red card": "Suspendu (carton rouge)",

    # Générique blessure
    "injury": "Blessé",

    # Détails blessures
    "thigh injury": "Blessure à la cuisse",
    "muscle injury": "Blessure musculaire",
    "foot injury": "Blessure au pied",
    "knee injury": "Blessure au genou",
    "ankle injury": "Blessure à la cheville",
    "hamstring injury": "Ischio-jambiers",
    "back injury": "Dos",
    "groin injury": "Adducteurs",
    "calf injury": "Blessure au mollet",

    # Autres
    "illness": "Maladie",
}

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

def _safe_prob(x: Any, default: float = 0.0) -> float:
    """
    Convertit x en probabilité float [0..1] si possible.
    Supporte: 0.13, "13%", "13.0%", "0.13"
    """
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float)):
            v = float(x)
            # si quelqu'un passe 13 au lieu de 0.13 -> on interprète comme %
            if v > 1.0 and v <= 100.0:
                return max(0.0, min(1.0, v / 100.0))
            return max(0.0, min(1.0, v))
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return float(default)
            if s.endswith("%"):
                s2 = s[:-1].strip()
                v = float(s2)
                return max(0.0, min(1.0, v / 100.0))
            v = float(s)
            if v > 1.0 and v <= 100.0:
                return max(0.0, min(1.0, v / 100.0))
            return max(0.0, min(1.0, v))
        return float(default)
    except Exception:
        return float(default)


def _norm_team_name(name: Any) -> str:
    """Normalize team name for safer matching (accent/spacing/case)."""
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_match_date(match_date: Any) -> Optional[dt.date]:
    """Accepts date, datetime, or common string formats; returns date or None."""
    if match_date is None or match_date == "":
        return None
    if isinstance(match_date, dt.date) and not isinstance(match_date, dt.datetime):
        return match_date
    if isinstance(match_date, dt.datetime):
        return match_date.date()
    s = str(match_date).strip()
    # try ISO first
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(s[:19], fmt).date()
        except Exception:
            continue
    # last resort: try fromisoformat
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        return None


def _season_from_date(match_date: Any) -> Optional[int]:
    """
    API-FOOTBALL season is the start year of the season.
    For European leagues: Jan–Jun belongs to previous start year (e.g., Jan 2026 -> season 2025).
    """
    # _parse_match_date must be defined in your codebase
    d = _parse_match_date(match_date)  # noqa: F821
    if d is None:
        return None
    try:
        month = int(getattr(d, "month", 0))
    except Exception:
        month = 0
    return int(d.year - 1) if month <= 6 else int(d.year)

def _safe_get_first(obj, key, default=None):
    try:
        # dict
        if isinstance(obj, dict):
            return obj.get(key, default)

        # pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            if key in obj.columns and len(obj) > 0:
                v = obj[key].iloc[0]
                return v if v is not None else default
            return default

        # pandas Series
        if isinstance(obj, pd.Series):
            v = obj.get(key, default)
            # si v est une Series (cas rare), prends le 1er élément
            if isinstance(v, pd.Series):
                return v.iloc[0] if len(v) else default
            return v

        # fallback attribute access
        if hasattr(obj, "get"):
            v = obj.get(key, default)
            if isinstance(v, pd.Series):
                return v.iloc[0] if len(v) else default
            return v

    except Exception:
        return default

    return default

def _resolve_fixture_id_from_df(
    season_df: Any,
    home_name: Any,
    away_name: Any,
    match_date: Any,
    league_code: Optional[str] = None,
) -> Optional[int]:
    """
    Try to resolve fixture_id from a season dataframe (if present).
    This is the preferred (offline) method used in apply_unexpected_layer().
    """
    if season_df is None:
        return None
    # candidate columns
    fid_cols = [c for c in ["fixture_id", "FixtureID", "fixture", "Fixture", "id", "ID"] if hasattr(season_df, "columns") and c in season_df.columns]
    if not fid_cols:
        return None

    # team/date column candidates
    home_cols = [c for c in ["HomeTeam", "home", "home_name", "Home", "HomeTeamName"] if hasattr(season_df, "columns") and c in season_df.columns]
    away_cols = [c for c in ["AwayTeam", "away", "away_name", "Away", "AwayTeamName"] if hasattr(season_df, "columns") and c in season_df.columns]
    date_cols = [c for c in ["Date", "date", "match_date", "MatchDate", "fixture_date"] if hasattr(season_df, "columns") and c in season_df.columns]

    if not home_cols or not away_cols or not date_cols:
        return None

    h = _norm_team_name(home_name)
    a = _norm_team_name(away_name)
    d = _parse_match_date(match_date)

    # if no date, try only teams (less precise)
    try:
        df = season_df
        # normalize to strings for compare
        # We keep it safe: any error -> None
        for hc in home_cols:
            for ac in away_cols:
                tmp = df
                try:
                    # filter by teams
                    mask = tmp[hc].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).eq(h) & \
                           tmp[ac].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).eq(a)
                    tmp2 = tmp[mask]
                except Exception:
                    continue

                if d is not None:
                    for dc in date_cols:
                        try:
                            tmp3 = tmp2.copy()
                            # parse date column safely
                            tmp3[dc] = tmp3[dc].astype(str).str.slice(0, 10)
                            maskd = tmp3[dc].apply(lambda x: _parse_match_date(x)).eq(d)
                            tmp4 = tmp2[maskd]
                            if len(tmp4) > 0:
                                fid = tmp4.iloc[0][fid_cols[0]]
                                try:
                                    return int(fid)
                                except Exception:
                                    return None
                        except Exception:
                            continue

                # no date match, but teams match
                if len(tmp2) > 0:
                    fid = tmp2.iloc[0][fid_cols[0]]
                    try:
                        return int(fid)
                    except Exception:
                        return None
    except Exception:
        return None

    return None

def _resolve_fixture_id_by_names________(
    home_name: Any,
    away_name: Any,
    match_date: Any,
    league_code: Optional[str] = None,
) -> Optional[int]:
    """
    Online fallback resolver.
    Strategy:
      - call /fixtures with date
      - if league provided, also pass season (critical)
      - try date delta (0, -1, +1)
      - if league filter returns 0, retry without league/season
      - match by normalized names
    """
    api_url = os.getenv("REALTIME_API_URL", REALTIME_API_URL).rstrip("/")  # noqa: F821
    api_key = os.getenv("REALTIME_API_KEY", REALTIME_API_KEY)  # noqa: F821
    if not api_url or not api_key or requests is None:
        return None

    d = _parse_match_date(match_date)  # noqa: F821
    if d is None:
        return None

    url = f"{api_url}/fixtures"
    headers = {"x-apisports-key": api_key}
    host = os.getenv("REALTIME_API_HOST", "").strip()
    if host:
        headers["x-rapidapi-host"] = host

    home_n = _norm_team_name(home_name)  # noqa: F821
    away_n = _norm_team_name(away_name)  # noqa: F821

    # league param (int or label)
    league_param: Optional[int] = None
    if league_code is not None:
        try:
            league_param = int(league_code)
        except Exception:
            try:
                if isinstance(league_code, str) and league_code in LEAGUES:  # noqa: F821
                    league_param = int(LEAGUES[league_code])  # noqa: F821
            except Exception:
                league_param = None

    season_param: Optional[int] = _season_from_date(match_date) if league_param is not None else None  # noqa: F821

    for delta in (0, -1, 1):
        date_str = (d + timedelta(days=delta)).strftime("%Y-%m-%d")
        params = {"date": date_str}

        if league_param is not None:
            params["league"] = int(league_param)
            if season_param is not None:
                params["season"] = int(season_param)

        # 1) attempt
        try:
            r = requests.get(url, headers=headers, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        resp = (data or {}).get("response", []) or []

        # 2) fallback if league filter too strict
        if league_param is not None and len(resp) == 0:
            try:
                params2 = {"date": date_str}
                r2 = requests.get(url, headers=headers, params=params2, timeout=8)
                r2.raise_for_status()
                data = r2.json()
                resp = (data or {}).get("response", []) or []
            except Exception:
                resp = []

        # 3) match
        try:
            for item in resp:
                th = _norm_team_name(item.get("teams", {}).get("home", {}).get("name"))  # noqa: F821
                ta = _norm_team_name(item.get("teams", {}).get("away", {}).get("name"))  # noqa: F821
                if (th == home_n and ta == away_n) or (th == away_n and ta == home_n):
                    fid = item.get("fixture", {}).get("id")
                    if fid is not None:
                        return int(fid)
        except Exception:
            continue

    return None

def normalize_team_name(name: str) -> str:
    import unicodedata
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = " ".join(name.split())
    return name


def resolve_fixture_id_local(fixtures_df, home, away, match_date, league_code=None):
    """
    fixtures_df doit contenir au minimum:
    - fixture_id
    - home
    - away
    - match_date
    - league_code
    """
    h = normalize_team_name(home)
    a = normalize_team_name(away)
    d = str(match_date)[:10]

    df = fixtures_df.copy()

    df["_home_norm"] = df["HomeTeam"].astype(str).map(normalize_team_name)
    df["_away_norm"] = df["AwayTeam"].astype(str).map(normalize_team_name)
    df["_date_norm"] = df["match_Date"].astype(str).str[:10]

    mask = (
        (df["_home_norm"] == h) &
        (df["_away_norm"] == a) &
        (df["_date_norm"] == d)
    )

    if league_code is not None and "league_code" in df.columns:
        mask = mask & (df["league_code"].astype(str) == str(league_code))

    found = df.loc[mask]

    if found.empty:
        return None

    return int(found.iloc[0]["fixture_id"])

def _safe_resolve_fixture_id(
    home_name: Any,
    away_name: Any,
    match_date: Any,
    league_code: Optional[str] = None,
    season_df: Any = None,
    features_df: Any = None,
) -> Optional[int]:
    """
    Safe wrapper:
      0) if fixture_id already present in features_df -> use it directly (best)
      1) offline resolution (season_df)
      2) online resolver by names
    """

    def _scalar(v):
        """Normalize pandas/array-likes to a single scalar (or None)."""
        try:
            if isinstance(v, pd.Series):
                return v.iloc[0] if len(v) else None
            if isinstance(v, (list, tuple, np.ndarray)):
                return v[0] if len(v) else None
        except Exception:
            return None
        return v

    # 0) Direct fixture_id (FAST + RELIABLE) — avoid `or` on Series
    try:
        fid = None

        if features_df is not None:

            # dict
            if isinstance(features_df, dict):
                for k in ("fixture_id", "_fixture_id"):
                    if k in features_df:
                        candidate = _scalar(features_df.get(k))
                        if candidate is not None and str(candidate).strip() != "":
                            fid = candidate
                            break

            # DataFrame
            elif isinstance(features_df, pd.DataFrame):
                if len(features_df) > 0:
                    for k in ("fixture_id", "_fixture_id"):
                        if k in features_df.columns:
                            candidate = _scalar(features_df[k].iloc[0])
                            if candidate is not None and str(candidate).strip() != "":
                                fid = candidate
                                break

            # Series (row)
            elif isinstance(features_df, pd.Series):
                for k in ("fixture_id", "_fixture_id"):
                    if k in features_df.index:
                        candidate = _scalar(features_df.get(k))
                        if candidate is not None and str(candidate).strip() != "":
                            fid = candidate
                            break

            # fallback: dict-like get, but NO boolean ops
            elif hasattr(features_df, "get"):
                for k in ("fixture_id", "_fixture_id"):
                    candidate = _scalar(features_df.get(k, None))
                    if candidate is not None and str(candidate).strip() != "":
                        fid = candidate
                        break

        if fid is not None and str(fid).strip() != "":
            return int(fid)
    except Exception:
        pass

    # 1) offline resolution (best)
    try:
        fid2 = _resolve_fixture_id_from_df(season_df, home_name, away_name, match_date, league_code=league_code)
        if fid2 is not None and str(fid2).strip() != "":
            return int(fid2)
    except Exception:
        pass

    # 2) online fallback
    try:
        return _resolve_fixture_id_by_names(home_name, away_name, match_date, league_code=league_code)
    except Exception:
        return None

class RealtimeFetchError(Exception):
    """Internal exception used to carry http/debug info without breaking the pipeline."""
    def __init__(self, code: str, detail: str = "", status: Optional[int] = None):
        super().__init__(code)
        self.code = code
        self.detail = detail
        self.status = status


def _get_realtime_api_config() -> Tuple[str, str, str]:
    """
    Returns (api_url, api_key, api_host).
    - api_url/api_key first from env
    - then fallback to module-level constants if you defined them (optional)
    """
    # Optional module-level fallbacks if you defined them elsewhere:
    fallback_url = globals().get("REALTIME_API_URL",REALTIME_API_URL)
    fallback_key = globals().get("REALTIME_API_KEY", "REALTIME_API_KEY")
    fallback_host = globals().get("REALTIME_API_HOST", "")

    api_url = os.getenv("REALTIME_API_URL", fallback_url).strip().rstrip("/")
    api_key = os.getenv("REALTIME_API_KEY", fallback_key).strip()
    api_host = os.getenv("REALTIME_API_HOST", fallback_host).strip()
    return api_url, api_key, api_host

def _api_get___________(endpoint: str, params: dict, timeout: int = 10) -> dict:
    api_url = os.getenv("REALTIME_API_URL", REALTIME_API_URL).rstrip("/")
    api_key = os.getenv("REALTIME_API_KEY")

    if not api_url or not api_key:
        raise RealtimeFetchError(code="key_missing", detail="Missing API URL/KEY", status=None)

    url = f"{api_url}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": api_key}
    host = os.getenv("REALTIME_API_HOST", "").strip()
    if host:
        headers["x-rapidapi-host"] = host

    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        status = r.status_code

        if status in (401, 403):
            raise RealtimeFetchError(code="unauthorized", detail="API key unauthorized", status=status)
        if status == 429:
            raise RealtimeFetchError(code="rate_limited", detail="Rate limit", status=status)

        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else {"response": js}

    except RealtimeFetchError:
        raise
    except Exception as e:
        raise RealtimeFetchError(code="http_error", detail=str(e), status=None)


def _api_get(endpoint: str, params: dict, timeout: int = 10, cache_ttl: int = 0) -> dict:
    api_url = os.getenv("REALTIME_API_URL", REALTIME_API_URL).rstrip("/")
    api_key = os.getenv("REALTIME_API_KEY")

    if not api_url or not api_key:
        raise RealtimeFetchError(code="key_missing", detail="Missing API URL/KEY", status=None)

    # cache
    key = _cache_key(endpoint, params)
    if cache_ttl and cache_ttl > 0:
        hit = _cache_get(key)
        if hit is not None:
            return hit

    url = f"{api_url}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": api_key}
    host = os.getenv("REALTIME_API_HOST", "").strip()
    if host:
        headers["x-rapidapi-host"] = host

    try:
        r = _HTTP.get(url, headers=headers, params=params, timeout=timeout)
        status = r.status_code

        if status in (401, 403):
            raise RealtimeFetchError(code="unauthorized", detail="API key unauthorized", status=status)
        if status == 429:
            raise RealtimeFetchError(code="rate_limited", detail="Rate limit", status=status)

        r.raise_for_status()
        js = r.json()
        out = js if isinstance(js, dict) else {"response": js}

        if cache_ttl and cache_ttl > 0:
            _cache_set(key, out, ttl=cache_ttl)

        return out

    except RealtimeFetchError:
        raise
    except Exception as e:
        raise RealtimeFetchError(code="http_error", detail=str(e), status=None)



def _fetch_realtime_context(fixture_id: int) -> Optional[dict]:
    """
    Fetch full realtime context for a fixture_id.
    Returns ctx dict or None if fixture not found / empty response.
    """
    fixture_id = int(fixture_id)
    ctx = {"meta": {"missing": [], "fixture_id": fixture_id}}

    # 1) fixture core (mandatory)
    data_fx = _api_get("fixtures", {"id": fixture_id}, timeout=10)
    if not isinstance(data_fx, dict):
        return None

    resp_fx = data_fx.get("response", []) or []
    if len(resp_fx) == 0:
        return None

    fx = resp_fx[0]
    if not isinstance(fx, dict):
        ctx["meta"]["missing"].append(f"fixture_shape_invalid:{type(fx).__name__}")
        return None

    ctx["fixture"] = fx.get("fixture") if isinstance(fx.get("fixture"), dict) else {}
    ctx["league"]  = fx.get("league")  if isinstance(fx.get("league"), dict)  else {}
    ctx["teams"]   = fx.get("teams")   if isinstance(fx.get("teams"), dict)   else {}
    ctx["goals"]   = fx.get("goals")   if isinstance(fx.get("goals"), dict)   else {}
    ctx["score"]   = fx.get("score")   if isinstance(fx.get("score"), dict)   else {}

    def _optional(name: str, endpoint: str, params: dict):
        try:
            d = _api_get(endpoint, params, timeout=10)
            if isinstance(d, dict):
                ctx[name] = d.get("response", []) or []
            else:
                ctx[name] = []
                ctx["meta"]["missing"].append(f"{name}_shape_invalid:{type(d).__name__}")

            if len(ctx[name]) == 0:
                ctx["meta"]["missing"].append(f"{name}_empty")

        except RealtimeFetchError as e:
            ctx[name] = []
            ctx["meta"]["missing"].append(f"{name}_err:{e.code}")
        except Exception:
            ctx[name] = []
            ctx["meta"]["missing"].append(f"{name}_err:unknown")

    # optional endpoints
    _optional("events", "fixtures/events", {"fixture": fixture_id})
    _optional("lineups", "fixtures/lineups", {"fixture": fixture_id})
    _optional("statistics", "fixtures/statistics", {"fixture": fixture_id})
    _optional("players", "fixtures/players", {"fixture": fixture_id})
    _optional("injuries", "injuries", {"fixture": fixture_id})

    #ctx["meta"]["fetched_at"] = datetime.utcnow().isoformat() + "Z"
    ctx["meta"]["fetched_at"] = datetime.datetime.utcnow().isoformat() + "Z"
   # ctx["meta"]["fetched_at"] = datetime.utcnow().isoformat() + "Z"
    return ctx

def _fetch_realtime_context_(fixture_id: int) -> Optional[dict]:
    """
    Fetch realtime context for a fixture_id.
    Version hybride:
    - fetch principal fixtures tolérant (comme ancienne version)
    - endpoints optionnels optimisés
    - mode light pré-match
    """

    fixture_id = int(fixture_id)

    ctx = {
        "meta": {
            "missing": [],
            "errors": [],
            "skipped": [],
            "fixture_id": fixture_id,
        },
        "fixture": {},
        "league": {},
        "teams": {},
        "goals": {},
        "score": {},
        "events": [],
        "lineups": [],
        "statistics": [],
        "players": [],
        "injuries": [],
    }

    # --------------------------------------------------
    # 1) fixtures core (reprend l'ancienne logique)
    # --------------------------------------------------
    try:
        # ✅ important : timeout fixe 10, pas de cache ici
        data_fx = _api_get("fixtures", {"id": fixture_id}, timeout=10, cache_ttl=0)

        if not isinstance(data_fx, dict):
            return None

        resp_fx = data_fx.get("response", []) or []
        if len(resp_fx) == 0:
            return None

        fx = resp_fx[0]
        if not isinstance(fx, dict):
            ctx["meta"]["missing"].append(f"fixture_shape_invalid:{type(fx).__name__}")
            return None

        ctx["fixture"] = fx.get("fixture") if isinstance(fx.get("fixture"), dict) else {}
        ctx["league"]  = fx.get("league")  if isinstance(fx.get("league"), dict)  else {}
        ctx["teams"]   = fx.get("teams")   if isinstance(fx.get("teams"), dict)   else {}
        ctx["goals"]   = fx.get("goals")   if isinstance(fx.get("goals"), dict)   else {}
        ctx["score"]   = fx.get("score")   if isinstance(fx.get("score"), dict)   else {}

    except RealtimeFetchError as e:
        ctx["meta"]["errors"].append(f"fixtures_error:{e.code}:{e.detail}")
        return ctx
    except Exception as e:
        ctx["meta"]["errors"].append(f"fixtures_error:{type(e).__name__}:{str(e)[:200]}")
        return ctx

    # --------------------------------------------------
    # 2) status / date
    # --------------------------------------------------
    status_short = ""
    fixture_date = None
    try:
        status_short = str(((ctx.get("fixture") or {}).get("status") or {}).get("short") or "")
        fixture_date = (ctx.get("fixture") or {}).get("date")
    except Exception:
        pass

    minutes_to_kickoff = None
    try:
        if isinstance(fixture_date, str) and fixture_date.strip():
            dt = datetime.fromisoformat(fixture_date.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            minutes_to_kickoff = int((dt - now).total_seconds() // 60)
    except Exception:
        minutes_to_kickoff = None

    # --------------------------------------------------
    # 3) helper optional endpoints
    # --------------------------------------------------
    def _optional(name: str, endpoint: str, params: dict, ttl: int = 20):
        try:
            d = _api_get(endpoint, params, timeout=10, cache_ttl=ttl)

            if isinstance(d, dict):
                ctx[name] = d.get("response", []) or []
            else:
                ctx[name] = []
                ctx["meta"]["missing"].append(f"{name}_shape_invalid:{type(d).__name__}")

            if len(ctx[name]) == 0:
                ctx["meta"]["missing"].append(f"{name}_empty")

        except RealtimeFetchError as e:
            ctx[name] = []
            ctx["meta"]["missing"].append(f"{name}_err:{e.code}:{e.detail}")
        except Exception as e:
            ctx[name] = []
            ctx["meta"]["missing"].append(f"{name}_err:{type(e).__name__}")

    # --------------------------------------------------
    # 4) realtime mode logic
    # --------------------------------------------------
   
    if REALTIME_MODE == "off":
        ctx["meta"]["fetched_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        return ctx

    _optional("injuries", "injuries", {"fixture": fixture_id}, ttl=60)

    if status_short == "NS":
        if minutes_to_kickoff is not None and minutes_to_kickoff <= REALTIME_LINEUPS_SOON_MINUTES:
            _optional("lineups", "fixtures/lineups", {"fixture": fixture_id}, ttl=20)
        else:
            ctx["meta"]["missing"].append("lineups_not_due_yet")

        ctx["meta"]["skipped"].extend(["events", "statistics", "players"])

    else:
        _optional("events", "fixtures/events", {"fixture": fixture_id}, ttl=10)
        _optional("lineups", "fixtures/lineups", {"fixture": fixture_id}, ttl=20)
        _optional("statistics", "fixtures/statistics", {"fixture": fixture_id}, ttl=10)

        if REALTIME_MODE == "full":
            _optional("players", "fixtures/players", {"fixture": fixture_id}, ttl=10)
        else:
            ctx["meta"]["skipped"].append("players")

    ctx["meta"]["fetched_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    return ctx

def _realtime_risk_score(ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert real-time context to a risk score.
    This function is intentionally conservative: it NEVER throws, and defaults to UNKNOWN.
    You can enrich later with your own signals without breaking the pipeline.
    """
    out = {
        "risk_level": "UNKNOWN",
        "risk_score": 0.0,
        "reasons": [],
    }
    if not ctx:
        out["reasons"].append("realtime_ctx_missing")
        return out

    # Example: if fixture status is not "Not Started" we flag (because prediction might be late)
    try:
        status = (ctx.get("fixture", {}).get("status", {}) or {}).get("short")  # e.g. NS, 1H, HT...
        if status and status != "NS":
            out["risk_level"] = "HIGH"
            out["risk_score"] = 0.9
            out["reasons"].append(f"fixture_status:{status}")
            return out
    except Exception:
        pass

    # If we have any injuries list (provider-specific), flag moderate.
    try:
        injuries = ctx.get("injuries") or ctx.get("players") or None
        if injuries:
            out["risk_level"] = "MEDIUM"
            out["risk_score"] = max(out["risk_score"], 0.4)
            out["reasons"].append("possible_injuries_or_lineup_changes")
    except Exception:
        pass

    return out

def resolve_fixture_id_from_user_input(
    home: Any,
    away: Any,
    match_date: Any,
    league_code: Optional[str] = None,
    season_df: Any = None,
) -> Optional[int]:
    """
    Résout un fixture_id à partir des entrées utilisateur (home, away, date, league).
    - essaie offline d'abord si season_df fourni
    - sinon fallback online via API (/fixtures?date=YYYY-MM-DD + league+season si possible)
    Retourne uniquement fixture_id (int) ou None.
    """
    # offline preferred
    try:
        if season_df is not None:
            fid = _resolve_fixture_id_from_df(  # noqa: F821
                season_df, home, away, match_date, league_code=league_code
            )
            if fid is not None and str(fid).strip() != "":
                return int(fid)
    except Exception:
        pass

    # online fallback
    try:
        fid = _resolve_fixture_id_by_names(  # noqa: F821
            home, away, match_date, league_code=league_code
        )
        if fid is not None and str(fid).strip() != "":
            return int(fid)
    except Exception:
        pass

    return None


def attach_fixture_id_if_missing(features_input: Any, league_code: Optional[str] = None, season_df: Any = None):
    """
    Backward-compatible wrapper:
    resolves fixture_id from home/away/match_date (+league) then injects it.
    """
    existing = _safe_get_first(features_input, "fixture_id")  # noqa: F821
    if isinstance(existing, pd.Series):
        existing = existing.iloc[0] if len(existing) else None
    elif isinstance(existing, (list, tuple, np.ndarray)):
        existing = existing[0] if len(existing) else None

    if existing is not None and str(existing).strip() != "":
        return features_input

    home = _safe_get_first(features_input, "home")  # noqa: F821
    away = _safe_get_first(features_input, "away")  # noqa: F821
    match_date = _safe_get_first(features_input, "match_date")  # noqa: F821

    if home is None or away is None or match_date is None:
        return features_input

    fid = resolve_fixture_id_from_user_input(  # noqa: F821
        home, away, match_date, league_code=league_code, season_df=season_df
    )
    if fid is None:
        return features_input  # ✅ never int(None)

    try:
        if isinstance(features_input, dict):
            features_input["fixture_id"] = int(fid)
        elif isinstance(features_input, pd.Series):
            features_input.loc["fixture_id"] = int(fid)
        elif isinstance(features_input, pd.DataFrame) and len(features_input) > 0:
            features_input.at[features_input.index[0], "fixture_id"] = int(fid)
    except Exception:
        pass

    return features_input

def translate_reason_fr(reason: str) -> str:
    """
    Traduction métier FR des raisons d'absence (API-Sports → BetSmart).
    """
    if not reason:
        return "Indisponible"

    r = reason.strip().lower()

    # ignorer libellés techniques inutiles
    if r in ("missing fixture",):
        return ""

    return REASON_TRANSLATION_FR.get(r, reason)


def format_absences_summary(summary: Dict[str, Any], max_players_per_team: int = 3) -> str:
    """
    Format UI-friendly absences summary:
      - counts home/away
      - lists up to N players per team with reason
    Works with summary produced by _realtime_summary_enriched (Option B).
    """
    missing_meta = summary.get("missing_meta") or []
    injuries_failed = any(str(x).startswith("injuries_err:") for x in missing_meta)    
    def _as_list(x):
        return x if isinstance(x, list) else []

    def _clean(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip()

    def _fmt_player(it: Dict[str, Any]) -> str:
        name = _clean(it.get("player"))
        reason_raw = _clean(it.get("reason"))
        status_type = _clean(it.get("status_type"))

        reason_fr = translate_reason_fr(reason_raw)

        if reason_fr:
            return f"{name} ({reason_fr})"

        # fallback (rare)
        if status_type and status_type.lower() != "missing fixture":
            return f"{name} ({status_type})"

        return f"{name}"

    home = _clean(summary.get("home")) or "Domicile"
    away = _clean(summary.get("away")) or "Extérieur"

    injuries_home = int(summary.get("injuries_home") or 0)
    injuries_away = int(summary.get("injuries_away") or 0)
    injuries_total = int(summary.get("injuries_total") or (injuries_home + injuries_away) or 0)

    home_list = _as_list(summary.get("top_injuries_home"))[:max_players_per_team]
    away_list = _as_list(summary.get("top_injuries_away"))[:max_players_per_team]

    home_players = ", ".join([_fmt_player(it) for it in home_list if isinstance(it, dict) and _clean(it.get("player"))])
    away_players = ", ".join([_fmt_player(it) for it in away_list if isinstance(it, dict) and _clean(it.get("player"))])

    # Status hints
    status_short = _clean(summary.get("status_short"))
    lineups_available = bool(summary.get("lineups_available"))
    lineups_expected_soon = bool(summary.get("lineups_expected_soon"))

    # If nothing at all
    
    if injuries_total <= 0 and not home_players and not away_players:
        if injuries_failed:
            return "Absences : indisponibles pour le moment (erreur de récupération des blessures)."
        if status_short == "NS" and not lineups_available:
            return "Absences : non disponibles pour l’instant (compositions non publiées)."
    return "Absences : aucune information notable."

    # Build lines
    line1 = f"Absences (pré-match) — {home}: {injuries_home} | {away}: {injuries_away}"

    # Build detail lines only if we have players
    lines = [line1]

    if home_players:
        lines.append(f"• {home}: {home_players}")
    if away_players:
        lines.append(f"• {away}: {away_players}")

    # Add small caution hint
    if status_short == "NS" and not lineups_available:
        if lineups_expected_soon:
            lines.append("ℹ️ Compositions attendues bientôt : prudence avant validation finale.")
        else:
            lines.append("ℹ️ Compositions non disponibles : prudence (infos peuvent évoluer).")

    return "\n".join(lines)

###--------- FONCTION SUR LA POSITION AU CLASSEMENT 

def _fetch_league_standings_________________(league_id: int, season: int) -> Optional[dict]:
    """
    API-Sports: GET /standings?league=..&season=..
    Returns raw response dict or None.
    """
    try:
        data = _api_get("standings", {"league": int(league_id), "season": int(season)}, timeout=10)
        resp = (data or {}).get("response", []) or []
        if not resp:
            return None
        return resp[0]  # usually one object with "league" + "standings"
    except Exception:
        return None

def _fetch_league_standings(league_id: int, season: int) -> Optional[dict]:
    """
    API-Sports: GET /standings?league=..&season=..
    Cached because standings are reused across matches.
    """
    try:
        data = _api_get(
            "standings",
            {"league": int(league_id), "season": int(season)},
            timeout=REALTIME_TIMEOUT_STANDINGS,
            cache_ttl=180,   # 3 minutes
        )
        resp = (data or {}).get("response", []) or []
        if not resp:
            return None
        return resp[0]
    except Exception:
        return None

def _extract_team_rank_from_standings(standings_payload: dict, team_id: int) -> Optional[dict]:
    """
    Extract rank/points/played for a team_id from standings payload.
    API often: payload["league"]["standings"] is a list of groups (list of lists).
    """
    try:
        league_obj = standings_payload.get("league") or {}
        groups = league_obj.get("standings") or []

        # groups can be: [[{...},{...}...]] or [{...},{...}]
        if isinstance(groups, list) and len(groups) > 0 and isinstance(groups[0], list):
            rows = [r for g in groups for r in (g or [])]
        else:
            rows = groups if isinstance(groups, list) else []

        for r in rows:
            if not isinstance(r, dict):
                continue
            t = (r.get("team") or {})
            if int(t.get("id") or -1) == int(team_id):
                all_ = r.get("all") or {}
                return {
                    "team_id": int(team_id),
                    "team": t.get("name"),
                    "rank": r.get("rank"),
                    "points": r.get("points"),
                    "played": all_.get("played"),
                    "win": all_.get("win"),
                    "draw": all_.get("draw"),
                    "lose": all_.get("lose"),
                    "goals_for": (all_.get("goals") or {}).get("for"),
                    "goals_against": (all_.get("goals") or {}).get("against"),
                    "form": r.get("form"),  # sometimes present like "WWDLW"
                }
        return None
    except Exception:
        return None


def _build_ranking_block_from_ctx(ctx: dict) -> dict:
    """
    Build ranking block (home/away) from realtime ctx.
    Safe: returns empty dict if unavailable.
    """
    try:
        league = ctx.get("league") or {}
        teams = ctx.get("teams") or {}

        league_id = league.get("id")
        season = league.get("season")

        home = (teams.get("home") or {})
        away = (teams.get("away") or {})

        home_id = home.get("id")
        away_id = away.get("id")

        if not league_id or season is None or not home_id or not away_id:
            return {}

        payload = _fetch_league_standings(int(league_id), int(season))
        if payload is None:
            return {
                "league_id": int(league_id),
                "season": int(season),
                "available": False,
                "missing": ["standings_empty"],
            }

        home_rank = _extract_team_rank_from_standings(payload, int(home_id))
        away_rank = _extract_team_rank_from_standings(payload, int(away_id))

        return {
            "league_id": int(league_id),
            "season": int(season),
            "available": True,
            "home": home_rank or {"team_id": int(home_id), "team": home.get("name"), "missing": ["team_not_in_standings"]},
            "away": away_rank or {"team_id": int(away_id), "team": away.get("name"), "missing": ["team_not_in_standings"]},
        }
    except Exception:
        return {}

def realtime_summary_enriched__________(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Résumé enrichi (pré-match / live / post-match) basé sur ctx API-Sports.
    Spécifique à ta structure injuries: item = {player, team, fixture, league}.
    Ne plante jamais.
    + Ajout: position au classement (ranking) home/away, sans casser les champs existants.
    """

    def _d(x): 
        return x if isinstance(x, dict) else {}

    def _l(x): 
        return x if isinstance(x, list) else []

    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip().lower()

    def _parse_iso_dt(s: Any) -> Optional[datetime]:
        if not isinstance(s, str) or not s.strip():
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    # -----------------------------
    # Base ctx
    # -----------------------------
    fixture = _d(ctx.get("fixture"))
    status = _d(fixture.get("status"))
    st_short = status.get("short")
    st_long = status.get("long")
    elapsed = status.get("elapsed")

    teams = _d(ctx.get("teams"))
    home_obj = _d(teams.get("home"))
    away_obj = _d(teams.get("away"))
    home_name = home_obj.get("name")
    away_name = away_obj.get("name")
    home_id = home_obj.get("id")
    away_id = away_obj.get("id")

    league = _d(ctx.get("league"))
    league_id = league.get("id")
    season = league.get("season")

    injuries_raw = _l(ctx.get("injuries"))
    injuries: List[Dict[str, Any]] = [it for it in injuries_raw if isinstance(it, dict)]

    # minutes to kickoff
    minutes_to_kickoff = None
    try:
        dt = _parse_iso_dt(fixture.get("date"))
        if dt is not None:
            now = datetime.now(timezone.utc)
            minutes_to_kickoff = int((dt - now).total_seconds() // 60)
    except Exception:
        pass

    # available blocks
    lineups = _l(ctx.get("lineups"))
    events = _l(ctx.get("events"))
    players = _l(ctx.get("players"))
    stats = _l(ctx.get("statistics"))

    # meta missing
    meta = _d(ctx.get("meta"))
    missing_meta = meta.get("missing", []) if isinstance(meta.get("missing"), list) else []

    # -----------------------------
    # Injuries split home/away (counts)
    # -----------------------------
    injuries_home = 0
    injuries_away = 0
    hn = _norm(home_name)
    an = _norm(away_name)

    for it in injuries:
        team_name = _d(it.get("team")).get("name")
        tn = _norm(team_name)
        if tn and hn and tn == hn:
            injuries_home += 1
        elif tn and an and tn == an:
            injuries_away += 1

    # build top injuries (up to 3) - keeps your existing behavior
    top_injuries = []
    for it in injuries[:3]:
        p = _d(it.get("player"))
        t = _d(it.get("team"))
        top_injuries.append({
            "team": t.get("name"),
            "player": p.get("name"),
            "status_type": p.get("type"),   # ex: Missing Fixture
            "reason": p.get("reason"),      # ex: Injury
        })
        
    # split detailed injuries lists
    injuries_home_list = []
    injuries_away_list = []

    for it in injuries:
        team_name = _d(it.get("team")).get("name")
        tn = _norm(team_name)
        if tn and hn and tn == hn:
            injuries_home_list.append(it)
        elif tn and an and tn == an:
            injuries_away_list.append(it)

    top_injuries_home = []
    for it in injuries_home_list[:3]:
        p = _d(it.get("player"))
        t = _d(it.get("team"))
        top_injuries_home.append({
            "team": t.get("name"),
            "player": p.get("name"),
            "status_type": p.get("type"),
            "reason": p.get("reason"),
        })

    top_injuries_away = []
    for it in injuries_away_list[:3]:
        p = _d(it.get("player"))
        t = _d(it.get("team"))
        top_injuries_away.append({
            "team": t.get("name"),
            "player": p.get("name"),
            "status_type": p.get("type"),
            "reason": p.get("reason"),
        })

    # lineups expected soon if match is close and still empty
    lineups_expected_soon = False
    try:
        if st_short == "NS" and minutes_to_kickoff is not None and minutes_to_kickoff <= 120 and len(lineups) == 0:
            lineups_expected_soon = True
    except Exception:
        pass

    # started/finished flags
    finished_set = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "SUSP", "INT"}
    is_finished = bool(st_short in finished_set)
    is_started = bool(st_short not in (None, "NS") and not is_finished)

    # -----------------------------
    # ✅ Ranking block (standings)
    # -----------------------------
    def _fetch_standings_payload(lid: int, seas: int) -> Optional[dict]:
        """
        Calls API-Sports standings endpoint via your existing _api_get.
        Returns resp[0] or None.
        """
        try:
            # _api_get must exist in your module
            if "_api_get" not in globals():
                return None
            #data = _api_get("standings", {"league": int(lid), "season": int(seas)}, timeout=10)
            data = _api_get(
                                "standings",
                                {"league": int(lid), "season": int(seas)},
                                timeout=REALTIME_TIMEOUT_STANDINGS,
                                cache_ttl=180
                            )
            resp = (data or {}).get("response", []) or []
            if not resp:
                return None
            return resp[0]
        except Exception:
            return None

    def _extract_team_rank(payload: dict, team_id_: int) -> Optional[dict]:
        """
        Extract minimal rank info for one team from standings payload.
        Safe across shapes: standings = [[...]] or [...]
        """
        try:
            league_obj = payload.get("league") or {}
            standings = league_obj.get("standings") or []

            # flatten if grouped
            rows: List[dict] = []
            if isinstance(standings, list) and standings and isinstance(standings[0], list):
                for g in standings:
                    if isinstance(g, list):
                        rows.extend([r for r in g if isinstance(r, dict)])
            elif isinstance(standings, list):
                rows = [r for r in standings if isinstance(r, dict)]

            for r in rows:
                t = r.get("team") or {}
                if int(t.get("id") or -1) == int(team_id_):
                    all_ = r.get("all") or {}
                    goals_ = all_.get("goals") or {}
                    return {
                        "team_id": int(team_id_),
                        "team": t.get("name"),
                        "rank": r.get("rank"),
                        "points": r.get("points"),
                        "played": all_.get("played"),
                        "win": all_.get("win"),
                        "draw": all_.get("draw"),
                        "lose": all_.get("lose"),
                        "goals_for": goals_.get("for"),
                        "goals_against": goals_.get("against"),
                        "form": r.get("form"),
                    }
            return None
        except Exception:
            return None

    ranking: Dict[str, Any] = {
        "available": False,
        "league_id": league_id,
        "season": season,
        "home": {"team_id": home_id, "team": home_name},
        "away": {"team_id": away_id, "team": away_name},
        "missing": []
    }

    try:
        # only attempt if we have necessary ids
        if league_id and season is not None and home_id and away_id:
            payload = _fetch_standings_payload(int(league_id), int(season))
            if payload is None:
                ranking["missing"].append("standings_empty_or_unavailable")
            else:
                home_rank = _extract_team_rank(payload, int(home_id))
                away_rank = _extract_team_rank(payload, int(away_id))
                ranking["available"] = True
                ranking["home"] = home_rank or {"team_id": int(home_id), "team": home_name, "missing": ["team_not_in_standings"]}
                ranking["away"] = away_rank or {"team_id": int(away_id), "team": away_name, "missing": ["team_not_in_standings"]}
        else:
            ranking["missing"].append("ranking_ids_missing")
    except Exception:
        ranking["available"] = False
        ranking["missing"].append("ranking_error")

    # -----------------------------
    # Return (unchanged fields + ranking)
    # -----------------------------
    return {
        "status_short": st_short,
        "status_long": st_long,
        "elapsed": elapsed,
        "home": home_name,
        "away": away_name,

        "is_started": is_started,
        "is_finished": is_finished,

        "minutes_to_kickoff": minutes_to_kickoff,
        "lineups_available": len(lineups) > 0,
        "events_available": len(events) > 0,
        "players_available": len(players) > 0,
        "statistics_available": len(stats) > 0,

        "injuries_total": len(injuries),
        "injuries_home": injuries_home,
        "injuries_away": injuries_away,
        "top_injuries": top_injuries,

        "lineups_expected_soon": bool(lineups_expected_soon),
        "missing_meta": missing_meta,

        # ✅ new (safe)
        "ranking": ranking,
        "top_injuries_home": top_injuries_home,
        "top_injuries_away": top_injuries_away,
    }

def realtime_summary_enriched(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Résumé enrichi (pré-match / live / post-match) basé sur ctx API-Sports.
    Compatible avec format_absences_summary(summary):
      - top_injuries_home
      - top_injuries_away
      - top_injuries
      - ranking
    Ne plante jamais.
    """

    def _d(x):
        return x if isinstance(x, dict) else {}

    def _l(x):
        return x if isinstance(x, list) else []

    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip().lower()

    def _parse_iso_dt(s: Any) -> Optional[datetime]:
        if not isinstance(s, str) or not s.strip():
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    # -----------------------------
    # Base ctx
    # -----------------------------
    fixture = _d(ctx.get("fixture"))
    status = _d(fixture.get("status"))
    st_short = status.get("short")
    st_long = status.get("long")
    elapsed = status.get("elapsed")

    teams = _d(ctx.get("teams"))
    home_obj = _d(teams.get("home"))
    away_obj = _d(teams.get("away"))
    home_name = home_obj.get("name")
    away_name = away_obj.get("name")
    home_id = home_obj.get("id")
    away_id = away_obj.get("id")

    league = _d(ctx.get("league"))
    league_id = league.get("id")
    season = league.get("season")

    injuries_raw = _l(ctx.get("injuries"))
    injuries: List[Dict[str, Any]] = [it for it in injuries_raw if isinstance(it, dict)]

    lineups = _l(ctx.get("lineups"))
    events = _l(ctx.get("events"))
    players = _l(ctx.get("players"))
    stats = _l(ctx.get("statistics"))

    meta = _d(ctx.get("meta"))
    missing_meta = meta.get("missing", []) if isinstance(meta.get("missing"), list) else []
    skipped_meta = meta.get("skipped", []) if isinstance(meta.get("skipped"), list) else []
    errors_meta = meta.get("errors", []) if isinstance(meta.get("errors"), list) else []

    # minutes to kickoff
    minutes_to_kickoff = None
    try:
        dt = _parse_iso_dt(fixture.get("date"))
        if dt is not None:
            now = datetime.now(timezone.utc)
            minutes_to_kickoff = int((dt - now).total_seconds() // 60)
    except Exception:
        pass

    # -----------------------------
    # Injuries split home/away
    # -----------------------------
    hn = _norm(home_name)
    an = _norm(away_name)

    injuries_home_list: List[Dict[str, Any]] = []
    injuries_away_list: List[Dict[str, Any]] = []

    for it in injuries:
        team_name = _d(it.get("team")).get("name")
        tn = _norm(team_name)
        if tn and hn and tn == hn:
            injuries_home_list.append(it)
        elif tn and an and tn == an:
            injuries_away_list.append(it)

    injuries_home = len(injuries_home_list)
    injuries_away = len(injuries_away_list)

    def _inj_to_rec(it: Dict[str, Any]) -> Dict[str, Any]:
        p = _d(it.get("player"))
        t = _d(it.get("team"))
        return {
            "team": t.get("name"),
            "player": p.get("name"),
            "status_type": p.get("type"),
            "reason": p.get("reason"),
        }

    top_injuries_home = [_inj_to_rec(it) for it in injuries_home_list[:3]]
    top_injuries_away = [_inj_to_rec(it) for it in injuries_away_list[:3]]
    top_injuries = [_inj_to_rec(it) for it in injuries[:3]]

    # lineups expected soon
    lineups_expected_soon = False
    try:
        if st_short == "NS" and minutes_to_kickoff is not None and minutes_to_kickoff <= 120 and len(lineups) == 0:
            lineups_expected_soon = True
    except Exception:
        pass

    # started/finished flags
    finished_set = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "SUSP", "INT"}
    is_finished = bool(st_short in finished_set)
    is_started = bool(st_short not in (None, "NS") and not is_finished)

    # -----------------------------
    # Ranking block
    # -----------------------------
    def _fetch_standings_payload(lid: int, seas: int) -> Optional[dict]:
        try:
            data = _api_get(
                "standings",
                {"league": int(lid), "season": int(seas)},
                timeout=REALTIME_TIMEOUT_STANDINGS,
                cache_ttl=0,   # debug: pas de cache
            )

            if not isinstance(data, dict):
                print("[standings] invalid data type:", type(data), "league=", lid, "season=", seas)
                return None

            resp = data.get("response", []) or []
            print("[standings] league=", lid, "season=", seas, "resp_len=", len(resp))

            if not resp:
                print("[standings] EMPTY response:", data)
                return None

            return resp[0]

        except Exception as e:
            print("[standings] ERROR:", type(e).__name__, str(e), "league=", lid, "season=", seas)
            return None
    
    def _extract_team_rank(payload: dict, team_id_: int) -> Optional[dict]:
        try:
            league_obj = payload.get("league") or {}
            standings = league_obj.get("standings") or []

            rows: List[dict] = []
            if isinstance(standings, list) and standings and isinstance(standings[0], list):
                for g in standings:
                    if isinstance(g, list):
                        rows.extend([r for r in g if isinstance(r, dict)])
            elif isinstance(standings, list):
                rows = [r for r in standings if isinstance(r, dict)]

            for r in rows:
                t = r.get("team") or {}
                if int(t.get("id") or -1) == int(team_id_):
                    all_ = r.get("all") or {}
                    goals_ = all_.get("goals") or {}
                    return {
                        "team_id": int(team_id_),
                        "team": t.get("name"),
                        "rank": r.get("rank"),
                        "points": r.get("points"),
                        "played": all_.get("played"),
                        "win": all_.get("win"),
                        "draw": all_.get("draw"),
                        "lose": all_.get("lose"),
                        "goals_for": goals_.get("for"),
                        "goals_against": goals_.get("against"),
                        "form": r.get("form"),
                    }
            return None
        except Exception:
            return None

    ranking: Dict[str, Any] = {
        "available": False,
        "league_id": league_id,
        "season": season,
        "home": {"team_id": home_id, "team": home_name},
        "away": {"team_id": away_id, "team": away_name},
        "missing": []
    }

    try:
        if league_id and season is not None and home_id and away_id:
            payload = _fetch_standings_payload(int(league_id), int(season))
            if payload is None:
                ranking["missing"].append(f"standings_empty_or_unavailable:league={league_id}:season={season}")
            else:
                home_rank = _extract_team_rank(payload, int(home_id))
                away_rank = _extract_team_rank(payload, int(away_id))
                ranking["available"] = True
                ranking["home"] = home_rank or {
                    "team_id": int(home_id),
                    "team": home_name,
                    "missing": [f"team_not_in_standings:league={league_id}:season={season}"]
                }
                ranking["away"] = away_rank or {
                    "team_id": int(away_id),
                    "team": away_name,
                    "missing": [f"team_not_in_standings:league={league_id}:season={season}"]
                }
        else:
            ranking["missing"].append(f"standings_empty_or_unavailable:league={league_id}:season={season}")
    except Exception:
        ranking["available"] = False
        ranking["missing"].append("ranking_error")

    # -----------------------------
    # Return
    # -----------------------------
    return {
        "status_short": st_short,
        "status_long": st_long,
        "elapsed": elapsed,
        "home": home_name,
        "away": away_name,

        "is_started": is_started,
        "is_finished": is_finished,

        "minutes_to_kickoff": minutes_to_kickoff,
        "lineups_available": len(lineups) > 0,
        "events_available": len(events) > 0,
        "players_available": len(players) > 0,
        "statistics_available": len(stats) > 0,

        "injuries_total": len(injuries),
        "injuries_home": injuries_home,
        "injuries_away": injuries_away,

        # ✅ compatible with format_absences_summary
        "top_injuries_home": top_injuries_home,
        "top_injuries_away": top_injuries_away,
        "top_injuries": top_injuries,

        "lineups_expected_soon": bool(lineups_expected_soon),

        "missing_meta": missing_meta,
        "skipped_meta": skipped_meta,
        "errors_meta": errors_meta,

        "ranking": ranking,
    }

def _build_realtime_block_____________(
    features_df: Any,
    league_code: Optional[str] = None,
    home_name: Any = None,
    away_name: Any = None,
    match_date: Any = None,
    season_df: Any = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Realtime enrichment block (Option B):
    - summary.top_injuries_home (max 3)
    - summary.top_injuries_away (max 3)

    Does NOT change prediction. Only enriches realtime_risk + notes.
    """

    # -----------------------------
    # helpers (safe & local)
    # -----------------------------
    def _as_scalar(v):
        try:
            if isinstance(v, pd.Series):
                return v.iloc[0] if len(v) else None
            if isinstance(v, (list, tuple, np.ndarray)):
                return v[0] if len(v) else None
        except Exception:
            return None
        return v

    def _as_dict(x):
        return x if isinstance(x, dict) else {}

    def _as_list(x):
        return x if isinstance(x, list) else []

    def _safe_len(x):
        try:
            return len(x) if x is not None else 0
        except Exception:
            return 0

    def _parse_iso_dt(s: Any) -> Optional[datetime]:
        if not isinstance(s, str) or not s.strip():
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _minutes_to_kickoff(ctx: Dict[str, Any]) -> Optional[int]:
        fixture = _as_dict(ctx.get("fixture"))
        dt = _parse_iso_dt(fixture.get("date"))
        if dt is None:
            return None
        now = datetime.now(timezone.utc)
        return int((dt - now).total_seconds() // 60)

    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip().lower()

    def _inj_to_rec(it: Dict[str, Any]) -> Dict[str, Any]:
        p = _as_dict(it.get("player"))
        t = _as_dict(it.get("team"))
        rec = {
            "team": t.get("name"),
            "player": p.get("name"),
            "status_type": p.get("type"),
            "reason": p.get("reason"),
            # optionnel (si tu veux UI + riche):
            # "player_id": p.get("id"),
            # "photo": p.get("photo"),
            # "team_id": t.get("id"),
            # "logo": t.get("logo"),
        }
        return {k: v for k, v in rec.items() if v is not None and str(v).strip() != ""}

    # --- summary (Option B) ---
    
    def _risk_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
        st = summary.get("status_short")
        injuries_total = int(summary.get("injuries_total") or 0)
        lineups_ok = bool(summary.get("lineups_available"))

        finished_set = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "SUSP", "INT"}
        if st in finished_set:
            return {"risk_level": "HIGH", "risk_score": 0.9, "reasons": [f"fixture_status:{st}"]}

        # Pre-match: injuries OR no lineups -> medium
        if st == "NS":
            if injuries_total > 0 or (not lineups_ok):
                score = 0.4
                if injuries_total >= 8:
                    score = 0.55
                return {"risk_level": "MEDIUM", "risk_score": score, "reasons": ["possible_injuries_or_lineup_changes"]}
            return {"risk_level": "LOW", "risk_score": 0.1, "reasons": ["pre_match_no_major_signals"]}

        return {"risk_level": "MEDIUM", "risk_score": 0.5, "reasons": [f"fixture_status:{st}"]}

    # -----------------------------
    # read inputs (df or args)
    # -----------------------------
    if home_name is None:
        home_name = _safe_get_first(features_df, "home")
    if away_name is None:
        away_name = _safe_get_first(features_df, "away")
    if match_date is None:
        match_date = _safe_get_first(features_df, "match_date")

    use_realtime_val = _as_scalar(_safe_get_first(features_df, "_use_realtime"))
    use_realtime = bool(use_realtime_val) if use_realtime_val is not None else False

    missing_fields = []
    if not home_name:
        missing_fields.append("home_name_missing")
    if not away_name:
        missing_fields.append("away_name_missing")
    if not match_date:
        missing_fields.append("match_date_missing")

    if not use_realtime:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": ["realtime_not_enabled_or_unavailable"],
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, "realtime: not enabled"

    if missing_fields:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": missing_fields,
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, f"realtime: skipped_missing_fields={missing_fields}"
    
        # -----------------------------
    # use existing fixture_id first
    # -----------------------------
    fixture_id_existing = None
    try:
        if isinstance(features_df, pd.DataFrame) and "fixture_id" in features_df.columns:
            v = _safe_get_first(features_df, "fixture_id")
            if v is not None and str(v).strip() != "":
                fixture_id_existing = int(v)
    except Exception:
        fixture_id_existing = None

    if fixture_id_existing is not None:
        fixture_id_int = fixture_id_existing
    else:
        try:
            fixture_id = _safe_resolve_fixture_id(
                home_name, away_name, match_date,
                league_code=league_code,
                season_df=season_df,
                features_df=features_df
            )
        except Exception as e:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": [f"fixture_resolve_error:{type(e).__name__}"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: resolve error={type(e).__name__}"

        if fixture_id is None or str(fixture_id).strip() == "":
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_not_found"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, "realtime: fixture not found"

        try:
            fixture_id_int = int(fixture_id)
        except Exception:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_invalid"],
                "reasons": ["fixture_id_invalid"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: fixture id invalid={fixture_id}"

    # -----------------------------
    # resolve fixture_id
    # -----------------------------
    try:
        fixture_id = _safe_resolve_fixture_id(
            home_name, away_name, match_date,
            league_code=league_code,
            season_df=season_df,
            features_df=features_df
        )
    except Exception as e:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": [f"fixture_resolve_error:{type(e).__name__}"],
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, f"realtime: resolve error={type(e).__name__}"

    if fixture_id is None or str(fixture_id).strip() == "":
        block = {
            "available": False,
            "fixture_id": None,
            "missing": ["fixture_id_not_found"],
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, "realtime: fixture not found"

    try:
        fixture_id_int = int(fixture_id)
    except Exception:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": ["fixture_id_invalid"],
            "reasons": ["fixture_id_invalid"],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, f"realtime: fixture id invalid={fixture_id}"

    # -----------------------------
    # fetch ctx & compute
    # -----------------------------
    debug_rt = os.getenv("DEBUG_REALTIME", "0") == "1"

    try:
        ctx = _fetch_realtime_context(fixture_id_int)

        if ctx is not None and not isinstance(ctx, dict):
            block = {
                "available": False,
                "fixture_id": fixture_id_int,
                "missing": [f"realtime_ctx_invalid:{type(ctx).__name__}"],
                "reasons": [f"realtime_ctx_invalid:{type(ctx).__name__}"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: ok fixture_id={fixture_id_int} but ctx invalid type={type(ctx).__name__}"

        if ctx is None:
            block = {
                "available": False,
                "fixture_id": fixture_id_int,
                "missing": ["realtime_ctx_empty"],
                "reasons": ["realtime_ctx_empty"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: ok fixture_id={fixture_id_int} but ctx empty"

        #summary = _realtime_summary_enriched(ctx)
        summary = realtime_summary_enriched(ctx)
        risk_pm = _risk_from_summary(summary)
        summary["absences_text"] = format_absences_summary(summary)

        # optional legacy scorer (guarded)
        risk_raw = {}
        try:
            rr = _realtime_risk_score(ctx)  # if exists
            if isinstance(rr, dict):
                risk_raw = rr
        except Exception:
            risk_raw = {}

        st_short = summary.get("status_short")
        chosen = risk_pm if st_short == "NS" else (risk_raw or risk_pm)

        block = {
            "available": True,
            "fixture_id": fixture_id_int,
            "missing": [],
            "reasons": chosen.get("reasons", []),
            "risk_level": chosen.get("risk_level", "UNKNOWN"),
            "risk_score": float(chosen.get("risk_score", 0.0) or 0.0),
            "summary": summary,
        }

        if debug_rt:
            try:
                fixture = _as_dict(ctx.get("fixture"))
                status = _as_dict(fixture.get("status"))
                block["debug"] = {
                    "ctx_keys": sorted(list(ctx.keys())),
                    "fixture_keys": sorted(list(fixture.keys())) if isinstance(fixture, dict) else [],
                    "status_short": status.get("short"),
                    "injuries_count": _safe_len(ctx.get("injuries")),
                    "injuries_home": summary.get("injuries_home"),
                    "injuries_away": summary.get("injuries_away"),
                    "missing_meta": summary.get("missing_meta"),
                    "risk_pm": risk_pm,
                    "risk_raw": risk_raw,
                    "risk_chosen": chosen,
                }
            except Exception:
                block["debug"] = {"debug_error": "failed_to_build_debug"}

        return block, f"realtime: ok fixture_id={fixture_id_int}"

    except Exception as e:
        block = {
            "available": False,
            "fixture_id": fixture_id_int,
            "missing": [f"realtime_error:{type(e).__name__}"],
            "reasons": [f"realtime_error:{type(e).__name__}"],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, f"realtime: ok fixture_id={fixture_id_int} but error={type(e).__name__}"


def _build_realtime_block(
    features_df: Any,
    league_code: Optional[str] = None,
    home_name: Any = None,
    away_name: Any = None,
    match_date: Any = None,
    season_df: Any = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Realtime enrichment block:
    - uses existing fixture_id if already available in features_df
    - fetches realtime context
    - enriches summary via realtime_summary_enriched
    - returns realtime_risk block + note
    """

    def _as_scalar(v):
        try:
            if isinstance(v, pd.Series):
                return v.iloc[0] if len(v) else None
            if isinstance(v, (list, tuple, np.ndarray)):
                return v[0] if len(v) else None
        except Exception:
            return None
        return v

    def _as_dict(x):
        return x if isinstance(x, dict) else {}

    def _safe_len(x):
        try:
            return len(x) if x is not None else 0
        except Exception:
            return 0

    def _risk_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
        st = summary.get("status_short")
        injuries_total = int(summary.get("injuries_total") or 0)
        lineups_ok = bool(summary.get("lineups_available"))

        finished_set = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "SUSP", "INT"}
        if st in finished_set:
            return {"risk_level": "HIGH", "risk_score": 0.9, "reasons": [f"fixture_status:{st}"]}

        if st == "NS":
            if injuries_total > 0 or (not lineups_ok):
                score = 0.4
                if injuries_total >= 8:
                    score = 0.55
                return {
                    "risk_level": "MEDIUM",
                    "risk_score": score,
                    "reasons": ["possible_injuries_or_lineup_changes"]
                }
            return {"risk_level": "LOW", "risk_score": 0.1, "reasons": ["pre_match_no_major_signals"]}

        return {"risk_level": "MEDIUM", "risk_score": 0.5, "reasons": [f"fixture_status:{st}"]}

    # -----------------------------
    # read inputs
    # -----------------------------
    if home_name is None:
        home_name = _safe_get_first(features_df, "home")
    if away_name is None:
        away_name = _safe_get_first(features_df, "away")
    if match_date is None:
        match_date = _safe_get_first(features_df, "match_date")

    use_realtime_val = _as_scalar(_safe_get_first(features_df, "_use_realtime"))
    use_realtime = bool(use_realtime_val) if use_realtime_val is not None else False

    missing_fields = []
    if not home_name:
        missing_fields.append("home_name_missing")
    if not away_name:
        missing_fields.append("away_name_missing")
    if not match_date:
        missing_fields.append("match_date_missing")

    if not use_realtime:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": ["realtime_not_enabled_or_unavailable"],
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, "realtime: not enabled"

    if missing_fields:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": missing_fields,
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, f"realtime: skipped_missing_fields={missing_fields}"

    
    # use existing fixture_id first
    # -----------------------------
    fixture_id_int = None
    try:
        if isinstance(features_df, pd.DataFrame) and "fixture_id" in features_df.columns:
            v = _safe_get_first(features_df, "fixture_id")
            if v is not None and str(v).strip() != "":
                fixture_id_int = int(v)
    except Exception:
        fixture_id_int = None
    """
    # fallback resolve only if needed
    if fixture_id_int is None:
        try:
            fixture_id = _safe_resolve_fixture_id(
                home_name, away_name, match_date,
                league_code=league_code,
                season_df=season_df,
                features_df=features_df
            )
        except Exception as e:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": [f"fixture_resolve_error:{type(e).__name__}"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: resolve error={type(e).__name__}"

        if fixture_id is None or str(fixture_id).strip() == "":
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_not_found"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, "realtime: fixture not found"

        try:
            fixture_id_int = int(fixture_id)
        except Exception:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_invalid"],
                "reasons": ["fixture_id_invalid"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: fixture id invalid={fixture_id}"
    """
    # fallback resolve only if needed
    if fixture_id_int is None:
        try:
            fixture_id = _safe_resolve_fixture_id(
                home_name, away_name, match_date,
                league_code=league_code,
                season_df=season_df,
                features_df=features_df
            )
        except Exception as e:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": [f"fixture_resolve_error:{type(e).__name__}"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: resolve error={type(e).__name__}"

        if fixture_id is None or str(fixture_id).strip() == "":
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_not_found"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, "realtime: fixture not found"

        try:
            fixture_id_int = int(fixture_id)
        except Exception:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_invalid"],
                "reasons": ["fixture_id_invalid"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: fixture id invalid={fixture_id}"

    # -----------------------------
    # fetch ctx & compute
    # -----------------------------
    debug_rt = os.getenv("DEBUG_REALTIME", "0") == "1"

    try:
        ctx = _fetch_realtime_context(fixture_id_int)

        if ctx is not None and not isinstance(ctx, dict):
            block = {
                "available": False,
                "fixture_id": fixture_id_int,
                "missing": [f"realtime_ctx_invalid:{type(ctx).__name__}"],
                "reasons": [f"realtime_ctx_invalid:{type(ctx).__name__}"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: ok fixture_id={fixture_id_int} but ctx invalid type={type(ctx).__name__}"

        if ctx is None:
            block = {
                "available": False,
                "fixture_id": fixture_id_int,
                "missing": ["realtime_ctx_empty"],
                "reasons": ["realtime_ctx_empty"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: ok fixture_id={fixture_id_int} but ctx empty"

        summary = realtime_summary_enriched(ctx)
        risk_pm = _risk_from_summary(summary)
        summary["absences_text"] = format_absences_summary(summary)

        risk_raw = {}
        try:
            rr = _realtime_risk_score(ctx)
            if isinstance(rr, dict):
                risk_raw = rr
        except Exception:
            risk_raw = {}

        st_short = summary.get("status_short")
        chosen = risk_pm if st_short == "NS" else (risk_raw or risk_pm)

        block = {
            "available": True,
            "fixture_id": fixture_id_int,
            "missing": [],
            "reasons": chosen.get("reasons", []),
            "risk_level": chosen.get("risk_level", "UNKNOWN"),
            "risk_score": float(chosen.get("risk_score", 0.0) or 0.0),
            "summary": summary,
        }

        if debug_rt:
            try:
                fixture = _as_dict(ctx.get("fixture"))
                status = _as_dict(fixture.get("status"))
                meta = _as_dict(ctx.get("meta"))
                block["debug"] = {
                    "ctx_keys": sorted(list(ctx.keys())),
                    "fixture_keys": sorted(list(fixture.keys())) if isinstance(fixture, dict) else [],
                    "status_short": status.get("short"),
                    "injuries_count": _safe_len(ctx.get("injuries")),
                    "injuries_home": summary.get("injuries_home"),
                    "injuries_away": summary.get("injuries_away"),
                    "missing_meta": summary.get("missing_meta"),
                    "skipped_meta": summary.get("skipped_meta"),
                    "errors_meta": summary.get("errors_meta"),
                    "ctx_meta_missing": meta.get("missing", []),
                    "ctx_meta_errors": meta.get("errors", []),
                    "ctx_meta_skipped": meta.get("skipped", []),
                    "risk_pm": risk_pm,
                    "risk_raw": risk_raw,
                    "risk_chosen": chosen,
                }
            except Exception:
                block["debug"] = {"debug_error": "failed_to_build_debug"}

        return block, f"realtime: ok fixture_id={fixture_id_int}"

    except Exception as e:
        block = {
            "available": False,
            "fixture_id": fixture_id_int,
            "missing": [f"realtime_error:{type(e).__name__}"],
            "reasons": [f"realtime_error:{type(e).__name__}"],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, f"realtime: ok fixture_id={fixture_id_int} but error={type(e).__name__}"

def log_prediction(prediction):
    log_data = {
        "request_date": datetime.datetime.utcnow().isoformat(),
        #"input": data,
        "prediction": prediction
    }
    print("➡️ Donnée à logger :", prediction)
    os.makedirs("logs", exist_ok=True)
    with open("logs/logs.jsonl", "a") as f:
        f.write(json.dumps(log_data) + "\n")
        
    
        
def log_dataframe_features_to_file(features_df, home, away, match_date, output_path="logs/features_log.jsonl"):
    os.makedirs("logs", exist_ok=True)
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "home_team": home,
        "away_team": away,
        "match_date": str(match_date),
        "features": features_df.to_dict(orient="records")[0]
    }
    with open(output_path, "a") as f:
        f.write(json.dumps(log_data) + "\n")


###----------- DEBUT DES FONCTIONS DE PREDICTION-----
# -*- coding: utf-8 -*-
REALTIME_API_KEY = os.getenv("REALTIME_API_KEY")


# -------------------------------------------------------------------
# 🔢 Conventions BetSmart (IMPORTANT: éviter toute confusion 0/1/2)
# 0 = Victoire domicile (Home)
# 1 = Match nul (Draw)
# 2 = Victoire extérieur (Away)
# -------------------------------------------------------------------


# =========================
# Utils probas / mapping
# =========================
def _proba_for_class(model, X, cls_label, default=0.0):
    """Récupère une probabilité de classe en utilisant model.classes_ (robuste à l'ordre)."""
    try:
        classes = list(getattr(model, "classes_", []))
        if cls_label not in classes:
            return float(default)
        idx = classes.index(cls_label)
        p = model.predict_proba(X)[0][idx]
        return float(p)
    except Exception:
        return float(default)


def _normalize3(p0, p1, p2):
    p0 = float(p0)
    p1 = float(p1)
    p2 = float(p2)
    s = p0 + p1 + p2
    if not np.isfinite(s) or s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (p0 / s, p1 / s, p2 / s)


def _final_prediction_from_probas(p0, p1, p2):
    arr = np.array([p0, p1, p2], dtype=float)
    if not np.isfinite(arr).all():
        return LABEL_DRAW
    return int([LABEL_HOME, LABEL_DRAW, LABEL_AWAY][int(np.argmax(arr))])


def _format_pct(p):
    try:
        return f"{round(float(p) * 100, 0)}%"
    except Exception:
        return "0%"


# =========================
# Config ligue
# =========================
RACINE_PROJET = pathlib.Path(__file__).resolve().parents[1]
chemin_csv = RACINE_PROJET / "data" / "champ_config.json"


@lru_cache(maxsize=1)
def _load_champ_config():
    with open(chemin_csv, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # double index: str et int
    cfg_by_str = {str(k): v for k, v in cfg.items()}
    cfg_by_int = {}
    for k, v in cfg.items():
        try:
            cfg_by_int[int(k)] = v
        except Exception:
            pass
    return {"by_str": cfg_by_str, "by_int": cfg_by_int}


def _get_params(league_code):
    cfg = _load_champ_config()
    if league_code in cfg["by_int"]:
        return cfg["by_int"][league_code]
    if league_code in cfg["by_str"]:
        return cfg["by_str"][league_code]
    try:
        return cfg["by_int"].get(int(league_code), cfg["by_str"].get(str(league_code), {}))
    except Exception:
        return cfg["by_str"].get(str(league_code), {})


def parametres(league_code):
    """
    Retourne 8 valeurs:
    (bookmaker_margin, uncertainty_threshold, importance, season_stage,
     upset_threshold, skip_threshold, bogey_weight, gki_weight)
    """
    p = _get_params(league_code)

    bookmaker_margin = float(p.get("bookmaker_margin", 0.0711))
    uncertainty_threshold = float(p.get("uncertainty_threshold", 0.12))
    importance = int(p.get("importance", 3))
    season_stage = str(p.get("season_stage", "mid"))

    upset_threshold = float(p.get("upset_threshold", 0.55))
    skip_threshold = float(p.get("skip_threshold", 1.50))
    bogey_weight = float(p.get("bogey_weight", 0.40))
    gki_weight = float(p.get("gki_weight", 0.60))

    return (
        bookmaker_margin,
        uncertainty_threshold,
        importance,
        season_stage,
        upset_threshold,
        skip_threshold,
        bogey_weight,
        gki_weight,
    )


# ---------- AJOUT: hyperparamètres de la porte de forme ----------
def parametres_form_gate(league_code):
    """
    Lit (si dispo) les hyperparamètres de la 'porte forme' depuis champ_config.json :
      - k_market_form  : intensité max de transfert H↔A (0..1)  (défaut 0.45)
      - gate_slope     : pente de la sigmoïde (défaut 14.0)
      - gate_tolerance : tolérance d’écart de forme avant d’agir (défaut 0.036)
    """
    p = _get_params(league_code)
    k = float(p.get("k_market_form", 0.45))
    slope = float(p.get("gate_slope", 14.0))
    tau = float(p.get("gate_tolerance", 0.036))
    return k, slope, tau


# =========================
# Safe adapters
# =========================
def _as_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _safe_parametres(league_code):
    """
    S'adapte à l'ancienne signature (4 valeurs) et la nouvelle (8).
    Force les types et fournit des valeurs par défaut sûres.
    """
    vals = parametres(league_code)

    if isinstance(vals, (list, tuple)) and len(vals) == 4:
        bookmaker_margin, uncertainty_threshold, importance, season_stage = vals
        upset_threshold, skip_threshold, bogey_weight, gki_weight = 0.55, 1.50, 0.40, 0.60
    elif isinstance(vals, (list, tuple)) and len(vals) >= 8:
        (
            bookmaker_margin,
            uncertainty_threshold,
            importance,
            season_stage,
            upset_threshold,
            skip_threshold,
            bogey_weight,
            gki_weight,
        ) = vals[:8]
    else:
        bookmaker_margin, uncertainty_threshold, importance, season_stage = 0.0711, 0.12, 3, "mid"
        upset_threshold, skip_threshold, bogey_weight, gki_weight = 0.55, 1.50, 0.40, 0.60

    bookmaker_margin = _as_float(bookmaker_margin, 0.0711)
    uncertainty_threshold = _as_float(uncertainty_threshold, 0.12)
    importance = _as_int(importance, 3)
    season_stage = str(season_stage) if season_stage is not None else "mid"
    upset_threshold = _as_float(upset_threshold, 0.55)
    skip_threshold = _as_float(skip_threshold, 1.50)
    bogey_weight = _as_float(bogey_weight, 0.40)
    gki_weight = _as_float(gki_weight, 0.60)

    return (
        bookmaker_margin,
        uncertainty_threshold,
        importance,
        season_stage,
        upset_threshold,
        skip_threshold,
        bogey_weight,
        gki_weight,
    )


def _fav_by_demarged(bh: float, bd: float, ba: float, eps: float = 0.02):
    """
    Détermine le favori via probabilités implicites dé-margées.
    Retourne (side, pH2, pA2, gap) où side ∈ {"home","away", None}.
    """
    bh = float(bh)
    bd = float(bd)
    ba = float(ba)
    if min(bh, bd, ba) <= 1.0 or any(not np.isfinite(x) for x in (bh, bd, ba)):
        return None, np.nan, np.nan, 0.0

    qH, qD, qA = 1.0 / bh, 1.0 / bd, 1.0 / ba
    s = qH + qD + qA
    if s <= 0:
        return None, np.nan, np.nan, 0.0

    pH, pD, pA = qH / s, qD / s, qA / s
    denom = pH + pA
    if denom <= 0:
        return None, np.nan, np.nan, 0.0
    pH2, pA2 = pH / denom, pA / denom
    gap = pH2 - pA2

    if gap > eps:
        side = "home"
    elif gap < -eps:
        side = "away"
    else:
        side = None
    return side, pH2, pA2, gap


# =========================
# Form stats
# =========================
def enrich_form_stats_dynamic(df, team, match_date, window=5):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    match_date = pd.to_datetime(match_date)

    recent_matches = (
        df[((df["HomeTeam"] == team) | (df["AwayTeam"] == team)) & (df["Date"] < match_date)]
        .sort_values("Date", ascending=False)
        .head(window)
    )
    if recent_matches.empty:
        return {"Form": 0.0, "GD": 0.0, "WinRate": 0.0, "DrawRate": 0.0, "GoalsAvg": 0.0}

    points = 0
    goals_diff = 0
    draws = 0
    wins = 0
    total_goals = 0

    for _, row in recent_matches.iterrows():
        is_home = row["HomeTeam"] == team

        if is_home:
            goals_for, goals_against = row["FTHG"], row["FTAG"]
            win = row["FTR"] == "H"
        else:
            goals_for, goals_against = row["FTAG"], row["FTHG"]
            win = row["FTR"] == "A"

        draw = row["FTR"] == "D"

        if draw:
            draws += 1
            points += 1
        elif win:
            wins += 1
            points += 3

        goals_diff += goals_for - goals_against
        total_goals += goals_for

    matches_played = len(recent_matches)
    return {
        "Form": points / (3 * matches_played),
        "GD": goals_diff / matches_played,
        "WinRate": wins / matches_played,
        "DrawRate": draws / matches_played,
        "GoalsAvg": total_goals / matches_played,
    }


# =========================
# Importance / ranks
# =========================
def _league_profile(league_code: str | int | None):
    try:
        code = int(league_code) if league_code is not None else None
    except Exception:
        code = None

    EURO = {
        39,
        61,
        78,
        140,
        135,
        88,
        207,
        94,
        203,
        144,
        197,
        119,
        179,
        180,
        253,
        2,
        3,
        233,
        62,
        40,
        79,
        136,
        141,
    }
    CAL_Y = {71, 98, 262, 292, 128}

    if code in EURO:
        return {"region": "europe", "late_months": {4, 5, 6}, "late_threshold": 0.70}
    elif code in CAL_Y:
        return {"region": "calendar_year", "late_months": {10, 11, 12}, "late_threshold": 0.70}
    else:
        return {"region": "unknown", "late_months": set(), "late_threshold": 0.70}


def _season_progress_by_dates(df_all: pd.DataFrame, asof) -> float:
    if df_all is None or df_all.empty:
        return 0.0
    d = df_all.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"])
    if d.empty:
        return 0.0
    asof = pd.to_datetime(asof)
    dmin, dmax = d["Date"].min(), d["Date"].max()
    total = (dmax - dmin).days
    if total <= 0:
        return 0.0
    prog = (asof - dmin).days / total
    return float(max(0.0, min(1.0, prog)))


def add_ranks_and_importance(df, home_team, away_team, match_date, league_code):
    if df is None or df.empty:
        return 10, 10, 0

    prof = _league_profile(league_code)
    late_months = prof["late_months"]
    late_th = prof["late_threshold"]

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    md = pd.to_datetime(match_date)
    d = d[d["Date"] < md].dropna(subset=["Date"])

    d["Points_H"] = d["FTR"].apply(lambda x: 3 if x == "H" else 1 if x == "D" else 0)
    d["Points_A"] = d["FTR"].apply(lambda x: 3 if x == "A" else 1 if x == "D" else 0)

    team_points = {}
    for _, row in d.iterrows():
        team_points[row["HomeTeam"]] = team_points.get(row["HomeTeam"], 0) + row["Points_H"]
        team_points[row["AwayTeam"]] = team_points.get(row["AwayTeam"], 0) + row["Points_A"]

    if not team_points:
        return 10, 10, 0

    sorted_teams = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
    ranks = {team: idx + 1 for idx, (team, _) in enumerate(sorted_teams)}
    n_teams = len(ranks)

    rank_home = ranks.get(home_team, min(10, n_teams))
    rank_away = ranks.get(away_team, min(10, n_teams))
    rank_diff = abs(rank_home - rank_away)

    season_prog = _season_progress_by_dates(df, md)
    late_season = (season_prog >= late_th) or (md.month in late_months)

    top_k = 5
    close_ranks = rank_diff <= 4
    top_clash = (rank_home <= top_k and rank_away <= top_k)

    releg_zone = max(3, int(round(0.12 * n_teams)))
    six_pointer_releg = late_season and ((rank_home > n_teams - releg_zone) or (rank_away > n_teams - releg_zone))
    euro_spot_fight = late_season and ((rank_home <= 7) or (rank_away <= 7)) and (rank_diff <= 6)

    importance = 1 if (top_clash or (late_season and (close_ranks or six_pointer_releg or euro_spot_fight))) else 0
    return rank_home, rank_away, importance


# =========================
# Features
# =========================
def prepare_input_features_enriched(home_team, away_team, match_date, b365h, b365a, b365d, season_df, league_code):
    df = season_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")
    all_teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()

    if (home_team not in all_teams) or (away_team not in all_teams):
        print(f"⚠️ Attention : {home_team} ou {away_team} n'a pas d'historique. Les stats seront neutres.")

    match_date = pd.to_datetime(match_date)
    df_past = df[df["Date"] < match_date]

    def safe_stats(d):
        d = dict(d or {})
        for key in ["Form", "GD", "WinRate", "DrawRate", "GoalsAvg"]:
            if d.get(key) is None:
                d[key] = 0.0
        return d

    home_stats = safe_stats(enrich_form_stats_dynamic(df_past, home_team, match_date))
    away_stats = safe_stats(enrich_form_stats_dynamic(df_past, away_team, match_date))

    odds_ratio_ha = b365h / b365a if b365a > 0 else 0
    odds_diff_hd = b365h - b365d
    odds_diff_ad = b365a - b365d
    odds_gap_min_delta = max(b365h, b365a, b365d) - min(b365h, b365a, b365d)
    form_diff = home_stats["Form"] - away_stats["Form"]

    rank_home, rank_away, match_importance = add_ranks_and_importance(df, home_team, away_team, match_date, league_code)

    features = pd.DataFrame(
        [
            {
                "HTHG": 0,
                "HTAG": 0,
                "HTR": 0,
                "B365H": b365h,
                "B365A": b365a,
                "B365D": b365d,
                "OddsRatio_HA": odds_ratio_ha,
                "OddsDiff_HD": odds_diff_hd,
                "OddsDiff_AD": odds_diff_ad,
                "OddsGap_MinDelta": odds_gap_min_delta,
                "Year": match_date.year,
                "Month": match_date.month,
                "Weekday": match_date.weekday(),
                "HomeForm": home_stats["Form"],
                "AwayForm": away_stats["Form"],
                "HomeGD": home_stats["GD"],
                "AwayGD": away_stats["GD"],
                "DrawRate_Home": home_stats["DrawRate"],
                "DrawRate_Away": away_stats["DrawRate"],
                "WinRate_Home": home_stats["WinRate"],
                "WinRate_Away": away_stats["WinRate"],
                "GoalsAvg_Home": home_stats["GoalsAvg"],
                "GoalsAvg_Away": away_stats["GoalsAvg"],
                "Form_Diff": form_diff,
                "Rank_Home": rank_home,
                "Rank_Away": rank_away,
                "MatchImportance": match_importance,
            }
        ]
    )

    return features


# =========================
# Règles auxiliaires
# =========================
def detect_double_chance(proba_0, proba_1, proba_2, final_prediction, league_code):
    (bookmaker_margin, uncertainty_threshold, importance, season_stage, upset_threshold, skip_threshold, bogey_weight, gki_weight) = _safe_parametres(
        league_code
    )

    seuil_incertitude = uncertainty_threshold - 0.02 * (importance / 5)

    probs = np.array([proba_0, proba_1, proba_2], dtype=float)
    sorted_probs = np.sort(probs)
    ecart = sorted_probs[-1] - sorted_probs[-2]

    if ecart <= seuil_incertitude:
        if final_prediction == 0 and proba_0 < 0.60:
            return "1X"
        elif final_prediction == 2 and proba_2 < 0.60:
            return "X2"
    return None



def _entropy(p0, p1, p2, eps=1e-12):
    p = np.array([p0, p1, p2], dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())  # max ~ 1.098

def detect_double_chance_v2(
    p0: float, p1: float, p2: float,
    pred_final: int,
    *,
    league_code: str = "default",
    bias_detected: bool = False,
    low_confidence: bool = False,
    upset_score: float = 0.0,
    upset_threshold: float = 0.52,
    override_tag: Optional[str] = None,
) -> Optional[str]:
    """
    DC renforcée (version pro).
    Retourne "1X", "X2" ou None.
    """

    # 1) seuils ligue
    try:
        params = _get_params(league_code)
    except Exception:
        params = {}

    # seuils par défaut
    base_gap = float(params.get("dc_gap_threshold", 0.12))          # plus strict que avant
    draw_th = float(params.get("dc_draw_threshold", 0.28))          # si nul >= 28% => DC
    ent_th  = float(params.get("dc_entropy_threshold", 1.03))       # proche du max (1.098)
    max_win_no_dc = float(params.get("dc_max_win_no_dc", 0.72))     # si win >= 72% => pas besoin

    # 2) métriques
    probs = np.array([p0, p1, p2], dtype=float)
    probs = probs / max(1e-9, probs.sum())

    top = float(np.max(probs))
    srt = np.sort(probs)
    gap = float(srt[-1] - srt[-2])
    ent = _entropy(probs[0], probs[1], probs[2])

    # 3) règles dures (force)
    force = False
    reasons = []

    if bias_detected:
        force = True; reasons.append("bias")
    if low_confidence:
        force = True; reasons.append("low_conf")
    if upset_score is not None and upset_score >= (upset_threshold * 0.85):
        force = True; reasons.append("upset_near_threshold")
    if override_tag is not None and "form_over_market" in str(override_tag):
        force = True; reasons.append("form_vs_market_conflict")

    # 4) règles probabilistes (force si risque)
    if float(p1) >= draw_th:
        force = True; reasons.append("high_draw")
    if gap <= base_gap:
        force = True; reasons.append("small_gap")
    if ent >= ent_th:
        force = True; reasons.append("high_entropy")

    # 5) si top win trop fort => on annule DC (sauf force métier)
    if (top >= max_win_no_dc) and (not (bias_detected or low_confidence)):
        return None

    if not force:
        return None

    # 6) sortie DC cohérente
    # pred_final: 0=home, 1=draw, 2=away
    if pred_final == 0:
        return "1X"
    if pred_final == 2:
        return "X2"

    # si draw prédit, DC dépend du plus fort entre home/away
    return "1X" if p0 >= p2 else "X2"


def detect_bias(features_df):
    odds = features_df[["B365H", "B365A", "B365D"]].values[0].astype(float)
    max_odds = np.max(odds)
    min_odds = np.min(odds)
    bias_score = abs(max_odds - min_odds) / np.mean(odds)
    return bias_score > 0.6


def is_confidence_low(proba_0, proba_1, proba_2):
    arr = np.array([proba_0, proba_1, proba_2], dtype=float)
    ecart_principal = np.max(arr) - np.median(arr)
    return ecart_principal < 0.07


def adjust_odds_weight_by_season(odds_gap, season_stage):
    if season_stage == "early":
        return odds_gap * 1.3
    elif season_stage == "mid":
        return odds_gap
    else:
        return odds_gap * 0.9


# =========================
# Porte "forme récente"
# =========================
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_explanation(rule_applied, features, user_profile):
    odds_ratio = features.get("OddsRatio_HA", 1)
    form_diff = features.get("Form_Diff", 0)
    match_importance = features.get("MatchImportance", 0)

    if isinstance(match_importance, pd.Series):
        match_importance = match_importance.values[0]

    if user_profile == "débutant":
        if rule_applied == "threshold":
            msg = "L'IA pense qu’il y aura un match nul car la probabilité dépasse le seuil fixé."
        elif rule_applied == "margin_adjusted":
            msg = "Les cotes sont très proches : cela suggère un match équilibré, donc nul."
        else:
            msg = "L’IA prédit une victoire car les chances sont déséquilibrées entre les équipes."
    elif user_profile == "expert":
        if rule_applied == "threshold":
            msg = f"Proba_nul = {features.get('proba_1', 0):.2f}, supérieur au seuil : nul prédit."
        elif rule_applied == "margin_adjusted":
            msg = f"Match ajusté à nul : cotes trop proches (écart ≈ {features.get('OddsGap_MinDelta', 0):.3f})."
        else:
            msg = (
                f"Proba_RF = [{features.get('proba_0', 0):.2f}, {features.get('proba_2', 0):.2f}], "
                f"écart de forme = {form_diff:.2f}"
            )
    else:
        if rule_applied == "threshold":
            msg = "Match nul probable : la probabilité dépasse le seuil."
        elif rule_applied == "margin_adjusted":
            msg = "Les cotes sont serrées, et l’IA anticipe un nul."
        else:
            msg = "Victoire probable : un déséquilibre a été détecté entre les deux équipes."

    if match_importance == 1:
        msg += " Ce match est considéré comme important."

    return msg


# -----------------------------
# Config (ENV friendly)
# -----------------------------

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY manquante. "
            "En local: mets-la dans un fichier .env. "
            "Sur Render: ajoute-la dans Environment Variables."
        )
    return OpenAI(api_key=api_key)

def explanation_from_pred_final(pred_final: Dict[str, Any], user_profile: str = "standard") -> Dict[str, Any]:
    """
    Prend le JSON FINAL (après apply_unexpected_layer) et renvoie le même JSON
    avec pred_final["explanation"] remplacé (4 à 8 phrases FR).
    - Fallback robuste offline
    - Optionnel LLM OpenAI si OPENAI_EXPLAIN_ENABLED=1 et clé ok
    - Ajoute des metas: explain_llm_used, explain_llm_model, explain_llm_error, explain_llm_debug
    """

    def _is_nan(x: Any) -> bool:
        try:
            return isinstance(x, float) and math.isnan(x)
        except Exception:
            return False

    def _get(d: Dict[str, Any], key: str, default=None):
        try:
            v = d.get(key, default)
            if _is_nan(v):
                return default
            return v
        except Exception:
            return default

    def _pct_from_any(v: Any) -> float:
        """
        Convertit:
          - 0.52 -> 0.52
          - "52%" -> 0.52
          - 52 -> 0.52 (si >1 on suppose %)
        """
        try:
            if v is None or _is_nan(v):
                return 0.0
            if isinstance(v, str):
                s = v.strip().replace(",", ".")
                if not s:
                    return 0.0
                if s.endswith("%"):
                    x = float(s[:-1].strip()) / 100.0
                    return max(0.0, min(1.0, x))
                x = float(s)
                if x > 1.0:
                    x /= 100.0
                return max(0.0, min(1.0, x))
            x = float(v)
            if x > 1.0:
                x /= 100.0
            return max(0.0, min(1.0, x))
        except Exception:
            return 0.0

    def _fmt_pct(x: float) -> str:
        try:
            return f"{round(float(x)*100,1)}%"
        except Exception:
            return "0.0%"

    # -----------------------------
    # Extract from FINAL JSON
    # -----------------------------
    home = str(_get(pred_final, "home", "") or "")
    away = str(_get(pred_final, "away", "") or "")
    match_date = str(_get(pred_final, "match_date", "") or _get(pred_final, "date", "") or "")

    form_home = str(_get(pred_final, "5_dern_perf_home", "") or "")
    form_away = str(_get(pred_final, "5_dern_perf_away", "") or "")

    bias_detected = bool(_get(pred_final, "bias_detected", False) or False)
    low_confidence = bool(_get(pred_final, "low_confidence", False) or False)
    double_chance = _get(pred_final, "double_chance", None)

    # probs: prefer proba_* if present, else p*_raw
    p0 = _pct_from_any(_get(pred_final, "proba_0", None))
    p1 = _pct_from_any(_get(pred_final, "proba_1", None))
    p2 = _pct_from_any(_get(pred_final, "proba_2", None))
    if (p0 + p1 + p2) <= 1e-6:
        p0 = _pct_from_any(_get(pred_final, "p0_raw", 0.0))
        p1 = _pct_from_any(_get(pred_final, "p1_raw", 0.0))
        p2 = _pct_from_any(_get(pred_final, "p2_raw", 0.0))

    # odds if present in pred_final
    odds = {}
    for k in ("B365H", "B365D", "B365A"):
        v = _get(pred_final, k, None)
        try:
            if v is not None and str(v).strip() != "":
                odds[k] = float(str(v).replace(",", "."))
        except Exception:
            pass

    rule_applied = str(_get(pred_final, "rule_applied", "") or "")
    upset_score = float(_get(pred_final, "_upset_score", 0.0) or 0.0)
    upset_threshold = float(_get(pred_final, "_upset_threshold", 0.52) or 0.52)

    # realtime summary
    realtime_risk = _get(pred_final, "realtime_risk", {}) or {}
    summary = {}
    try:
        summary = (realtime_risk or {}).get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
    except Exception:
        summary = {}

    absences_text = str(summary.get("absences_text") or "")
    missing_meta = summary.get("missing_meta") or []
    if not isinstance(missing_meta, list):
        missing_meta = []

    top_injuries = summary.get("top_injuries") or []
    if not isinstance(top_injuries, list):
        top_injuries = []

    ranking = summary.get("ranking") or {}
    if not isinstance(ranking, dict):
        ranking = {}

    rank_home = rank_away = None
    pts_home = pts_away = None
    if ranking.get("available") is True:
        try:
            rh = ranking.get("home") or {}
            ra = ranking.get("away") or {}
            rank_home = rh.get("rank")
            rank_away = ra.get("rank")
            pts_home = rh.get("points")
            pts_away = ra.get("points")
        except Exception:
            pass

    status_short = str(summary.get("status_short") or "")
    status_long = str(summary.get("status_long") or "")
    is_finished = bool(summary.get("is_finished") is True)
    is_started = bool(summary.get("is_started") is True)
    elapsed = summary.get("elapsed")

    # -----------------------------
    # OFFLINE fallback (4-8 phrases)
    # -----------------------------
    def _fallback() -> str:
        lines: List[str] = []

        title = f"{home} vs {away}" if home and away else "Match"
        if match_date:
            title += f" ({match_date})"
        lines.append(f"{title}.")

        if (p0 + p1 + p2) > 1e-6:
            lines.append(f"Probabilités (1/N/2) : {_fmt_pct(p0)}, {_fmt_pct(p1)}, {_fmt_pct(p2)}.")
        else:
            lines.append("Probabilités (1/N/2) : indisponibles.")

        # favorite
        if (p0 + p1 + p2) > 1e-6:
            fav = "home" if p0 >= max(p1, p2) else ("draw" if p1 >= max(p0, p2) else "away")
            if fav == "home":
                lines.append(f"Lecture modèle : avantage {home} (victoire à domicile).")
            elif fav == "away":
                lines.append(f"Lecture modèle : avantage {away} (victoire à l’extérieur).")
            else:
                lines.append("Lecture modèle : match équilibré (nul plausible).")

        if form_home or form_away:
            lines.append(f"Forme (5 derniers) : {home}={form_home or 'n/a'} ; {away}={form_away or 'n/a'}.")

        if isinstance(rank_home, int) and isinstance(rank_away, int):
            pts_txt = ""
            if isinstance(pts_home, int) and isinstance(pts_away, int):
                pts_txt = f" ({pts_home} pts vs {pts_away} pts)"
            lines.append(f"Classement : {home} est {rank_home}ᵉ, {away} est {rank_away}ᵉ{pts_txt}.")

        if odds:
            parts = []
            if "B365H" in odds: parts.append(f"H={odds['B365H']}")
            if "B365D" in odds: parts.append(f"N={odds['B365D']}")
            if "B365A" in odds: parts.append(f"A={odds['B365A']}")
            lines.append("Cotes (B365) : " + ", ".join(parts) + ".")

        if double_chance:
            lines.append(f"Double chance : {double_chance} (filet de sécurité).")

        if bias_detected:
            lines.append("Biais de cotes détecté : prudence (effet popularité / surcote possible).")

        if upset_score > 0 and upset_score >= upset_threshold:
            lines.append("Risque de surprise (upset) élevé : éviter les mises agressives.")

        if absences_text:
            lines.append(absences_text.strip())

        # status (live/FT)
        if status_short:
            if is_finished or status_short == "FT":
                lines.append("Note : le match est terminé (infos temps réel post-match).")
            elif is_started:
                lines.append(f"Note : match en cours ({status_short}), minute ≈ {elapsed}.")

        # prudence if missing key live info
        if isinstance(missing_meta, list) and len(missing_meta) > 0:
            lines.append("Certaines données temps réel manquent encore (compos/stats/événements) : prudence avant de valider un pari.")

        return " ".join(lines[:8]).strip()

    fallback_text = _fallback()
   

    # -----------------------------
    # LLM explanation (uses FINAL JSON only)
    # -----------------------------
    api_key = get_openai_client()
    if not OPENAI_EXPLAIN_ENABLED or not api_key:
        pred_final["explanation"] = fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = ""
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = {"enabled": OPENAI_EXPLAIN_ENABLED, "has_key": bool(api_key), "payload": None}
        return pred_final

    # Build "facts" to reduce hallucination
    facts: List[str] = []
    facts.append(f"Match: {home} vs {away} | date={match_date}")
    facts.append(f"Probas 1/N/2: {_fmt_pct(p0)}, {_fmt_pct(p1)}, {_fmt_pct(p2)}")
    if form_home or form_away:
        facts.append(f"Forme(5): {home}={form_home or 'n/a'} ; {away}={form_away or 'n/a'}")
    if isinstance(rank_home, int) and isinstance(rank_away, int):
        facts.append(f"Classement: {home} rank={rank_home} pts={pts_home} | {away} rank={rank_away} pts={pts_away}")
    if odds:
        facts.append(f"Cotes B365: H={odds.get('B365H')} N={odds.get('B365D')} A={odds.get('B365A')}")
    facts.append(f"Flags: bias_detected={bias_detected} double_chance={double_chance} low_confidence={low_confidence}")
    if absences_text:
        facts.append(f"Absences: {absences_text}")
    if top_injuries:
        inj_txt = "; ".join([f"{x.get('team','')}:{x.get('player','')}({x.get('reason','')})" for x in top_injuries[:3]])
        facts.append(f"Top injuries: {inj_txt}")
    if status_short:
        facts.append(f"Status: {status_short} ({status_long}) started={is_started} finished={is_finished} elapsed={elapsed}")
    if upset_score:
        facts.append(f"Upset score: {upset_score} (threshold={upset_threshold})")

    payload = {
        "pred_final": pred_final,          # JSON final complet (source of truth)
        "facts": facts,                   # facts verrouillés anti-hallucination
        "user_profile": user_profile,
    }

    try:
       
        #client = OpenAI(api_key=api_key)
        client =get_openai_client()

        sys_msg = """Tu es un analyste professionnel de football ET un parieur expérimenté.
                    Ton style est celui d’un consultant TV + trader de marché des cotes.

                    Mission :
                    Produire 6 à 9 phrases en français, structurées, claires, avec une vraie prise de position.

                    Règles STRICTES :
                    - Utilise uniquement les données du JSON. N’invente jamais.
                    - Si une donnée manque, dis-le explicitement.
                    - Analyse la cohérence entre probabilités du modèle et cotes bookmakers.
                    - Détecte s’il existe une VALUE BET (écart modèle vs marché).
                    - Mentionne obligatoirement : probabilités 1/N/2, forme 5 matchs, classement (rank + points),
                    absences (absences_text + top_injuries), risk_level/risk_score, double_chance,
                    bias_detected, low_confidence, statut match (NS/1H/HT/FT).
                    - Si match ≠ NS → préciser que c’est du live/post-match.

                    Structure obligatoire :

                    1) Résumé du match + favori.
                    2) Lecture du nul (si ≥25% → "nul non négligeable").
                    3) Classement + écart de points + interprétation.
                    4) Forme récente convertie en bilan (ex: 2 Victoire- 2 Null - 1 Défaite).
                    5) Absences majeures et impact potentiel pour les deux équipes.
                    6) Prédiction du modèle vs côte du marché
                    7) Recommandation EXPERTE :
                    - Niveau de confiance (faible / modéré / élevé)
                    - Gestion de mise (prudente / standard / agressive)

                    Style :
                    - Ton professionnel.
                    - Décision claire.
                    - Pas de blabla.
                    - Conclusion ferme comme un expert parieur.
                    """

        user_msg = (
            "FACTS (à respecter strictement):\n"
            + "\n".join([f"- {x}" for x in facts])
            + "\n\npred_final JSON:\n"
            + json.dumps(pred_final, ensure_ascii=False)
        )

        resp = client.chat.completions.create(
            model=OPENAI_EXPLAIN_MODEL,
            temperature=OPENAI_EXPLAIN_TEMPERATURE,
            max_tokens=OPENAI_EXPLAIN_MAX_TOKENS,
            timeout=OPENAI_EXPLAIN_TIMEOUT,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        text_out = (resp.choices[0].message.content or "").strip()
        
        def _one_line(s: str) -> str:
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            s = re.sub(r"\n+", " ", s)      # remplace tous les retours ligne par espace
            s = re.sub(r"\s{2,}", " ", s)   # compact espaces multiples
            return s.strip()
        
        text_out = _one_line(text_out)

        if text_out:
            pred_final["explanation"] = text_out
            pred_final["explain_llm_used"] = 1
            pred_final["explain_llm_model"] = OPENAI_EXPLAIN_MODEL
            pred_final["explain_llm_error"] = ""
            if LLM_DEBUG:
                pred_final["explain_llm_debug"] = payload
            return pred_final

        # empty => fallback
        pred_final["explanation"] = fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = "empty_response"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = payload
        return pred_final

    except Exception as e:
        pred_final["explanation"] = fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = f"{type(e).__name__}: {e}"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = payload
        return pred_final


def explanation_from_pred_final_________(pred_final: Dict[str, Any], user_profile: str = "standard") -> Dict[str, Any]:
    """
    Prend le JSON FINAL (après apply_unexpected_layer) et renvoie le même JSON
    avec pred_final["explanation"] remplacé (4 à 8 phrases FR).
    - Fallback robuste offline
    - Optionnel LLM OpenAI si OPENAI_EXPLAIN_ENABLED=1 et clé ok
    - Ajoute des metas: explain_llm_used, explain_llm_model, explain_llm_error, explain_llm_debug
    """

    def _is_nan(x: Any) -> bool:
        try:
            return isinstance(x, float) and math.isnan(x)
        except Exception:
            return False

    def _get(d: Dict[str, Any], key: str, default=None):
        try:
            v = d.get(key, default)
            if _is_nan(v):
                return default
            return v
        except Exception:
            return default

    def _pct_from_any(v: Any) -> float:
        """
        Convertit:
          - 0.52 -> 0.52
          - "52%" -> 0.52
          - 52 -> 0.52 (si >1 on suppose %)
        """
        try:
            if v is None or _is_nan(v):
                return 0.0
            if isinstance(v, str):
                s = v.strip().replace(",", ".")
                if not s:
                    return 0.0
                if s.endswith("%"):
                    x = float(s[:-1].strip()) / 100.0
                    return max(0.0, min(1.0, x))
                x = float(s)
                if x > 1.0:
                    x /= 100.0
                return max(0.0, min(1.0, x))
            x = float(v)
            if x > 1.0:
                x /= 100.0
            return max(0.0, min(1.0, x))
        except Exception:
            return 0.0

    def _fmt_pct(x: float) -> str:
        try:
            return f"{round(float(x)*100,1)}%"
        except Exception:
            return "0.0%"

    # -----------------------------
    # Extract from FINAL JSON
    # -----------------------------
    home = str(_get(pred_final, "home", "") or "")
    away = str(_get(pred_final, "away", "") or "")
    match_date = str(_get(pred_final, "match_date", "") or _get(pred_final, "date", "") or "")

    form_home = str(_get(pred_final, "5_dern_perf_home", "") or "")
    form_away = str(_get(pred_final, "5_dern_perf_away", "") or "")

    bias_detected = bool(_get(pred_final, "bias_detected", False) or False)
    low_confidence = bool(_get(pred_final, "low_confidence", False) or False)
    double_chance = _get(pred_final, "double_chance", None)

    # probs: prefer proba_* if present, else p*_raw
    p0 = _pct_from_any(_get(pred_final, "proba_0", None))
    p1 = _pct_from_any(_get(pred_final, "proba_1", None))
    p2 = _pct_from_any(_get(pred_final, "proba_2", None))
    if (p0 + p1 + p2) <= 1e-6:
        p0 = _pct_from_any(_get(pred_final, "p0_raw", 0.0))
        p1 = _pct_from_any(_get(pred_final, "p1_raw", 0.0))
        p2 = _pct_from_any(_get(pred_final, "p2_raw", 0.0))

    # odds if present in pred_final
    odds = {}
    for k in ("B365H", "B365D", "B365A"):
        v = _get(pred_final, k, None)
        try:
            if v is not None and str(v).strip() != "":
                odds[k] = float(str(v).replace(",", "."))
        except Exception:
            pass

    rule_applied = str(_get(pred_final, "rule_applied", "") or "")
    upset_score = float(_get(pred_final, "_upset_score", 0.0) or 0.0)
    upset_threshold = float(_get(pred_final, "_upset_threshold", 0.52) or 0.52)

    # realtime summary
    realtime_risk = _get(pred_final, "realtime_risk", {}) or {}
    summary = {}
    try:
        summary = (realtime_risk or {}).get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
    except Exception:
        summary = {}

    absences_text = str(summary.get("absences_text") or "")
    missing_meta = summary.get("missing_meta") or []
    if not isinstance(missing_meta, list):
        missing_meta = []

    top_injuries = summary.get("top_injuries") or []
    if not isinstance(top_injuries, list):
        top_injuries = []

    ranking = summary.get("ranking") or {}
    if not isinstance(ranking, dict):
        ranking = {}

    rank_home = rank_away = None
    pts_home = pts_away = None
    if ranking.get("available") is True:
        try:
            rh = ranking.get("home") or {}
            ra = ranking.get("away") or {}
            rank_home = rh.get("rank")
            rank_away = ra.get("rank")
            pts_home = rh.get("points")
            pts_away = ra.get("points")
        except Exception:
            pass

    status_short = str(summary.get("status_short") or "")
    status_long = str(summary.get("status_long") or "")
    is_finished = bool(summary.get("is_finished") is True)
    is_started = bool(summary.get("is_started") is True)
    elapsed = summary.get("elapsed")

    # -----------------------------
    # OFFLINE fallback (4-8 phrases)
    # -----------------------------
    def _fallback() -> str:
        lines: List[str] = []

        title = f"{home} vs {away}" if home and away else "Match"
        if match_date:
            title += f" ({match_date})"
        lines.append(f"{title}.")

        if (p0 + p1 + p2) > 1e-6:
            lines.append(f"Probabilités (1/N/2) : {_fmt_pct(p0)}, {_fmt_pct(p1)}, {_fmt_pct(p2)}.")
        else:
            lines.append("Probabilités (1/N/2) : indisponibles.")

        # favorite
        if (p0 + p1 + p2) > 1e-6:
            fav = "home" if p0 >= max(p1, p2) else ("draw" if p1 >= max(p0, p2) else "away")
            if fav == "home":
                lines.append(f"Lecture modèle : avantage {home} (victoire à domicile).")
            elif fav == "away":
                lines.append(f"Lecture modèle : avantage {away} (victoire à l’extérieur).")
            else:
                lines.append("Lecture modèle : match équilibré (nul plausible).")

        if form_home or form_away:
            lines.append(f"Forme (5 derniers) : {home}={form_home or 'n/a'} ; {away}={form_away or 'n/a'}.")

        if isinstance(rank_home, int) and isinstance(rank_away, int):
            pts_txt = ""
            if isinstance(pts_home, int) and isinstance(pts_away, int):
                pts_txt = f" ({pts_home} pts vs {pts_away} pts)"
            lines.append(f"Classement : {home} est {rank_home}ᵉ, {away} est {rank_away}ᵉ{pts_txt}.")

        if odds:
            parts = []
            if "B365H" in odds: parts.append(f"H={odds['B365H']}")
            if "B365D" in odds: parts.append(f"N={odds['B365D']}")
            if "B365A" in odds: parts.append(f"A={odds['B365A']}")
            lines.append("Cotes (B365) : " + ", ".join(parts) + ".")

        if double_chance:
            lines.append(f"Double chance : {double_chance} (filet de sécurité).")

        if bias_detected:
            lines.append("Biais de cotes détecté : prudence (effet popularité / surcote possible).")

        if upset_score > 0 and upset_score >= upset_threshold:
            lines.append("Risque de surprise (upset) élevé : éviter les mises agressives.")

        if absences_text:
            lines.append(absences_text.strip())

        # status (live/FT)
        if status_short:
            if is_finished or status_short == "FT":
                lines.append("Note : le match est terminé (infos temps réel post-match).")
            elif is_started:
                lines.append(f"Note : match en cours ({status_short}), minute ≈ {elapsed}.")

        # prudence if missing key live info
        if isinstance(missing_meta, list) and len(missing_meta) > 0:
            lines.append("Certaines données temps réel manquent encore (compos/stats/événements) : prudence avant de valider un pari.")

        return " ".join(lines[:8]).strip()

    fallback_text = _fallback()

    def _should_use_llm() -> bool:
        """
        Active le LLM uniquement pour les cas qui méritent une analyse riche.
        """
        try:
            # 1) cas explicites
            if low_confidence:
                return True
            if bias_detected:
                return True

            # 2) match équilibré
            # - nul important
            # - home et away proches
            # - aucune issue très dominante
            max_p = max(p0, p1, p2)
            min_p = min(p0, p1, p2)
            spread = max_p - min_p

            balanced_match = (
                p1 >= 0.28              # nul significatif
                or abs(p0 - p2) <= 0.10 # home/away proches
                or spread <= 0.12       # distribution serrée
            )

            if balanced_match:
                return True

            return False

        except Exception:
            return False

    # -----------------------------
    # LLM explanation (uses FINAL JSON only)
    # -----------------------------
    if not OPENAI_EXPLAIN_ENABLED:
        pred_final["explanation"] = fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = "disabled"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = {"enabled": False, "payload": None}
        return pred_final

    # ✅ nouveau : fallback direct si le match n'a pas besoin du LLM
    if not _should_use_llm():
        pred_final["explanation"] = fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = "skipped_not_needed"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = {"enabled": True, "skipped": True, "payload": None}
        return pred_final

    client = get_openai_client()
    if not client:
        pred_final["explanation"] = fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = "missing_client"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = {"enabled": True, "has_client": False, "payload": None}
        return pred_final

    # Build "facts" to reduce hallucination
    facts: List[str] = []
    facts.append(f"Match: {home} vs {away} | date={match_date}")
    facts.append(f"Probas 1/N/2: {_fmt_pct(p0)}, {_fmt_pct(p1)}, {_fmt_pct(p2)}")
    if form_home or form_away:
        facts.append(f"Forme(5): {home}={form_home or 'n/a'} ; {away}={form_away or 'n/a'}")
    if isinstance(rank_home, int) and isinstance(rank_away, int):
        facts.append(f"Classement: {home} rank={rank_home} pts={pts_home} | {away} rank={rank_away} pts={pts_away}")
    if odds:
        facts.append(f"Cotes B365: H={odds.get('B365H')} N={odds.get('B365D')} A={odds.get('B365A')}")
    facts.append(f"Flags: bias_detected={bias_detected} double_chance={double_chance} low_confidence={low_confidence}")
    if absences_text:
        facts.append(f"Absences: {absences_text}")
    if top_injuries:
        inj_txt = "; ".join([f"{x.get('team','')}:{x.get('player','')}({x.get('reason','')})" for x in top_injuries[:3]])
        facts.append(f"Top injuries: {inj_txt}")
    if status_short:
        facts.append(f"Status: {status_short} ({status_long}) started={is_started} finished={is_finished} elapsed={elapsed}")
    if upset_score:
        facts.append(f"Upset score: {upset_score} (threshold={upset_threshold})")

    payload = {
        "pred_final": pred_final,          # JSON final complet (source of truth)
        "facts": facts,                   # facts verrouillés anti-hallucination
        "user_profile": user_profile,
    }

    try:
       
        #client = OpenAI(api_key=api_key)
        #client =get_openai_client()

        sys_msg = """Tu es un analyste professionnel de football ET un parieur expérimenté.
                    Ton style est celui d’un consultant TV + trader de marché des cotes.

                    Mission :
                    Produire 6 à 9 phrases en français, structurées, claires, avec une vraie prise de position.

                    Règles STRICTES :
                    - Utilise uniquement les données du JSON. N’invente jamais.
                    - Si une donnée manque, dis-le explicitement.
                    - Analyse la cohérence entre probabilités du modèle et cotes bookmakers.
                    - Détecte s’il existe une VALUE BET (écart modèle vs marché).
                    - Mentionne obligatoirement : probabilités 1/N/2, forme 5 matchs, classement (rank + points),
                    absences (absences_text + top_injuries), risk_level/risk_score, double_chance,
                    bias_detected, low_confidence, statut match (NS/1H/HT/FT).
                    - Si match ≠ NS → préciser que c’est du live/post-match.

                    Structure obligatoire :

                    1) Résumé du match + favori.
                    2) Lecture du nul (si ≥25% → "nul non négligeable").
                    3) Classement + écart de points + interprétation.
                    4) Forme récente convertie en bilan (ex: 2 Victoire- 2 Null - 1 Défaite).
                    5) Absences majeures et impact potentiel pour les deux équipes.
                    6) Prédiction du modèle vs côte du marché
                    7) Recommandation EXPERTE :
                    - Niveau de confiance (faible / modéré / élevé)
                    - Gestion de mise (prudente / standard / agressive)

                    Style :
                    - Ton professionnel.
                    - Décision claire.
                    - Pas de blabla.
                    - Conclusion ferme comme un expert parieur.
                    """

        user_msg = (
            "FACTS (à respecter strictement):\n"
            + "\n".join([f"- {x}" for x in facts])
            + "\n\npred_final JSON:\n"
            + json.dumps(pred_final, ensure_ascii=False)
        )

        resp = client.chat.completions.create(
            model=OPENAI_EXPLAIN_MODEL,
            temperature=OPENAI_EXPLAIN_TEMPERATURE,
            max_tokens=OPENAI_EXPLAIN_MAX_TOKENS,
            timeout=OPENAI_EXPLAIN_TIMEOUT,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        text_out = (resp.choices[0].message.content or "").strip()
        
        def _one_line(s: str) -> str:
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            s = re.sub(r"\n+", " ", s)      # remplace tous les retours ligne par espace
            s = re.sub(r"\s{2,}", " ", s)   # compact espaces multiples
            return s.strip()
        
        text_out = _one_line(text_out)

        if text_out:
            pred_final["explanation"] = text_out
            pred_final["explain_llm_used"] = 1
            pred_final["explain_llm_model"] = OPENAI_EXPLAIN_MODEL
            pred_final["explain_llm_error"] = ""
            if LLM_DEBUG:
                pred_final["explain_llm_debug"] = payload
            return pred_final

        # empty => fallback
        pred_final["explanation"] = fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = "empty_response"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = payload
        return pred_final

    except Exception as e:
        pred_final["explanation"] = fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = f"{type(e).__name__}: {e}"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = payload
        return pred_final



def clean_extract_final_result(pred_final: dict) -> dict:
    """
    Retourne un JSON final propre à partir de pred_final,
    en conservant uniquement les champs stratégiques.

    - Supprime tout objet potentiellement circulaire
    - Garantit que les clés existent
    - Ne modifie pas l'objet original
    """

    if not isinstance(pred_final, dict):
        raise ValueError("pred_final doit être un dictionnaire")

    # Helper pour éviter KeyError
    def _get(key, default=None):
        return pred_final.get(key, default)

    # Construction du JSON final propre
    result = {
        "5_dern_perf_away": _get("5_dern_perf_away"),
        "5_dern_perf_home": _get("5_dern_perf_home"),
        "_upset_score": _get("_upset_score", 0.0),
        "_upset_threshold": _get("_upset_threshold", 0.52),
        "_use_realtime": _get("_use_realtime", False),

        "away": _get("away"),
        "home": _get("home"),

        "bias_detected": _get("bias_detected", False),
        "low_confidence": _get("low_confidence", False),

        "double_chance": _get("double_chance"),
        "prediction": _get("prediction"),
        "prediction_model": _get("prediction_model"),
        "explanation": _get("explanation", ""),
        "proba_0": _get("proba_0"),
        "proba_1": _get("proba_1"),
        "proba_2": _get("proba_2"),

        "plus_but": _get("plus_but"),
        "mess_but": _get("mess_but"),

        "rule_applied": _get("rule_applied"),
        # ⚠️ explanation laissée vide si absente
       
    }

    return result

## DC
def _apply_form_gate(
    p0, p1, p2,
    features_df: pd.DataFrame,
    league_code="default",
    *,
    # --- règle métier: forme prioritaire si contradiction ---
    form_pick_threshold: float = 0.20,         # seuil contradiction forme
    # --- intensité gate (transfert de masse sur H/A) ---
    k_market_form: float = 0.35,               # intensité max (raisonnable)
    gate_slope: float = 14.0,
    gate_tolerance: float = 0.036,
    # --- sécurité ---
    preserve_draw_mass: bool = True            # on ne touche jamais p1
):
    """
    Gate = ajuste UNIQUEMENT p0/p2 (Home/Away) en fonction de la FORME,
    en restant compatible avec tes outputs (notes: form_vs_market etc.)

    ✅ Règle respectée:
    - La FORME peut déplacer la décision (Home/Away) si contradiction significative.
    - Le draw (p1) est conservé (on ne le gonfle pas), car ton verrou stage2 gère déjà le nul.
    """

    # safe extraction
    try:
        home_form = float(features_df["HomeForm"].values[0])
        away_form = float(features_df["AwayForm"].values[0])
    except Exception:
        return p0, p1, p2, {"form_gate": "skipped_missing_form"}

    # contradiction forme ?
    form_diff = home_form - away_form  # + => home mieux
    if abs(form_diff) < float(form_pick_threshold):
        return p0, p1, p2, {
            "form_gate": "skipped_no_strong_form_signal",
            "home_form": round(home_form, 3),
            "away_form": round(away_form, 3),
            "form_diff": round(form_diff, 3),
            "th": float(form_pick_threshold),
        }

    # gate strength via sigmoid
    gate_strength = float(k_market_form) * _sigmoid(float(gate_slope) * (abs(form_diff) - float(gate_tolerance)))
    gate_strength = float(np.clip(gate_strength, 0.0, 1.0))

    h, d, a = float(p0), float(p1), float(p2)

    # masse HA uniquement
    mass_HA = max(1e-9, (h + a))
    transfer = gate_strength * mass_HA

    # direction: vers l'équipe en forme
    if form_diff > 0:   # home en forme
        # transfère de Away -> Home
        take = min(transfer, a)
        a_new = a - take
        h_new = h + take
    else:               # away en forme
        take = min(transfer, h)
        h_new = h - take
        a_new = a + take

    # renormalise HA pour garder d identique
    if preserve_draw_mass:
        scale = (h + a) / max(1e-9, (h_new + a_new))
        h_new *= scale
        a_new *= scale
        d_new = d
    else:
        # (non utilisé ici)
        h_new, d_new, a_new = _normalize3(h_new, d, a_new)

    return h_new, d_new, a_new, {
        "form_gate": "applied",
        "home_form": round(home_form, 3),
        "away_form": round(away_form, 3),
        "form_diff": round(form_diff, 3),
        "gate_strength": round(gate_strength, 4),
        "transfer": round(float(transfer), 4),
    }


def predict_match_with_proba(
    features_df: pd.DataFrame,
    model_stage1,
    model_stage2,
    threshold_draw=0.63,
    user_profile="standard",
    league_code="default",
) -> dict:
    """
    ✅ LOGIQUE BETSMART (verrouillée) — version stable
    + Anti-0%: floor+renormalize appliqué en FIN de pipeline (draw ET non-draw)
    + REALTIME (info only) sans impacter la prédiction
    """

    (bookmaker_margin, uncertainty_threshold, importance, season_stage,
     upset_threshold, skip_threshold, bogey_weight, gki_weight) = _safe_parametres(league_code)

    # ---- params ligue / config ----
    try:
        params = _get_params(league_code)
    except Exception:
        params = {}

    form_pick_threshold = float(params.get("form_pick_threshold", 0.20))
    strong_conf_threshold = float(params.get("strong_conf_threshold", 0.70))
    strong_conf_draw_cap = float(params.get("strong_conf_draw_cap", 0.12))
    dc_disable_if_strong_conf = bool(params.get("dc_disable_if_strong_conf", True))

    # ✅ nouveau: floor proba (évite 0% / 100% strict)
    min_prob_floor = float(params.get("min_prob_floor", 0.01))  # 1% par défaut

    # ---- util explication ---
    
    def _explain(rule_tag, p0, p1, p2, extra=None):
        f = features_df.copy()

        # IMPORTANT: expose les probas au format attendu
        f["p0_raw"] = float(p0)
        f["p1_raw"] = float(p1)
        f["p2_raw"] = float(p2)

        
        if isinstance(extra, dict):
            for k, v in extra.items():
                try:
                    # ✅ dict/list => forcer "scalaire" en cellule, pas alignement par index
                    if isinstance(v, (dict, list)):
                        if len(f) == 1:
                            f.at[f.index[0], k] = v
                        else:
                            f[k] = [v] * len(f)
                    else:
                        f[k] = v
                except Exception:
                    pass

        
        
        assert "realtime_risk" in f.columns, "realtime_risk missing at explain-time"
        text = generate_explanation(rule_tag, f, user_profile)
        
        

        # ✅ PATCH: copier les metas LLM dans features_df pour que _notes_llm_debug(features_df) marche
        for col in ["_llm_used", "_llm_mode", "_llm_model", "_llm_error", "_llm_debug"]:
            try:
                if col in f.columns:
                    features_df.loc[:, col] = f[col].values[0]
            except Exception:
                pass

        return text

    # ---- clamp draw non-dominant (stage2) ----
    def _clamp_draw_not_dominant(p0, p1, p2, eps=1e-6):
        p0, p1, p2 = float(p0), float(p1), float(p2)
        p0, p1, p2 = _normalize3(p0, p1, p2)
        max_ha = max(p0, p2)
        if p1 >= max_ha:
            target = max(0.0, max_ha - float(eps))
            if p1 > 0:
                scale = target / p1
                p1 = p1 * scale
                rest = 1.0 - p1
                ha_sum = max(1e-9, (p0 + p2))
                p0 = rest * (p0 / ha_sum)
                p2 = rest * (p2 / ha_sum)
        return _normalize3(p0, p1, p2)

    # ---- helper marché dé-margé + DC protection marché ----
    def _market_fav_and_dc():
        try:
            b365h = float(features_df["B365H"].values[0])
            b365d = float(features_df["B365D"].values[0])
            b365a = float(features_df["B365A"].values[0])
            eps_m = max(0.02, 0.5 * float(bookmaker_margin))
            fav_side, pH2, pA2, fav_gap = _fav_by_demarged(b365h, b365d, b365a, eps=eps_m)
            dc_market = None
            if fav_side == "home":
                dc_market = "1X"
            elif fav_side == "away":
                dc_market = "X2"
            return fav_side, float(fav_gap), dc_market
        except Exception:
            return None, 0.0, None

    # ✅ floor + renormalize (anti 0% / 100%)
    def _clip_and_normalize_probs(p0, p1, p2, *, min_prob=0.01):
        p = np.array([float(p0), float(p1), float(p2)], dtype=float)
        if not np.isfinite(p).all():
            p = np.array([1/3, 1/3, 1/3], dtype=float)

        # clip
        min_prob = float(min_prob)
        min_prob = max(0.0, min(min_prob, 0.10))  # sécurité
        p = np.clip(p, min_prob, 1.0 - min_prob)

        # renormalize
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            p = np.array([1/3, 1/3, 1/3], dtype=float)
        else:
            p = p / s
        return float(p[0]), float(p[1]), float(p[2])
    
    def _notes_llm_debug(df):
        try:
            import pandas as pd
            if not isinstance(df, pd.DataFrame) or df.empty:
                return []
            used = df.get("_llm_used")
            err = df.get("_llm_error")
            model = df.get("_llm_model")
            mode = df.get("_llm_mode")

            used_v = used.values[0] if hasattr(used, "values") else used
            err_v = err.values[0] if hasattr(err, "values") else err
            model_v = model.values[0] if hasattr(model, "values") else model
            mode_v = mode.values[0] if hasattr(mode, "values") else mode

            if str(used_v) == "1":
                return [f"explain: llm_used=1 model={model_v}"]
            else:
                if err_v:
                    return [f"explain: llm_used=0 err={str(err_v)[:160]}"]
                return [f"explain: llm_used=0 mode={mode_v}"]
        except Exception:
            return []
    ### nouvels ajouts
    def soften_probs_temperature(p0, p1, p2, T=1.6, eps=1e-12):
        p = np.array([float(p0), float(p1), float(p2)], dtype=float)
        p = np.clip(p, eps, 1.0)
        p = p / p.sum()

        # log-softmax avec température
        logits = np.log(p + eps)
        logits = logits / float(T)
        exp = np.exp(logits - np.max(logits))
        q = exp / exp.sum()
        return float(q[0]), float(q[1]), float(q[2])
    
    def shrink_to_prior(p0, p1, p2, alpha=0.15, prior=(1/3, 1/3, 1/3)):
        # alpha = part de prior (0.10 à 0.30 en pratique)
        p = np.array([p0, p1, p2], dtype=float)
        p = p / p.sum()
        pr = np.array(prior, dtype=float)
        pr = pr / pr.sum()
        q = (1 - alpha) * p + alpha * pr
        q = q / q.sum()
        return float(q[0]), float(q[1]), float(q[2])

    def clip_probs(p0, p1, p2, min_p=0.05, max_p=0.90):
        p = np.array([p0, p1, p2], dtype=float)
        p = p / p.sum()
        p = np.clip(p, float(min_p), float(max_p))
        p = p / p.sum()
        return float(p[0]), float(p[1]), float(p[2])

    # ------------------------------------------------------------------
    # STAGE 1 : pDraw
    # ------------------------------------------------------------------
    X1 = features_df.copy()
    for col in model_stage1.feature_names_in_:
        if col not in X1.columns:
            X1[col] = 0
    X1 = X1[model_stage1.feature_names_in_]

    p_draw = _proba_for_class(model_stage1, X1, LABEL_DRAW, default=0.0)
    p_draw = float(np.clip(p_draw, 0.0, 1.0))

    # ------------------------------------------------------------------
    # CAS DRAW DIRECT
    # ------------------------------------------------------------------
    if p_draw >= float(threshold_draw):
        p1 = p_draw
        p0 = p2 = (1.0 - p1) / 2.0

        p0, p1, p2, meta_gate = _apply_form_gate(
            p0, p1, p2, features_df, league_code,
            form_pick_threshold=form_pick_threshold
        )

        # ✅ anti-0%
        p0_raw, p1_raw, p2_raw = float(p0), float(p1), float(p2)
        p0, p1, p2 = _clip_and_normalize_probs(p0, p1, p2, min_prob=min_prob_floor)
        # adoucissement
        p0, p1, p2 = soften_probs_temperature(p0, p1, p2, T=1.6)

        # shrink léger
        p0, p1, p2 = shrink_to_prior(p0, p1, p2, alpha=0.12)

        pred_final = LABEL_DRAW
        dc = detect_double_chance(p0, p1, p2, pred_final, league_code)

        
        rt_block, rt_note = _build_realtime_block(features_df, league_code=league_code)

        explanation_text = _explain(
            "threshold",
            p0, p1, p2,
            extra={
                "form_gate_meta": str(meta_gate),
                "double_chance": dc,
                "realtime_risk": rt_block,
            }
        )
        
        debug_payload = None
        try:
            if bool(os.getenv("LLM_DEBUG", "").strip() in ("1","true","True")) and "_llm_debug" in features_df.columns:
                debug_payload = features_df["_llm_debug"].values[0]
        except Exception:
            pass

        notes = []
        if rt_note:
            notes.append(rt_note)
        
        notes += _notes_llm_debug(features_df)
        
        return {
            "prediction": int(pred_final),
            "prediction_model": LABEL_DRAW,
            "proba_0": _format_pct(p0),
            "proba_1": _format_pct(p1),
            "proba_2": _format_pct(p2),
            "p0_raw": p0_raw, "p1_raw": p1_raw, "p2_raw": p2_raw,
            "rule_applied": "threshold|draw_dominant|form_gate",
            #"explanation": _explain("threshold", p0, p1, p2, extra={"form_gate_meta": str(meta_gate)}),
            "explanation": generate_explanation("margin_adjusted", features_df, user_profile),
            "double_chance": dc,
            "realtime_risk": rt_block,
            "notes": notes,
            "llm_debug": debug_payload,
        }

    # ------------------------------------------------------------------
    # STAGE 2 : Home vs Away (ND)
    # ------------------------------------------------------------------
    X2 = features_df.copy()
    for col in model_stage2.feature_names_in_:
        if col not in X2.columns:
            X2[col] = 0
    X2 = X2[model_stage2.feature_names_in_]

    pH_nd = _proba_for_class(model_stage2, X2, LABEL_HOME, default=0.5)
    pA_nd = _proba_for_class(model_stage2, X2, LABEL_AWAY, default=0.5)
    s = float(pH_nd) + float(pA_nd)
    if not np.isfinite(s) or s <= 0:
        pH_nd, pA_nd = 0.5, 0.5
    else:
        pH_nd, pA_nd = float(pH_nd) / s, float(pA_nd) / s

    prediction_rf = int(model_stage2.predict(X2)[0])

    p1 = p_draw
    pND = max(0.0, 1.0 - p1)
    p0 = pND * pH_nd
    p2 = pND * pA_nd
    p0, p1, p2 = _normalize3(p0, p1, p2)

    p0, p1, p2 = _clamp_draw_not_dominant(p0, p1, p2)

    p0, p1, p2, meta_gate = _apply_form_gate(
        p0, p1, p2, features_df, league_code,
        form_pick_threshold=form_pick_threshold
    )
    p0, p1, p2 = _normalize3(p0, p1, p2)
    p0, p1, p2 = _clamp_draw_not_dominant(p0, p1, p2)

    # strong conf draw cap
    strong_side = max(float(p0), float(p2))
    strong_conf = (strong_side >= float(strong_conf_threshold))
    strong_tag = None
    if strong_conf:
        cap = float(np.clip(strong_conf_draw_cap, 0.0, 0.30))
        if float(p1) > cap:
            rest = 1.0 - cap
            ha_sum = max(1e-9, (float(p0) + float(p2)))
            p0 = rest * (float(p0) / ha_sum)
            p2 = rest * (float(p2) / ha_sum)
            p1 = cap
            p0, p1, p2 = _normalize3(p0, p1, p2)
        strong_tag = "strong_conf_draw_cut"

    # ✅ anti-0% (IMPORTANT: après TOUS les gates/caps)
    p0_raw, p1_raw, p2_raw = float(p0), float(p1), float(p2)
    p0, p1, p2 = _clip_and_normalize_probs(p0, p1, p2, min_prob=min_prob_floor)
    
     # adoucissement
    p0, p1, p2 = soften_probs_temperature(p0, p1, p2, T=1.6)

        # shrink léger
    p0, p1, p2 = shrink_to_prior(p0, p1, p2, alpha=0.12)

    pred_final = LABEL_HOME if float(p0) >= float(p2) else LABEL_AWAY

    # marché / forme override tag (NE change pas les probas, uniquement la décision)
    fav_side, fav_gap, dc_market = _market_fav_and_dc()

    try:
        home_form = float(features_df["HomeForm"].values[0])
        away_form = float(features_df["AwayForm"].values[0])
        form_diff = home_form - away_form
    except Exception:
        home_form = away_form = form_diff = 0.0

    override_tag = None
    dc_override = None
    if abs(float(form_diff)) >= float(form_pick_threshold):
        form_side = "home" if form_diff > 0 else "away"
        if fav_side is not None and form_side != fav_side:
            pred_final = LABEL_HOME if form_side == "home" else LABEL_AWAY
            override_tag = "form_over_market_pick_" + ("home" if form_side == "home" else "away")

            # ✅ DC cohérente avec la décision finale (forme)
            dc_override = "1X" if form_side == "home" else "X2"
            
            

    bd = detect_bias(features_df)
    if isinstance(bd, pd.Series):
        bias_detected = bool(bd.iloc[0])
    elif isinstance(bd, (list, tuple, np.ndarray)):
        bias_detected = bool(np.any(bd))
    else:
        bias_detected = bool(bd)
        
    low_confidence = bool(is_confidence_low(p0, p1, p2))

    #dc = detect_double_chance(p0, p1, p2, pred_final, league_code)
    dc = detect_double_chance_v2(
        p0, p1, p2, pred_final,
        league_code=league_code,
        bias_detected=bias_detected,
        low_confidence=low_confidence,
        upset_score=float(_safe_get_first(features_df, "_upset_score") or 0.0),
        upset_threshold=float(_safe_get_first(features_df, "_upset_threshold") or 0.52),
        override_tag=override_tag
    )
    if dc_override is not None:
        dc = dc_override

    if strong_conf and dc_disable_if_strong_conf and (not bias_detected) and (not low_confidence):
        dc = None

    if (bias_detected or low_confidence) and dc is None:
        dc = "1X" if pred_final == LABEL_HOME else "X2"
    
    if dc == "1X" and pred_final == LABEL_AWAY:
        dc = "X2"

    if dc == "X2" and pred_final == LABEL_HOME:
        dc = "1X"

    rule_parts = ["rf_decision", "stage2_locked_no_draw", "form_gate"]
    if strong_tag:
        rule_parts.append(strong_tag)
    if override_tag:
        rule_parts.append(override_tag)
    rule_applied = "|".join(rule_parts)

    extra = {
        "bias_detected": int(bias_detected),
        "low_confidence": int(low_confidence),
        "form_gate_meta": str(meta_gate),
        "strong_conf": int(bool(strong_conf)),
        "fav_side": str(fav_side),
        "fav_gap": float(fav_gap),
        "form_diff": float(form_diff),
    }
   
    rt_block, rt_note = _build_realtime_block(features_df, league_code=league_code)

    extra = {
        "bias_detected": int(bias_detected),
        "low_confidence": int(low_confidence),
        "form_gate_meta": str(meta_gate),
        "strong_conf": int(bool(strong_conf)),
        "fav_side": str(fav_side),
        "fav_gap": float(fav_gap),
        "form_diff": float(form_diff),
        "double_chance": dc,

        # ✅ TRÈS IMPORTANT : passer le realtime au générateur d'explication
        "realtime_risk": rt_block,
    }

    # ✅ ensuite explication (maintenant elle voit summary/ranking/absences)
    explanation_text = _explain("rf_decision", p0, p1, p2, extra=extra)
    
    debug_payload = None
    try:
        if bool(os.getenv("LLM_DEBUG", "").strip() in ("1","true","True")) and "_llm_debug" in features_df.columns:
            debug_payload = features_df["_llm_debug"].values[0]
    except Exception:
        pass
    
    
    notes = []
    if rt_note:
        notes.append(rt_note)
    
    notes += _notes_llm_debug(features_df)

    return {
        "prediction": int(pred_final),
        "prediction_model": prediction_rf,
        "proba_0": _format_pct(p0),
        "proba_1": _format_pct(p1),
        "proba_2": _format_pct(p2),
        "p0_raw": p0_raw, "p1_raw": p1_raw, "p2_raw": p2_raw,
        "rule_applied": rule_applied,
        #"explanation": _explain("rf_decision", p0, p1, p2, extra=extra),
        "explanation": explanation_text,
        "double_chance": dc,
        "bias_detected": bias_detected,
        "low_confidence": low_confidence,
        "realtime_risk": rt_block,
        "notes": notes,
        "llm_debug": debug_payload,
    }


# =========================
# Unexpected / anti-OC layer
# =========================
def apply_unexpected_layer(
    base_pred: dict,
    season_current_df=None,
    season_past_list=None,
    home: str = None,
    away: str = None,
    match_date=None,
    feats_df=None,
    league_code: str = None,
    X_ref_features=None,
    upset_threshold: float = 0.52,
):
    """
    Safe post-layer that can enrich the base prediction with:
      - real-time risk block (without overwriting a valid fixture_id from predict_match_with_proba)
      - conservative 'unexpected' score placeholders

    Constraint respected: does NOT change your existing 1N2 prediction logic;
    it only enriches output fields and notes.
    """
    out = dict(base_pred or {})
    notes = list(out.get("notes", []))

    # ---- feats_df must exist for safe getters ----
    feats_df = feats_df if feats_df is not None else {}

    # ---- propagate home/away (prefer base_pred if already set) ----
    if out.get("home") is None and home is not None:
        out["home"] = str(home)
    if out.get("away") is None and away is not None:
        out["away"] = str(away)

    # Resolve canonical names for realtime calls:
    home_name = str(home) if home is not None else (str(out.get("home")) if out.get("home") is not None else None)
    away_name = str(away) if away is not None else (str(out.get("away")) if out.get("away") is not None else None)

    # match_date: prefer explicit arg, else feats_df["match_date"] if present
    if match_date is None:
        try:
            md = _safe_get_first(feats_df, "match_date")
            match_date = md if md not in ("", None) else None
        except Exception:
            match_date = None

    # _use_realtime: prefer feats_df flag if present else keep existing else False
    try:
        if _safe_get_first(feats_df, "_use_realtime") is not None:
            out["_use_realtime"] = bool(_safe_get_first(feats_df, "_use_realtime"))
        else:
            out["_use_realtime"] = bool(out.get("_use_realtime", False))
    except Exception:
        out["_use_realtime"] = bool(out.get("_use_realtime", False))

    # ------------------------------------------------------------------
    # REALTIME BLOCK (FIXED):
    # Rule: if base already contains a fixture_id, NEVER overwrite it.
    # Even if ctx missing / available False / missing not empty -> keep fixture_id.
    # Only build realtime if fixture_id is absent.
    # ------------------------------------------------------------------
    rt_existing = out.get("realtime_risk")

    existing_fixture_id = None
    if isinstance(rt_existing, dict):
        existing_fixture_id = rt_existing.get("fixture_id", None)

    # If realtime isn't enabled, keep block as-is (or set minimal) and do not try to resolve.
    if not out["_use_realtime"]:
        # keep existing if any, else set minimal
        if not isinstance(rt_existing, dict):
            out["realtime_risk"] = {
                "available": False,
                "fixture_id": None,
                "missing": ["realtime_not_enabled_or_unavailable"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
            }
        notes.append("realtime: not enabled")
    else:
        # realtime enabled
        if existing_fixture_id is not None:
            # ✅ KEEP, do not overwrite (prevents your contradictory logs)
            notes.append(f"realtime: kept_fixture_id={existing_fixture_id}")
            # keep existing block untouched
            out["realtime_risk"] = rt_existing
        else:
            # Only now we attempt to build realtime
            try:
                rt_block, rt_note = _build_realtime_block(
                    feats_df,
                    league_code=league_code,
                    home_name=home_name,
                    away_name=away_name,
                    match_date=match_date,
                    season_df=season_current_df,
                )
                out["realtime_risk"] = rt_block
                notes.append(rt_note)
            except Exception as e:
                out["realtime_risk"] = {
                    "available": False,
                    "fixture_id": None,
                    "missing": [f"realtime_error:{type(e).__name__}"],
                    "reasons": [],
                    "risk_level": "UNKNOWN",
                    "risk_score": 0.0,
                }
                notes.append(f"realtime: error={type(e).__name__}")

    # ------------------------------------------------------------------
    # Unexpected score (unchanged behaviour)
    # ------------------------------------------------------------------
    out.setdefault("_upset_threshold", float(upset_threshold))

    upset_score = out.get("_upset_score")
    if upset_score is None:
        try:
            odd_h = float(_safe_get_first(feats_df, "B365H") or 0.0)
            odd_a = float(_safe_get_first(feats_df, "B365A") or 0.0)
            odd_d = float(_safe_get_first(feats_df, "B365D") or 0.0)

            inv = []
            for o in (odd_h, odd_d, odd_a):
                inv.append(1.0 / o if o and o > 0 else 0.0)
            s = sum(inv) if sum(inv) > 0 else 1.0
            p_h, p_d, p_a = [v / s for v in inv]

            pred = out.get("prediction")
            best = max(p_h, p_a)
            if pred == 0:
                outsider = p_h
            elif pred == 2:
                outsider = p_a
            else:
                outsider = p_d

            upset_score = float(max(0.0, min(1.0, best - outsider)))
        except Exception:
            upset_score = 0.0

    out["_upset_score"] = float(upset_score)

    out["notes"] = notes
    return out


##---------------------- FIN FONCTIONS  ------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _extract_row(features: Any) -> Dict[str, Any]:
    # features peut être DataFrame (1 ligne), dict, etc.
    try:
        import pandas as pd
        if isinstance(features, pd.DataFrame):
            if features.empty:
                return {}
            return features.to_dict(orient="records")[0] or {}
    except Exception:
        pass

    if isinstance(features, dict):
        return dict(features)
    return {}

# ============================================================
# Helpers (assume you already have these in fonction.py)
# - _safe_get_first(df, col)
# - detect_bias(df)
# - _format_pct(x)  # if x in [0..1] => "13.0%" etc
# ============================================================
def _safe_get_first(df: Any, col: str):
    try:
        if isinstance(df, pd.DataFrame) and col in df.columns and len(df) > 0:
            v = df[col].iloc[0]
            # unwrap numpy scalars
            if isinstance(v, (np.generic,)):
                return v.item()
            return v
    except Exception:
        pass
    return None

def _format_pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "0.0%"

def get_valid_date(user_input):
    """
    Convertit différentes représentations de date en format 'YYYY-MM-DD'.
    """
    try:
        # Parse intelligent (fonctionne avec des formats très variés)
        date_obj = parser.parse(user_input)
        return date_obj.strftime("%Y-%m-%d")
    except Exception:
        raise ValueError("⛔ Format de date non reconnu. Essayez par exemple : '2025-02-14' ou '14/02/2025'")



##---------------------- NOMBRE DE BUTS MARQUES PAR EQUIPE ------------------------------------------


def entree_utilisateur(home_team, away_team, b365h,b365a,b365d, season_current, season_previous):
    # 🔧 Chargement des arguments
    # ---------------------
    home_team=str(home_team)
    away_team=str(away_team)
    b365h=float(b365h)
    b365a=float(b365a)
    b365d=float(b365d)
    
    #df_curr = pd.read_csv(args.season_current, parse_dates=["Date"])
    df_curr = season_current.copy()
    df_curr['Date']=pd.to_datetime(df_curr['Date'])
    df_curr=df_curr.sort_values(by='Date')
    df_prev = season_previous.copy()
    df_prev['Date']=pd.to_datetime(df_prev['Date'])
    df_prev=df_prev.sort_values(by='Date')
    
    df_prev["goals_1s"] = df_prev["HTHG"] + df_prev["HTAG"]
    df_prev["goals_2n"] = (df_prev["FTHG"] + df_prev["FTAG"]) - df_prev["goals_1s"]

    df_prev["conceded_1s"] = df_prev["goals_1s"]  # pour les moyennes globales, c’est la même chose
    df_prev["conceded_2n"] = df_prev["goals_2n"]

    # Calcul des points (pts) par match selon le résultat
    df_prev["pts"] = df_prev["FTR"].map({"H": 3, "D": 1, "A": 0})
    
    # 📊 Moyennes globales
    # ---------------------
    league_avg = {
        "goals_1st": round(df_prev["goals_1s"].mean(), 2),
        "goals_2nd": round(df_prev["goals_2n"].mean(), 2),
        "conceded_1st": round(df_prev["conceded_1s"].mean(), 2),
        "conceded_2nd": round(df_prev["conceded_2n"].mean(), 2),
        "pts": round(df_prev["pts"].mean(), 2)
    }
    
    def compute_form(team, df, window=5):
        
        df_team = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date", ascending=True)
        if len(df_team) == 0:
            return None
        form = []
        for _, row in df_team.iterrows():
            is_home = row["HomeTeam"] == team
            hthg, fthg = row["HTHG"], row["FTHG"]
            htag, ftag = row["HTAG"], row["FTAG"]

            g1 = hthg if is_home else htag
            g2 = (fthg - hthg) if is_home else (ftag - htag)
            c1 = htag if is_home else hthg
            c2 = (ftag - htag) if is_home else (fthg - hthg)

            if (fthg == ftag): pts = 1
            elif (is_home and fthg > ftag) or (not is_home and ftag > fthg): pts = 3
            else: pts = 0

            form.append((g1, g2, c1, c2, pts))

        if len(form) < 3:
            return None
        last = form[-window:]
        return {
            "goals_1st": np.mean([x[0] for x in last]),
            "goals_2nd": np.mean([x[1] for x in last]),
            "conceded_1st": np.mean([x[2] for x in last]),
            "conceded_2nd": np.mean([x[3] for x in last]),
            "pts": np.mean([x[4] for x in last])
            }
    def get_final_form(team):
        
        # Priorité : saison en cours
        f1 = compute_form(team, df_curr)
        if f1: return f1
        # Sinon, saison précédente
        f2 = compute_form(team, df_prev)
        if f2: return f2
        # Sinon, valeurs moyennes
        return league_avg
    home_stats = get_final_form(home_team)
    away_stats = get_final_form(away_team)
    
    input_features = {
    "total_avg_goals_home": home_stats["goals_1st"] + home_stats["goals_2nd"] + home_stats["conceded_1st"] + home_stats["conceded_2nd"],
    "total_avg_goals_away": away_stats["goals_1st"] + away_stats["goals_2nd"] + away_stats["conceded_1st"] + away_stats["conceded_2nd"],
    "goal_diff_home": (home_stats["goals_1st"] + home_stats["goals_2nd"]) - (home_stats["conceded_1st"] + home_stats["conceded_2nd"]),
    "goal_diff_away": (away_stats["goals_1st"] + away_stats["goals_2nd"]) - (away_stats["conceded_1st"] + away_stats["conceded_2nd"]),
    "pts_recent_home": home_stats["pts"],
    "pts_recent_away": away_stats["pts"],
    "odds_diff": b365h - b365a,
    "odds_draw_gap": b365d - np.mean([b365h,b365a]),
    "odds_mean": np.mean([b365h, b365d, b365a])}
    
    return pd.DataFrame([input_features])

def to_serializable(obj):
    if isinstance(obj, floating):
        return float(obj)
    elif isinstance(obj, integer):
        return int(obj)
    elif isinstance(obj, ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj

def get_last5_results_pattern(df, team_name, match_date):
    """
    Retourne les 5 derniers résultats ('W', 'L', 'D') d'une équipe donnée avant une date donnée.
    Si aucun match joué avant la date → 'MMMMM'.
    Sinon → complète les matchs manquants avec 'M'.
    """
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    match_date = pd.to_datetime(match_date)

    past_matches = df[
        ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) &
        (df['Date'] < match_date)
    ].sort_values(by='Date', ascending=False).head(5)

    if past_matches.empty:
        return "MMMMM"

    results = []

    for _, row in past_matches.iterrows():
        if row['HomeTeam'] == team_name:
            if row['FTR'] == 'H':
                results.append('W')
            elif row['FTR'] == 'D':
                results.append('D')
            else:
                results.append('L')
        elif row['AwayTeam'] == team_name:
            if row['FTR'] == 'A':
                results.append('W')
            elif row['FTR'] == 'D':
                results.append('D')
            else:
                results.append('L')

    # Compléter avec 'M' si moins de 5 matchs
    while len(results) < 5:
        results.append('M')

    return ''.join(results)