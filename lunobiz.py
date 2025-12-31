"""
LunoBiz - Single-file Windows application (v0.3.1)

Key features:
- GUI dashboard (values & tables only)
- Robust candle caching in SQLite
- Backtesting with walk-forward validation
- Paper trading engine (light loop; backtester is primary evaluation tool)
- Paper-only leverage simulation (no live execution)
- Option A: Volatility / Mean-Reversion Engine (StrategyV3)
- GUI-controllable strategy parameters (no .env editing required)
- Strategy parameters persisted in SQLite and embedded into backtest reports
- Safety controls: kill switch, live gate (still locked by default)

Public-repo safe:
- No secrets in code
- Reads secrets from .env (git-ignored) or environment variables
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import json
import math
import os
import queue
import random
import re
import sqlite3
import sys
import threading
import time
import traceback
import typing as t
from dataclasses import dataclass
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception as e:
    raise RuntimeError("Tkinter is required. On Windows it should be included with Python.") from e

try:
    import requests
except Exception as e:
    raise RuntimeError("Missing dependency: requests. Install with: pip install requests") from e


# =========================
# Constants / Globals
# =========================

APP_NAME = "LunoBiz"
APP_VERSION = "0.3.1"
DEFAULT_API_BASE = "https://api.luno.com"

# Luno endpoints
CANDLES_PATH = "/api/exchange/1/candles"
TICKER_PATH = "/api/1/ticker"
TICKERS_PATH = "/api/1/tickers"
BALANCE_PATH = "/api/1/balance"
LIST_ORDERS_PATH = "/api/1/listorders"
CREATE_ORDER_PATH = "/api/1/postorder"
STOP_ORDER_PATH = "/api/1/stoporder"

# Safety toggles (local files, not tracked)
LIVE_MODE_UNLOCK_FILE = "LIVE_UNLOCK.ok"
KILL_SWITCH_FILE = "KILL_SWITCH"

# Reporting
REPORTS_DIRNAME = "reports"

# DB keys
STRATEGY_PARAMS_KEY = "strategy_params_v3"


# =========================
# Utilities
# =========================

def now_utc() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


def utc_ms(ts: dt.datetime | None = None) -> int:
    if ts is None:
        ts = now_utc()
    return int(ts.timestamp() * 1000)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x: t.Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def safe_int(x: t.Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def fmt_money(x: float, currency: str = "MYR") -> str:
    try:
        return f"{currency} {x:,.2f}"
    except Exception:
        return f"{currency} {x}"


def fmt_pct(x: float) -> str:
    try:
        return f"{x * 100:.2f}%"
    except Exception:
        return f"{x}"


def human_ts(ms: int) -> str:
    try:
        d = dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.timezone.utc)
        return d.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ms)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_kill_switch_on(data_dir: Path) -> bool:
    try:
        return (data_dir / KILL_SWITCH_FILE).exists()
    except Exception:
        return False


def open_folder_in_explorer(folder: Path) -> None:
    try:
        if not folder.exists():
            return
        if sys.platform.startswith("win"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{folder}"')
        else:
            os.system(f'xdg-open "{folder}"')
    except Exception:
        pass


def safe_json_dumps(obj: t.Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            return "{}"


# =========================
# .env loader (no extra deps)
# =========================

_ENV_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def load_dotenv(dotenv_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not dotenv_path.exists():
        return env

    try:
        text = dotenv_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return env

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE_RE.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if v and not (v.startswith('"') and v.endswith('"')) and "#" in v:
            v = v.split("#", 1)[0].strip()
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        env[k] = v
    return env


def get_env(key: str, fallback: str = "") -> str:
    val = os.environ.get(key)
    if val is None or str(val).strip() == "":
        return fallback
    return str(val)


# =========================
# Timeframes
# =========================

def timeframe_to_seconds(tf: str) -> int:
    tf = (tf or "").strip().lower()
    m = re.match(r"^(\d+)\s*([smhdw])$", tf)
    if not m:
        return 300
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "s":
        return n
    if unit == "m":
        return n * 60
    if unit == "h":
        return n * 3600
    if unit == "d":
        return n * 86400
    if unit == "w":
        return n * 604800
    return 300


# =========================
# Logging
# =========================

class AppLogger:
    def __init__(self, data_dir: Path):
        self._q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._data_dir = data_dir
        self._log_file = data_dir / "lunobiz.log"

    def _emit(self, level: str, msg: str) -> None:
        ts = now_utc().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts} UTC] {level.upper():<5} {msg}"
        self._q.put((level.upper(), line))
        try:
            ensure_dir(self._data_dir)
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def info(self, msg: str) -> None:
        self._emit("INFO", msg)

    def warn(self, msg: str) -> None:
        self._emit("WARN", msg)

    def error(self, msg: str) -> None:
        self._emit("ERROR", msg)

    def exception(self, msg: str, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._emit("ERROR", f"{msg}\n{tb}")

    def drain(self, max_items: int = 200) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        for _ in range(max_items):
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        return items


# =========================
# Config
# =========================

@dataclass
class AppConfig:
    luno_api_key_readonly: str
    luno_api_secret_readonly: str
    luno_api_key_live: str
    luno_api_secret_live: str

    telegram_bot_token: str
    telegram_chat_id: str

    app_mode: str  # PAPER or LIVE

    default_view_pair: str
    scan_pairs_csv: str

    # Portfolio risk controls
    risk_per_trade: float           # fraction of equity risked to stop (e.g. 0.005 = 0.5%)
    daily_loss_cap: float           # fraction of equity (e.g. 0.02 = 2% daily max loss)
    max_open_positions: int
    cooldown_minutes: int

    # Backtest parameters
    backtest_timeframe: str         # entry timeframe
    signal_timeframe: str           # higher timeframe for regime
    backtest_slippage_bps: float
    backtest_fee_bps: float

    # Paper leverage simulation
    leverage_enabled_paper: bool
    leverage_max: float             # hard cap
    leverage_min: float             # typically 1
    margin_alloc_cap: float         # cap margin usage fraction of equity per position
    maint_margin_ratio: float       # simplistic maintenance margin ratio

    data_dir: Path
    db_filename: str

    poll_interval_seconds: int
    http_timeout_seconds: int = 15
    http_max_retries: int = 4

    def is_paper(self) -> bool:
        return self.app_mode.strip().upper() == "PAPER"

    def is_live(self) -> bool:
        return self.app_mode.strip().upper() == "LIVE"


def load_config(repo_root: Path, log: AppLogger) -> AppConfig:
    dotenv_path = repo_root / ".env"
    dotenv = load_dotenv(dotenv_path)
    for k, v in dotenv.items():
        if os.environ.get(k) is None:
            os.environ[k] = v

    def envs(k: str, default: str = "") -> str:
        return get_env(k, default)

    data_dir = Path(envs("DATA_DIR", "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    cfg = AppConfig(
        luno_api_key_readonly=envs("LUNO_API_KEY_READONLY", ""),
        luno_api_secret_readonly=envs("LUNO_API_SECRET_READONLY", ""),
        luno_api_key_live=envs("LUNO_API_KEY_LIVE", ""),
        luno_api_secret_live=envs("LUNO_API_SECRET_LIVE", ""),

        telegram_bot_token=envs("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=envs("TELEGRAM_CHAT_ID", ""),

        app_mode=envs("APP_MODE", "PAPER"),

        default_view_pair=envs("DEFAULT_VIEW_PAIR", "ETHMYR"),
        scan_pairs_csv=envs("SCAN_PAIRS", ""),

        risk_per_trade=safe_float(envs("RISK_PER_TRADE", "0.005"), 0.005),
        daily_loss_cap=safe_float(envs("DAILY_LOSS_CAP", "0.02"), 0.02),
        max_open_positions=max(1, safe_int(envs("MAX_OPEN_POSITIONS", "2"), 2)),
        cooldown_minutes=max(1, safe_int(envs("COOLDOWN_MINUTES", "240"), 240)),

        backtest_timeframe=envs("BACKTEST_TIMEFRAME", "15m"),
        signal_timeframe=envs("SIGNAL_TIMEFRAME", "1h"),
        backtest_slippage_bps=safe_float(envs("BACKTEST_SLIPPAGE_BPS", "8"), 8.0),
        backtest_fee_bps=safe_float(envs("BACKTEST_FEE_BPS", "30"), 30.0),

        leverage_enabled_paper=(envs("PAPER_LEVERAGE_ENABLED", "1").strip() not in ("0", "false", "False")),
        leverage_max=clamp(safe_float(envs("PAPER_LEVERAGE_MAX", "3.0"), 3.0), 1.0, 10.0),
        leverage_min=clamp(safe_float(envs("PAPER_LEVERAGE_MIN", "1.0"), 1.0), 1.0, 10.0),
        margin_alloc_cap=clamp(safe_float(envs("MARGIN_ALLOC_CAP", "0.60"), 0.60), 0.05, 0.95),
        maint_margin_ratio=clamp(safe_float(envs("MAINT_MARGIN_RATIO", "0.35"), 0.35), 0.05, 0.95),

        data_dir=data_dir,
        db_filename=envs("DB_FILENAME", "lunobiz.sqlite3"),

        poll_interval_seconds=max(2, safe_int(envs("POLL_INTERVAL_SECONDS", "10"), 10)),
        http_timeout_seconds=max(5, safe_int(envs("HTTP_TIMEOUT_SECONDS", "15"), 15)),
        http_max_retries=max(1, safe_int(envs("HTTP_MAX_RETRIES", "4"), 4)),
    )

    ensure_dir(cfg.data_dir)
    log.info(
        f"Config loaded. mode={cfg.app_mode.upper()} data_dir={cfg.data_dir} "
        f"bt_tf={cfg.backtest_timeframe} signal_tf={cfg.signal_timeframe} "
        f"paper_leverage={'ON' if cfg.leverage_enabled_paper else 'OFF'}"
    )
    return cfg


# =========================
# SQLite storage
# =========================

class Storage:
    def __init__(self, db_path: Path, log: AppLogger):
        self.db_path = db_path
        self.log = log
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path), check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.executescript("""
                CREATE TABLE IF NOT EXISTS candles (
                    pair TEXT NOT NULL,
                    duration_sec INTEGER NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY(pair, duration_sec, ts_ms)
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    pair TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    meta_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    qty REAL NOT NULL,
                    fee REAL NOT NULL,
                    equity_after REAL NOT NULL,
                    meta_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS gates (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_ms INTEGER NOT NULL
                );
                """)
                con.commit()
            finally:
                con.close()
        self.log.info(f"SQLite ready: {self.db_path}")

    def upsert_candles(self, pair: str, duration_sec: int, candles: list[dict[str, t.Any]]) -> int:
        if not candles:
            return 0
        with self._lock:
            con = self._connect()
            try:
                cur = con.cursor()
                n = 0
                for c in candles:
                    ts_ms = safe_int(c.get("timestamp"), 0)
                    if ts_ms <= 0:
                        continue
                    o = safe_float(c.get("open"), 0.0)
                    h = safe_float(c.get("high"), 0.0)
                    l = safe_float(c.get("low"), 0.0)
                    cl = safe_float(c.get("close"), 0.0)
                    v = safe_float(c.get("volume"), 0.0)
                    cur.execute("""
                        INSERT OR REPLACE INTO candles(pair, duration_sec, ts_ms, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (pair, duration_sec, ts_ms, o, h, l, cl, v))
                    n += 1
                con.commit()
                return n
            finally:
                con.close()

    def fetch_candles(self, pair: str, duration_sec: int, since_ms: int, until_ms: int) -> list[sqlite3.Row]:
        with self._lock:
            con = self._connect()
            try:
                cur = con.cursor()
                cur.execute("""
                    SELECT ts_ms, open, high, low, close, volume
                    FROM candles
                    WHERE pair=? AND duration_sec=? AND ts_ms>=? AND ts_ms<=?
                    ORDER BY ts_ms ASC
                """, (pair, duration_sec, since_ms, until_ms))
                return cur.fetchall()
            finally:
                con.close()

    def add_decision(self, ts_ms: int, pair: str, action: str, confidence: float, reason: str, meta: dict[str, t.Any]) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.execute("""
                    INSERT INTO decisions(ts_ms, pair, action, confidence, reason, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ts_ms, pair, action, float(confidence), reason, json.dumps(meta, ensure_ascii=False)))
                con.commit()
            finally:
                con.close()

    def add_paper_trade(self, ts_ms: int, pair: str, side: str, price: float, qty: float, fee: float, equity_after: float, meta: dict[str, t.Any]) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.execute("""
                    INSERT INTO paper_trades(ts_ms, pair, side, price, qty, fee, equity_after, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (ts_ms, pair, side, float(price), float(qty), float(fee), float(equity_after), json.dumps(meta, ensure_ascii=False)))
                con.commit()
            finally:
                con.close()

    def set_gate(self, key: str, value: str) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.execute("""
                    INSERT INTO gates(key, value, updated_ms)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ms=excluded.updated_ms
                """, (key, value, utc_ms()))
                con.commit()
            finally:
                con.close()

    def get_gate(self, key: str, default: str = "") -> str:
        with self._lock:
            con = self._connect()
            try:
                cur = con.cursor()
                cur.execute("SELECT value FROM gates WHERE key=?", (key,))
                row = cur.fetchone()
                return str(row["value"]) if row else default
            finally:
                con.close()

    # ---- KV store for persisted GUI parameters ----

    def kv_set(self, key: str, value: str) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.execute("""
                    INSERT INTO kv_store(key, value, updated_ms)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ms=excluded.updated_ms
                """, (key, value, utc_ms()))
                con.commit()
            finally:
                con.close()

    def kv_get(self, key: str, default: str = "") -> str:
        with self._lock:
            con = self._connect()
            try:
                cur = con.cursor()
                cur.execute("SELECT value FROM kv_store WHERE key=?", (key,))
                row = cur.fetchone()
                return str(row["value"]) if row else default
            finally:
                con.close()


# =========================
# Notifications (Telegram optional)
# =========================

class Notifier:
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log

    def telegram_enabled(self) -> bool:
        return bool(self.cfg.telegram_bot_token.strip()) and bool(self.cfg.telegram_chat_id.strip())

    def send_telegram(self, text: str) -> None:
        if not self.telegram_enabled():
            return
        try:
            url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
            r = requests.post(url, data={"chat_id": self.cfg.telegram_chat_id, "text": text}, timeout=10)
            if r.status_code != 200:
                self.log.warn(f"Telegram send failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            self.log.exception("Telegram send exception", e)

    def notify(self, title: str, message: str) -> None:
        self.log.info(f"{title}: {message}")
        if self.telegram_enabled():
            self.send_telegram(f"{title}\n{message}")


# =========================
# Luno REST Client
# =========================

class LunoClient:
    def __init__(self, cfg: AppConfig, log: AppLogger, use_live: bool):
        self.cfg = cfg
        self.log = log
        self.base = DEFAULT_API_BASE.rstrip("/")
        self.use_live = use_live

    def _auth_header(self) -> dict[str, str]:
        key = self.cfg.luno_api_key_live if self.use_live else self.cfg.luno_api_key_readonly
        sec = self.cfg.luno_api_secret_live if self.use_live else self.cfg.luno_api_secret_readonly
        if not key or not sec:
            return {}
        token = base64.b64encode(f"{key}:{sec}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}"}

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, t.Any] | None = None,
        data: dict[str, t.Any] | None = None
    ) -> dict[str, t.Any]:
        url = f"{self.base}{path}"
        headers = {"Accept": "application/json"}
        headers.update(self._auth_header())

        timeout = self.cfg.http_timeout_seconds
        retries = self.cfg.http_max_retries
        backoff = 0.6

        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                if method.upper() == "GET":
                    r = requests.get(url, params=params, headers=headers, timeout=timeout)
                else:
                    r = requests.post(url, params=params, data=data, headers=headers, timeout=timeout)

                if r.status_code == 429:
                    self.log.warn(f"HTTP 429 rate-limited on {path}. attempt={attempt}/{retries}")
                    time.sleep(backoff * attempt + random.random() * 0.2)
                    continue
                if r.status_code >= 500:
                    self.log.warn(f"HTTP {r.status_code} server error on {path}. attempt={attempt}/{retries}")
                    time.sleep(backoff * attempt)
                    continue
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                try:
                    return r.json()
                except Exception:
                    raise RuntimeError(f"Non-JSON response: {r.text[:200]}")
            except Exception as e:
                last_exc = e if isinstance(e, Exception) else Exception(str(e))
                self.log.warn(f"HTTP error on {path}: {e}. attempt={attempt}/{retries}")
                time.sleep(backoff * attempt)
        raise RuntimeError(f"Request failed after retries: {path}. last={last_exc}")

    def tickers(self) -> dict[str, t.Any]:
        return self._request("GET", TICKERS_PATH)

    def ticker(self, pair: str) -> dict[str, t.Any]:
        return self._request("GET", TICKER_PATH, params={"pair": pair})

    def candles(self, pair: str, since_ms: int, duration_sec: int) -> dict[str, t.Any]:
        return self._request("GET", CANDLES_PATH, params={"pair": pair, "since": int(since_ms), "duration": int(duration_sec)})

    def balance(self) -> dict[str, t.Any]:
        return self._request("GET", BALANCE_PATH)

    def list_orders(self, state: str = "PENDING", pair: str | None = None) -> dict[str, t.Any]:
        params = {"state": state}
        if pair:
            params["pair"] = pair
        return self._request("GET", LIST_ORDERS_PATH, params=params)

    def post_order_limit(self, pair: str, side: str, volume: str, price: str) -> dict[str, t.Any]:
        data = {"pair": pair, "type": "LIMIT", "side": side, "volume": volume, "price": price}
        return self._request("POST", CREATE_ORDER_PATH, data=data)

    def stop_order(self, order_id: str) -> dict[str, t.Any]:
        return self._request("POST", STOP_ORDER_PATH, data={"order_id": order_id})


# =========================
# Indicators
# =========================

def sma(values: list[float], period: int) -> list[float]:
    if period <= 1 or not values:
        return values[:]
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > period:
            s -= q.pop(0)
        if len(q) < period:
            out.append(q[-1])
        else:
            out.append(s / period)
    return out


def stdev(values: list[float], period: int) -> list[float]:
    if period <= 1 or not values:
        return [0.0] * len(values)
    out: list[float] = []
    q: list[float] = []
    for v in values:
        q.append(v)
        if len(q) > period:
            q.pop(0)
        if len(q) < period:
            out.append(0.0)
        else:
            m = sum(q) / period
            var = sum((x - m) ** 2 for x in q) / period
            out.append(math.sqrt(max(0.0, var)))
    return out


def ema(values: list[float], period: int) -> list[float]:
    if period <= 1 or len(values) == 0:
        return values[:]
    out: list[float] = []
    k = 2.0 / (period + 1.0)
    prev = values[0]
    out.append(prev)
    for v in values[1:]:
        prev = prev + k * (v - prev)
        out.append(prev)
    return out


def rsi(values: list[float], period: int = 14) -> list[float]:
    if len(values) < period + 1:
        return [50.0] * len(values)
    gains: list[float] = [0.0]
    losses: list[float] = [0.0]
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(0.0, ch))
        losses.append(max(0.0, -ch))
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period
    out: list[float] = [50.0] * (period)
    rs = avg_gain / (avg_loss + 1e-12)
    out.append(100.0 - (100.0 / (1.0 + rs)))
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / (avg_loss + 1e-12)
        out.append(100.0 - (100.0 / (1.0 + rs)))
    return out


def true_range(high: list[float], low: list[float], close: list[float]) -> list[float]:
    tr: list[float] = [0.0]
    for i in range(1, len(close)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    return tr


def atr(high: list[float], low: list[float], close: list[float], period: int = 14) -> list[float]:
    tr = true_range(high, low, close)
    return ema(tr, period)


def adx(high: list[float], low: list[float], close: list[float], period: int = 14) -> list[float]:
    """
    Wilder's ADX implementation (sufficient for regime filtering).
    Returns list aligned to inputs length, using 0 until enough data accumulates.
    """
    n = len(close)
    if n < period * 3:
        return [0.0] * n

    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n

    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    def wilder_smooth(series: list[float], p: int) -> list[float]:
        out = [0.0] * n
        s = sum(series[1:p + 1])
        out[p] = s
        for i in range(p + 1, n):
            out[i] = out[i - 1] - (out[i - 1] / p) + series[i]
        return out

    atr_s = wilder_smooth(tr, period)
    pdm_s = wilder_smooth(plus_dm, period)
    mdm_s = wilder_smooth(minus_dm, period)

    pdi = [0.0] * n
    mdi = [0.0] * n
    dx = [0.0] * n
    for i in range(period, n):
        denom = atr_s[i] if atr_s[i] != 0 else 1e-12
        pdi[i] = 100.0 * (pdm_s[i] / denom)
        mdi[i] = 100.0 * (mdm_s[i] / denom)
        d = pdi[i] + mdi[i]
        dx[i] = 100.0 * (abs(pdi[i] - mdi[i]) / (d if d != 0 else 1e-12))

    adx_out = [0.0] * n
    start = period * 2
    adx_out[start] = sum(dx[period + 1:start + 1]) / period
    for i in range(start + 1, n):
        adx_out[i] = ((adx_out[i - 1] * (period - 1)) + dx[i]) / period

    return adx_out


# =========================
# Candle helpers
# =========================

@dataclass
class CandleSeries:
    pair: str
    tf: str
    duration_sec: int
    ts: list[int]
    o: list[float]
    h: list[float]
    l: list[float]
    c: list[float]
    v: list[float]


def build_series_from_rows(pair: str, tf: str, duration_sec: int, rows: list[sqlite3.Row]) -> CandleSeries:
    ts: list[int] = []
    o: list[float] = []
    h: list[float] = []
    l: list[float] = []
    c: list[float] = []
    v: list[float] = []
    for r in rows:
        ts.append(int(r["ts_ms"]))
        o.append(float(r["open"]))
        h.append(float(r["high"]))
        l.append(float(r["low"]))
        c.append(float(r["close"]))
        v.append(float(r["volume"]))
    return CandleSeries(pair=pair, tf=tf, duration_sec=duration_sec, ts=ts, o=o, h=h, l=l, c=c, v=v)


def find_last_index_leq(ts_list: list[int], t_ms: int) -> int:
    """
    Binary search: last index where ts <= t_ms, else -1.
    """
    lo, hi = 0, len(ts_list) - 1
    ans = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if ts_list[mid] <= t_ms:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


# =========================
# Strategy Parameters (GUI-controlled, persisted)
# =========================

@dataclass
class StrategyParamsV3:
    """
    Option A (Volatility / Mean-Reversion Engine) parameters.
    All fields are GUI-editable and persisted in SQLite.

    Notes:
    - This is long-only (spot-like), leverage is simulated (paper only).
    - Parameters must be tuned via backtest + walk-forward validation.
    """

    # Core mean-reversion signal (z-score of log price vs rolling mean)
    z_lookback: int = 120
    z_enter: float = 1.75          # enter when z <= -z_enter
    z_exit: float = 0.35           # exit when z >= -z_exit (reversion toward mean)

    # Bollinger confirmation / fallback
    bb_len: int = 120
    bb_mult: float = 2.0           # BB = SMA +/- mult*stdev
    bb_enter_pct: float = 0.02     # require price <= lowerBB*(1+bb_enter_pct) for extra confirmation (0.02=2%)
    bb_exit_mid: bool = True       # allow exit at mid-band

    # RSI guard
    rsi_len: int = 14
    rsi_oversold: float = 33.0     # require RSI <= oversold to enter

    # Volatility filters (ATR%)
    atr_len: int = 14
    atr_min_pct: float = 0.0012    # avoid dead chop
    atr_max_pct: float = 0.0600    # avoid extreme spikes
    atr_stop_mult: float = 2.2     # stop distance = atr_stop_mult * ATR(entry_tf)

    # Trend filter (avoid strong trends that crush MR)
    adx_len: int = 14
    adx_max_for_mr: float = 22.0   # only mean-revert if ADX <= this threshold on signal TF

    # Position/exit management
    confidence_min: float = 0.62
    max_bars_hold: int = 220       # time stop
    take_partial_at_r: float = 0.75  # partial at +X R
    partial_qty_frac: float = 0.35

    # Cost-awareness gate
    min_r_vs_cost_mult: float = 1.35  # require 1R >= cost * mult

    # Leverage heuristic (paper only)
    lev_base: float = 1.0
    lev_max: float = 2.0
    lev_scale_by_signal: float = 1.0  # add up to (lev_max-lev_base) scaled by confidence

    def sanitize(self) -> "StrategyParamsV3":
        p = dataclasses.replace(self)

        p.z_lookback = int(clamp(float(p.z_lookback), 20, 800))
        p.z_enter = clamp(float(p.z_enter), 0.6, 6.0)
        p.z_exit = clamp(float(p.z_exit), 0.05, 2.0)

        p.bb_len = int(clamp(float(p.bb_len), 20, 800))
        p.bb_mult = clamp(float(p.bb_mult), 0.8, 4.0)
        p.bb_enter_pct = clamp(float(p.bb_enter_pct), 0.0, 0.20)

        p.rsi_len = int(clamp(float(p.rsi_len), 2, 50))
        p.rsi_oversold = clamp(float(p.rsi_oversold), 5.0, 55.0)

        p.atr_len = int(clamp(float(p.atr_len), 2, 100))
        p.atr_min_pct = clamp(float(p.atr_min_pct), 0.0, 0.03)
        p.atr_max_pct = clamp(float(p.atr_max_pct), 0.01, 0.20)
        if p.atr_max_pct < p.atr_min_pct:
            p.atr_max_pct = max(p.atr_min_pct + 0.005, p.atr_max_pct)

        p.atr_stop_mult = clamp(float(p.atr_stop_mult), 0.8, 8.0)

        p.adx_len = int(clamp(float(p.adx_len), 2, 50))
        p.adx_max_for_mr = clamp(float(p.adx_max_for_mr), 8.0, 45.0)

        p.confidence_min = clamp(float(p.confidence_min), 0.20, 0.95)
        p.max_bars_hold = int(clamp(float(p.max_bars_hold), 10, 5000))
        p.take_partial_at_r = clamp(float(p.take_partial_at_r), 0.25, 3.0)
        p.partial_qty_frac = clamp(float(p.partial_qty_frac), 0.05, 0.90)

        p.min_r_vs_cost_mult = clamp(float(p.min_r_vs_cost_mult), 0.50, 5.0)

        p.lev_base = clamp(float(p.lev_base), 1.0, 6.0)
        p.lev_max = clamp(float(p.lev_max), p.lev_base, 10.0)
        p.lev_scale_by_signal = clamp(float(p.lev_scale_by_signal), 0.0, 2.0)

        return p

    def to_json(self) -> str:
        return safe_json_dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(s: str) -> "StrategyParamsV3":
        try:
            d = json.loads(s) if s else {}
            if not isinstance(d, dict):
                return StrategyParamsV3()
            # tolerate missing keys
            base = StrategyParamsV3()
            kw = {}
            for f in dataclasses.fields(base):
                if f.name in d:
                    kw[f.name] = d[f.name]
            p = StrategyParamsV3(**kw)  # type: ignore[arg-type]
            return p.sanitize()
        except Exception:
            return StrategyParamsV3()


# =========================
# Strategy v0.3 (Option A: Volatility / Mean-Reversion Engine)
# =========================

@dataclass
class RegimeState:
    """
    Regime computed on signal timeframe.
    For mean-reversion, we mainly use ADX as "trend strength" filter.
    """
    adx: float
    atr_pct: float


@dataclass
class EntrySignal:
    action: str      # BUY/HOLD
    confidence: float
    reason: str
    leverage: float
    stop_price: float
    r_value: float
    meta: dict[str, t.Any]


class StrategyV3:
    """
    Option A: Volatility / Mean-Reversion Engine

    - Uses z-score of log-price vs rolling mean on ENTRY TF as primary entry trigger.
    - Uses signal TF ADX to avoid strong trends (mean reversion tends to fail in strong trends).
    - Uses ATR% band to avoid dead chop and extreme spikes.
    - Uses RSI as oversold guard to reduce false entries.
    - Exits on reversion (z crosses toward mean), time stop, stop-loss, and optional BB mid-band.

    All parameters are GUI-controlled via StrategyParamsV3 and persisted in SQLite.
    """

    def __init__(self, cfg: AppConfig, log: AppLogger, get_params: t.Callable[[], StrategyParamsV3]):
        self.cfg = cfg
        self.log = log
        self.get_params = get_params

    def compute_regime(self, sig: CandleSeries) -> list[RegimeState]:
        p = self.get_params().sanitize()
        if len(sig.c) < max(60, p.adx_len * 3 + 10):
            return [RegimeState(adx=0.0, atr_pct=0.0) for _ in sig.c]

        a = atr(sig.h, sig.l, sig.c, p.atr_len)
        adxv = adx(sig.h, sig.l, sig.c, p.adx_len)

        out: list[RegimeState] = []
        for i in range(len(sig.c)):
            price = sig.c[i]
            atr_pct = (a[i] / max(1e-12, price)) if price > 0 else 0.0
            out.append(RegimeState(adx=float(adxv[i]), atr_pct=float(atr_pct)))
        return out

    def _zscore(self, series: list[float], lookback: int) -> float:
        if len(series) < max(lookback, 20):
            return 0.0
        # z-score of log price: (logP - mean(logP))/std(logP)
        window = series[-lookback:]
        logw = [math.log(max(1e-12, x)) for x in window]
        m = sum(logw) / lookback
        v = sum((x - m) ** 2 for x in logw) / lookback
        sd = math.sqrt(max(1e-12, v))
        z = (logw[-1] - m) / sd
        return float(z)

    def _bollinger(self, series: list[float], length: int, mult: float) -> tuple[float, float, float]:
        if len(series) < max(length, 20):
            x = series[-1] if series else 0.0
            return x, x, x
        w = series[-length:]
        m = sum(w) / length
        v = sum((x - m) ** 2 for x in w) / length
        sd = math.sqrt(max(1e-12, v))
        upper = m + mult * sd
        lower = m - mult * sd
        return float(lower), float(m), float(upper)

    def choose_leverage(self, confidence: float, equity_dd: float) -> float:
        """
        Conservative leverage for mean-reversion:
        - Base leverage is typically 1x.
        - Add up to (lev_max-lev_base) scaled by confidence (and dampened in drawdown).
        - Hard-capped by cfg.leverage_max and params.lev_max.
        """
        p = self.get_params().sanitize()

        if not self.cfg.leverage_enabled_paper:
            return 1.0

        # Confidence scaling (0..1)
        conf_scale = clamp((confidence - p.confidence_min) / max(1e-12, (0.95 - p.confidence_min)), 0.0, 1.0)
        lev = p.lev_base + (p.lev_max - p.lev_base) * (conf_scale * p.lev_scale_by_signal)

        # Drawdown dampener
        if equity_dd > 0.02:
            lev *= clamp(1.0 - equity_dd * 7.0, 0.35, 1.0)

        lev = clamp(lev, self.cfg.leverage_min, min(self.cfg.leverage_max, p.lev_max))
        return float(lev)

    def entry_signal(
        self,
        pair: str,
        entry: CandleSeries,
        idx_entry: int,
        sig: CandleSeries,
        idx_sig: int,
        regime: RegimeState,
        equity_dd: float,
    ) -> EntrySignal:
        p = self.get_params().sanitize()

        # Need enough entry history
        if idx_entry < max(p.z_lookback, p.bb_len, p.rsi_len, p.atr_len) + 5:
            return EntrySignal("HOLD", 0.0, "Insufficient entry history", 1.0, 0.0, 0.0, {"need": "more_candles"})

        price = entry.c[idx_entry]
        if price <= 0:
            return EntrySignal("HOLD", 0.0, "Bad price", 1.0, 0.0, 0.0, {})

        # Regime filter: avoid strong trends (ADX high)
        if regime.adx > p.adx_max_for_mr:
            return EntrySignal("HOLD", 0.0, "ADX too high (trend risk)", 1.0, 0.0, 0.0, {"adx": regime.adx})

        # Volatility sanity (using entry TF ATR%)
        close = entry.c[:idx_entry + 1]
        high = entry.h[:idx_entry + 1]
        low = entry.l[:idx_entry + 1]

        a = atr(high, low, close, p.atr_len)
        atr_now = a[-1] if a else 0.0
        atr_pct = atr_now / max(1e-12, price)

        if atr_pct < p.atr_min_pct:
            return EntrySignal("HOLD", 0.0, "ATR% too low (chop risk)", 1.0, 0.0, 0.0, {"atr_pct": atr_pct})
        if atr_pct > p.atr_max_pct:
            return EntrySignal("HOLD", 0.0, "ATR% too high (spike risk)", 1.0, 0.0, 0.0, {"atr_pct": atr_pct})

        # RSI guard
        r = rsi(close, p.rsi_len)
        rsi_now = r[-1] if r else 50.0
        if rsi_now > p.rsi_oversold:
            return EntrySignal("HOLD", 0.0, "RSI not oversold", 1.0, 0.0, 0.0, {"rsi": rsi_now})

        # Z-score (mean-reversion trigger)
        z = self._zscore(close, p.z_lookback)
        if z > -p.z_enter:
            return EntrySignal("HOLD", 0.0, "Z-score not extreme enough", 1.0, 0.0, 0.0, {"z": z})

        # Bollinger confirmation (optional but helpful on MR)
        lower, mid, upper = self._bollinger(close, p.bb_len, p.bb_mult)
        if lower > 0:
            # require price to be near/below lower band
            if price > lower * (1.0 + p.bb_enter_pct):
                return EntrySignal(
                    "HOLD", 0.0,
                    "Not near lower band",
                    1.0, 0.0, 0.0,
                    {"price": price, "bb_lower": lower, "bb_enter_pct": p.bb_enter_pct}
                )

        # Stop distance using ATR
        stop_dist = p.atr_stop_mult * atr_now
        stop_price = price - stop_dist
        if stop_price <= 0:
            return EntrySignal("HOLD", 0.0, "Invalid stop price", 1.0, 0.0, 0.0, {"atr": atr_now})

        r_value = price - stop_price  # 1R in price terms

        # Fee-aware minimum R gate
        bps_total = (self.cfg.backtest_fee_bps + self.cfg.backtest_slippage_bps) / 10000.0
        est_round_trip_cost = price * bps_total * 2.2  # padded
        if r_value < est_round_trip_cost * p.min_r_vs_cost_mult:
            return EntrySignal(
                "HOLD", 0.0,
                "R too small vs estimated costs",
                1.0, 0.0, 0.0,
                {"r_value": r_value, "est_cost": est_round_trip_cost, "price": price}
            )

        # Confidence model:
        # - deeper z => higher confidence
        # - lower RSI => higher confidence
        # - lower ADX => higher confidence
        z_depth = clamp((-z - p.z_enter) / max(1e-12, (p.z_enter * 1.2)), 0.0, 1.0)
        rsi_score = clamp((p.rsi_oversold - rsi_now) / max(1e-12, p.rsi_oversold), 0.0, 1.0)
        adx_score = clamp((p.adx_max_for_mr - regime.adx) / max(1e-12, p.adx_max_for_mr), 0.0, 1.0)

        conf = clamp(0.45 + 0.30 * z_depth + 0.20 * rsi_score + 0.15 * adx_score, 0.0, 0.95)

        lev = self.choose_leverage(conf, equity_dd)

        return EntrySignal(
            action="BUY",
            confidence=float(conf),
            reason="Mean-reversion (z-score + RSI + low ADX)",
            leverage=float(lev),
            stop_price=float(stop_price),
            r_value=float(r_value),
            meta={
                "pair": pair,
                "price": price,
                "z": z,
                "rsi": rsi_now,
                "atr": atr_now,
                "atr_pct": atr_pct,
                "bb_lower": lower,
                "bb_mid": mid,
                "bb_upper": upper,
                "regime_adx": regime.adx,
                "regime_atr_pct": regime.atr_pct,
                "equity_dd": equity_dd,
                "params": dataclasses.asdict(p),
            }
        )


# =========================
# Backtesting (v0.3 - StrategyV3)
# =========================

@dataclass
class BacktestResult:
    pair: str
    entry_tf: str
    signal_tf: str
    start_ms: int
    end_ms: int

    trades: int
    win_rate: float
    total_return: float
    max_drawdown: float

    profit_factor: float
    avg_trade_return: float

    notes: str
    equity_curve: list[tuple[int, float]]
    extra: dict[str, t.Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "pair": self.pair,
            "entry_tf": self.entry_tf,
            "signal_tf": self.signal_tf,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_utc": human_ts(self.start_ms) if self.start_ms else "",
            "end_utc": human_ts(self.end_ms) if self.end_ms else "",
            "trades": self.trades,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "notes": self.notes,
            "extra": self.extra,
            "equity_curve": [{"ts_ms": ts, "ts_utc": human_ts(ts), "equity": eq} for ts, eq in self.equity_curve],
        }

    def summary_line(self) -> str:
        return (
            f"{self.pair} entry={self.entry_tf} signal={self.signal_tf} trades={self.trades} "
            f"win={self.win_rate*100:.1f}% ret={self.total_return*100:.2f}% "
            f"dd={self.max_drawdown*100:.2f}% notes={self.notes}"
        )


@dataclass
class SimPosition:
    pair: str
    qty: float
    entry_price: float
    stop_price: float
    r_value: float
    leverage: float
    margin_used: float
    entry_ts_ms: int

    took_partial: bool = False
    qty_remaining: float = 0.0
    bars_held: int = 0


class BacktesterV3:
    """
    Long-only with paper leverage simulation.
    - Position sized by risk_per_trade: risk amount equals max loss at stop.
    - Leverage affects margin used; liquidation if loss exceeds margin_used*(1-maint_margin_ratio).
    - Partial exit at +X R (params.take_partial_at_r); then managed exit on reversion or time stop.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger, strategy: StrategyV3, get_params: t.Callable[[], StrategyParamsV3]):
        self.cfg = cfg
        self.log = log
        self.strategy = strategy
        self.get_params = get_params

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = self.cfg.backtest_slippage_bps / 10000.0
        if side.upper() == "BUY":
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    def _fee(self, notional: float) -> float:
        return notional * (self.cfg.backtest_fee_bps / 10000.0)

    def _liquidation_price_long(self, pos: SimPosition) -> float:
        """
        Simplified liquidation:
        - Liquidation occurs when equity in position <= maintenance margin.
        - loss_at_liq = margin_used * (1 - maint_margin_ratio)
        """
        loss_cap = pos.margin_used * (1.0 - self.cfg.maint_margin_ratio)
        if pos.qty <= 0 or pos.entry_price <= 0:
            return 0.0
        return max(0.0, pos.entry_price - (loss_cap / pos.qty))

    def run(
        self,
        pair: str,
        entry: CandleSeries,
        signal: CandleSeries,
        regime_series: list[RegimeState],
        initial_equity: float = 100.0
    ) -> BacktestResult:
        p = self.get_params().sanitize()

        if len(entry.c) < 400 or len(signal.c) < 260 or len(regime_series) != len(signal.c):
            return BacktestResult(
                pair=pair, entry_tf=entry.tf, signal_tf=signal.tf,
                start_ms=0, end_ms=0,
                trades=0, win_rate=0.0, total_return=0.0, max_drawdown=0.0,
                profit_factor=0.0, avg_trade_return=0.0,
                notes="Insufficient candles",
                equity_curve=[]
            )

        equity = float(initial_equity)
        peak = equity
        max_dd = 0.0

        pos: SimPosition | None = None

        trades = 0
        wins = 0
        trade_returns: list[float] = []
        gross_profit = 0.0
        gross_loss = 0.0

        eq_curve: list[tuple[int, float]] = []

        def equity_drawdown() -> float:
            return (peak - equity) / max(1e-12, peak)

        # Pre-compute z / bb mid to allow exit logic without recomputing too heavy
        # (still O(n*lookback) if naive; we keep it lightweight for v0.3.1)
        for i in range(0, len(entry.ts)):
            ts = entry.ts[i]
            idx_sig = find_last_index_leq(signal.ts, ts)
            if idx_sig < 0:
                continue
            regime = regime_series[idx_sig]

            price = entry.c[i]
            hi = entry.h[i]
            lo = entry.l[i]
            if price <= 0:
                continue

            # Update open position
            if pos is not None:
                pos.bars_held += 1

                # Liquidation check (intrabar)
                liq_px = self._liquidation_price_long(pos)
                if liq_px > 0 and lo <= liq_px:
                    exit_px = self._apply_slippage(liq_px, "SELL")
                    notional = pos.qty_remaining * exit_px
                    fee = self._fee(notional)
                    pnl = (exit_px - pos.entry_price) * pos.qty_remaining - fee
                    equity += pnl

                    trades += 1
                    if pnl > 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    trade_returns.append(pnl / max(1e-12, equity))

                    pos = None
                    eq_curve.append((ts, equity))
                    peak = max(peak, equity)
                    max_dd = max(max_dd, equity_drawdown())
                    break

                # Hard stop
                if lo <= pos.stop_price:
                    exit_px = self._apply_slippage(pos.stop_price, "SELL")
                    notional = pos.qty_remaining * exit_px
                    fee = self._fee(notional)
                    pnl = (exit_px - pos.entry_price) * pos.qty_remaining - fee
                    equity += pnl

                    trades += 1
                    if pnl > 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    trade_returns.append(pnl / max(1e-12, equity))
                    pos = None
                else:
                    # Partial at +X R
                    tp_r = pos.entry_price + p.take_partial_at_r * pos.r_value
                    if (not pos.took_partial) and hi >= tp_r and pos.qty_remaining > 0:
                        take_qty = pos.qty_remaining * p.partial_qty_frac
                        exit_px = self._apply_slippage(tp_r, "SELL")
                        notional = take_qty * exit_px
                        fee = self._fee(notional)
                        pnl = (exit_px - pos.entry_price) * take_qty - fee
                        equity += pnl

                        pos.qty_remaining = max(0.0, pos.qty_remaining - take_qty)
                        pos.took_partial = True

                        # Reduce risk by tightening stop to entry (break-even) after partial
                        pos.stop_price = max(pos.stop_price, pos.entry_price)

                    # Mean-reversion exit logic: z crosses toward mean OR BB mid-band (optional)
                    # Compute on the fly using current history slice
                    close_slice = entry.c[:i + 1]
                    z = self.strategy._zscore(close_slice, p.z_lookback)
                    bb_lower, bb_mid, bb_upper = self.strategy._bollinger(close_slice, p.bb_len, p.bb_mult)

                    exit_reason = ""
                    should_exit = False

                    # Z-based reversion exit
                    if z >= -p.z_exit:
                        should_exit = True
                        exit_reason = "Z reversion"

                    # Optional BB mid exit
                    if (not should_exit) and p.bb_exit_mid and bb_mid > 0 and price >= bb_mid:
                        should_exit = True
                        exit_reason = "BB mid exit"

                    # Time stop
                    if (not should_exit) and pos.bars_held >= p.max_bars_hold:
                        should_exit = True
                        exit_reason = "Time stop"

                    # If regime becomes too trending mid-trade (ADX rises), exit to reduce MR failure cases
                    if (not should_exit) and regime.adx > (p.adx_max_for_mr + 6.0):
                        should_exit = True
                        exit_reason = "ADX trend risk"

                    if should_exit and pos is not None and pos.qty_remaining > 0:
                        exit_px = self._apply_slippage(price, "SELL")
                        notional = pos.qty_remaining * exit_px
                        fee = self._fee(notional)
                        pnl = (exit_px - pos.entry_price) * pos.qty_remaining - fee
                        equity += pnl

                        trades += 1
                        if pnl > 0:
                            wins += 1
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                        trade_returns.append(pnl / max(1e-12, equity))
                        pos = None

            # Entry logic if flat
            if pos is None:
                if equity_drawdown() >= 0.25:
                    eq_curve.append((ts, equity))
                    break

                idx_sig = find_last_index_leq(signal.ts, ts)
                if idx_sig < 0:
                    eq_curve.append((ts, equity))
                    continue
                regime = regime_series[idx_sig]

                es = self.strategy.entry_signal(
                    pair=pair,
                    entry=entry,
                    idx_entry=i,
                    sig=signal,
                    idx_sig=idx_sig,
                    regime=regime,
                    equity_dd=equity_drawdown(),
                )

                if es.action == "BUY" and es.confidence >= p.confidence_min:
                    risk_amount = equity * clamp(self.cfg.risk_per_trade, 0.0005, 0.05)
                    stop_dist = max(1e-9, (entry.c[i] - es.stop_price))
                    qty = risk_amount / stop_dist

                    lev = es.leverage if self.cfg.leverage_enabled_paper else 1.0
                    lev = clamp(lev, self.cfg.leverage_min, self.cfg.leverage_max)

                    entry_px = self._apply_slippage(entry.c[i], "BUY")
                    notional = qty * entry_px
                    if notional <= 0:
                        eq_curve.append((ts, equity))
                        continue

                    margin_required = notional / lev
                    margin_cap = equity * self.cfg.margin_alloc_cap

                    if margin_required > margin_cap:
                        scale = margin_cap / max(1e-12, margin_required)
                        qty *= scale
                        notional = qty * entry_px
                        margin_required = notional / lev

                    if notional < max(5.0, equity * 0.02):
                        eq_curve.append((ts, equity))
                        continue

                    entry_fee = self._fee(notional)
                    equity -= entry_fee

                    pos = SimPosition(
                        pair=pair,
                        qty=qty,
                        entry_price=entry_px,
                        stop_price=es.stop_price,
                        r_value=es.r_value,
                        leverage=lev,
                        margin_used=margin_required,
                        entry_ts_ms=ts,
                        took_partial=False,
                        qty_remaining=qty,
                        bars_held=0,
                    )

            eq_curve.append((ts, equity))
            peak = max(peak, equity)
            max_dd = max(max_dd, equity_drawdown())

        total_return = (equity - initial_equity) / max(1e-12, initial_equity)
        win_rate = wins / trades if trades > 0 else 0.0
        profit_factor = (gross_profit / max(1e-12, gross_loss)) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        avg_tr = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0

        start_ms = entry.ts[0] if entry.ts else 0
        end_ms = entry.ts[-1] if entry.ts else 0

        notes = "OK"
        if trades < 10:
            notes = "Low trades"
        if total_return < 0:
            notes = "Negative"

        return BacktestResult(
            pair=pair,
            entry_tf=entry.tf,
            signal_tf=signal.tf,
            start_ms=start_ms,
            end_ms=end_ms,
            trades=trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            avg_trade_return=avg_tr,
            notes=notes,
            equity_curve=eq_curve,
            extra={
                "paper_leverage_enabled": self.cfg.leverage_enabled_paper,
                "leverage_cap": self.cfg.leverage_max,
                "margin_alloc_cap": self.cfg.margin_alloc_cap,
                "maint_margin_ratio": self.cfg.maint_margin_ratio,
                "strategy_params": dataclasses.asdict(self.get_params().sanitize()),
            }
        )

    def walk_forward(
        self,
        pair: str,
        entry: CandleSeries,
        signal: CandleSeries,
        regime_series: list[RegimeState],
        initial_equity: float = 100.0
    ) -> BacktestResult:
        n = len(entry.ts)
        if n < 800:
            return self.run(pair, entry, signal, regime_series, initial_equity)

        segments = 5
        seg_size = n // segments
        eq = initial_equity

        all_curve: list[tuple[int, float]] = []
        total_trades = 0
        weighted_wins = 0.0
        worst_dd = 0.0
        avg_trs: list[float] = []
        g_profit = 0.0
        g_loss = 0.0

        for s in range(segments):
            a = s * seg_size
            b = (s + 1) * seg_size if s < segments - 1 else n

            e_seg = CandleSeries(
                pair=entry.pair, tf=entry.tf, duration_sec=entry.duration_sec,
                ts=entry.ts[a:b], o=entry.o[a:b], h=entry.h[a:b], l=entry.l[a:b], c=entry.c[a:b], v=entry.v[a:b],
            )

            res = self.run(pair, e_seg, signal, regime_series, eq)
            eq = eq * (1.0 + res.total_return)

            total_trades += res.trades
            weighted_wins += res.win_rate * res.trades
            worst_dd = max(worst_dd, res.max_drawdown)
            avg_trs.append(res.avg_trade_return)

            all_curve.extend(res.equity_curve)

            if res.total_return > 0:
                g_profit += res.total_return
            else:
                g_loss += abs(res.total_return)

        win_rate = (weighted_wins / total_trades) if total_trades > 0 else 0.0
        total_return = (eq - initial_equity) / max(1e-12, initial_equity)
        avg_tr = sum(avg_trs) / len(avg_trs) if avg_trs else 0.0
        profit_factor = (g_profit / max(1e-12, g_loss)) if g_loss > 0 else (g_profit if g_profit > 0 else 0.0)

        return BacktestResult(
            pair=pair,
            entry_tf=entry.tf,
            signal_tf=signal.tf,
            start_ms=entry.ts[0] if entry.ts else 0,
            end_ms=entry.ts[-1] if entry.ts else 0,
            trades=total_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=worst_dd,
            profit_factor=profit_factor,
            avg_trade_return=avg_tr,
            notes="Walk-forward OK",
            equity_curve=all_curve,
            extra={
                "paper_leverage_enabled": self.cfg.leverage_enabled_paper,
                "leverage_cap": self.cfg.leverage_max,
                "segments": segments,
                "strategy_params": dataclasses.asdict(self.get_params().sanitize()),
            }
        )


# =========================
# Reporting
# =========================

class ReportWriter:
    def __init__(self, data_dir: Path, log: AppLogger):
        self.data_dir = data_dir
        self.log = log
        self.reports_dir = data_dir / REPORTS_DIRNAME
        ensure_dir(self.reports_dir)

    def write_backtest_reports(self, results: list[BacktestResult], context: dict[str, t.Any]) -> dict[str, Path]:
        stamp = now_utc().strftime("%Y%m%d_%H%M%S")
        csv_path = self.reports_dir / f"backtest_{stamp}.csv"
        json_path = self.reports_dir / f"backtest_{stamp}.json"
        txt_path = self.reports_dir / f"backtest_{stamp}_summary.txt"

        # CSV summary
        try:
            header = "pair,entry_tf,signal_tf,start_utc,end_utc,trades,win_rate,total_return,max_drawdown,profit_factor,avg_trade_return,notes\n"
            lines = [header]
            for r in results:
                lines.append(
                    f"{r.pair},{r.entry_tf},{r.signal_tf},"
                    f"\"{human_ts(r.start_ms)}\",\"{human_ts(r.end_ms)}\","
                    f"{r.trades},{r.win_rate:.6f},{r.total_return:.6f},{r.max_drawdown:.6f},"
                    f"{r.profit_factor:.6f},{r.avg_trade_return:.6f},\"{r.notes}\"\n"
                )
            csv_path.write_text("".join(lines), encoding="utf-8")
        except Exception as e:
            self.log.exception("Failed writing CSV report", e)

        # JSON full
        try:
            payload = {
                "generated_utc": now_utc().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "context": context,
                "results": [r.to_dict() for r in results],
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.exception("Failed writing JSON report", e)

        # TXT compact
        try:
            top_lines = []
            top_lines.append(f"{APP_NAME} {APP_VERSION} BACKTEST SUMMARY")
            top_lines.append(f"Generated: {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            for k, v in context.items():
                if k == "strategy_params":
                    continue
                top_lines.append(f"{k}: {v}")
            top_lines.append("")
            if "strategy_params" in context:
                top_lines.append("Strategy Params (v3):")
                try:
                    params_pretty = json.dumps(context["strategy_params"], ensure_ascii=False, indent=2)
                    for line in params_pretty.splitlines():
                        top_lines.append("  " + line)
                except Exception:
                    top_lines.append("  (unavailable)")
                top_lines.append("")
            top_lines.append("Top Results:")
            for r in results[:10]:
                top_lines.append(" - " + r.summary_line())
            txt_path.write_text("\n".join(top_lines) + "\n", encoding="utf-8")
        except Exception as e:
            self.log.exception("Failed writing TXT report", e)

        out = {"csv": csv_path, "json": json_path, "txt": txt_path}
        self.log.info(f"Backtest reports saved: csv={csv_path} json={json_path} txt={txt_path}")
        return out


# =========================
# Paper Portfolio (for GUI engine loop)
# =========================

@dataclass
class PaperPosition:
    pair: str
    qty: float
    entry_price: float
    entry_ts_ms: int


@dataclass
class PaperPortfolio:
    equity: float = 100.0
    cash: float = 100.0
    positions: dict[str, PaperPosition] = dataclasses.field(default_factory=dict)
    day_start_equity: float = 100.0
    day_utc: str = ""


# =========================
# Engine
# =========================

class TradingEngine:
    def __init__(self, cfg: AppConfig, log: AppLogger, storage: Storage, notifier: Notifier):
        self.cfg = cfg
        self.log = log
        self.storage = storage
        self.notifier = notifier

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._client_ro = LunoClient(cfg, log, use_live=False)
        self._client_live = LunoClient(cfg, log, use_live=True)

        self._pairs_cache: list[str] = []
        self._last_health: str = "INIT"

        self.paper = PaperPortfolio(equity=100.0, cash=100.0)

        # Strategy params are GUI-controlled and persisted
        self._params_lock = threading.Lock()
        self._strategy_params: StrategyParamsV3 = self._load_or_init_strategy_params()

        # Strategy and backtester use a params provider (so GUI apply updates instantly)
        self.strategy_v3 = StrategyV3(cfg, log, get_params=self.get_strategy_params)
        self.backtester_v3 = BacktesterV3(cfg, log, self.strategy_v3, get_params=self.get_strategy_params)

    # ---- Strategy params persistence ----

    def _load_or_init_strategy_params(self) -> StrategyParamsV3:
        try:
            raw = self.storage.kv_get(STRATEGY_PARAMS_KEY, "")
            if raw.strip():
                p = StrategyParamsV3.from_json(raw).sanitize()
                self.log.info("Loaded strategy params from SQLite.")
                return p
        except Exception as e:
            self.log.warn(f"Failed loading strategy params: {e}")

        p = StrategyParamsV3().sanitize()
        try:
            self.storage.kv_set(STRATEGY_PARAMS_KEY, p.to_json())
            self.log.info("Initialized default strategy params into SQLite.")
        except Exception:
            pass
        return p

    def get_strategy_params(self) -> StrategyParamsV3:
        with self._params_lock:
            return dataclasses.replace(self._strategy_params).sanitize()

    def set_strategy_params(self, new_params: StrategyParamsV3, persist: bool = True) -> StrategyParamsV3:
        p = new_params.sanitize()
        with self._params_lock:
            self._strategy_params = p
        if persist:
            try:
                self.storage.kv_set(STRATEGY_PARAMS_KEY, p.to_json())
            except Exception as e:
                self.log.warn(f"Persist strategy params failed: {e}")
        self.log.info("Strategy params updated (GUI).")
        return p

    # ---- Live safety gates ----

    def live_unlocked(self) -> bool:
        unlock_file = self.cfg.data_dir / LIVE_MODE_UNLOCK_FILE
        gate = self.storage.get_gate("LIVE_GATE", "0")
        try:
            return unlock_file.exists() and gate == "1"
        except Exception:
            return False

    def allow_live(self) -> bool:
        if not self.cfg.is_live():
            return False
        if not self.live_unlocked():
            return False
        if is_kill_switch_on(self.cfg.data_dir):
            return False
        return True

    # ---- Pairs / candles ----

    def get_scan_pairs(self) -> list[str]:
        raw = (self.cfg.scan_pairs_csv or "").strip()
        if raw:
            pairs = [p.strip().upper() for p in raw.split(",") if p.strip()]
            return sorted(list(dict.fromkeys(pairs)))

        if self._pairs_cache:
            return self._pairs_cache[:]

        try:
            data = self._client_ro.tickers()
            tickers = data.get("tickers") or []
            pairs: list[str] = []
            if isinstance(tickers, list):
                for tkr in tickers:
                    p = str(tkr.get("pair", "")).upper()
                    if p:
                        pairs.append(p)
            self._pairs_cache = sorted(list(dict.fromkeys(pairs)))
            return self._pairs_cache[:]
        except Exception as e:
            self.log.warn(f"Failed to auto-discover pairs: {e}")
            return [self.cfg.default_view_pair.upper()]

    def ingest_candles(self, pair: str, duration_sec: int, start_ms: int, end_ms: int) -> int:
        total = 0
        since = start_ms
        for _ in range(1000):
            if since >= end_ms:
                break
            data = self._client_ro.candles(pair=pair, since_ms=since, duration_sec=duration_sec)
            candles = data.get("candles") or []
            if not isinstance(candles, list) or not candles:
                break
            n = self.storage.upsert_candles(pair, duration_sec, candles)
            total += n
            last_ts = safe_int(candles[-1].get("timestamp"), since)
            next_since = max(last_ts + duration_sec * 1000, since + duration_sec * 1000)
            if next_since <= since:
                break
            since = next_since
            time.sleep(0.12)
            if n < 4:
                break
        return total

    def load_series(self, pair: str, tf: str, start_ms: int, end_ms: int) -> CandleSeries:
        duration = timeframe_to_seconds(tf)
        rows = self.storage.fetch_candles(pair, duration, start_ms, end_ms)
        return build_series_from_rows(pair, tf, duration, rows)

    # ---- Backtest ----

    def run_backtest(self, pairs: list[str], entry_tf: str, days: int, initial_equity: float, walk_forward: bool) -> list[BacktestResult]:
        entry_duration = timeframe_to_seconds(entry_tf)
        signal_tf = self.cfg.signal_timeframe
        signal_duration = timeframe_to_seconds(signal_tf)

        end_ms = utc_ms()
        start_ms = end_ms - int(days * 86400 * 1000)

        results: list[BacktestResult] = []
        for pair in pairs:
            pair = pair.upper().strip()
            if not pair:
                continue

            try:
                self.log.info(f"Backtest ingest candles: {pair} entry={entry_tf} signal={signal_tf} days={days}")
                n1 = self.ingest_candles(pair, entry_duration, start_ms, end_ms)
                n2 = self.ingest_candles(pair, signal_duration, start_ms, end_ms)
                self.log.info(f"Ingested candles: {pair} entry={n1} signal={n2}")

                entry_series = self.load_series(pair, entry_tf, start_ms, end_ms)
                signal_series = self.load_series(pair, signal_tf, start_ms, end_ms)

                regimes = self.strategy_v3.compute_regime(signal_series)

                if walk_forward:
                    res = self.backtester_v3.walk_forward(pair, entry_series, signal_series, regimes, initial_equity=initial_equity)
                else:
                    res = self.backtester_v3.run(pair, entry_series, signal_series, regimes, initial_equity=initial_equity)

                results.append(res)

            except Exception as e:
                self.log.exception(f"Backtest failed for {pair}", e)

        results.sort(key=lambda r: r.total_return, reverse=True)
        return results

    # ---- Engine loop (lightweight) ----

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, name="EngineLoop", daemon=True)
            self._thread.start()
        self.log.info("Engine started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        self.log.info("Engine stopping...")

    def running(self) -> bool:
        with self._lock:
            return self._running

    def _paper_reset_day_if_needed(self) -> None:
        day = now_utc().strftime("%Y-%m-%d")
        if self.paper.day_utc != day:
            self.paper.day_utc = day
            self.paper.day_start_equity = self.paper.equity
            self.log.info(f"New UTC day: {day}. day_start_equity={self.paper.day_start_equity:.2f}")

    def _paper_daily_pnl_pct(self) -> float:
        return (self.paper.equity - self.paper.day_start_equity) / max(1e-12, self.paper.day_start_equity)

    def _run_loop(self) -> None:
        self._paper_reset_day_if_needed()

        while self.running():
            try:
                if is_kill_switch_on(self.cfg.data_dir):
                    self._last_health = "KILL_SWITCH"
                    time.sleep(1.0)
                    continue

                pair = self.cfg.default_view_pair.upper().strip()
                tkr = self._client_ro.ticker(pair)
                last_trade = safe_float(tkr.get("last_trade"), 0.0)
                self._last_health = f"OK last={last_trade}"

                if self.cfg.is_paper():
                    self._paper_reset_day_if_needed()
                    # v0.3.1: keep paper engine conservative (no auto entries/exits yet).
                    # Backtester remains the primary optimization tool.
                    pass
                else:
                    self._last_health = "LIVE_LOCKED"

            except Exception as e:
                self.log.exception("Engine loop exception", e)
                self._last_health = "ERROR"

            time.sleep(self.cfg.poll_interval_seconds)

    def health(self) -> str:
        return self._last_health


# =========================
# GUI
# =========================

class Dashboard(tk.Tk):
    def __init__(self, repo_root: Path):
        super().__init__()
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1280x820")
        self.minsize(1100, 720)

        self.repo_root = repo_root

        # initial logger for config bootstrap
        self.log = AppLogger(data_dir=(repo_root / "data"))
        self.cfg = load_config(repo_root, self.log)
        ensure_dir(self.cfg.data_dir)
        self.log = AppLogger(data_dir=self.cfg.data_dir)
        self.cfg = load_config(repo_root, self.log)

        self.storage = Storage(self.cfg.data_dir / self.cfg.db_filename, self.log)
        self.notifier = Notifier(self.cfg, self.log)
        self.engine = TradingEngine(self.cfg, self.log, self.storage, self.notifier)
        self.reports = ReportWriter(self.cfg.data_dir, self.log)

        self._ui_queue: "queue.Queue[t.Callable[[], None]]" = queue.Queue()
        self._backtest_thread: threading.Thread | None = None

        self._last_backtest_results: list[BacktestResult] = []
        self._last_backtest_context: dict[str, t.Any] = {}
        self._last_backtest_report_paths: dict[str, Path] = {}

        # GUI-bound params
        self._param_vars: dict[str, tk.StringVar] = {}

        self._build_style()
        self._build_widgets()

        self.after(250, self._poll_logs)
        self.after(500, self._refresh_status)
        self.after(1000, self._drain_ui_queue)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_startup_notice()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

    def _build_widgets(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.var_mode = tk.StringVar(value=self.cfg.app_mode.upper())
        self.var_health = tk.StringVar(value="INIT")
        self.var_equity = tk.StringVar(value=fmt_money(0.0))
        self.var_cash = tk.StringVar(value=fmt_money(0.0))
        self.var_daily_pnl = tk.StringVar(value="0.00%")

        ttk.Label(top, text="Mode:").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.var_mode, width=7).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text="Health:").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.var_health, width=26).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text="Equity:").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.var_equity, width=16).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text="Cash:").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.var_cash, width=16).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text="Daily PnL:").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.var_daily_pnl, width=10).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self.tbl_positions = self._make_table(
            left,
            title="Open Positions (Paper)",
            columns=[("pair", 90), ("qty", 120), ("entry", 120), ("age", 140)]
        )
        self.tbl_positions.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.tbl_rank = self._make_table(
            left,
            title="Scanner / Latest Decisions",
            columns=[("time", 160), ("pair", 90), ("action", 80), ("conf", 80), ("reason", 360)]
        )
        self.tbl_rank.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(right, text="Controls", padding=10)
        controls.pack(fill=tk.X, pady=(0, 8))

        btn_row = ttk.Frame(controls)
        btn_row.pack(fill=tk.X)

        self.btn_start = ttk.Button(btn_row, text="Start Engine", command=self._start_engine)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop = ttk.Button(btn_row, text="Stop Engine", command=self._stop_engine)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(btn_row, text="Export Logs", command=self._export_logs).pack(side=tk.LEFT)

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # ---- Strategy Parameters (GUI-controlled) ----
        params_box = ttk.LabelFrame(right, text="Strategy Parameters (Option A - Mean Reversion)", padding=10)
        params_box.pack(fill=tk.X, pady=(0, 8))

        self._build_params_panel(params_box)

        # Grid-compatible separator
        sep = ttk.Separator(params_box, orient=tk.HORIZONTAL)
        sep.grid(row=12, column=0, columnspan=2, sticky="ew", pady=10)

        btnp = ttk.Frame(params_box)
        btnp.grid(row=13, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Button(
            btnp,
            text="Apply & Save Params",
            command=self._apply_save_params
        ).grid(row=0, column=0, padx=(0, 8), sticky="w")

        ttk.Button(
            btnp,
            text="Reload Saved Params",
            command=self._reload_params_from_engine
        ).grid(row=0, column=1, padx=(0, 8), sticky="w")

        ttk.Button(
            btnp,
            text="Reset to Defaults",
            command=self._reset_params_defaults
        ).grid(row=0, column=2, sticky="w")

        # ---- Backtest ----
        bt = ttk.LabelFrame(right, text="Backtest", padding=10)
        bt.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(bt)
        row1.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row1, text="Pairs (comma):").pack(side=tk.LEFT)
        self.var_bt_pairs = tk.StringVar(value=self.cfg.default_view_pair)
        ttk.Entry(row1, textvariable=self.var_bt_pairs).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        row2 = ttk.Frame(bt)
        row2.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row2, text="Entry TF:").pack(side=tk.LEFT)
        self.var_bt_tf = tk.StringVar(value=self.cfg.backtest_timeframe)
        ttk.Entry(row2, textvariable=self.var_bt_tf, width=8).pack(side=tk.LEFT, padx=8)

        ttk.Label(row2, text="Days:").pack(side=tk.LEFT)
        self.var_bt_days = tk.StringVar(value="180")
        ttk.Entry(row2, textvariable=self.var_bt_days, width=6).pack(side=tk.LEFT, padx=8)

        ttk.Label(row2, text="Initial equity:").pack(side=tk.LEFT)
        self.var_bt_equity = tk.StringVar(value="100")
        ttk.Entry(row2, textvariable=self.var_bt_equity, width=10).pack(side=tk.LEFT, padx=8)

        row3 = ttk.Frame(bt)
        row3.pack(fill=tk.X, pady=(0, 6))

        self.var_bt_walk = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="Walk-forward", variable=self.var_bt_walk).pack(side=tk.LEFT)

        ttk.Button(row3, text="Run Backtest", command=self._run_backtest).pack(side=tk.LEFT, padx=10)
        ttk.Button(row3, text="Copy Summary", command=self._copy_backtest_summary).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row3, text="Export Backtest Report", command=self._export_backtest_report).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row3, text="Open Reports Folder", command=self._open_reports_folder).pack(side=tk.LEFT)

        self.tbl_bt = self._make_table(
            right,
            title="Backtest Results (sorted by return)",
            columns=[("pair", 90), ("entry", 70), ("signal", 70), ("trades", 70), ("win", 70), ("ret", 90), ("dd", 80), ("notes", 170)]
        )
        self.tbl_bt.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        logs = ttk.LabelFrame(right, text="Logs", padding=8)
        logs.pack(fill=tk.BOTH, expand=True)

        self.txt_logs = tk.Text(logs, height=12, wrap=tk.NONE)
        self.txt_logs.pack(fill=tk.BOTH, expand=True)

        # Load GUI param fields from engine-saved params at startup
        self._reload_params_from_engine()

    def _make_table(self, parent: tk.Widget, title: str, columns: list[tuple[str, int]]) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        cols = [c[0] for c in columns]
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=vsb.set)

        for name, width in columns:
            tree.heading(name, text=name.upper())
            tree.column(name, width=width, anchor=tk.W)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        frame.tree = tree  # type: ignore[attr-defined]
        return frame

    def _show_startup_notice(self) -> None:
        msg = (
            "PAPER mode is recommended while optimizing backtests.\n\n"
            "This software includes risk controls and paper leverage simulation.\n"
            "Use read-only keys for data access and testing.\n"
        )
        messagebox.showinfo(f"{APP_NAME}", msg)

    # ---- Strategy params panel ----

    def _build_params_panel(self, parent: ttk.LabelFrame) -> None:
        """
        Creates GUI fields for StrategyParamsV3.
        Fields are string vars, then sanitized and converted on Apply.
        """
        # We pack in a compact grid (label + entry), 2 columns of fields.
        p = self.engine.get_strategy_params()

        def add_field(row: int, col: int, key: str, label: str, default_val: t.Any) -> None:
            if key not in self._param_vars:
                self._param_vars[key] = tk.StringVar(value=str(default_val))
            fr = ttk.Frame(parent)
            fr.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
            fr.columnconfigure(1, weight=1)
            ttk.Label(fr, text=label, width=18).grid(row=0, column=0, sticky="w")
            ttk.Entry(fr, textvariable=self._param_vars[key], width=12).grid(row=0, column=1, sticky="ew")

        # Column 0 fields
        add_field(0, 0, "z_lookback", "Z Lookback", p.z_lookback)
        add_field(1, 0, "z_enter", "Z Enter", p.z_enter)
        add_field(2, 0, "z_exit", "Z Exit", p.z_exit)

        add_field(3, 0, "rsi_len", "RSI Len", p.rsi_len)
        add_field(4, 0, "rsi_oversold", "RSI Oversold", p.rsi_oversold)

        add_field(5, 0, "atr_len", "ATR Len", p.atr_len)
        add_field(6, 0, "atr_min_pct", "ATR Min %", p.atr_min_pct)
        add_field(7, 0, "atr_max_pct", "ATR Max %", p.atr_max_pct)
        add_field(8, 0, "atr_stop_mult", "ATR Stop Mult", p.atr_stop_mult)

        # Column 1 fields
        add_field(0, 1, "bb_len", "BB Len", p.bb_len)
        add_field(1, 1, "bb_mult", "BB Mult", p.bb_mult)
        add_field(2, 1, "bb_enter_pct", "BB Enter %", p.bb_enter_pct)

        add_field(3, 1, "adx_len", "ADX Len", p.adx_len)
        add_field(4, 1, "adx_max_for_mr", "ADX Max MR", p.adx_max_for_mr)

        add_field(5, 1, "confidence_min", "Conf Min", p.confidence_min)
        add_field(6, 1, "max_bars_hold", "Max Bars Hold", p.max_bars_hold)
        add_field(7, 1, "take_partial_at_r", "Partial at R", p.take_partial_at_r)
        add_field(8, 1, "partial_qty_frac", "Partial Qty", p.partial_qty_frac)

        # Row 9 for leverage (compact)
        add_field(9, 0, "lev_base", "Lev Base", p.lev_base)
        add_field(9, 1, "lev_max", "Lev Max", p.lev_max)

        # Row 10 for cost gate + scale
        add_field(10, 0, "min_r_vs_cost_mult", "Min R vs Cost", p.min_r_vs_cost_mult)
        add_field(10, 1, "lev_scale_by_signal", "Lev Scale", p.lev_scale_by_signal)

        # Row 11: bb_exit_mid as dropdown boolean
        fr = ttk.Frame(parent)
        fr.grid(row=11, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        fr.columnconfigure(1, weight=1)
        ttk.Label(fr, text="BB Exit Mid", width=18).grid(row=0, column=0, sticky="w")
        self._param_vars["bb_exit_mid"] = tk.StringVar(value="1" if p.bb_exit_mid else "0")
        dd = ttk.Combobox(fr, textvariable=self._param_vars["bb_exit_mid"], width=10, values=["1", "0"], state="readonly")
        dd.grid(row=0, column=1, sticky="w")
        ttk.Label(fr, text="(1=yes, 0=no)").grid(row=0, column=2, sticky="w", padx=6)

    def _params_from_gui(self) -> StrategyParamsV3:
        """
        Reads current GUI fields, builds StrategyParamsV3, sanitizes, returns it.
        """
        # Helper to read stringvar safely
        def gv(k: str, default: str = "") -> str:
            try:
                return (self._param_vars.get(k) or tk.StringVar(value=default)).get().strip()
            except Exception:
                return default

        p = StrategyParamsV3(
            z_lookback=safe_int(gv("z_lookback", "120"), 120),
            z_enter=safe_float(gv("z_enter", "1.75"), 1.75),
            z_exit=safe_float(gv("z_exit", "0.35"), 0.35),

            bb_len=safe_int(gv("bb_len", "120"), 120),
            bb_mult=safe_float(gv("bb_mult", "2.0"), 2.0),
            bb_enter_pct=safe_float(gv("bb_enter_pct", "0.02"), 0.02),
            bb_exit_mid=(gv("bb_exit_mid", "1") not in ("0", "false", "False")),

            rsi_len=safe_int(gv("rsi_len", "14"), 14),
            rsi_oversold=safe_float(gv("rsi_oversold", "33.0"), 33.0),

            atr_len=safe_int(gv("atr_len", "14"), 14),
            atr_min_pct=safe_float(gv("atr_min_pct", "0.0012"), 0.0012),
            atr_max_pct=safe_float(gv("atr_max_pct", "0.06"), 0.06),
            atr_stop_mult=safe_float(gv("atr_stop_mult", "2.2"), 2.2),

            adx_len=safe_int(gv("adx_len", "14"), 14),
            adx_max_for_mr=safe_float(gv("adx_max_for_mr", "22.0"), 22.0),

            confidence_min=safe_float(gv("confidence_min", "0.62"), 0.62),
            max_bars_hold=safe_int(gv("max_bars_hold", "220"), 220),
            take_partial_at_r=safe_float(gv("take_partial_at_r", "0.75"), 0.75),
            partial_qty_frac=safe_float(gv("partial_qty_frac", "0.35"), 0.35),

            min_r_vs_cost_mult=safe_float(gv("min_r_vs_cost_mult", "1.35"), 1.35),

            lev_base=safe_float(gv("lev_base", "1.0"), 1.0),
            lev_max=safe_float(gv("lev_max", "2.0"), 2.0),
            lev_scale_by_signal=safe_float(gv("lev_scale_by_signal", "1.0"), 1.0),
        ).sanitize()

        return p

    def _apply_save_params(self) -> None:
        try:
            p = self._params_from_gui()
            self.engine.set_strategy_params(p, persist=True)
            messagebox.showinfo("Strategy Params", "Applied and saved strategy parameters.")
            self.log.info("GUI applied & saved strategy params.")
        except Exception as e:
            self.log.exception("Apply/save params failed", e)
            messagebox.showerror("Strategy Params", f"Failed to apply/save: {e}")

    def _reload_params_from_engine(self) -> None:
        try:
            p = self.engine.get_strategy_params().sanitize()
            for k, v in dataclasses.asdict(p).items():
                if k not in self._param_vars:
                    self._param_vars[k] = tk.StringVar()
                self._param_vars[k].set(str(v if not isinstance(v, bool) else (1 if v else 0)))
            self.log.info("GUI reloaded params from engine.")
        except Exception as e:
            self.log.warn(f"Reload params failed: {e}")

    def _reset_params_defaults(self) -> None:
        try:
            p = StrategyParamsV3().sanitize()
            self.engine.set_strategy_params(p, persist=True)
            self._reload_params_from_engine()
            messagebox.showinfo("Strategy Params", "Reset to defaults and saved.")
            self.log.info("GUI reset params to defaults.")
        except Exception as e:
            self.log.exception("Reset params failed", e)
            messagebox.showerror("Strategy Params", f"Failed: {e}")

    # ---- Controls ----

    def _start_engine(self) -> None:
        self.engine.start()
        self.log.info("GUI requested engine start")

    def _stop_engine(self) -> None:
        self.engine.stop()
        self.log.info("GUI requested engine stop")

    def _export_logs(self) -> None:
        src = self.cfg.data_dir / "lunobiz.log"
        if not src.exists():
            messagebox.showwarning("Export Logs", "Log file not found yet.")
            return
        dst = filedialog.asksaveasfilename(
            title="Export Logs",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if not dst:
            return
        try:
            Path(dst).write_bytes(src.read_bytes())
            messagebox.showinfo("Export Logs", f"Exported to: {dst}")
        except Exception as e:
            messagebox.showerror("Export Logs", f"Failed: {e}")

    def _open_reports_folder(self) -> None:
        open_folder_in_explorer(self.reports.reports_dir)

    # ---- Backtest UI ----

    def _copy_backtest_summary(self) -> None:
        if not self._last_backtest_results:
            messagebox.showwarning("Copy Summary", "Run a backtest first.")
            return
        summary = self._compose_backtest_summary_text()
        try:
            self.clipboard_clear()
            self.clipboard_append(summary)
            self.update()
            messagebox.showinfo("Copy Summary", "Copied backtest summary to clipboard. Paste it into chat.")
        except Exception as e:
            messagebox.showerror("Copy Summary", f"Failed: {e}")

    def _compose_backtest_summary_text(self) -> str:
        ctx = self._last_backtest_context or {}
        lines: list[str] = []
        lines.append(f"{APP_NAME} {APP_VERSION} BACKTEST SUMMARY")
        lines.append(f"Generated: {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        for k, v in ctx.items():
            if k == "strategy_params":
                continue
            lines.append(f"{k}: {v}")

        if "strategy_params" in ctx:
            lines.append("")
            lines.append("Strategy Params (v3):")
            try:
                params_pretty = json.dumps(ctx["strategy_params"], ensure_ascii=False, indent=2)
                lines.extend(params_pretty.splitlines())
            except Exception:
                lines.append("(unavailable)")

        lines.append("")
        lines.append("Top Results:")
        for r in self._last_backtest_results[:10]:
            lines.append(" - " + r.summary_line())
        if self._last_backtest_report_paths:
            lines.append("")
            lines.append("Latest report files:")
            for k, p in self._last_backtest_report_paths.items():
                lines.append(f" - {k}: {p}")
        return "\n".join(lines)

    def _export_backtest_report(self) -> None:
        if not self._last_backtest_results:
            messagebox.showwarning("Export Backtest Report", "Run a backtest first.")
            return
        try:
            paths = self.reports.write_backtest_reports(self._last_backtest_results, self._last_backtest_context)
            self._last_backtest_report_paths = paths
            messagebox.showinfo(
                "Export Backtest Report",
                "Saved CSV/JSON/TXT reports.\n\n"
                f"CSV: {paths.get('csv')}\n"
                f"JSON: {paths.get('json')}\n"
                f"TXT: {paths.get('txt')}\n"
            )
        except Exception as e:
            messagebox.showerror("Export Backtest Report", f"Failed: {e}")

    def _run_backtest(self) -> None:
        if self._backtest_thread and self._backtest_thread.is_alive():
            messagebox.showwarning("Backtest", "Backtest already running.")
            return

        raw_pairs = self.var_bt_pairs.get().strip()
        pairs = [p.strip().upper() for p in raw_pairs.split(",") if p.strip()]
        if not pairs:
            messagebox.showwarning("Backtest", "Enter at least one pair.")
            return

        entry_tf = self.var_bt_tf.get().strip()
        days = safe_int(self.var_bt_days.get().strip(), 180)
        eq = safe_float(self.var_bt_equity.get().strip(), 100.0)
        walk = bool(self.var_bt_walk.get())

        # Freeze strategy params snapshot for report context
        params_snapshot = dataclasses.asdict(self.engine.get_strategy_params().sanitize())

        def worker() -> None:
            try:
                self.log.info(f"Backtest start: pairs={pairs} entry_tf={entry_tf} days={days} equity={eq} walk={walk}")
                res = self.engine.run_backtest(pairs=pairs, entry_tf=entry_tf, days=days, initial_equity=eq, walk_forward=walk)

                ctx = {
                    "pairs": ",".join(pairs),
                    "entry_timeframe": entry_tf,
                    "signal_timeframe": self.cfg.signal_timeframe,
                    "days": days,
                    "initial_equity": eq,
                    "walk_forward": walk,
                    "slippage_bps": self.cfg.backtest_slippage_bps,
                    "fee_bps": self.cfg.backtest_fee_bps,
                    "paper_leverage_enabled": self.cfg.leverage_enabled_paper,
                    "paper_leverage_max": self.cfg.leverage_max,
                    "margin_alloc_cap": self.cfg.margin_alloc_cap,
                    "maint_margin_ratio": self.cfg.maint_margin_ratio,
                    "strategy_params": params_snapshot,
                }
                self._last_backtest_results = res
                self._last_backtest_context = ctx
                self._last_backtest_report_paths = {}

                self._ui_queue.put(lambda: self._render_backtest_results(res))

                # Conservative live gate (still not executing live in v0.3.1):
                if res:
                    top = res[0]
                    gate_ok = (top.total_return > 0.10 and top.max_drawdown < 0.20 and top.trades >= 20)
                    self.engine.storage.set_gate("LIVE_GATE", "1" if gate_ok else "0")
                    self.log.info(
                        f"Gate eval: top_return={fmt_pct(top.total_return)} dd={fmt_pct(top.max_drawdown)} "
                        f"trades={top.trades} -> LIVE_GATE={int(gate_ok)}"
                    )

            except Exception as e:
                self.log.exception("Backtest worker failed", e)

        self._backtest_thread = threading.Thread(target=worker, daemon=True, name="BacktestWorker")
        self._backtest_thread.start()

    def _render_backtest_results(self, results: list[BacktestResult]) -> None:
        tree: ttk.Treeview = self.tbl_bt.tree  # type: ignore[attr-defined]
        for item in tree.get_children():
            tree.delete(item)

        for r in results[:50]:
            tree.insert("", tk.END, values=(
                r.pair,
                r.entry_tf,
                r.signal_tf,
                r.trades,
                f"{r.win_rate * 100:.1f}%",
                f"{r.total_return * 100:.1f}%",
                f"{r.max_drawdown * 100:.1f}%",
                r.notes
            ))

    # ---- Logs & status ----

    def _poll_logs(self) -> None:
        for _, line in self.log.drain(200):
            try:
                self.txt_logs.insert(tk.END, line + "\n")
                self.txt_logs.see(tk.END)
            except Exception:
                pass
        self.after(250, self._poll_logs)

    def _refresh_status(self) -> None:
        try:
            self.var_mode.set(self.cfg.app_mode.upper())
            self.var_health.set(self.engine.health())

            eq = self.engine.paper.equity
            cash = self.engine.paper.cash
            self.var_equity.set(fmt_money(eq, "MYR"))
            self.var_cash.set(fmt_money(cash, "MYR"))
            self.var_daily_pnl.set(fmt_pct((eq - self.engine.paper.day_start_equity) / max(1e-12, self.engine.paper.day_start_equity)))

            self._render_positions()
            self._render_recent_decisions()
        except Exception as e:
            self.log.warn(f"UI status update failed: {e}")

        self.after(700, self._refresh_status)

    def _render_positions(self) -> None:
        tree: ttk.Treeview = self.tbl_positions.tree  # type: ignore[attr-defined]
        for item in tree.get_children():
            tree.delete(item)

        for p, pos in self.engine.paper.positions.items():
            age_s = max(0, int((utc_ms() - pos.entry_ts_ms) / 1000))
            age = f"{age_s // 3600:02d}:{(age_s % 3600) // 60:02d}:{age_s % 60:02d}"
            tree.insert("", tk.END, values=(
                p,
                f"{pos.qty:.6f}",
                f"{pos.entry_price:.4f}",
                age
            ))

    def _render_recent_decisions(self) -> None:
        tree: ttk.Treeview = self.tbl_rank.tree  # type: ignore[attr-defined]
        for item in tree.get_children():
            tree.delete(item)

        try:
            con = sqlite3.connect(str(self.storage.db_path))
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute("""
                SELECT ts_ms, pair, action, confidence, reason
                FROM decisions
                ORDER BY ts_ms DESC
                LIMIT 20
            """)
            rows = cur.fetchall()
            con.close()

            for r in rows:
                tree.insert("", tk.END, values=(
                    human_ts(int(r["ts_ms"])),
                    r["pair"],
                    r["action"],
                    f"{float(r['confidence']):.2f}",
                    r["reason"]
                ))
        except Exception:
            pass

    def _drain_ui_queue(self) -> None:
        for _ in range(50):
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as e:
                self.log.warn(f"UI task failed: {e}")
        self.after(300, self._drain_ui_queue)

    def _on_close(self) -> None:
        try:
            self.engine.stop()
            time.sleep(0.2)
        except Exception:
            pass
        self.destroy()


# =========================
# Entry point
# =========================

def find_repo_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    repo_root = find_repo_root()
    ensure_dir(repo_root / "data")
    app = Dashboard(repo_root)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
