"""
LunoBiz - Single-file Windows application (v0.3.2)

Key features:
- GUI dashboard (values & tables; no charts)
- Robust candle caching in SQLite
- Backtesting with walk-forward validation
- Strategy parameters editable via GUI (persisted to SQLite)
- Volatility Continuation Engine (long-only; paper/backtest)
- Paper-only leverage simulation (no live execution)
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
APP_VERSION = "0.3.2"

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

# Storage keys
GATE_STRAT_PARAMS_JSON = "STRAT_PARAMS_JSON"
GATE_LIVE_GATE = "LIVE_GATE"


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

@dataclass(frozen=True)
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
    backtest_timeframe: str
    signal_timeframe: str
    backtest_slippage_bps: float
    backtest_fee_bps: float

    # Paper leverage simulation
    leverage_enabled_paper: bool
    leverage_max: float
    leverage_min: float
    margin_alloc_cap: float
    maint_margin_ratio: float

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

    def _request(self, method: str, path: str, params: dict[str, t.Any] | None = None, data: dict[str, t.Any] | None = None) -> dict[str, t.Any]:
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


def sma(values: list[float], period: int) -> list[float]:
    if period <= 1:
        return values[:]
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > period:
            s -= q.pop(0)
        out.append(s / len(q))
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


def stdev(values: list[float], period: int) -> list[float]:
    if period <= 1:
        return [0.0] * len(values)
    out: list[float] = []
    q: list[float] = []
    for v in values:
        q.append(v)
        if len(q) > period:
            q.pop(0)
        if len(q) < 2:
            out.append(0.0)
        else:
            m = sum(q) / len(q)
            var = sum((x - m) ** 2 for x in q) / max(1, (len(q) - 1))
            out.append(math.sqrt(max(0.0, var)))
    return out


def rolling_median(values: list[float], period: int) -> list[float]:
    if period <= 1:
        return values[:]
    out: list[float] = []
    q: list[float] = []
    for v in values:
        q.append(v)
        if len(q) > period:
            q.pop(0)
        s = sorted(q)
        mid = len(s) // 2
        if len(s) % 2 == 1:
            out.append(s[mid])
        else:
            out.append(0.5 * (s[mid - 1] + s[mid]))
    return out


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
        for j in range(p + 1, n):
            out[j] = out[j - 1] - (out[j - 1] / p) + series[j]
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


def zscore(values: list[float], lookback: int) -> list[float]:
    if lookback <= 2:
        return [0.0] * len(values)
    out: list[float] = []
    q: list[float] = []
    s = 0.0
    ss = 0.0
    for v in values:
        q.append(v)
        s += v
        ss += v * v
        if len(q) > lookback:
            old = q.pop(0)
            s -= old
            ss -= old * old
        n = len(q)
        if n < 3:
            out.append(0.0)
            continue
        mean = s / n
        var = max(0.0, (ss / n) - (mean * mean))
        sd = math.sqrt(var) if var > 1e-18 else 0.0
        out.append((v - mean) / (sd + 1e-12))
    return out


def rolling_vwap(close: list[float], volume: list[float], lookback: int) -> list[float]:
    """
    Rolling VWAP using close*volume.
    """
    if lookback <= 1:
        return close[:]
    out: list[float] = []
    q_pv: list[float] = []
    q_v: list[float] = []
    s_pv = 0.0
    s_v = 0.0
    for p, v in zip(close, volume):
        pv = p * v
        q_pv.append(pv)
        q_v.append(v)
        s_pv += pv
        s_v += v
        if len(q_pv) > lookback:
            s_pv -= q_pv.pop(0)
            s_v -= q_v.pop(0)
        out.append((s_pv / (s_v + 1e-12)) if s_v > 0 else p)
    return out


def bollinger(close: list[float], length: int, mult: float) -> tuple[list[float], list[float], list[float]]:
    mid = sma(close, length)
    sd = stdev(close, length)
    upper: list[float] = []
    lower: list[float] = []
    for m, s in zip(mid, sd):
        upper.append(m + mult * s)
        lower.append(m - mult * s)
    return mid, upper, lower


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
# Strategy Params (GUI-controllable)
# =========================

@dataclass
class StrategyParamsV32:
    # Core filters
    z_lookback: int = 60
    z_reentry: float = 0.55              # breakout strength in zscore terms
    z_neutral_band: float = 0.35         # require prior z within ±band (compression)
    z_exit_revert: float = 0.15          # exit when z reverts toward 0

    vwap_lookback: int = 80

    bb_len: int = 60
    bb_mult: float = 1.6
    squeeze_bw_lookback: int = 120       # for percentile-ish squeeze check via median
    squeeze_bw_mult: float = 0.80        # current bb_width < median(bb_width)*mult => squeeze

    atr_len: int = 14
    atr_expand_mult: float = 1.15        # current atr% > median(atr%)*mult => expansion
    atr_min_pct: float = 0.0010
    atr_max_pct: float = 0.055
    atr_stop_mult: float = 1.35

    rsi_len: int = 7
    rsi_min_for_long: float = 52.0       # momentum confirmation
    rsi_overheat: float = 78.0           # avoid too stretched entries

    adx_len: int = 14
    adx_min_rising: float = 14.0         # allow if adx above threshold
    adx_max_entry: float = 38.0          # avoid late-stage blowoff

    momentum_confirm_bars: int = 3       # close now > close[-k]
    cooldown_bars_after_exit: int = 6    # prevent immediate re-entry chop

    # Execution & management
    confidence_min: float = 0.48
    max_bars_hold: int = 140
    take_partial_at_r: float = 0.70
    partial_qty_frac: float = 0.40
    trail_atr_mult: float = 1.80
    min_r_vs_cost_mult: float = 1.10

    # Leverage (paper only)
    lev_base: float = 1.0
    lev_max: float = 1.6
    lev_scale_by_signal: float = 0.8     # scale by regime strength

    def sanitize(self) -> "StrategyParamsV32":
        # Defensive clamps (prevents GUI bad input from crashing / invalidating runs)
        p = dataclasses.replace(self)
        p.z_lookback = int(clamp(p.z_lookback, 20, 400))
        p.z_reentry = float(clamp(p.z_reentry, 0.10, 3.0))
        p.z_neutral_band = float(clamp(p.z_neutral_band, 0.05, 2.0))
        p.z_exit_revert = float(clamp(p.z_exit_revert, 0.01, 1.0))
        p.vwap_lookback = int(clamp(p.vwap_lookback, 20, 400))
        p.bb_len = int(clamp(p.bb_len, 20, 400))
        p.bb_mult = float(clamp(p.bb_mult, 0.8, 4.0))
        p.squeeze_bw_lookback = int(clamp(p.squeeze_bw_lookback, 40, 600))
        p.squeeze_bw_mult = float(clamp(p.squeeze_bw_mult, 0.20, 1.20))
        p.atr_len = int(clamp(p.atr_len, 5, 50))
        p.atr_expand_mult = float(clamp(p.atr_expand_mult, 0.80, 1.80))
        p.atr_min_pct = float(clamp(p.atr_min_pct, 0.0001, 0.02))
        p.atr_max_pct = float(clamp(p.atr_max_pct, 0.01, 0.20))
        p.atr_stop_mult = float(clamp(p.atr_stop_mult, 0.5, 5.0))
        p.rsi_len = int(clamp(p.rsi_len, 3, 30))
        p.rsi_min_for_long = float(clamp(p.rsi_min_for_long, 40.0, 70.0))
        p.rsi_overheat = float(clamp(p.rsi_overheat, 60.0, 95.0))
        p.adx_len = int(clamp(p.adx_len, 5, 30))
        p.adx_min_rising = float(clamp(p.adx_min_rising, 5.0, 40.0))
        p.adx_max_entry = float(clamp(p.adx_max_entry, 10.0, 80.0))
        p.momentum_confirm_bars = int(clamp(p.momentum_confirm_bars, 1, 12))
        p.cooldown_bars_after_exit = int(clamp(p.cooldown_bars_after_exit, 0, 60))
        p.confidence_min = float(clamp(p.confidence_min, 0.05, 0.95))
        p.max_bars_hold = int(clamp(p.max_bars_hold, 20, 1000))
        p.take_partial_at_r = float(clamp(p.take_partial_at_r, 0.10, 3.0))
        p.partial_qty_frac = float(clamp(p.partial_qty_frac, 0.05, 0.95))
        p.trail_atr_mult = float(clamp(p.trail_atr_mult, 0.5, 8.0))
        p.min_r_vs_cost_mult = float(clamp(p.min_r_vs_cost_mult, 0.5, 5.0))
        p.lev_base = float(clamp(p.lev_base, 1.0, 3.0))
        p.lev_max = float(clamp(p.lev_max, 1.0, 6.0))
        p.lev_scale_by_signal = float(clamp(p.lev_scale_by_signal, 0.0, 2.0))
        return p

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(text: str) -> "StrategyParamsV32":
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                return StrategyParamsV32()
            # Use defaults for missing keys
            base = dataclasses.asdict(StrategyParamsV32())
            base.update({k: obj.get(k, base.get(k)) for k in base.keys()})
            return StrategyParamsV32(**base).sanitize()
        except Exception:
            return StrategyParamsV32()


# =========================
# Strategy v0.3.2 (Volatility Continuation)
# =========================

@dataclass
class RegimeState:
    strength: float  # 0..1
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


class StrategyV32:
    """
    Volatility Continuation Engine (Long-only):
    - Identify volatility compression (squeeze)
    - Enter on expansion + directional breakout confirmation
    - Exit on mean reversion (z back toward 0) or momentum fade/time stop

    Uses entry timeframe candles only for speed/robustness.
    Signal timeframe is used only for simple "regime strength" gating (ADX + ATR%).
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log

    def compute_regime(self, sig: CandleSeries, p: StrategyParamsV32) -> list[RegimeState]:
        n = len(sig.c)
        if n < max(120, p.adx_len * 3, p.atr_len * 3):
            return [RegimeState(strength=0.0, adx=0.0, atr_pct=0.0) for _ in range(n)]

        a = atr(sig.h, sig.l, sig.c, p.atr_len)
        adxv = adx(sig.h, sig.l, sig.c, p.adx_len)

        out: list[RegimeState] = []
        for i in range(n):
            price = sig.c[i]
            atr_pct = (a[i] / max(1e-12, price)) if price > 0 else 0.0
            # Strength: prefers healthy but not extreme volatility + decent ADX
            adx_score = clamp((adxv[i] - 10.0) / 25.0, 0.0, 1.0)     # 10..35
            vol_score = 1.0 - clamp(abs(atr_pct - 0.010) / 0.020, 0.0, 1.0)  # sweet spot around 1%
            strength = clamp(0.65 * adx_score + 0.35 * vol_score, 0.0, 1.0)
            out.append(RegimeState(strength=strength, adx=adxv[i], atr_pct=atr_pct))
        return out

    def choose_leverage(self, regime: RegimeState, equity_dd: float, p: StrategyParamsV32) -> float:
        if not self.cfg.leverage_enabled_paper:
            return 1.0
        # Conservative: lev_base + scaled strength, reduced in drawdown
        lev = p.lev_base + (p.lev_scale_by_signal * regime.strength)
        if equity_dd > 0.02:
            lev *= clamp(1.0 - equity_dd * 6.0, 0.35, 1.0)
        lev = clamp(lev, 1.0, min(self.cfg.leverage_max, p.lev_max))
        return float(lev)

    def entry_signal(
        self,
        pair: str,
        entry: CandleSeries,
        i: int,
        regime: RegimeState,
        equity_dd: float,
        p: StrategyParamsV32
    ) -> EntrySignal:
        """
        Entry rules (continuation):
        - Prior z is neutral (compression)
        - BB width indicates squeeze (current bw < median bw * mult)
        - ATR% expands above its median (expansion)
        - Breakout: z crosses above z_reentry, close > vwap, close > bb_mid
        - Momentum: close > close[-k], RSI above threshold, ADX within [min, max]
        """
        if i < max(p.z_lookback, p.bb_len, p.vwap_lookback, p.squeeze_bw_lookback, 80):
            return EntrySignal("HOLD", 0.0, "Insufficient history", 1.0, 0.0, 0.0, {})

        close = entry.c[:i + 1]
        high = entry.h[:i + 1]
        low = entry.l[:i + 1]
        vol = entry.v[:i + 1]
        price = close[-1]
        if price <= 0:
            return EntrySignal("HOLD", 0.0, "Bad price", 1.0, 0.0, 0.0, {})

        # Compute indicators
        z = zscore(close, p.z_lookback)
        z_now = z[-1]
        z_prev = z[-2] if len(z) >= 2 else 0.0

        mid, upper, lower = bollinger(close, p.bb_len, p.bb_mult)
        bb_mid = mid[-1]
        bb_u = upper[-1]
        bb_l = lower[-1]
        bb_width = (bb_u - bb_l) / max(1e-12, bb_mid)

        # Squeeze: compare to rolling median of bb width
        bw_series: list[float] = []
        for j in range(len(close)):
            if j < p.bb_len:
                bw_series.append(0.0)
            else:
                m, u, l = bollinger(close[:j + 1], p.bb_len, p.bb_mult)
                bw = (u[-1] - l[-1]) / max(1e-12, m[-1])
                bw_series.append(bw)

        bw_med = rolling_median(bw_series, p.squeeze_bw_lookback)[-1]
        is_squeeze = (bw_med > 0) and (bb_width < (bw_med * p.squeeze_bw_mult))

        a = atr(high, low, close, p.atr_len)
        atr_now = a[-1]
        atr_pct_now = atr_now / max(1e-12, price)
        atr_pct_series = [(a_k / max(1e-12, c_k)) if c_k > 0 else 0.0 for a_k, c_k in zip(a, close)]
        atr_pct_med = rolling_median(atr_pct_series, p.squeeze_bw_lookback)[-1]
        is_expansion = (atr_pct_med > 0) and (atr_pct_now > atr_pct_med * p.atr_expand_mult)

        vwap = rolling_vwap(close, vol, p.vwap_lookback)
        vwap_now = vwap[-1]

        rs = rsi(close, p.rsi_len)
        rsi_now = rs[-1]

        adxv = adx(high, low, close, p.adx_len)
        adx_now = adxv[-1]
        adx_prev = adxv[-2] if len(adxv) >= 2 else adx_now
        adx_rising = adx_now >= adx_prev

        # Basic sanity bands
        if atr_pct_now < p.atr_min_pct:
            return EntrySignal("HOLD", 0.0, "ATR% too low", 1.0, 0.0, 0.0, {"atr_pct": atr_pct_now})
        if atr_pct_now > p.atr_max_pct:
            return EntrySignal("HOLD", 0.0, "ATR% too high", 1.0, 0.0, 0.0, {"atr_pct": atr_pct_now})

        # Regime gating from signal timeframe
        if not (p.adx_min_rising <= adx_now <= p.adx_max_entry):
            return EntrySignal("HOLD", 0.0, "ADX out of entry band", 1.0, 0.0, 0.0, {"adx": adx_now})
        if not adx_rising:
            return EntrySignal("HOLD", 0.0, "ADX not rising", 1.0, 0.0, 0.0, {"adx": adx_now})

        # Compression prerequisites
        if not is_squeeze:
            return EntrySignal("HOLD", 0.0, "No squeeze", 1.0, 0.0, 0.0, {"bb_width": bb_width, "bw_med": bw_med})

        # Require that we were recently neutral (avoid chasing late trends)
        if abs(z_prev) > p.z_neutral_band:
            return EntrySignal("HOLD", 0.0, "Prior z not neutral", 1.0, 0.0, 0.0, {"z_prev": z_prev})

        # Expansion trigger
        if not is_expansion:
            return EntrySignal("HOLD", 0.0, "No vol expansion", 1.0, 0.0, 0.0, {"atr_pct": atr_pct_now, "atr_med": atr_pct_med})

        # Directional breakout confirmation
        crossed = (z_prev <= p.z_reentry and z_now > p.z_reentry)
        if not crossed:
            return EntrySignal("HOLD", 0.0, "No z breakout cross", 1.0, 0.0, 0.0, {"z_now": z_now, "z_prev": z_prev})

        if not (price > vwap_now and price > bb_mid):
            return EntrySignal("HOLD", 0.0, "Price not above VWAP/BB mid", 1.0, 0.0, 0.0, {"price": price, "vwap": vwap_now, "bb_mid": bb_mid})

        # Momentum confirm
        k = max(1, p.momentum_confirm_bars)
        if i - k < 0:
            return EntrySignal("HOLD", 0.0, "Momentum window short", 1.0, 0.0, 0.0, {})
        if not (price > close[-1 - k]):
            return EntrySignal("HOLD", 0.0, "No momentum confirm", 1.0, 0.0, 0.0, {"k": k})

        if rsi_now < p.rsi_min_for_long:
            return EntrySignal("HOLD", 0.0, "RSI too weak", 1.0, 0.0, 0.0, {"rsi": rsi_now})
        if rsi_now > p.rsi_overheat:
            return EntrySignal("HOLD", 0.0, "RSI overheated", 1.0, 0.0, 0.0, {"rsi": rsi_now})

        # Define stop using ATR (continuation: tighter stop, but not too tight)
        stop_dist = max(1e-9, p.atr_stop_mult * atr_now)
        stop_price = price - stop_dist
        if stop_price <= 0:
            return EntrySignal("HOLD", 0.0, "Invalid stop price", 1.0, 0.0, 0.0, {"atr": atr_now})

        r_value = price - stop_price

        # Cost gate (round trip)
        bps_total = (self.cfg.backtest_fee_bps + self.cfg.backtest_slippage_bps) / 10000.0
        est_round_trip_cost = price * bps_total * 2.2
        if r_value < est_round_trip_cost * p.min_r_vs_cost_mult:
            return EntrySignal("HOLD", 0.0, "R too small vs costs", 1.0, 0.0, 0.0,
                               {"r_value": r_value, "est_cost": est_round_trip_cost})

        # Confidence score (balanced)
        squeeze_score = clamp(1.0 - (bb_width / (bw_med + 1e-12)), 0.0, 1.0) if bw_med > 0 else 0.0
        z_score = clamp((z_now - p.z_reentry) / max(1e-9, (2.5 - p.z_reentry)), 0.0, 1.0)
        adx_score = clamp((adx_now - p.adx_min_rising) / max(1e-9, (p.adx_max_entry - p.adx_min_rising)), 0.0, 1.0)
        rsi_score = clamp((rsi_now - p.rsi_min_for_long) / max(1e-9, (p.rsi_overheat - p.rsi_min_for_long)), 0.0, 1.0)

        conf = clamp(0.20 + 0.35 * squeeze_score + 0.25 * z_score + 0.20 * adx_score + 0.10 * rsi_score, 0.0, 0.95)

        lev = self.choose_leverage(regime, equity_dd, p)

        return EntrySignal(
            action="BUY",
            confidence=float(conf),
            reason="Squeeze->Expansion breakout (continuation)",
            leverage=float(lev),
            stop_price=float(stop_price),
            r_value=float(r_value),
            meta={
                "price": price,
                "z_prev": z_prev,
                "z_now": z_now,
                "bb_mid": bb_mid,
                "bb_width": bb_width,
                "bw_med": bw_med,
                "atr": atr_now,
                "atr_pct": atr_pct_now,
                "atr_med": atr_pct_med,
                "vwap": vwap_now,
                "rsi": rsi_now,
                "adx": adx_now,
                "equity_dd": equity_dd,
                "regime_strength": regime.strength,
            }
        )


# =========================
# Backtesting (v0.3.2)
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
    entry_bar_index: int

    # exit management
    took_partial: bool = False
    qty_remaining: float = 0.0
    trail_stop: float = 0.0
    last_exit_bar: int = -10_000


class BacktesterV32:
    """
    Long-only with paper leverage simulation.
    - Position sized by cfg.risk_per_trade: max loss at stop equals risk amount
    - Leverage affects margin used; liquidation if low breaches liquidation price
    - Partial exit at +R; trailing stop thereafter
    - Exit if z reverts toward 0 or time stop
    """
    def __init__(self, cfg: AppConfig, log: AppLogger, strategy: StrategyV32):
        self.cfg = cfg
        self.log = log
        self.strategy = strategy

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
        - loss_cap = margin_used * (1 - maint_margin_ratio)
        - liq_price = entry_price - loss_cap/qty
        """
        loss_cap = pos.margin_used * (1.0 - self.cfg.maint_margin_ratio)
        if pos.qty_remaining <= 0 or pos.entry_price <= 0:
            return 0.0
        return max(0.0, pos.entry_price - (loss_cap / pos.qty_remaining))

    def run(
        self,
        pair: str,
        entry: CandleSeries,
        signal: CandleSeries,
        regime_series: list[RegimeState],
        params: StrategyParamsV32,
        initial_equity: float = 100.0
    ) -> BacktestResult:
        p = params.sanitize()

        if len(entry.c) < 600 or len(signal.c) < 200 or len(regime_series) != len(signal.c):
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

        closed_trades = 0
        wins = 0
        trade_returns: list[float] = []
        gross_profit = 0.0
        gross_loss = 0.0

        eq_curve: list[tuple[int, float]] = []

        # Precompute z for exit condition (efficient)
        z_series = zscore(entry.c, p.z_lookback)
        atr_series = atr(entry.h, entry.l, entry.c, p.atr_len)

        def equity_drawdown() -> float:
            return (peak - equity) / max(1e-12, peak)

        last_exit_bar = -10_000

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

            # --- Manage open position ---
            if pos is not None:
                # liquidation
                liq_px = self._liquidation_price_long(pos)
                if liq_px > 0 and lo <= liq_px:
                    exit_px = self._apply_slippage(liq_px, "SELL")
                    notional = pos.qty_remaining * exit_px
                    fee = self._fee(notional)
                    pnl = (exit_px - pos.entry_price) * pos.qty_remaining - fee

                    equity += pnl
                    closed_trades += 1
                    if pnl > 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    trade_returns.append(pnl / max(1e-12, initial_equity))
                    pos = None
                    last_exit_bar = i
                    eq_curve.append((ts, equity))
                    peak = max(peak, equity)
                    max_dd = max(max_dd, equity_drawdown())
                    break

                # exit by stop / trailing stop
                active_stop = max(pos.stop_price, pos.trail_stop)
                if lo <= active_stop:
                    exit_px = self._apply_slippage(active_stop, "SELL")
                    notional = pos.qty_remaining * exit_px
                    fee = self._fee(notional)
                    pnl = (exit_px - pos.entry_price) * pos.qty_remaining - fee

                    equity += pnl
                    closed_trades += 1
                    if pnl > 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    trade_returns.append(pnl / max(1e-12, initial_equity))
                    pos = None
                    last_exit_bar = i
                else:
                    # partial at +R
                    tp_r = max(0.10, p.take_partial_at_r)
                    tp1 = pos.entry_price + tp_r * pos.r_value
                    if (not pos.took_partial) and hi >= tp1:
                        take_qty = pos.qty_remaining * clamp(p.partial_qty_frac, 0.05, 0.95)
                        exit_px = self._apply_slippage(tp1, "SELL")
                        notional = take_qty * exit_px
                        fee = self._fee(notional)
                        pnl = (exit_px - pos.entry_price) * take_qty - fee
                        equity += pnl

                        pos.qty_remaining = max(0.0, pos.qty_remaining - take_qty)
                        pos.took_partial = True

                        # Stop to breakeven after partial
                        pos.stop_price = max(pos.stop_price, pos.entry_price)

                    # trailing update once partial taken
                    if pos is not None and pos.took_partial and pos.qty_remaining > 0:
                        atr_now = atr_series[i] if i < len(atr_series) else 0.0
                        if atr_now > 0:
                            new_trail = price - p.trail_atr_mult * atr_now
                            pos.trail_stop = max(pos.trail_stop, new_trail)

                    # exit on z reversion (mean reversion used for exit)
                    z_now = z_series[i] if i < len(z_series) else 0.0
                    if pos is not None and abs(z_now) <= p.z_exit_revert and pos.qty_remaining > 0 and (i - pos.entry_bar_index) >= 3:
                        exit_px = self._apply_slippage(price, "SELL")
                        notional = pos.qty_remaining * exit_px
                        fee = self._fee(notional)
                        pnl = (exit_px - pos.entry_price) * pos.qty_remaining - fee

                        equity += pnl
                        closed_trades += 1
                        if pnl > 0:
                            wins += 1
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                        trade_returns.append(pnl / max(1e-12, initial_equity))
                        pos = None
                        last_exit_bar = i

                    # time stop
                    if pos is not None and (i - pos.entry_bar_index) >= p.max_bars_hold:
                        exit_px = self._apply_slippage(price, "SELL")
                        notional = pos.qty_remaining * exit_px
                        fee = self._fee(notional)
                        pnl = (exit_px - pos.entry_price) * pos.qty_remaining - fee

                        equity += pnl
                        closed_trades += 1
                        if pnl > 0:
                            wins += 1
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                        trade_returns.append(pnl / max(1e-12, initial_equity))
                        pos = None
                        last_exit_bar = i

            # --- Entry if flat ---
            if pos is None:
                # cooldown after exit
                if (i - last_exit_bar) < p.cooldown_bars_after_exit:
                    eq_curve.append((ts, equity))
                    peak = max(peak, equity)
                    max_dd = max(max_dd, equity_drawdown())
                    continue

                # hard backtest “damage” cap (avoid unrealistic recovery)
                if equity_drawdown() >= 0.35:
                    eq_curve.append((ts, equity))
                    break

                es = self.strategy.entry_signal(pair, entry, i, regime, equity_drawdown(), p)
                if es.action == "BUY" and es.confidence >= p.confidence_min:
                    # Size by risk
                    risk_amount = equity * clamp(self.cfg.risk_per_trade, 0.0005, 0.05)
                    stop_dist = max(1e-9, (price - es.stop_price))
                    qty = risk_amount / stop_dist

                    lev = clamp(es.leverage, 1.0, self.cfg.leverage_max)
                    entry_px = self._apply_slippage(price, "BUY")
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
                        entry_bar_index=i,
                        took_partial=False,
                        qty_remaining=qty,
                        trail_stop=0.0,
                    )

            eq_curve.append((ts, equity))
            peak = max(peak, equity)
            max_dd = max(max_dd, equity_drawdown())

        total_return = (equity - initial_equity) / max(1e-12, initial_equity)
        win_rate = (wins / closed_trades) if closed_trades > 0 else 0.0
        profit_factor = (gross_profit / max(1e-12, gross_loss)) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        avg_tr = (sum(trade_returns) / len(trade_returns)) if trade_returns else 0.0

        start_ms = entry.ts[0] if entry.ts else 0
        end_ms = entry.ts[-1] if entry.ts else 0

        notes = "OK"
        if closed_trades < 12:
            notes = "Low trades"
        if total_return < 0:
            notes = "Negative"

        return BacktestResult(
            pair=pair,
            entry_tf=entry.tf,
            signal_tf=signal.tf,
            start_ms=start_ms,
            end_ms=end_ms,
            trades=closed_trades,
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
                "strategy_params": dataclasses.asdict(p),
            }
        )

    def walk_forward(
        self,
        pair: str,
        entry: CandleSeries,
        signal: CandleSeries,
        regime_series: list[RegimeState],
        params: StrategyParamsV32,
        initial_equity: float = 100.0
    ) -> BacktestResult:
        n = len(entry.ts)
        if n < 1200:
            return self.run(pair, entry, signal, regime_series, params, initial_equity)

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

            res = self.run(pair, e_seg, signal, regime_series, params, eq)
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
                "strategy_params": dataclasses.asdict(params.sanitize()),
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

        try:
            payload = {
                "generated_utc": now_utc().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "context": context,
                "results": [r.to_dict() for r in results],
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.exception("Failed writing JSON report", e)

        try:
            top_lines = []
            top_lines.append(f"{APP_NAME} {APP_VERSION} BACKTEST SUMMARY")
            top_lines.append(f"Generated: {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            for k, v in context.items():
                if k == "strategy_params_json":
                    continue
                top_lines.append(f"{k}: {v}")
            spj = context.get("strategy_params_json", "")
            if spj:
                top_lines.append("")
                top_lines.append("Strategy Params (v0.3.2):")
                top_lines.append(spj)
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
# Paper Portfolio (GUI engine loop)
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

        self.strategy = StrategyV32(cfg, log)
        self.backtester = BacktesterV32(cfg, log, self.strategy)

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._client_ro = LunoClient(cfg, log, use_live=False)
        self._client_live = LunoClient(cfg, log, use_live=True)

        self._pairs_cache: list[str] = []
        self._last_health: str = "INIT"

        self.paper = PaperPortfolio(equity=100.0, cash=100.0)

        # Strategy params persisted in DB (GUI writes here)
        self.params = self._load_or_init_params()

    def _load_or_init_params(self) -> StrategyParamsV32:
        raw = self.storage.get_gate(GATE_STRAT_PARAMS_JSON, "")
        if raw.strip():
            p = StrategyParamsV32.from_json(raw).sanitize()
            self.log.info("Loaded strategy params from DB.")
            return p
        p = StrategyParamsV32().sanitize()
        self.storage.set_gate(GATE_STRAT_PARAMS_JSON, p.to_json())
        self.log.info("Initialized default strategy params in DB.")
        return p

    def save_params(self, params: StrategyParamsV32) -> StrategyParamsV32:
        p = params.sanitize()
        self.params = p
        self.storage.set_gate(GATE_STRAT_PARAMS_JSON, p.to_json())
        self.log.info("Saved strategy params to DB.")
        return p

    def live_unlocked(self) -> bool:
        unlock_file = self.cfg.data_dir / LIVE_MODE_UNLOCK_FILE
        gate = self.storage.get_gate(GATE_LIVE_GATE, "0")
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
        for _ in range(1200):
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

    def run_backtest(self, pairs: list[str], entry_tf: str, days: int, initial_equity: float, walk_forward: bool) -> list[BacktestResult]:
        entry_duration = timeframe_to_seconds(entry_tf)
        signal_tf = self.cfg.signal_timeframe
        signal_duration = timeframe_to_seconds(signal_tf)

        end_ms = utc_ms()
        start_ms = end_ms - int(days * 86400 * 1000)

        # snapshot params (thread-safe behavior for GUI edits mid-run)
        p = self.params.sanitize()

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

                regimes = self.strategy.compute_regime(signal_series, p)

                if walk_forward:
                    res = self.backtester.walk_forward(pair, entry_series, signal_series, regimes, p, initial_equity=initial_equity)
                else:
                    res = self.backtester.run(pair, entry_series, signal_series, regimes, p, initial_equity=initial_equity)
                results.append(res)
            except Exception as e:
                self.log.exception(f"Backtest failed for {pair}", e)

        results.sort(key=lambda r: r.total_return, reverse=True)
        return results

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
                    # v0.3.x keeps live/paper execution minimal. Backtest is the main optimizer.
                    # (Future versions can promote backtest-validated logic into paper execution.)
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
        self.geometry("1320x820")
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

        self._build_style()
        self._build_widgets()

        self.after(250, self._poll_logs)
        self.after(500, self._refresh_status)
        self.after(500, self._drain_ui_queue)

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
        ttk.Label(top, textvariable=self.var_health, width=28).pack(side=tk.LEFT, padx=(4, 12))

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

        # LEFT
        self.tbl_positions = self._make_table(
            left,
            title="Open Positions (Paper)",
            columns=[("pair", 90), ("qty", 120), ("entry", 120), ("age", 140)]
        )
        self.tbl_positions.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.tbl_rank = self._make_table(
            left,
            title="Latest Decisions (DB)",
            columns=[("time", 170), ("pair", 90), ("action", 90), ("conf", 70), ("reason", 420)]
        )
        self.tbl_rank.pack(fill=tk.BOTH, expand=True)

        # RIGHT
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

        # Strategy params (GRID ONLY inside this frame)
        params_box = ttk.LabelFrame(right, text="Strategy Parameters (v0.3.2) - GUI Controlled", padding=10)
        params_box.pack(fill=tk.X, pady=(0, 8))

        self._build_params_panel(params_box)

        # --- GRID separator (do not use pack here) ---
        sep = ttk.Separator(params_box, orient=tk.HORIZONTAL)
        sep.grid(row=99, column=0, columnspan=4, sticky="ew", pady=10)

        # Buttons in their own child frame using GRID
        btnp = ttk.Frame(params_box)
        btnp.grid(row=100, column=0, columnspan=4, sticky="ew")
        ttk.Button(btnp, text="Apply & Save Params", command=self._apply_save_params).grid(row=0, column=0, padx=(0, 8), sticky="w")
        ttk.Button(btnp, text="Reload Saved Params", command=self._reload_params_from_engine).grid(row=0, column=1, padx=(0, 8), sticky="w")
        ttk.Button(btnp, text="Reset to Defaults", command=self._reset_params_defaults).grid(row=0, column=2, sticky="w")

        # Backtest
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
            "Strategy parameters are editable via GUI and persisted locally.\n"
            "Use read-only keys for data access and testing.\n"
        )
        messagebox.showinfo(f"{APP_NAME}", msg)

    # ---------- Params panel ----------
    def _build_params_panel(self, parent: ttk.LabelFrame) -> None:
        # Grid layout: 4 columns (label, entry) x 2
        for col in range(4):
            parent.grid_columnconfigure(col, weight=1 if col in (1, 3) else 0)

        p = self.engine.params.sanitize()

        self._param_vars: dict[str, tk.StringVar] = {}

        def add_row(row: int, key: str, label: str, value: t.Any, col_offset: int = 0) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=0 + col_offset, sticky="w", padx=(0, 8), pady=2)
            var = tk.StringVar(value=str(value))
            self._param_vars[key] = var
            ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1 + col_offset, sticky="ew", pady=2)

        # Left block
        r = 0
        add_row(r, "z_lookback", "z_lookback", p.z_lookback, 0); add_row(r, "z_reentry", "z_reentry", p.z_reentry, 2); r += 1
        add_row(r, "z_neutral_band", "z_neutral_band", p.z_neutral_band, 0); add_row(r, "z_exit_revert", "z_exit_revert", p.z_exit_revert, 2); r += 1
        add_row(r, "vwap_lookback", "vwap_lookback", p.vwap_lookback, 0); add_row(r, "bb_len", "bb_len", p.bb_len, 2); r += 1
        add_row(r, "bb_mult", "bb_mult", p.bb_mult, 0); add_row(r, "squeeze_bw_lookback", "squeeze_bw_lookback", p.squeeze_bw_lookback, 2); r += 1
        add_row(r, "squeeze_bw_mult", "squeeze_bw_mult", p.squeeze_bw_mult, 0); add_row(r, "atr_len", "atr_len", p.atr_len, 2); r += 1
        add_row(r, "atr_expand_mult", "atr_expand_mult", p.atr_expand_mult, 0); add_row(r, "atr_stop_mult", "atr_stop_mult", p.atr_stop_mult, 2); r += 1
        add_row(r, "atr_min_pct", "atr_min_pct", p.atr_min_pct, 0); add_row(r, "atr_max_pct", "atr_max_pct", p.atr_max_pct, 2); r += 1
        add_row(r, "rsi_len", "rsi_len", p.rsi_len, 0); add_row(r, "rsi_min_for_long", "rsi_min_for_long", p.rsi_min_for_long, 2); r += 1
        add_row(r, "rsi_overheat", "rsi_overheat", p.rsi_overheat, 0); add_row(r, "adx_len", "adx_len", p.adx_len, 2); r += 1
        add_row(r, "adx_min_rising", "adx_min_rising", p.adx_min_rising, 0); add_row(r, "adx_max_entry", "adx_max_entry", p.adx_max_entry, 2); r += 1
        add_row(r, "momentum_confirm_bars", "momentum_confirm_bars", p.momentum_confirm_bars, 0); add_row(r, "cooldown_bars_after_exit", "cooldown_bars_after_exit", p.cooldown_bars_after_exit, 2); r += 1
        add_row(r, "confidence_min", "confidence_min", p.confidence_min, 0); add_row(r, "max_bars_hold", "max_bars_hold", p.max_bars_hold, 2); r += 1
        add_row(r, "take_partial_at_r", "take_partial_at_r", p.take_partial_at_r, 0); add_row(r, "partial_qty_frac", "partial_qty_frac", p.partial_qty_frac, 2); r += 1
        add_row(r, "trail_atr_mult", "trail_atr_mult", p.trail_atr_mult, 0); add_row(r, "min_r_vs_cost_mult", "min_r_vs_cost_mult", p.min_r_vs_cost_mult, 2); r += 1
        add_row(r, "lev_base", "lev_base", p.lev_base, 0); add_row(r, "lev_max", "lev_max", p.lev_max, 2); r += 1
        add_row(r, "lev_scale_by_signal", "lev_scale_by_signal", p.lev_scale_by_signal, 0);  # last row left only

    def _vars_to_params(self) -> StrategyParamsV32:
        # Parse with safe conversion
        def gi(k: str, d: int) -> int:
            return safe_int(self._param_vars.get(k, tk.StringVar(value=str(d))).get(), d)

        def gf(k: str, d: float) -> float:
            return safe_float(self._param_vars.get(k, tk.StringVar(value=str(d))).get(), d)

        p = StrategyParamsV32(
            z_lookback=gi("z_lookback", 60),
            z_reentry=gf("z_reentry", 0.55),
            z_neutral_band=gf("z_neutral_band", 0.35),
            z_exit_revert=gf("z_exit_revert", 0.15),
            vwap_lookback=gi("vwap_lookback", 80),
            bb_len=gi("bb_len", 60),
            bb_mult=gf("bb_mult", 1.6),
            squeeze_bw_lookback=gi("squeeze_bw_lookback", 120),
            squeeze_bw_mult=gf("squeeze_bw_mult", 0.80),
            atr_len=gi("atr_len", 14),
            atr_expand_mult=gf("atr_expand_mult", 1.15),
            atr_min_pct=gf("atr_min_pct", 0.0010),
            atr_max_pct=gf("atr_max_pct", 0.055),
            atr_stop_mult=gf("atr_stop_mult", 1.35),
            rsi_len=gi("rsi_len", 7),
            rsi_min_for_long=gf("rsi_min_for_long", 52.0),
            rsi_overheat=gf("rsi_overheat", 78.0),
            adx_len=gi("adx_len", 14),
            adx_min_rising=gf("adx_min_rising", 14.0),
            adx_max_entry=gf("adx_max_entry", 38.0),
            momentum_confirm_bars=gi("momentum_confirm_bars", 3),
            cooldown_bars_after_exit=gi("cooldown_bars_after_exit", 6),
            confidence_min=gf("confidence_min", 0.48),
            max_bars_hold=gi("max_bars_hold", 140),
            take_partial_at_r=gf("take_partial_at_r", 0.70),
            partial_qty_frac=gf("partial_qty_frac", 0.40),
            trail_atr_mult=gf("trail_atr_mult", 1.80),
            min_r_vs_cost_mult=gf("min_r_vs_cost_mult", 1.10),
            lev_base=gf("lev_base", 1.0),
            lev_max=gf("lev_max", 1.6),
            lev_scale_by_signal=gf("lev_scale_by_signal", 0.8),
        )
        return p.sanitize()

    def _apply_save_params(self) -> None:
        try:
            p = self._vars_to_params()
            self.engine.save_params(p)
            messagebox.showinfo("Strategy Params", "Saved strategy parameters.")
        except Exception as e:
            messagebox.showerror("Strategy Params", f"Failed to save: {e}")

    def _reload_params_from_engine(self) -> None:
        try:
            p = StrategyParamsV32.from_json(self.storage.get_gate(GATE_STRAT_PARAMS_JSON, "")).sanitize()
            self.engine.params = p
            for k, v in dataclasses.asdict(p).items():
                if k in self._param_vars:
                    self._param_vars[k].set(str(v))
            messagebox.showinfo("Strategy Params", "Reloaded saved parameters from DB.")
        except Exception as e:
            messagebox.showerror("Strategy Params", f"Failed to reload: {e}")

    def _reset_params_defaults(self) -> None:
        try:
            p = StrategyParamsV32().sanitize()
            self.engine.save_params(p)
            for k, v in dataclasses.asdict(p).items():
                if k in self._param_vars:
                    self._param_vars[k].set(str(v))
            messagebox.showinfo("Strategy Params", "Reset to defaults.")
        except Exception as e:
            messagebox.showerror("Strategy Params", f"Failed to reset: {e}")

    # ---------- Controls ----------
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

    # ---------- Backtest ----------
    def _compose_backtest_summary_text(self) -> str:
        ctx = self._last_backtest_context or {}
        lines: list[str] = []
        lines.append(f"{APP_NAME} {APP_VERSION} BACKTEST SUMMARY")
        lines.append(f"Generated: {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        for k, v in ctx.items():
            if k == "strategy_params_json":
                continue
            lines.append(f"{k}: {v}")
        spj = ctx.get("strategy_params_json", "")
        if spj:
            lines.append("")
            lines.append("Strategy Params (v0.3.2):")
            lines.append(spj)
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

        # Always save GUI params before running, so the backtest matches UI
        try:
            self.engine.save_params(self._vars_to_params())
        except Exception as e:
            messagebox.showwarning("Strategy Params", f"Could not save params; using last saved. ({e})")

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
                    "strategy_params_json": self.engine.params.sanitize().to_json(),
                }
                self._last_backtest_results = res
                self._last_backtest_context = ctx
                self._last_backtest_report_paths = {}

                self._ui_queue.put(lambda: self._render_backtest_results(res))

                # Conservative live gate (still not executing live in v0.3.2):
                if res:
                    top = res[0]
                    gate_ok = (top.total_return > 0.10 and top.max_drawdown < 0.20 and top.trades >= 25 and top.win_rate >= 0.42)
                    self.engine.storage.set_gate(GATE_LIVE_GATE, "1" if gate_ok else "0")
                    self.log.info(
                        f"Gate eval: top_return={fmt_pct(top.total_return)} dd={fmt_pct(top.max_drawdown)} "
                        f"trades={top.trades} win={top.win_rate*100:.1f}% -> LIVE_GATE={int(gate_ok)}"
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

    # ---------- Logs / status ----------
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
