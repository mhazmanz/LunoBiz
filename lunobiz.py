"""
LunoBiz - Single-file Windows application
- GUI dashboard (no charts; values and tables only)
- Backtesting + walk-forward validation
- Paper trading engine (default)
- Live trading scaffold with strict safety locks

Public-repo safe:
- No secrets hard-coded
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
APP_VERSION = "0.1.1"
DEFAULT_API_BASE = "https://api.luno.com"

# Luno candles endpoint:
# GET /api/exchange/1/candles?pair=...&since=...&duration=...
CANDLES_PATH = "/api/exchange/1/candles"
TICKER_PATH = "/api/1/ticker"
TICKERS_PATH = "/api/1/tickers"
ORDERBOOK_PATH = "/api/1/orderbook"
BALANCE_PATH = "/api/1/balance"
LIST_TRADES_PATH = "/api/1/trades"
LIST_ORDERS_PATH = "/api/1/listorders"
CREATE_ORDER_PATH = "/api/1/postorder"
STOP_ORDER_PATH = "/api/1/stoporder"

# Safety: never live trade unless explicitly enabled and gates pass
LIVE_MODE_UNLOCK_FILE = "LIVE_UNLOCK.ok"  # local file toggle (not tracked)
KILL_SWITCH_FILE = "KILL_SWITCH"          # if present => trading disabled immediately


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


def file_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def is_kill_switch_on(data_dir: Path) -> bool:
    return file_exists(data_dir / KILL_SWITCH_FILE)


def open_folder_in_explorer(folder: Path) -> None:
    """
    Opens folder on Windows/macOS/Linux. Best effort; never raises.
    """
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

    risk_per_trade: float
    daily_loss_cap: float
    max_open_positions: int
    cooldown_minutes: int

    backtest_timeframe: str
    backtest_slippage_bps: float
    backtest_fee_bps: float

    data_dir: Path
    db_filename: str

    poll_interval_seconds: int
    http_timeout_seconds: int = 15
    http_max_retries: int = 4

    def is_paper(self) -> bool:
        return self.app_mode.strip().upper() == "PAPER"

    def is_live(self) -> bool:
        return self.app_mode.strip().upper() == "LIVE"


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


def load_config(repo_root: Path, log: "AppLogger") -> AppConfig:
    dotenv_path = repo_root / ".env"
    dotenv = load_dotenv(dotenv_path)
    for k, v in dotenv.items():
        if os.environ.get(k) is None:
            os.environ[k] = v

    def envf(k: str, default: str = "") -> str:
        return get_env(k, default)

    data_dir = Path(envf("DATA_DIR", "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    cfg = AppConfig(
        luno_api_key_readonly=envf("LUNO_API_KEY_READONLY", ""),
        luno_api_secret_readonly=envf("LUNO_API_SECRET_READONLY", ""),
        luno_api_key_live=envf("LUNO_API_KEY_LIVE", ""),
        luno_api_secret_live=envf("LUNO_API_SECRET_LIVE", ""),

        telegram_bot_token=envf("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=envf("TELEGRAM_CHAT_ID", ""),

        app_mode=envf("APP_MODE", "PAPER"),

        default_view_pair=envf("DEFAULT_VIEW_PAIR", "ETHMYR"),
        scan_pairs_csv=envf("SCAN_PAIRS", ""),

        risk_per_trade=safe_float(envf("RISK_PER_TRADE", "0.005"), 0.005),
        daily_loss_cap=safe_float(envf("DAILY_LOSS_CAP", "0.02"), 0.02),
        max_open_positions=max(1, safe_int(envf("MAX_OPEN_POSITIONS", "2"), 2)),
        cooldown_minutes=max(1, safe_int(envf("COOLDOWN_MINUTES", "240"), 240)),

        backtest_timeframe=envf("BACKTEST_TIMEFRAME", "5m"),
        backtest_slippage_bps=safe_float(envf("BACKTEST_SLIPPAGE_BPS", "8"), 8.0),
        backtest_fee_bps=safe_float(envf("BACKTEST_FEE_BPS", "30"), 30.0),

        data_dir=data_dir,
        db_filename=envf("DB_FILENAME", "lunobiz.sqlite3"),

        poll_interval_seconds=max(2, safe_int(envf("POLL_INTERVAL_SECONDS", "10"), 10)),
        http_timeout_seconds=max(5, safe_int(envf("HTTP_TIMEOUT_SECONDS", "15"), 15)),
        http_max_retries=max(1, safe_int(envf("HTTP_MAX_RETRIES", "4"), 4)),
    )

    ensure_dir(cfg.data_dir)
    log.info(f"Config loaded. mode={cfg.app_mode.upper()} data_dir={cfg.data_dir}")
    return cfg


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

                CREATE TABLE IF NOT EXISTS metrics_daily (
                    day_utc TEXT PRIMARY KEY,
                    equity_start REAL NOT NULL,
                    equity_end REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    notes TEXT NOT NULL
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

    def orderbook(self, pair: str) -> dict[str, t.Any]:
        return self._request("GET", ORDERBOOK_PATH, params={"pair": pair})

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


# =========================
# Strategy / Scoring
# =========================

@dataclass
class Signal:
    pair: str
    action: str
    confidence: float
    reason: str
    stop_price: float | None = None
    take_profit_price: float | None = None
    meta: dict[str, t.Any] = dataclasses.field(default_factory=dict)


class StrategyEngine:
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log

    def generate_signal_from_candles(self, pair: str, candles: list[dict[str, float]]) -> Signal:
        if len(candles) < 80:
            return Signal(pair=pair, action="HOLD", confidence=0.0, reason="Insufficient candle history")

        close = [c["close"] for c in candles]
        high_ = [c["high"] for c in candles]
        low_ = [c["low"] for c in candles]

        ema_fast = ema(close, 12)
        ema_slow = ema(close, 26)
        r = rsi(close, 14)
        a = atr(high_, low_, close, 14)

        i = len(close) - 1
        price = close[i]
        trend = (ema_fast[i] - ema_slow[i]) / max(1e-9, ema_slow[i])
        vol = a[i] / max(1e-9, price)
        mom = (close[i] - close[i - 10]) / max(1e-9, close[i - 10])

        confidence = 0.0
        action = "HOLD"
        reason = "No edge detected"

        if trend > 0.002 and mom > 0.003 and r[i] < 78:
            action = "BUY"
            confidence = clamp(0.55 + trend * 40 + mom * 30 - vol * 2, 0.0, 0.95)
            reason = "Trend+momentum alignment"
        elif trend < -0.002 and mom < -0.003 and r[i] > 22:
            action = "SELL"
            confidence = clamp(0.55 + abs(trend) * 40 + abs(mom) * 30 - vol * 2, 0.0, 0.95)
            reason = "Downtrend detected (exit/avoid)"

        stop = None
        tp = None
        if action == "BUY":
            stop = price - 2.2 * a[i]
            tp = price + 3.2 * a[i]
        elif action == "SELL":
            stop = price + 2.2 * a[i]
            tp = price - 3.2 * a[i]

        meta = {
            "price": price,
            "ema_fast": ema_fast[i],
            "ema_slow": ema_slow[i],
            "trend": trend,
            "rsi": r[i],
            "atr": a[i],
            "volatility": vol,
            "momentum_10": mom,
        }
        return Signal(pair=pair, action=action, confidence=float(confidence), reason=reason, stop_price=stop, take_profit_price=tp, meta=meta)


# =========================
# Backtesting
# =========================

@dataclass
class BacktestResult:
    pair: str
    timeframe: str
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

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "pair": self.pair,
            "timeframe": self.timeframe,
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
            "equity_curve": [{"ts_ms": ts, "ts_utc": human_ts(ts), "equity": eq} for ts, eq in self.equity_curve],
        }

    def summary_line(self) -> str:
        return (
            f"{self.pair} tf={self.timeframe} trades={self.trades} "
            f"win={self.win_rate*100:.1f}% ret={self.total_return*100:.2f}% "
            f"dd={self.max_drawdown*100:.2f}% notes={self.notes}"
        )


class Backtester:
    def __init__(self, cfg: AppConfig, log: AppLogger, storage: Storage, strategy: StrategyEngine):
        self.cfg = cfg
        self.log = log
        self.storage = storage
        self.strategy = strategy

    def _apply_costs(self, price: float, side: str) -> float:
        slip = self.cfg.backtest_slippage_bps / 10000.0
        if side.upper() == "BUY":
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    def _fee(self, notional: float) -> float:
        return notional * (self.cfg.backtest_fee_bps / 10000.0)

    def run_single_pair(self, pair: str, candles: list[dict[str, float]], initial_equity: float = 100.0) -> BacktestResult:
        if len(candles) < 100:
            return BacktestResult(pair, self.cfg.backtest_timeframe, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "Insufficient candles", [])

        equity = float(initial_equity)
        peak = equity
        max_dd = 0.0

        in_pos = False
        entry_price = 0.0
        qty = 0.0
        trade_returns: list[float] = []
        gross_profit = 0.0
        gross_loss = 0.0

        equity_curve: list[tuple[int, float]] = []
        trades = 0
        wins = 0

        for i in range(80, len(candles)):
            window = candles[:i + 1]
            ts_ms = int(window[-1]["ts_ms"])
            price = float(window[-1]["close"])

            sig = self.strategy.generate_signal_from_candles(pair, window)

            if in_pos:
                if sig.action == "SELL" and sig.confidence > 0.55:
                    exit_px = self._apply_costs(price, "SELL")
                    notional = qty * exit_px
                    fee = self._fee(notional)
                    pnl = (exit_px - entry_price) * qty - fee
                    ret = pnl / max(1e-9, equity)
                    equity += pnl
                    trades += 1
                    if pnl > 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    trade_returns.append(ret)
                    in_pos = False
                    qty = 0.0
                else:
                    if price >= entry_price * 1.015:
                        exit_px = self._apply_costs(price, "SELL")
                        notional = qty * exit_px
                        fee = self._fee(notional)
                        pnl = (exit_px - entry_price) * qty - fee
                        ret = pnl / max(1e-9, equity)
                        equity += pnl
                        trades += 1
                        if pnl > 0:
                            wins += 1
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                        trade_returns.append(ret)
                        in_pos = False
                        qty = 0.0
                    elif price <= entry_price * 0.990:
                        exit_px = self._apply_costs(price, "SELL")
                        notional = qty * exit_px
                        fee = self._fee(notional)
                        pnl = (exit_px - entry_price) * qty - fee
                        ret = pnl / max(1e-9, equity)
                        equity += pnl
                        trades += 1
                        if pnl > 0:
                            wins += 1
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                        trade_returns.append(ret)
                        in_pos = False
                        qty = 0.0

            if not in_pos and sig.action == "BUY" and sig.confidence >= 0.60:
                alloc = clamp(self.cfg.risk_per_trade * 10.0, 0.01, 0.35)
                cash_to_use = equity * alloc
                entry_px = self._apply_costs(price, "BUY")
                if entry_px > 0 and cash_to_use > 1e-6:
                    qty = (cash_to_use / entry_px)
                    notional = qty * entry_px
                    fee = self._fee(notional)
                    equity -= fee
                    entry_price = entry_px
                    in_pos = True

            equity_curve.append((ts_ms, equity))
            peak = max(peak, equity)
            dd = (peak - equity) / max(1e-9, peak)
            max_dd = max(max_dd, dd)

        total_return = (equity - initial_equity) / max(1e-9, initial_equity)
        win_rate = wins / trades if trades > 0 else 0.0
        profit_factor = (gross_profit / max(1e-9, gross_loss)) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        avg_tr = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0

        start_ms = int(candles[0]["ts_ms"])
        end_ms = int(candles[-1]["ts_ms"])

        return BacktestResult(
            pair=pair,
            timeframe=self.cfg.backtest_timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            trades=trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            avg_trade_return=avg_tr,
            notes="OK",
            equity_curve=equity_curve,
        )

    def walk_forward(self, pair: str, candles: list[dict[str, float]], initial_equity: float = 100.0) -> BacktestResult:
        if len(candles) < 400:
            return self.run_single_pair(pair, candles, initial_equity)

        n_windows = 5
        window = len(candles) // n_windows
        eq = initial_equity
        all_curve: list[tuple[int, float]] = []
        total_trades = 0
        weighted_wins = 0.0
        worst_dd = 0.0
        trade_returns: list[float] = []

        for w in range(n_windows):
            seg = candles[w * window: (w + 1) * window] if w < n_windows - 1 else candles[w * window:]
            res = self.run_single_pair(pair, seg, eq)
            total_trades += res.trades
            weighted_wins += res.win_rate * res.trades
            worst_dd = max(worst_dd, res.max_drawdown)
            eq = eq * (1.0 + res.total_return)
            all_curve.extend(res.equity_curve)
            trade_returns.append(res.avg_trade_return)

        win_rate = (weighted_wins / total_trades) if total_trades > 0 else 0.0
        avg_tr = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
        total_return = (eq - initial_equity) / max(1e-9, initial_equity)

        return BacktestResult(
            pair=pair,
            timeframe=self.cfg.backtest_timeframe,
            start_ms=int(candles[0]["ts_ms"]),
            end_ms=int(candles[-1]["ts_ms"]),
            trades=total_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=worst_dd,
            profit_factor=0.0,
            avg_trade_return=avg_tr,
            notes="Walk-forward OK",
            equity_curve=all_curve,
        )


# =========================
# Paper Portfolio State
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

        self.strategy = StrategyEngine(cfg, log)
        self.backtester = Backtester(cfg, log, storage, self.strategy)

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._client_ro = LunoClient(cfg, log, use_live=False)
        self._client_live = LunoClient(cfg, log, use_live=True)

        self._pairs_cache: list[str] = []
        self._last_health: str = "INIT"

        self.paper = PaperPortfolio(equity=100.0, cash=100.0)

    def live_unlocked(self) -> bool:
        unlock_file = self.cfg.data_dir / LIVE_MODE_UNLOCK_FILE
        gate = self.storage.get_gate("LIVE_GATE", "0")
        return unlock_file.exists() and gate == "1"

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
        while since < end_ms:
            data = self._client_ro.candles(pair=pair, since_ms=since, duration_sec=duration_sec)
            candles = data.get("candles") or []
            if not isinstance(candles, list) or not candles:
                break
            n = self.storage.upsert_candles(pair, duration_sec, candles)
            total += n
            last_ts = safe_int(candles[-1].get("timestamp"), since)
            since = max(last_ts + duration_sec * 1000, since + duration_sec * 1000)
            time.sleep(0.15)
            if n < 5:
                break
        return total

    def load_candles_for_backtest(self, pair: str, duration_sec: int, start_ms: int, end_ms: int) -> list[dict[str, float]]:
        rows = self.storage.fetch_candles(pair, duration_sec, start_ms, end_ms)
        out: list[dict[str, float]] = []
        for r in rows:
            out.append({
                "ts_ms": int(r["ts_ms"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            })
        return out

    def run_backtest(self, pairs: list[str], tf: str, days: int, initial_equity: float, walk_forward: bool) -> list[BacktestResult]:
        duration_sec = timeframe_to_seconds(tf)
        end_ms = utc_ms()
        start_ms = end_ms - int(days * 86400 * 1000)

        results: list[BacktestResult] = []
        for pair in pairs:
            pair = pair.upper().strip()
            if not pair:
                continue
            self.log.info(f"Backtest ingest candles: {pair} tf={tf} days={days}")
            try:
                n = self.ingest_candles(pair, duration_sec, start_ms, end_ms)
                self.log.info(f"Ingested {n} candles into cache for {pair}")
                candles = self.load_candles_for_backtest(pair, duration_sec, start_ms, end_ms)
                if walk_forward:
                    res = self.backtester.walk_forward(pair, candles, initial_equity=initial_equity)
                else:
                    res = self.backtester.run_single_pair(pair, candles, initial_equity=initial_equity)
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
                    scan_pairs = self.get_scan_pairs()[:10]
                    self._paper_scan_and_maybe_trade(scan_pairs)
                else:
                    if self.allow_live():
                        self._last_health = "LIVE_UNLOCKED (not executing in v1)"
                    else:
                        self._last_health = "LIVE_LOCKED"

            except Exception as e:
                self.log.exception("Engine loop exception", e)
                self._last_health = "ERROR"

            time.sleep(self.cfg.poll_interval_seconds)

    def health(self) -> str:
        return self._last_health

    def _paper_reset_day_if_needed(self) -> None:
        day = now_utc().strftime("%Y-%m-%d")
        if self.paper.day_utc != day:
            self.paper.day_utc = day
            self.paper.day_start_equity = self.paper.equity
            self.log.info(f"New UTC day: {day}. day_start_equity={self.paper.day_start_equity:.2f}")

    def _paper_daily_pnl_pct(self) -> float:
        return (self.paper.equity - self.paper.day_start_equity) / max(1e-9, self.paper.day_start_equity)

    def _paper_scan_and_maybe_trade(self, pairs: list[str]) -> None:
        if self._paper_daily_pnl_pct() <= -abs(self.cfg.daily_loss_cap):
            self.notifier.notify("Daily halt", f"Daily loss cap hit. pnl={fmt_pct(self._paper_daily_pnl_pct())}. Cooling down.")
            time.sleep(self.cfg.cooldown_minutes * 60)
            return

        if len(self.paper.positions) >= self.cfg.max_open_positions:
            return

        ranked: list[tuple[float, str]] = []
        duration_sec = timeframe_to_seconds(self.cfg.backtest_timeframe)
        end_ms = utc_ms()
        start_ms = end_ms - int(3 * 86400 * 1000)

        for p in pairs:
            try:
                self.ingest_candles(p, duration_sec, start_ms, end_ms)
                cs = self.load_candles_for_backtest(p, duration_sec, start_ms, end_ms)
                if len(cs) < 80:
                    continue
                mom = (cs[-1]["close"] - cs[-10]["close"]) / max(1e-9, cs[-10]["close"])
                ranked.append((mom, p))
            except Exception:
                continue

        ranked.sort(key=lambda x: x[0], reverse=True)
        top = [p for _, p in ranked[:5]]

        for p in top:
            if p in self.paper.positions:
                continue
            cs = self.load_candles_for_backtest(p, duration_sec, start_ms, end_ms)
            sig = self.strategy.generate_signal_from_candles(p, cs)
            self.storage.add_decision(
                ts_ms=utc_ms(),
                pair=p,
                action=sig.action,
                confidence=sig.confidence,
                reason=sig.reason,
                meta=sig.meta
            )
            if sig.action == "BUY" and sig.confidence >= 0.65:
                price = float(sig.meta.get("price", cs[-1]["close"]))
                alloc = clamp(self.cfg.risk_per_trade * 10.0, 0.02, 0.35)
                cash_to_use = self.paper.cash * alloc
                if cash_to_use <= 0:
                    continue
                qty = cash_to_use / max(1e-12, price)
                fee = (cash_to_use * (self.cfg.backtest_fee_bps / 10000.0))
                self.paper.cash = max(0.0, self.paper.cash - cash_to_use - fee)
                pos = PaperPosition(pair=p, qty=qty, entry_price=price, entry_ts_ms=utc_ms())
                self.paper.positions[p] = pos
                self.log.info(f"PAPER BUY {p} qty={qty:.8f} price={price:.4f} alloc={fmt_pct(alloc)}")
                self.notifier.notify("PAPER BUY", f"{p} @ {price:.4f} qty={qty:.6f} conf={sig.confidence:.2f}")
                self.storage.add_paper_trade(
                    ts_ms=utc_ms(),
                    pair=p,
                    side="BUY",
                    price=price,
                    qty=qty,
                    fee=fee,
                    equity_after=self.paper.equity,
                    meta={"alloc": alloc, "confidence": sig.confidence, "reason": sig.reason}
                )
                break

        to_close: list[str] = []
        for p, pos in self.paper.positions.items():
            try:
                tkr = self._client_ro.ticker(p)
                price = safe_float(tkr.get("last_trade"), 0.0)
                if price <= 0:
                    continue
                if price >= pos.entry_price * 1.015 or price <= pos.entry_price * 0.990:
                    notional = pos.qty * price
                    fee = notional * (self.cfg.backtest_fee_bps / 10000.0)
                    pnl = (price - pos.entry_price) * pos.qty - fee
                    self.paper.cash += max(0.0, notional - fee)
                    self.paper.equity += pnl
                    to_close.append(p)
                    self.log.info(f"PAPER SELL {p} price={price:.4f} pnl={pnl:.2f} equity={self.paper.equity:.2f}")
                    self.notifier.notify("PAPER SELL", f"{p} @ {price:.4f} pnl={pnl:.2f} equity={self.paper.equity:.2f}")
                    self.storage.add_paper_trade(
                        ts_ms=utc_ms(),
                        pair=p,
                        side="SELL",
                        price=price,
                        qty=pos.qty,
                        fee=fee,
                        equity_after=self.paper.equity,
                        meta={"entry_price": pos.entry_price}
                    )
            except Exception:
                continue

        for p in to_close:
            self.paper.positions.pop(p, None)


# =========================
# Reporting (Backtest export)
# =========================

class ReportWriter:
    def __init__(self, data_dir: Path, log: AppLogger):
        self.data_dir = data_dir
        self.log = log
        self.reports_dir = data_dir / "reports"
        ensure_dir(self.reports_dir)

    def write_backtest_reports(
        self,
        results: list[BacktestResult],
        context: dict[str, t.Any],
    ) -> dict[str, Path]:
        """
        Writes:
        - CSV (summary rows)
        - JSON (full results incl equity curve)
        - TXT (small summary for pasting)
        Returns dict of produced paths.
        """
        stamp = now_utc().strftime("%Y%m%d_%H%M%S")
        csv_path = self.reports_dir / f"backtest_{stamp}.csv"
        json_path = self.reports_dir / f"backtest_{stamp}.json"
        txt_path = self.reports_dir / f"backtest_{stamp}_summary.txt"

        # CSV: summary only
        try:
            header = "pair,timeframe,start_utc,end_utc,trades,win_rate,total_return,max_drawdown,profit_factor,avg_trade_return,notes\n"
            lines = [header]
            for r in results:
                lines.append(
                    f"{r.pair},{r.timeframe},"
                    f"\"{human_ts(r.start_ms)}\",\"{human_ts(r.end_ms)}\","
                    f"{r.trades},{r.win_rate:.6f},{r.total_return:.6f},{r.max_drawdown:.6f},"
                    f"{r.profit_factor:.6f},{r.avg_trade_return:.6f},\"{r.notes}\"\n"
                )
            csv_path.write_text("".join(lines), encoding="utf-8")
        except Exception as e:
            self.log.exception("Failed writing CSV report", e)

        # JSON: context + full results
        try:
            payload = {
                "generated_utc": now_utc().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "context": context,
                "results": [r.to_dict() for r in results],
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.exception("Failed writing JSON report", e)

        # TXT: compact summary
        try:
            top_lines = []
            top_lines.append(f"{APP_NAME} {APP_VERSION} BACKTEST SUMMARY")
            top_lines.append(f"Generated: {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            for k, v in context.items():
                top_lines.append(f"{k}: {v}")
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
# GUI
# =========================

class Dashboard(tk.Tk):
    def __init__(self, repo_root: Path):
        super().__init__()
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1250x780")
        self.minsize(1050, 680)

        self.repo_root = repo_root

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

        # Backtest output cache
        self._last_backtest_results: list[BacktestResult] = []
        self._last_backtest_context: dict[str, t.Any] = {}
        self._last_backtest_report_paths: dict[str, Path] = {}

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
            columns=[("time", 160), ("pair", 90), ("action", 80), ("conf", 80), ("reason", 340)]
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

        bt = ttk.LabelFrame(right, text="Backtest", padding=10)
        bt.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(bt)
        row1.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(row1, text="Pairs (comma):").pack(side=tk.LEFT)
        self.var_bt_pairs = tk.StringVar(value=self.cfg.default_view_pair)
        ttk.Entry(row1, textvariable=self.var_bt_pairs).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        row2 = ttk.Frame(bt)
        row2.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row2, text="Timeframe:").pack(side=tk.LEFT)
        self.var_bt_tf = tk.StringVar(value=self.cfg.backtest_timeframe)
        ttk.Entry(row2, textvariable=self.var_bt_tf, width=8).pack(side=tk.LEFT, padx=8)

        ttk.Label(row2, text="Days:").pack(side=tk.LEFT)
        self.var_bt_days = tk.StringVar(value="30")
        ttk.Entry(row2, textvariable=self.var_bt_days, width=6).pack(side=tk.LEFT, padx=8)

        ttk.Label(row2, text="Initial equity:").pack(side=tk.LEFT)
        self.var_bt_equity = tk.StringVar(value="100")
        ttk.Entry(row2, textvariable=self.var_bt_equity, width=10).pack(side=tk.LEFT, padx=8)

        row3 = ttk.Frame(bt)
        row3.pack(fill=tk.X, pady=(0, 6))

        self.var_bt_walk = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="Walk-forward", variable=self.var_bt_walk).pack(side=tk.LEFT)

        ttk.Button(row3, text="Run Backtest", command=self._run_backtest).pack(side=tk.LEFT, padx=10)

        # New: export/copy buttons
        ttk.Button(row3, text="Copy Summary", command=self._copy_backtest_summary).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row3, text="Export Backtest Report", command=self._export_backtest_report).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row3, text="Open Reports Folder", command=self._open_reports_folder).pack(side=tk.LEFT)

        self.tbl_bt = self._make_table(
            right,
            title="Backtest Results (sorted by return)",
            columns=[("pair", 90), ("tf", 60), ("trades", 70), ("win", 70), ("ret", 90), ("dd", 80), ("notes", 230)]
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
            "PAPER mode is recommended until backtests and walk-forward validation pass.\n\n"
            "This software enforces safety controls but cannot guarantee daily returns or zero losses.\n"
            "Use read-only Luno API keys for testing.\n"
        )
        messagebox.showinfo(f"{APP_NAME}", msg)

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

    def _copy_backtest_summary(self) -> None:
        if not self._last_backtest_results:
            messagebox.showwarning("Copy Summary", "Run a backtest first.")
            return
        summary = self._compose_backtest_summary_text()
        try:
            self.clipboard_clear()
            self.clipboard_append(summary)
            self.update()  # keep clipboard after app exits
            messagebox.showinfo("Copy Summary", "Copied backtest summary to clipboard. Paste it into chat.")
        except Exception as e:
            messagebox.showerror("Copy Summary", f"Failed: {e}")

    def _compose_backtest_summary_text(self) -> str:
        ctx = self._last_backtest_context or {}
        lines = []
        lines.append(f"{APP_NAME} {APP_VERSION} BACKTEST SUMMARY")
        lines.append(f"Generated: {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        for k, v in ctx.items():
            lines.append(f"{k}: {v}")
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

        tf = self.var_bt_tf.get().strip()
        days = safe_int(self.var_bt_days.get().strip(), 30)
        eq = safe_float(self.var_bt_equity.get().strip(), 100.0)
        walk = bool(self.var_bt_walk.get())

        def worker() -> None:
            try:
                self.log.info(f"Backtest start: pairs={pairs} tf={tf} days={days} equity={eq} walk={walk}")
                res = self.engine.run_backtest(pairs=pairs, tf=tf, days=days, initial_equity=eq, walk_forward=walk)

                # cache results and context for export/copy
                ctx = {
                    "pairs": ",".join(pairs),
                    "timeframe": tf,
                    "days": days,
                    "initial_equity": eq,
                    "walk_forward": walk,
                    "slippage_bps": self.cfg.backtest_slippage_bps,
                    "fee_bps": self.cfg.backtest_fee_bps,
                }
                self._last_backtest_results = res
                self._last_backtest_context = ctx
                self._last_backtest_report_paths = {}

                self._ui_queue.put(lambda: self._render_backtest_results(res))

                # Gate example (conservative): keep locked unless strong
                if res:
                    top = res[0]
                    gate_ok = (top.total_return > 0.10 and top.max_drawdown < 0.20 and top.trades >= 10)
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
                r.timeframe,
                r.trades,
                f"{r.win_rate * 100:.1f}%",
                f"{r.total_return * 100:.1f}%",
                f"{r.max_drawdown * 100:.1f}%",
                r.notes
            ))

    def _poll_logs(self) -> None:
        for level, line in self.log.drain(200):
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
            self.var_daily_pnl.set(fmt_pct(self.engine._paper_daily_pnl_pct()))

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
    here = Path(__file__).resolve()
    return here.parent


def main() -> int:
    repo_root = find_repo_root()
    ensure_dir(repo_root / "data")
    app = Dashboard(repo_root)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
