"""
LunoBiz - Single-file Windows application (v0.3.0)

Key features:
- GUI dashboard (values & tables only)
- Robust candle caching in SQLite
- Backtesting with walk-forward validation
- Paper trading engine scaffold (paper-only)
- Paper-only leverage simulation (no live execution)
- Mean-reversion / volatility-expansion strategy (v0.3.0)
- Safety controls: kill switch, live gate (still locked by default)

Public-repo safe:
- No secrets in code
- Reads secrets from .env (git-ignored) or environment variables

Notes:
- This software is for research/testing. Markets are risky.
- Paper leverage simulation is only a simulation; live execution remains locked.
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
import zipfile
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
APP_VERSION = "0.3.0"
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


def safe_json_loads(s: str, default: t.Any) -> t.Any:
    try:
        return json.loads(s)
    except Exception:
        return default


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
    backtest_timeframe: str         # entry timeframe
    signal_timeframe: str           # higher timeframe for regime / volatility checks
    backtest_slippage_bps: float
    backtest_fee_bps: float

    # Paper leverage simulation
    leverage_enabled_paper: bool
    leverage_max: float             # hard cap
    leverage_min: float             # typically 1
    margin_alloc_cap: float         # cap margin usage fraction of equity per position
    maint_margin_ratio: float       # simplistic maintenance margin ratio

    # Strategy v0.3 (mean reversion / vol expansion)
    mr_window: int                  # rolling window for z-score / bands
    mr_z_enter: float               # enter when z <= -mr_z_enter (long-only)
    mr_z_exit: float                # exit when z >= -mr_z_exit (closer to mean, mr_z_exit < mr_z_enter)
    mr_bb_k: float                  # bollinger k
    mr_atr_period: int              # ATR period on entry TF
    mr_atr_stop_mult: float         # stop = entry - atr*mult
    mr_tp_mode: str                 # "VWAP" or "MID" (bollinger mid)
    mr_tp_min_r: float              # minimum expected reward in R before allowing trade
    mr_max_hold_bars: int           # time stop
    mr_min_atr_pct: float           # avoid dead chop
    mr_max_atr_pct: float           # avoid extreme spikes
    mr_vol_expand_ratio: float      # ATR now / ATR slow must be >= this
    mr_cooldown_after_loss_bars: int  # cooldown bars after a losing exit

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
        leverage_max=clamp(safe_float(envs("PAPER_LEVERAGE_MAX", "2.0"), 2.0), 1.0, 10.0),
        leverage_min=clamp(safe_float(envs("PAPER_LEVERAGE_MIN", "1.0"), 1.0), 1.0, 10.0),
        margin_alloc_cap=clamp(safe_float(envs("MARGIN_ALLOC_CAP", "0.60"), 0.60), 0.05, 0.95),
        maint_margin_ratio=clamp(safe_float(envs("MAINT_MARGIN_RATIO", "0.35"), 0.35), 0.05, 0.95),

        # Strategy v0.3 defaults (safe starter values)
        mr_window=max(20, safe_int(envs("MR_WINDOW", "80"), 80)),
        mr_z_enter=clamp(safe_float(envs("MR_Z_ENTER", "2.0"), 2.0), 0.8, 6.0),
        mr_z_exit=clamp(safe_float(envs("MR_Z_EXIT", "0.6"), 0.6), 0.1, 3.0),
        mr_bb_k=clamp(safe_float(envs("MR_BB_K", "2.0"), 2.0), 0.8, 4.0),
        mr_atr_period=max(5, safe_int(envs("MR_ATR_PERIOD", "14"), 14)),
        mr_atr_stop_mult=clamp(safe_float(envs("MR_ATR_STOP_MULT", "1.4"), 1.4), 0.6, 5.0),
        mr_tp_mode=(envs("MR_TP_MODE", "VWAP").strip().upper() or "VWAP"),
        mr_tp_min_r=clamp(safe_float(envs("MR_TP_MIN_R", "0.9"), 0.9), 0.1, 5.0),
        mr_max_hold_bars=max(6, safe_int(envs("MR_MAX_HOLD_BARS", "24"), 24)),
        mr_min_atr_pct=clamp(safe_float(envs("MR_MIN_ATR_PCT", "0.0010"), 0.0010), 0.0, 0.02),
        mr_max_atr_pct=clamp(safe_float(envs("MR_MAX_ATR_PCT", "0.0600"), 0.0600), 0.01, 0.50),
        mr_vol_expand_ratio=clamp(safe_float(envs("MR_VOL_EXPAND_RATIO", "1.15"), 1.15), 0.8, 3.0),
        mr_cooldown_after_loss_bars=max(0, safe_int(envs("MR_COOLDOWN_AFTER_LOSS_BARS", "6"), 6)),

        data_dir=data_dir,
        db_filename=envs("DB_FILENAME", "lunobiz.sqlite3"),

        poll_interval_seconds=max(2, safe_int(envs("POLL_INTERVAL_SECONDS", "10"), 10)),
        http_timeout_seconds=max(5, safe_int(envs("HTTP_TIMEOUT_SECONDS", "15"), 15)),
        http_max_retries=max(1, safe_int(envs("HTTP_MAX_RETRIES", "4"), 4)),
    )

    ensure_dir(cfg.data_dir)
    log.info(
        f"Config loaded. mode={cfg.app_mode.upper()} data_dir={cfg.data_dir} "
        f"bt_tf={cfg.backtest_timeframe} signal_tf={cfg.signal_timeframe} fee_bps={cfg.backtest_fee_bps} "
        f"mr_window={cfg.mr_window} z_enter={cfg.mr_z_enter} z_exit={cfg.mr_z_exit} "
        f"paper_leverage={'ON' if cfg.leverage_enabled_paper else 'OFF'} lev_cap={cfg.leverage_max}"
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
# Indicators (lightweight, list-based)
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
    out: list[float] = [50.0] * period
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


def rolling_mean_std(values: list[float], window: int) -> tuple[list[float], list[float]]:
    """
    Rolling mean/std with O(n) using running sums.
    std is population std over window.
    For indices < window-1, returns 0 mean/std to keep alignment.
    """
    n = len(values)
    if n == 0:
        return [], []
    w = max(2, int(window))
    means = [0.0] * n
    stds = [0.0] * n

    s = 0.0
    ss = 0.0
    for i, x in enumerate(values):
        s += x
        ss += x * x
        if i >= w:
            x0 = values[i - w]
            s -= x0
            ss -= x0 * x0
        if i >= w - 1:
            mu = s / w
            var = max(0.0, (ss / w) - (mu * mu))
            means[i] = mu
            stds[i] = math.sqrt(var)
    return means, stds


def rolling_vwap(close: list[float], volume: list[float], window: int) -> list[float]:
    """
    Rolling VWAP using close as typical price proxy.
    vwap = sum(price*vol)/sum(vol)
    """
    n = len(close)
    if n == 0:
        return []
    w = max(2, int(window))
    out = [0.0] * n
    pv = 0.0
    vv = 0.0
    for i in range(n):
        p = close[i]
        v = max(0.0, volume[i] if i < len(volume) else 0.0)
        pv += p * v
        vv += v
        if i >= w:
            p0 = close[i - w]
            v0 = max(0.0, volume[i - w] if (i - w) < len(volume) else 0.0)
            pv -= p0 * v0
            vv -= v0
        if i >= w - 1:
            out[i] = (pv / vv) if vv > 1e-12 else close[i]
    return out


def bollinger(close: list[float], window: int, k: float) -> tuple[list[float], list[float], list[float]]:
    mu, sd = rolling_mean_std(close, window)
    n = len(close)
    mid = mu
    up = [0.0] * n
    lo = [0.0] * n
    kk = float(k)
    for i in range(n):
        up[i] = mid[i] + kk * sd[i]
        lo[i] = mid[i] - kk * sd[i]
    return lo, mid, up


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
# Strategy v0.3 (Volatility Expansion + Mean Reversion)
# =========================

@dataclass
class VolState:
    """
    Computed on signal timeframe (higher TF) to avoid trading in dead regimes.
    """
    atr_pct: float
    expand_ratio: float   # ATR_fast / ATR_slow
    ok: bool
    score: float          # 0..1


@dataclass
class EntrySignal:
    action: str           # BUY/HOLD
    confidence: float
    reason: str
    leverage: float
    stop_price: float
    take_profit: float
    r_value: float
    max_hold_bars: int
    meta: dict[str, t.Any]


class StrategyV3:
    """
    Mean Reversion (long-only) with volatility expansion gate.

    High-level:
    - Use signal timeframe (e.g., 1h) to compute if volatility is "alive"
      via ATR expansion ratio + ATR% band.
    - Use entry timeframe (e.g., 15m) to detect overextension:
        * Z-score vs rolling VWAP (or rolling mean)
        * Bollinger Band touch/penetration
        * Short RSI exhaustion filter
    - Exit quickly at mean target (VWAP or mid-band), with strict ATR stop and time stop.

    This is deliberately conservative and fee-aware.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log

    def compute_vol_state(self, sig: CandleSeries) -> list[VolState]:
        n = len(sig.c)
        if n < max(60, self.cfg.mr_atr_period * 6):
            return [VolState(atr_pct=0.0, expand_ratio=0.0, ok=False, score=0.0) for _ in range(n)]

        atr_fast = atr(sig.h, sig.l, sig.c, self.cfg.mr_atr_period)
        atr_slow = atr(sig.h, sig.l, sig.c, max(self.cfg.mr_atr_period * 4, 40))

        out: list[VolState] = []
        for i in range(n):
            px = sig.c[i]
            af = atr_fast[i] if i < len(atr_fast) else 0.0
            aslow = atr_slow[i] if i < len(atr_slow) else 0.0
            atr_pct = (af / px) if px > 1e-12 else 0.0
            ratio = (af / aslow) if aslow > 1e-12 else 0.0

            band_ok = (atr_pct >= self.cfg.mr_min_atr_pct) and (atr_pct <= self.cfg.mr_max_atr_pct)
            expand_ok = ratio >= self.cfg.mr_vol_expand_ratio

            ok = bool(band_ok and expand_ok)

            # Score: blend how much above thresholds it is (capped)
            atr_score = clamp((atr_pct - self.cfg.mr_min_atr_pct) / max(1e-12, (self.cfg.mr_max_atr_pct - self.cfg.mr_min_atr_pct)), 0.0, 1.0)
            exp_score = clamp((ratio - self.cfg.mr_vol_expand_ratio) / max(1e-12, (2.0 - self.cfg.mr_vol_expand_ratio)), 0.0, 1.0) if self.cfg.mr_vol_expand_ratio < 2.0 else clamp(ratio / max(1e-12, self.cfg.mr_vol_expand_ratio), 0.0, 1.0)
            score = clamp(0.55 * exp_score + 0.45 * atr_score, 0.0, 1.0)

            out.append(VolState(atr_pct=float(atr_pct), expand_ratio=float(ratio), ok=ok, score=float(score)))
        return out

    def choose_leverage(self, vol: VolState, equity_dd: float) -> float:
        """
        Conservative leverage:
        - base 1x
        - allow up to cfg.leverage_max when vol score high AND drawdown low
        - reduce leverage quickly in drawdown
        """
        if not self.cfg.leverage_enabled_paper:
            return 1.0

        lev = 1.0
        if vol.ok:
            # scale from 1.0 up to cap by score
            lev = 1.0 + (self.cfg.leverage_max - 1.0) * clamp(vol.score, 0.0, 1.0)

        # drawdown dampener
        if equity_dd > 0.02:
            lev *= clamp(1.0 - equity_dd * 7.0, 0.35, 1.0)

        lev = clamp(lev, self.cfg.leverage_min, self.cfg.leverage_max)
        return float(lev)

    def entry_signal(
        self,
        pair: str,
        entry: CandleSeries,
        idx_entry: int,
        sig: CandleSeries,
        idx_sig: int,
        vol: VolState,
        equity_dd: float,
        cooldown_until_idx: int,
    ) -> EntrySignal:
        # Signal timeframe must be OK
        if not vol.ok:
            return EntrySignal(
                action="HOLD", confidence=0.0,
                reason="Volatility gate off",
                leverage=1.0, stop_price=0.0, take_profit=0.0, r_value=0.0, max_hold_bars=self.cfg.mr_max_hold_bars,
                meta={"vol": dataclasses.asdict(vol)}
            )

        # Cooldown after a loss
        if idx_entry < cooldown_until_idx:
            return EntrySignal(
                action="HOLD", confidence=0.0,
                reason="Cooldown after loss",
                leverage=1.0, stop_price=0.0, take_profit=0.0, r_value=0.0, max_hold_bars=self.cfg.mr_max_hold_bars,
                meta={"cooldown_until_idx": cooldown_until_idx, "idx_entry": idx_entry}
            )

        # Need enough entry TF candles for rolling computations
        w = self.cfg.mr_window
        if idx_entry < max(w + 5, self.cfg.mr_atr_period + 10):
            return EntrySignal(
                action="HOLD", confidence=0.0,
                reason="Insufficient entry TF history",
                leverage=1.0, stop_price=0.0, take_profit=0.0, r_value=0.0, max_hold_bars=self.cfg.mr_max_hold_bars,
                meta={"need": max(w + 5, self.cfg.mr_atr_period + 10), "idx_entry": idx_entry}
            )

        # Slice up to idx_entry (inclusive)
        close = entry.c[:idx_entry + 1]
        high = entry.h[:idx_entry + 1]
        low = entry.l[:idx_entry + 1]
        volm = entry.v[:idx_entry + 1]

        px = close[-1]
        if px <= 0:
            return EntrySignal("HOLD", 0.0, "Bad price", 1.0, 0.0, 0.0, 0.0, self.cfg.mr_max_hold_bars, {})

        # ATR sanity (entry TF)
        atrs = atr(high, low, close, self.cfg.mr_atr_period)
        atr_now = atrs[-1] if atrs else 0.0
        atr_pct = (atr_now / px) if px > 1e-12 else 0.0
        if atr_pct < self.cfg.mr_min_atr_pct:
            return EntrySignal("HOLD", 0.0, "ATR% too low", 1.0, 0.0, 0.0, 0.0, self.cfg.mr_max_hold_bars, {"atr_pct": atr_pct})
        if atr_pct > self.cfg.mr_max_atr_pct:
            return EntrySignal("HOLD", 0.0, "ATR% too high", 1.0, 0.0, 0.0, 0.0, self.cfg.mr_max_hold_bars, {"atr_pct": atr_pct})

        # Mean reference
        vwap = rolling_vwap(close, volm, w)
        mu, sd = rolling_mean_std(close, w)
        mid = mu[-1]
        st = sd[-1]
        vwap_now = vwap[-1] if vwap else mid

        # Guard: must have valid rolling stats
        if st <= 1e-12 or mid <= 0:
            return EntrySignal("HOLD", 0.0, "Rolling stats not ready", 1.0, 0.0, 0.0, 0.0, self.cfg.mr_max_hold_bars, {})

        z = (px - vwap_now) / st if st > 1e-12 else 0.0

        # Bollinger
        bb_lo, bb_mid, bb_up = bollinger(close, w, self.cfg.mr_bb_k)
        lo_band = bb_lo[-1]
        mid_band = bb_mid[-1]
        up_band = bb_up[-1]

        # Short RSI exhaustion
        rsi_short = rsi(close, period=7)
        rsi_now = rsi_short[-1] if rsi_short else 50.0

        # Overextension condition (long-only):
        # - z below threshold OR price below lower band (with slight penetration)
        z_ok = z <= -self.cfg.mr_z_enter
        bb_ok = (px <= lo_band * 1.0005)  # allow tiny tolerance
        if not (z_ok or bb_ok):
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="No overextension",
                leverage=1.0, stop_price=0.0, take_profit=0.0, r_value=0.0, max_hold_bars=self.cfg.mr_max_hold_bars,
                meta={"z": z, "z_enter": self.cfg.mr_z_enter, "px": px, "bb_lo": lo_band}
            )

        # Exhaustion filter: avoid buying when short RSI still high
        if rsi_now > 48.0:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="RSI not exhausted",
                leverage=1.0, stop_price=0.0, take_profit=0.0, r_value=0.0, max_hold_bars=self.cfg.mr_max_hold_bars,
                meta={"rsi7": rsi_now}
            )

        # Determine take profit target
        tp_mode = (self.cfg.mr_tp_mode or "VWAP").upper()
        if tp_mode == "MID":
            tp = mid_band if mid_band > 0 else vwap_now
            tp_reason = "TP=BB_MID"
        else:
            tp = vwap_now if vwap_now > 0 else mid_band
            tp_reason = "TP=VWAP"

        # Stop: ATR based
        stop = px - self.cfg.mr_atr_stop_mult * atr_now
        if stop <= 0:
            return EntrySignal("HOLD", 0.0, "Invalid stop", 1.0, 0.0, 0.0, 0.0, self.cfg.mr_max_hold_bars, {"atr": atr_now})

        r_value = px - stop
        if r_value <= 0:
            return EntrySignal("HOLD", 0.0, "Invalid R", 1.0, 0.0, 0.0, 0.0, self.cfg.mr_max_hold_bars, {})

        # Ensure TP offers minimum reward in R
        exp_reward = max(0.0, tp - px)
        exp_r = exp_reward / max(1e-12, r_value)

        # Fee/slippage gate: require R and expected reward to dominate costs
        bps_total = (self.cfg.backtest_fee_bps + self.cfg.backtest_slippage_bps) / 10000.0
        est_round_trip_cost = px * bps_total * 2.2  # padded
        # convert cost into "R units"
        cost_r = est_round_trip_cost / max(1e-12, r_value)

        if exp_r < self.cfg.mr_tp_min_r:
            return EntrySignal(
                action="HOLD", confidence=0.0,
                reason="TP too small vs stop (low R)",
                leverage=1.0, stop_price=0.0, take_profit=0.0, r_value=0.0, max_hold_bars=self.cfg.mr_max_hold_bars,
                meta={"exp_r": exp_r, "min_r": self.cfg.mr_tp_min_r, "tp": tp, "px": px, "stop": stop}
            )

        if cost_r > 0.55:
            return EntrySignal(
                action="HOLD", confidence=0.0,
                reason="Costs too high vs stop distance",
                leverage=1.0, stop_price=0.0, take_profit=0.0, r_value=0.0, max_hold_bars=self.cfg.mr_max_hold_bars,
                meta={"cost_r": cost_r, "r_value": r_value, "est_cost": est_round_trip_cost}
            )

        # Confidence: deeper oversold and higher vol score => higher confidence
        z_score = clamp((-z) / max(1e-12, self.cfg.mr_z_enter), 0.0, 1.8)  # can exceed 1 a bit
        bb_penetration = 0.0
        if lo_band > 0:
            bb_penetration = clamp((lo_band - px) / lo_band / 0.01, 0.0, 1.0)  # 1% penetration -> 1

        rsi_score = clamp((48.0 - rsi_now) / 20.0, 0.0, 1.0)

        conf = clamp(
            0.35
            + 0.35 * clamp(vol.score, 0.0, 1.0)
            + 0.20 * clamp(z_score, 0.0, 1.0)
            + 0.10 * clamp(bb_penetration, 0.0, 1.0)
            + 0.10 * rsi_score
            - 0.10 * clamp(cost_r, 0.0, 1.0),
            0.0,
            0.95
        )

        lev = self.choose_leverage(vol, equity_dd)

        return EntrySignal(
            action="BUY",
            confidence=float(conf),
            reason=f"Mean reversion entry ({tp_reason})",
            leverage=float(lev),
            stop_price=float(stop),
            take_profit=float(tp),
            r_value=float(r_value),
            max_hold_bars=int(self.cfg.mr_max_hold_bars),
            meta={
                "px": px,
                "z": z,
                "z_enter": self.cfg.mr_z_enter,
                "z_exit": self.cfg.mr_z_exit,
                "bb_lo": lo_band,
                "bb_mid": mid_band,
                "bb_up": up_band,
                "vwap": vwap_now,
                "mid": mid,
                "std": st,
                "rsi7": rsi_now,
                "atr": atr_now,
                "atr_pct": atr_pct,
                "exp_r": exp_r,
                "cost_r": cost_r,
                "vol": dataclasses.asdict(vol),
                "equity_dd": equity_dd,
            }
        )


# =========================
# Backtesting (v0.3)
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
    take_profit: float
    r_value: float
    leverage: float
    margin_used: float
    entry_ts_ms: int
    entry_idx: int

    max_hold_bars: int
    # tracking
    bars_held: int = 0


class BacktesterV3:
    """
    Long-only mean reversion backtest with paper leverage simulation.

    Mechanics:
    - Size by risk_per_trade: risk amount equals max loss at stop.
    - Leverage affects margin used; liquidation if loss exceeds margin_used*(1-maint_margin_ratio).
    - Exit rules:
        * Take profit at target (VWAP or BB mid), intrabar high check.
        * Stop at stop_price, intrabar low check.
        * Time stop after max_hold_bars (exit at close).
        * Reversion exit: if z >= -mr_z_exit (i.e., sufficiently reverted), exit at close.
    - Cooldown after a losing exit (bars-based).
    """
    def __init__(self, cfg: AppConfig, log: AppLogger, strategy: StrategyV3):
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
        - Liquidation occurs when loss exceeds margin_used*(1-maint_margin_ratio).
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
        vol_series: list[VolState],
        initial_equity: float = 100.0
    ) -> BacktestResult:
        # Basic data checks
        if len(entry.c) < max(500, self.cfg.mr_window + 120) or len(signal.c) < 120 or len(vol_series) != len(signal.c):
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
        cooldown_until_idx = -1

        trades = 0
        wins = 0
        trade_pnls: list[float] = []
        trade_returns: list[float] = []
        gross_profit = 0.0
        gross_loss = 0.0

        eq_curve: list[tuple[int, float]] = []

        # Precompute entry rolling series for reversion exit check:
        close = entry.c
        volm = entry.v
        w = self.cfg.mr_window
        vwap_arr = rolling_vwap(close, volm, w)
        mu_arr, sd_arr = rolling_mean_std(close, w)

        # ATR for stop sanity and optional analysis
        atr_arr = atr(entry.h, entry.l, entry.c, self.cfg.mr_atr_period)

        def equity_drawdown() -> float:
            return (peak - equity) / max(1e-12, peak)

        for i in range(0, len(entry.ts)):
            ts = entry.ts[i]
            idx_sig = find_last_index_leq(signal.ts, ts)
            if idx_sig < 0:
                continue
            vol = vol_series[idx_sig]

            px = entry.c[i]
            hi = entry.h[i]
            lo = entry.l[i]
            if px <= 0:
                continue

            # Update open position
            if pos is not None:
                pos.bars_held += 1

                # Liquidation (intrabar) check
                liq_px = self._liquidation_price_long(pos)
                if liq_px > 0 and lo <= liq_px:
                    exit_px = self._apply_slippage(liq_px, "SELL")
                    notional = pos.qty * exit_px
                    fee = self._fee(notional)
                    pnl = (exit_px - pos.entry_price) * pos.qty - fee
                    equity += pnl

                    trades += 1
                    trade_pnls.append(pnl)
                    trade_returns.append(pnl / max(1e-12, (equity - pnl)))  # return vs pre-exit equity
                    gross_profit += pnl if pnl > 0 else 0.0
                    gross_loss += abs(pnl) if pnl < 0 else 0.0
                    if pnl > 0:
                        wins += 1
                    else:
                        cooldown_until_idx = i + self.cfg.mr_cooldown_after_loss_bars

                    pos = None
                    eq_curve.append((ts, equity))
                    peak = max(peak, equity)
                    max_dd = max(max_dd, equity_drawdown())
                    break  # stop after liquidation for realism

                # Stop check (intrabar)
                if pos is not None and lo <= pos.stop_price:
                    exit_px = self._apply_slippage(pos.stop_price, "SELL")
                    notional = pos.qty * exit_px
                    fee = self._fee(notional)
                    pnl = (exit_px - pos.entry_price) * pos.qty - fee
                    equity += pnl

                    trades += 1
                    trade_pnls.append(pnl)
                    trade_returns.append(pnl / max(1e-12, (equity - pnl)))
                    if pnl > 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                        cooldown_until_idx = i + self.cfg.mr_cooldown_after_loss_bars
                    pos = None

                # Take profit check (intrabar)
                if pos is not None and hi >= pos.take_profit:
                    exit_px = self._apply_slippage(pos.take_profit, "SELL")
                    notional = pos.qty * exit_px
                    fee = self._fee(notional)
                    pnl = (exit_px - pos.entry_price) * pos.qty - fee
                    equity += pnl

                    trades += 1
                    trade_pnls.append(pnl)
                    trade_returns.append(pnl / max(1e-12, (equity - pnl)))
                    if pnl > 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                        cooldown_until_idx = i + self.cfg.mr_cooldown_after_loss_bars
                    pos = None

                # Reversion exit or time stop (at close)
                if pos is not None:
                    # z-score vs vwap
                    if i < len(vwap_arr) and i < len(sd_arr) and sd_arr[i] > 1e-12:
                        z = (px - vwap_arr[i]) / sd_arr[i]
                    else:
                        z = 0.0

                    # Exit when reverted sufficiently
                    if z >= -self.cfg.mr_z_exit:
                        exit_px = self._apply_slippage(px, "SELL")
                        notional = pos.qty * exit_px
                        fee = self._fee(notional)
                        pnl = (exit_px - pos.entry_price) * pos.qty - fee
                        equity += pnl

                        trades += 1
                        trade_pnls.append(pnl)
                        trade_returns.append(pnl / max(1e-12, (equity - pnl)))
                        if pnl > 0:
                            wins += 1
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                            cooldown_until_idx = i + self.cfg.mr_cooldown_after_loss_bars
                        pos = None

                    # Time stop
                    elif pos is not None and pos.bars_held >= pos.max_hold_bars:
                        exit_px = self._apply_slippage(px, "SELL")
                        notional = pos.qty * exit_px
                        fee = self._fee(notional)
                        pnl = (exit_px - pos.entry_price) * pos.qty - fee
                        equity += pnl

                        trades += 1
                        trade_pnls.append(pnl)
                        trade_returns.append(pnl / max(1e-12, (equity - pnl)))
                        if pnl > 0:
                            wins += 1
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                            cooldown_until_idx = i + self.cfg.mr_cooldown_after_loss_bars
                        pos = None

            # Entry logic if flat
            if pos is None:
                # drawdown safety: if deep drawdown, stop trading
                if equity_drawdown() >= 0.30:
                    eq_curve.append((ts, equity))
                    break

                idx_sig = find_last_index_leq(signal.ts, ts)
                if idx_sig < 0:
                    eq_curve.append((ts, equity))
                    continue
                vol = vol_series[idx_sig]

                es = self.strategy.entry_signal(
                    pair=pair,
                    entry=entry,
                    idx_entry=i,
                    sig=signal,
                    idx_sig=idx_sig,
                    vol=vol,
                    equity_dd=equity_drawdown(),
                    cooldown_until_idx=cooldown_until_idx,
                )

                # Threshold is intentionally moderate: mean reversion needs opportunities
                if es.action == "BUY" and es.confidence >= 0.58:
                    risk_amount = equity * clamp(self.cfg.risk_per_trade, 0.0005, 0.05)

                    stop_dist = max(1e-9, (px - es.stop_price))
                    qty = risk_amount / stop_dist

                    lev = es.leverage if self.cfg.leverage_enabled_paper else 1.0
                    lev = clamp(lev, self.cfg.leverage_min, self.cfg.leverage_max)

                    entry_px = self._apply_slippage(px, "BUY")
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

                    # Entry fee
                    entry_fee = self._fee(notional)
                    equity -= entry_fee

                    pos = SimPosition(
                        pair=pair,
                        qty=float(qty),
                        entry_price=float(entry_px),
                        stop_price=float(es.stop_price),
                        take_profit=float(es.take_profit),
                        r_value=float(es.r_value),
                        leverage=float(lev),
                        margin_used=float(margin_required),
                        entry_ts_ms=int(ts),
                        entry_idx=int(i),
                        max_hold_bars=int(es.max_hold_bars),
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
        if trades < 12:
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
                "mr_window": self.cfg.mr_window,
                "mr_z_enter": self.cfg.mr_z_enter,
                "mr_z_exit": self.cfg.mr_z_exit,
                "mr_tp_mode": self.cfg.mr_tp_mode,
                "mr_tp_min_r": self.cfg.mr_tp_min_r,
                "mr_max_hold_bars": self.cfg.mr_max_hold_bars,
                "mr_vol_expand_ratio": self.cfg.mr_vol_expand_ratio,
                "fee_bps": self.cfg.backtest_fee_bps,
                "slippage_bps": self.cfg.backtest_slippage_bps,
            }
        )

    def walk_forward(
        self,
        pair: str,
        entry: CandleSeries,
        signal: CandleSeries,
        vol_series: list[VolState],
        initial_equity: float = 100.0
    ) -> BacktestResult:
        # Split by time into 5 segments
        n = len(entry.ts)
        if n < 900:
            return self.run(pair, entry, signal, vol_series, initial_equity)

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

            res = self.run(pair, e_seg, signal, vol_series, eq)
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

        notes = "Walk-forward OK"
        if total_trades < 12:
            notes = "Walk-forward OK (Low trades)"
        if total_return < 0:
            notes = "Walk-forward OK (Negative)"

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
            notes=notes,
            equity_curve=all_curve,
            extra={
                "paper_leverage_enabled": self.cfg.leverage_enabled_paper,
                "leverage_cap": self.cfg.leverage_max,
                "segments": segments,
                "mr_window": self.cfg.mr_window,
                "mr_z_enter": self.cfg.mr_z_enter,
                "mr_z_exit": self.cfg.mr_z_exit,
                "fee_bps": self.cfg.backtest_fee_bps,
                "slippage_bps": self.cfg.backtest_slippage_bps,
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
                "app": {"name": APP_NAME, "version": APP_VERSION},
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

    def make_share_bundle(
        self,
        bundle_name: str,
        include_paths: list[Path],
        meta: dict[str, t.Any],
    ) -> Path:
        """
        Create a zip bundle in reports folder:
        - includes selected files (e.g., json/txt/csv/log)
        - includes a meta.json file with sanitized context (no secrets)
        """
        ensure_dir(self.reports_dir)
        stamp = now_utc().strftime("%Y%m%d_%H%M%S")
        zip_path = self.reports_dir / f"{bundle_name}_{stamp}.zip"

        # Sanitize meta aggressively (never store env/secrets here)
        safe_meta = {
            "app": {"name": APP_NAME, "version": APP_VERSION},
            "generated_utc": now_utc().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "meta": meta,
            "files": [str(p.name) for p in include_paths if p and p.exists()],
        }

        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                # meta.json
                z.writestr("meta.json", json.dumps(safe_meta, ensure_ascii=False, indent=2))
                for p in include_paths:
                    try:
                        if p and p.exists() and p.is_file():
                            z.write(p, arcname=p.name)
                    except Exception:
                        continue
            self.log.info(f"Share bundle created: {zip_path}")
        except Exception as e:
            self.log.exception("Failed to create share bundle", e)
            raise

        return zip_path


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

        self.strategy_v3 = StrategyV3(cfg, log)
        self.backtester_v3 = BacktesterV3(cfg, log, self.strategy_v3)

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

                # Compute vol state on signal TF
                vol_states = self.strategy_v3.compute_vol_state(signal_series)

                if walk_forward:
                    res = self.backtester_v3.walk_forward(pair, entry_series, signal_series, vol_states, initial_equity=initial_equity)
                else:
                    res = self.backtester_v3.run(pair, entry_series, signal_series, vol_states, initial_equity=initial_equity)

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

                # Live execution remains locked (by design)
                if not self.cfg.is_paper():
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
        self.geometry("1250x820")
        self.minsize(1050, 700)

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

        ttk.Button(btn_row, text="Export Logs", command=self._export_logs).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Create Share Bundle", command=self._create_share_bundle).pack(side=tk.LEFT)

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
        ttk.Button(row3, text="Copy JSON (Latest)", command=self._copy_backtest_json).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row3, text="Export Backtest Report", command=self._export_backtest_report).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row3, text="Open Reports Folder", command=self._open_reports_folder).pack(side=tk.LEFT)

        self.tbl_bt = self._make_table(
            right,
            title="Backtest Results (sorted by return)",
            columns=[("pair", 90), ("entry", 70), ("signal", 70), ("trades", 70), ("win", 70), ("ret", 90), ("dd", 80), ("notes", 200)]
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
            "This build uses a volatility-expansion + mean-reversion engine.\n"
            "Live execution remains locked.\n"
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
            self.update()
            messagebox.showinfo("Copy Summary", "Copied backtest summary to clipboard.")
        except Exception as e:
            messagebox.showerror("Copy Summary", f"Failed: {e}")

    def _copy_backtest_json(self) -> None:
        """
        Copies a compact JSON payload (context + results summary) to clipboard.
        This makes it easy to paste results without screenshots.
        """
        if not self._last_backtest_results:
            messagebox.showwarning("Copy JSON", "Run a backtest first.")
            return
        payload = {
            "app": {"name": APP_NAME, "version": APP_VERSION},
            "generated_utc": now_utc().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "context": self._last_backtest_context,
            "top": [r.to_dict() for r in self._last_backtest_results[:10]],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()
            messagebox.showinfo("Copy JSON", "Copied JSON (top 10 + context) to clipboard.")
        except Exception as e:
            messagebox.showerror("Copy JSON", f"Failed: {e}")

    def _compose_backtest_summary_text(self) -> str:
        ctx = self._last_backtest_context or {}
        lines: list[str] = []
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

    def _create_share_bundle(self) -> None:
        """
        Creates a single zip file in reports/ that you can upload or share.
        It includes:
        - latest backtest txt/json/csv if available
        - lunobiz.log (if exists)
        - meta.json (sanitized, no secrets)
        """
        include: list[Path] = []
        try:
            if self._last_backtest_report_paths:
                for k in ("txt", "json", "csv"):
                    p = self._last_backtest_report_paths.get(k)
                    if p and p.exists():
                        include.append(p)
            # include logs
            log_path = self.cfg.data_dir / "lunobiz.log"
            if log_path.exists():
                include.append(log_path)

            if not include:
                messagebox.showwarning("Share Bundle", "No reports/logs found yet. Run backtest and export report first.")
                return

            meta = {
                "context": self._last_backtest_context,
                "note": "Bundle contains reports and logs only. No secrets are included.",
            }
            zip_path = self.reports.make_share_bundle("share_bundle", include, meta)
            messagebox.showinfo("Share Bundle", f"Created:\n{zip_path}\n\nYou can upload this zip for review.")
        except Exception as e:
            messagebox.showerror("Share Bundle", f"Failed: {e}")

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
        days = max(1, safe_int(self.var_bt_days.get().strip(), 180))
        eq = max(1.0, safe_float(self.var_bt_equity.get().strip(), 100.0))
        walk = bool(self.var_bt_walk.get())

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
                    "mr_window": self.cfg.mr_window,
                    "mr_z_enter": self.cfg.mr_z_enter,
                    "mr_z_exit": self.cfg.mr_z_exit,
                    "mr_bb_k": self.cfg.mr_bb_k,
                    "mr_tp_mode": self.cfg.mr_tp_mode,
                    "mr_tp_min_r": self.cfg.mr_tp_min_r,
                    "mr_max_hold_bars": self.cfg.mr_max_hold_bars,
                    "mr_vol_expand_ratio": self.cfg.mr_vol_expand_ratio,
                }
                self._last_backtest_results = res
                self._last_backtest_context = ctx
                self._last_backtest_report_paths = {}

                self._ui_queue.put(lambda: self._render_backtest_results(res))

                # Conservative gate: still locked by default
                # Only set LIVE_GATE=1 if top result is strongly positive with controlled DD and enough trades.
                if res:
                    top = res[0]
                    gate_ok = (top.total_return > 0.12 and top.max_drawdown < 0.18 and top.trades >= 30)
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
