# LunoBiz — Living System Plan (Public Repo Safe)

## 1) Mission
Build a Windows app (single-file Python) that:
- Attempts aggressive daily returns (target 3–5% per day) while enforcing hard risk controls
- Runs extensive backtesting + walk-forward validation
- Starts in PAPER trading and only enables LIVE trading after objective gates pass
- Produces a professional GUI dashboard (values only, no charts required)

## 2) Reality Check (Engineering Constraints)
- No system can guarantee 3–5% daily return or “close to zero losses.”
- Our engineering goal: maximize risk-adjusted return, and prevent catastrophic loss with strict safeguards.
- If performance gates fail in backtest/walk-forward, LIVE mode remains locked.

## 3) Operating Modes
### PAPER (default)
- Simulated orders, fees/slippage model
- Uses the same strategy logic and risk logic as LIVE
- Writes full audit logs and performance stats

### LIVE (locked behind gates)
- Requires explicit enable + passing performance gates + manual confirmation
- Uses separate LIVE API keys
- Hard kill-switch: a local file can immediately disable trading

## 4) Core Modules (Inside a Single Python File)
1. **Config Loader**
   - Loads `.env` safely
   - Validates required fields
   - Refuses LIVE if missing keys or gates not satisfied

2. **Logging + Audit**
   - Structured logs (JSON-lines)
   - SQLite: trades, decisions, snapshots, metrics

3. **Luno API Client**
   - HTTPS only (mandatory)
   - Timeouts, retries with exponential backoff
   - Rate limit handling (429) with respectful retry
   - Read-only vs live credentials separation

4. **Market Data**
   - Candle fetching + caching
   - Data integrity checks (missing candles, time gaps)

5. **Strategy Engine (Aggressive Target)**
   - Multi-strategy ensemble with regime filter:
     - Momentum/trend breakout
     - Volatility expansion breakout
     - Mean reversion (only in range regime)
   - Multi-market scoring: pick best opportunities across allowed pairs
   - Confidence score + recommended position size

6. **Risk Manager (Non-Negotiable)**
   - Per-trade risk sizing (fraction of equity)
   - Daily loss cap -> immediate stop + cooldown
   - Max open positions
   - Spread/liquidity filters
   - Slippage stress scenario checks
   - “Uncertain state” => no trade

7. **Execution Engine**
   - Paper fills model or live order placement
   - Order status reconciliation
   - Detect manual interventions via balance/trade history reconciliation

8. **Backtesting + Walk-Forward Validation**
   - Multi-pair, multi-timeframe backtest
   - Walk-forward windows:
     - Optimize on training window
     - Validate on next window
     - Roll forward
   - Metrics:
     - Total return, daily return distribution
     - Max drawdown
     - Profit factor / expectancy
     - Worst-day loss
     - Exposure time
     - Stability across windows

9. **GUI Dashboard (No charts)**
   - Top status panel: mode, equity, cash, exposure, API health
   - Tables:
     - Open positions
     - Market rankings (scanner)
     - Recent signals and decisions
     - Recent fills
   - Controls:
     - Start/stop bot
     - Run backtest
     - Toggle paper/live (live locked until gates pass)
     - Update risk settings
     - Export logs

10. **Notifications**
   - Telegram if configured
   - Otherwise local desktop notifications
   - Triggers:
     - Entry/exit/stop
     - Daily halt/cooldown
     - API errors/retries
     - Gate pass/fail summaries

## 5) Performance Gates Before LIVE Unlock
LIVE stays locked unless:
- Walk-forward validation passes across multiple windows
- Max drawdown <= configured threshold (e.g., <= 10–15%)
- Worst-day loss <= configured threshold
- Minimum trade count reached
- No “fragile” dependence on a single market regime

## 6) Secrets & Public Repo Safety
- Never commit `.env` or keys
- `.env.example` is a template only
- Use separate API keys for read-only and live
- If a secret is ever committed: rotate keys immediately (history is permanent)

## 7) Change Log
- 2025-12-30: Initialized full planning doc
