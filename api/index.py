"""Reversion Lab — Vercel serverless FastAPI backend"""
import io, os, secrets, sqlite3
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import requests as httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from mangum import Mangum
from pydantic import BaseModel

# ── config ────────────────────────────────────────────────────────────────
DB_PATH           = os.getenv("REVERSION_DB",       "/tmp/reversion_lab.db")
TWELVE_DATA_KEY   = os.getenv("TWELVE_DATA_KEY",    "")
POLYGON_KEY       = os.getenv("POLYGON_KEY",        "")
ALPACA_KEY_ID     = os.getenv("ALPACA_KEY_ID",      "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY",  "")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL",    "https://paper-api.alpaca.markets")
DEFAULT_SYMBOLS   = [s.strip().upper() for s in os.getenv("SYMBOLS", "SPY,QQQ,AAPL,IWM,NVDA").split(",") if s.strip()]
DASHBOARD_USER    = os.getenv("DASHBOARD_USER",     "admin")
DASHBOARD_PASS    = os.getenv("DASHBOARD_PASS",     "changeme")
RISK_FRAC         = float(os.getenv("RISK_FRAC",    "0.01"))
CRON_SECRET       = os.getenv("CRON_SECRET",        "")

# ── app ───────────────────────────────────────────────────────────────────
security = HTTPBasic()
app = FastAPI(title="Reversion Lab", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Mangum adapter — wraps ASGI app for Vercel serverless
handler = Mangum(app, lifespan="off")

# ── auth ──────────────────────────────────────────────────────────────────
def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(credentials.username.encode(), DASHBOARD_USER.encode())
    ok_pass = secrets.compare_digest(credentials.password.encode(), DASHBOARD_PASS.encode())
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

# ── models ────────────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    symbols: List[str]
    ibs_threshold: float = 0.25
    range_factor_threshold: float = 3.5
    lookback_high: int = 10
    lookback_range: int = 25
    source_tolerance: float = 0.05
    sma_filter: Optional[int] = None
    max_hold_bars: int = 5
    fee_bps: float = 2.0
    slippage_bps: float = 3.0
    account_equity: float = 100000.0
    risk_frac: float = RISK_FRAC
    auto_execute: bool = False

class JournalTrade(BaseModel):
    symbol: str
    trade_date: str
    side: str = "LONG"
    entry_price: float
    exit_price: Optional[float] = None
    shares: Optional[float] = None
    r_multiple: Optional[float] = None
    source_1_close: Optional[float] = None
    source_2_close: Optional[float] = None
    verified: int = 0
    notes: Optional[str] = None

# ── db ────────────────────────────────────────────────────────────────────
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS verification_audit (
      symbol TEXT NOT NULL, date TEXT NOT NULL,
      primary_open REAL, primary_high REAL, primary_low REAL, primary_close REAL,
      secondary_open REAL, secondary_high REAL, secondary_low REAL, secondary_close REAL,
      open_diff REAL, high_diff REAL, low_diff REAL, close_diff REAL,
      verified INTEGER NOT NULL, PRIMARY KEY (symbol, date));
    CREATE TABLE IF NOT EXISTS signals (
      symbol TEXT NOT NULL, date TEXT NOT NULL,
      ibs REAL, range_factor REAL, verified INTEGER NOT NULL,
      signal INTEGER NOT NULL, close REAL, PRIMARY KEY (symbol, date));
    CREATE TABLE IF NOT EXISTS trades (
      trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
      symbol TEXT NOT NULL, signal_date TEXT NOT NULL,
      entry_date TEXT NOT NULL, exit_date TEXT NOT NULL,
      entry_price REAL NOT NULL, exit_price REAL NOT NULL,
      shares REAL, dollar_risk REAL,
      gross_return REAL, net_return REAL, pnl_dollars REAL,
      bars_held INTEGER, verified INTEGER NOT NULL,
      ibs REAL, range_factor REAL, equity_after REAL);
    CREATE TABLE IF NOT EXISTS broker_orders (
      order_id TEXT PRIMARY KEY, symbol TEXT NOT NULL,
      signal_date TEXT NOT NULL, qty REAL, side TEXT,
      order_type TEXT, status TEXT, submitted_at TEXT, alpaca_response TEXT);
    CREATE TABLE IF NOT EXISTS trade_journal (
      journal_id INTEGER PRIMARY KEY AUTOINCREMENT,
      symbol TEXT NOT NULL, trade_date TEXT NOT NULL, side TEXT NOT NULL,
      entry_price REAL, exit_price REAL, shares REAL, r_multiple REAL,
      source_1_close REAL, source_2_close REAL,
      verified INTEGER NOT NULL DEFAULT 0, notes TEXT);
    """)
    conn.commit()
    conn.close()

init_db()

# ── data sources ──────────────────────────────────────────────────────────
def twelve_data_daily(symbol: str, outputsize: int = 500) -> pd.DataFrame:
    if not TWELVE_DATA_KEY:
        raise HTTPException(500, detail="TWELVE_DATA_KEY not configured")
    r = httpx.get("https://api.twelvedata.com/time_series", timeout=30, params={
        "symbol": symbol, "interval": "1day",
        "outputsize": outputsize, "order": "ASC", "apikey": TWELVE_DATA_KEY,
    })
    r.raise_for_status()
    vals = r.json().get("values", [])
    if not vals:
        raise HTTPException(400, detail=f"No Twelve Data values for {symbol}")
    return pd.DataFrame([{
        "date": pd.to_datetime(v["datetime"]).normalize(),
        "open": float(v["open"]), "high": float(v["high"]),
        "low": float(v["low"]), "close": float(v["close"]),
        "volume": float(v.get("volume") or 0),
    } for v in vals]).sort_values("date").reset_index(drop=True)

def polygon_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    if not POLYGON_KEY:
        raise HTTPException(500, detail="POLYGON_KEY not configured")
    r = httpx.get(
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
        timeout=30,
        params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY},
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise HTTPException(400, detail=f"No Polygon results for {symbol}")
    return pd.DataFrame([{
        "date": pd.to_datetime(item["t"], unit="ms").normalize(),
        "open": float(item["o"]), "high": float(item["h"]),
        "low": float(item["l"]), "close": float(item["c"]),
        "volume": float(item.get("v") or 0),
    } for item in results]).sort_values("date").reset_index(drop=True)

# ── verification ──────────────────────────────────────────────────────────
def verify_sources(primary: pd.DataFrame, secondary: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    a = primary.rename(columns={c: f"primary_{c}" for c in ["open", "high", "low", "close", "volume"]})
    b = secondary.rename(columns={c: f"secondary_{c}" for c in ["open", "high", "low", "close", "volume"]})
    df = a.merge(b, on="date", how="inner")
    for f in ["open", "high", "low", "close"]:
        df[f"{f}_diff"] = (df[f"primary_{f}"] - df[f"secondary_{f}"]).abs()
    df["verified"] = (df[["open_diff", "high_diff", "low_diff", "close_diff"]].max(axis=1) <= tolerance).astype(int)
    for f in ["open", "high", "low", "close"]:
        df[f] = df[f"primary_{f}"]
    return df.sort_values("date").reset_index(drop=True)

# ── signal engine ─────────────────────────────────────────────────────────
def enrich_and_signal(df: pd.DataFrame, cfg: ScanRequest) -> pd.DataFrame:
    out = df.copy()
    rng = out["high"] - out["low"]
    out["ibs"] = np.where(rng != 0, (out["close"] - out["low"]) / rng, np.nan)
    out["avg_range"] = rng.rolling(cfg.lookback_range).mean()
    out["rolling_high"] = out["high"].rolling(cfg.lookback_high).max()
    out["range_factor"] = (out["close"] - out["rolling_high"]).abs() / out["avg_range"]
    cond = (
        (out["verified"] == 1)
        & (out["ibs"] < cfg.ibs_threshold)
        & (out["close"] < (out["rolling_high"] - cfg.range_factor_threshold * out["avg_range"]))
    )
    if cfg.sma_filter:
        out["sma"] = out["close"].rolling(cfg.sma_filter).mean()
        cond &= out["close"] > out["sma"]
    out["signal"] = cond.fillna(False).astype(int)
    return out

# ── position sizing ───────────────────────────────────────────────────────
def calc_position_size(entry: float, stop: float, equity: float, risk_frac: float) -> dict:
    if stop >= entry or entry <= 0:
        return {"shares": 0, "dollar_risk": 0, "notional": 0}
    dollar_risk = equity * risk_frac
    risk_per_share = entry - stop
    shares = max(1, int(dollar_risk / risk_per_share))
    return {"shares": shares, "dollar_risk": round(dollar_risk, 2), "notional": round(shares * entry, 2)}

# ── backtest ──────────────────────────────────────────────────────────────
def backtest(signals: pd.DataFrame, symbol: str, cfg: ScanRequest):
    trades, curve = [], []
    equity = cfg.account_equity
    peak = equity
    max_dd = 0.0
    i = 0
    while i < len(signals) - 1:
        row = signals.iloc[i]
        if int(row.get("signal", 0)) != 1:
            curve.append({"date": str(row["date"].date()), "equity": equity})
            i += 1
            continue
        entry_idx = i + 1
        if entry_idx >= len(signals):
            break
        entry_row = signals.iloc[entry_idx]
        entry = float(entry_row["close"])
        stop = float(entry_row["low"])
        sizing = calc_position_size(entry, stop, equity, cfg.risk_frac)
        shares = sizing["shares"]
        exit_idx = None
        for j in range(entry_idx + 1, min(len(signals), entry_idx + 1 + cfg.max_hold_bars)):
            if float(signals.iloc[j]["close"]) > float(signals.iloc[j - 1]["high"]):
                exit_idx = j
                break
        if exit_idx is None:
            exit_idx = min(len(signals) - 1, entry_idx + cfg.max_hold_bars)
        exit_row = signals.iloc[exit_idx]
        exit_price = float(exit_row["close"])
        gross = exit_price / entry - 1
        cost = (cfg.fee_bps + cfg.slippage_bps) * 2 / 10000
        net = gross - cost
        pnl = shares * (exit_price - entry) - cost * shares * entry
        equity += pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, (equity - peak) / peak)
        trades.append({
            "symbol": symbol,
            "signal_date": str(row["date"].date()),
            "entry_date": str(entry_row["date"].date()),
            "exit_date": str(exit_row["date"].date()),
            "entry_price": entry, "exit_price": exit_price,
            "shares": shares, "dollar_risk": sizing["dollar_risk"],
            "gross_return": round(gross, 6), "net_return": round(net, 6),
            "pnl_dollars": round(pnl, 2), "bars_held": int(exit_idx - entry_idx),
            "verified": int(row["verified"]), "ibs": float(row["ibs"]),
            "range_factor": float(row["range_factor"]) if pd.notna(row.get("range_factor", float("nan"))) else None,
            "equity_after": round(equity, 2),
        })
        curve.append({"date": str(exit_row["date"].date()), "equity": round(equity, 2)})
        i = exit_idx + 1
    return pd.DataFrame(trades), pd.DataFrame(curve), round(max_dd * 100, 2)

# ── portfolio analytics ───────────────────────────────────────────────────
def portfolio_analytics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    wins = df[df["net_return"] > 0]
    losses = df[df["net_return"] <= 0]
    gw = wins["net_return"].sum()
    gl = losses["net_return"].abs().sum()
    by_symbol = (
        df.groupby("symbol")
        .agg(
            trades=("net_return", "count"),
            win_rate=("net_return", lambda x: round(float((x > 0).mean()), 4)),
            avg_net=("net_return", "mean"),
            total_pnl=("pnl_dollars", "sum"),
        )
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "total_trades": int(len(df)),
        "win_rate": round(float((df["net_return"] > 0).mean()), 4),
        "avg_net_return": round(float(df["net_return"].mean()), 6),
        "profit_factor": round(float(gw / gl), 4) if gl > 0 else None,
        "total_pnl_dollars": round(float(df["pnl_dollars"].sum()), 2),
        "avg_hold_bars": round(float(df["bars_held"].mean()), 2),
        "by_symbol": by_symbol,
    }

# ── persist ───────────────────────────────────────────────────────────────
def persist_symbol(symbol, verified, signals, trades_df):
    conn = get_conn()
    def d(v):
        return str(v.date()) if hasattr(v, "date") else str(v)
    conn.executemany(
        "INSERT OR REPLACE INTO verification_audit VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(symbol, d(r["date"]), r.get("primary_open"), r.get("primary_high"), r.get("primary_low"), r.get("primary_close"),
          r.get("secondary_open"), r.get("secondary_high"), r.get("secondary_low"), r.get("secondary_close"),
          r.get("open_diff"), r.get("high_diff"), r.get("low_diff"), r.get("close_diff"), int(r["verified"]))
         for _, r in verified.tail(500).iterrows()],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?)",
        [(symbol, d(r["date"]),
          None if pd.isna(r["ibs"]) else float(r["ibs"]),
          None if pd.isna(r.get("range_factor", float("nan"))) else float(r["range_factor"]),
          int(r["verified"]), int(r["signal"]), float(r["close"]))
         for _, r in signals.tail(500).iterrows()],
    )
    if not trades_df.empty:
        conn.executemany(
            """INSERT INTO trades
               (symbol,signal_date,entry_date,exit_date,entry_price,exit_price,shares,dollar_risk,
                gross_return,net_return,pnl_dollars,bars_held,verified,ibs,range_factor,equity_after)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(r["symbol"], r["signal_date"], r["entry_date"], r["exit_date"],
              r["entry_price"], r["exit_price"], r.get("shares"), r.get("dollar_risk"),
              r["gross_return"], r["net_return"], r.get("pnl_dollars"),
              int(r["bars_held"]), int(r["verified"]), r["ibs"],
              r.get("range_factor"), r["equity_after"])
             for _, r in trades_df.iterrows()],
        )
    conn.commit()
    conn.close()

# ── broker ────────────────────────────────────────────────────────────────
def submit_alpaca_order(symbol, qty, signal_date):
    if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
        return {"status": "skipped", "reason": "Alpaca keys not configured"}
    headers = {"APCA-API-KEY-ID": ALPACA_KEY_ID, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}
    payload = {"symbol": symbol, "qty": str(qty), "side": "buy", "type": "market", "time_in_force": "day"}
    r = httpx.post(f"{ALPACA_BASE_URL}/v2/orders", json=payload, headers=headers, timeout=15)
    resp = r.json()
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO broker_orders VALUES (?,?,?,?,?,?,?,?,?)",
        (resp.get("id", ""), symbol, signal_date, qty, "buy", "market",
         resp.get("status", ""), datetime.utcnow().isoformat(), str(resp)),
    )
    conn.commit()
    conn.close()
    return resp

# ── core scan ─────────────────────────────────────────────────────────────
def run_scan(cfg: ScanRequest):
    symbols = [s.strip().upper() for s in cfg.symbols if s.strip()]
    results, all_trades = [], []
    for symbol in symbols:
        try:
            primary = twelve_data_daily(symbol)
            start = str(primary["date"].min().date())
            end = str(primary["date"].max().date())
            secondary = polygon_daily(symbol, start, end)
            verified = verify_sources(primary, secondary, cfg.source_tolerance)
            signals = enrich_and_signal(verified, cfg)
            trades_df, _, max_dd = backtest(signals, symbol, cfg)
            persist_symbol(symbol, verified, signals, trades_df)
            all_trades.append(trades_df)
            latest = signals.iloc[-1]
            latest_v = verified.iloc[-1]
            sig_val = int(latest["signal"])
            ver_val = int(latest["verified"])
            broker_resp = None
            if sig_val == 1 and ver_val == 1 and cfg.auto_execute:
                entry = float(primary["close"].iloc[-1])
                stop = float(primary["low"].iloc[-1])
                sz = calc_position_size(entry, stop, cfg.account_equity, cfg.risk_frac)
                if sz["shares"] > 0:
                    broker_resp = submit_alpaca_order(
                        symbol, sz["shares"],
                        str(latest["date"].date()) if hasattr(latest["date"], "date") else str(latest["date"]),
                    )
            results.append({
                "symbol": symbol,
                "date": str(latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]),
                "ibs": None if pd.isna(latest["ibs"]) else round(float(latest["ibs"]), 4),
                "range_factor": None if pd.isna(latest.get("range_factor", float("nan"))) else round(float(latest["range_factor"]), 4),
                "verified": ver_val, "signal": sig_val,
                "close": round(float(latest["close"]), 4),
                "close_diff": round(float(latest_v["close_diff"]), 4),
                "max_drawdown_pct": max_dd,
                "trade_count": int(len(trades_df)),
                "win_rate": round(float((trades_df["net_return"] > 0).mean()), 4) if len(trades_df) else 0.0,
                "broker_order": broker_resp,
            })
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})
    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return results, portfolio_analytics(combined)

# ── routes ────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat(), "version": "1.0.0"}

@app.get("/api/config", dependencies=[Depends(require_auth)])
def get_config():
    return {
        "symbols": DEFAULT_SYMBOLS, "primary_source": "TWELVE_DATA",
        "secondary_source": "POLYGON", "risk_frac": RISK_FRAC,
        "alpaca_configured": bool(ALPACA_KEY_ID),
    }

@app.post("/api/scan", dependencies=[Depends(require_auth)])
def scan(req: ScanRequest):
    results, analytics = run_scan(req)
    return {"results": results, "portfolio_analytics": analytics}

# ── cron endpoint — accepts GET (Vercel built-in) AND POST (external cron) ──
@app.api_route("/api/cron/eod-scan", methods=["GET", "POST"])
async def cron_eod_scan(request: Request):
    authorization = request.headers.get("Authorization", "")
    if CRON_SECRET and authorization != f"Bearer {CRON_SECRET}":
        raise HTTPException(401, detail="Unauthorized")
    cfg = ScanRequest(symbols=DEFAULT_SYMBOLS, auto_execute=True)
    results, analytics = run_scan(cfg)
    return {"ok": True, "scanned": len(results), "analytics": analytics, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/dashboard-summary", dependencies=[Depends(require_auth)])
def dashboard_summary():
    conn = get_conn()
    latest_signals = [dict(r) for r in conn.execute("SELECT * FROM signals ORDER BY date DESC LIMIT 100").fetchall()]
    latest_trades  = [dict(r) for r in conn.execute("SELECT * FROM trades ORDER BY trade_id DESC LIMIT 500").fetchall()]
    journal        = [dict(r) for r in conn.execute("SELECT * FROM trade_journal ORDER BY journal_id DESC LIMIT 50").fetchall()]
    broker_orders  = [dict(r) for r in conn.execute("SELECT * FROM broker_orders ORDER BY submitted_at DESC LIMIT 20").fetchall()]
    conn.close()
    trades_df = pd.DataFrame(latest_trades)
    analytics = portfolio_analytics(trades_df) if not trades_df.empty else {}
    return {
        "signals_count": sum(1 for s in latest_signals if s.get("signal") == 1),
        "verified_count": sum(1 for s in latest_signals if s.get("verified") == 1),
        "portfolio_analytics": analytics, "latest_signals": latest_signals,
        "latest_trades": latest_trades, "journal": journal, "broker_orders": broker_orders,
    }

@app.get("/api/equity-curve", dependencies=[Depends(require_auth)])
def equity_curve(limit: int = Query(300, ge=10, le=2000)):
    conn = get_conn()
    rows = conn.execute("SELECT exit_date as date, equity_after as equity FROM trades ORDER BY exit_date ASC").fetchall()
    conn.close()
    return {"points": [dict(r) for r in rows][-limit:]}

@app.get("/api/verification/{symbol}", dependencies=[Depends(require_auth)])
def verification(symbol: str, limit: int = Query(50, ge=1, le=500)):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM verification_audit WHERE symbol=? ORDER BY date DESC LIMIT ?",
        (symbol.upper(), limit),
    ).fetchall()
    conn.close()
    return {"rows": [dict(r) for r in rows]}

@app.get("/api/portfolio/analytics", dependencies=[Depends(require_auth)])
def port_analytics():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM trades").fetchall()
    conn.close()
    return portfolio_analytics(pd.DataFrame([dict(r) for r in rows]))

@app.get("/api/journal", dependencies=[Depends(require_auth)])
def get_journal(limit: int = Query(100, ge=1, le=500)):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM trade_journal ORDER BY journal_id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return {"rows": [dict(r) for r in rows]}

@app.post("/api/journal", dependencies=[Depends(require_auth)])
def add_journal(trade: JournalTrade):
    conn = get_conn()
    conn.execute(
        """INSERT INTO trade_journal
        (symbol,trade_date,side,entry_price,exit_price,shares,r_multiple,
         source_1_close,source_2_close,verified,notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (trade.symbol.upper(), trade.trade_date, trade.side,
         trade.entry_price, trade.exit_price, trade.shares, trade.r_multiple,
         trade.source_1_close, trade.source_2_close, trade.verified, trade.notes),
    )
    conn.commit()
    conn.close()
    return {"ok": True}

@app.get("/api/broker/orders", dependencies=[Depends(require_auth)])
def broker_orders_list():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM broker_orders ORDER BY submitted_at DESC LIMIT 100").fetchall()
    conn.close()
    return {"orders": [dict(r) for r in rows]}

# ── CSV exports ───────────────────────────────────────────────────────────
def _csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

@app.get("/api/export/trades.csv", dependencies=[Depends(require_auth)])
def export_trades():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM trades ORDER BY trade_id ASC", con=conn)
    conn.close()
    return _csv_response(df, "trades.csv")

@app.get("/api/export/signals.csv", dependencies=[Depends(require_auth)])
def export_signals():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM signals ORDER BY date ASC", con=conn)
    conn.close()
    return _csv_response(df, "signals.csv")

@app.get("/api/export/verification.csv", dependencies=[Depends(require_auth)])
def export_verification():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM verification_audit ORDER BY date ASC", con=conn)
    conn.close()
    return _csv_response(df, "verification_audit.csv")

@app.get("/api/export/journal.csv", dependencies=[Depends(require_auth)])
def export_journal():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM trade_journal ORDER BY journal_id ASC", con=conn)
    conn.close()
    return _csv_response(df, "journal.csv")

@app.get("/api/export/portfolio-analytics.csv", dependencies=[Depends(require_auth)])
def export_portfolio():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM trades", con=conn)
    conn.close()
    a = portfolio_analytics(df)
    return _csv_response(pd.DataFrame(a.get("by_symbol", [])), "portfolio_analytics.csv")
