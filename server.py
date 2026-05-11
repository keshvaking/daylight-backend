"""
╔══════════════════════════════════════════════════════════════════╗
║   DAYLIGHT STRATEGY v2 — BACKEND SERVER                        ║
║   Flask REST API + SSE for the web dashboard                   ║
║   Runs backtests & paper trading in background threads         ║
║   Closing the browser does NOT stop the test.                  ║
╚══════════════════════════════════════════════════════════════════╝

USAGE:
  pip install flask flask-cors python-binance pandas numpy matplotlib
  python server.py

Then open the dashboard at:  http://localhost:5000
"""

import os
import sys
import json
import time
import queue
import threading
import itertools
import warnings
import traceback
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

# ── Try to import Binance; warn if missing ────────────────────────
try:
    from binance.client import Client as BinanceClient
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("[WARN] python-binance not installed. Install: pip install python-binance")

# ══════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder="static")
CORS(app)

# ══════════════════════════════════════════════════════════════════
# GLOBAL STATE  (thread-safe with Lock)
# ══════════════════════════════════════════════════════════════════
STATE_FILE       = "daylight_state.json"
PAPER_STATE_FILE = "paper_trades.json"

_lock = threading.Lock()

# Live broadcast queues (one per SSE client)
_sse_queues: list[queue.Queue] = []

# Current job state
_job = {
    "type"       : None,    # "backtest" | "paper" | None
    "status"     : "idle",  # idle | running | done | error
    "progress"   : 0,
    "message"    : "",
    "thread"     : None,
    "results"    : None,    # final backtest results
    "paper_state": None,    # live paper state
    "logs"       : [],      # last 200 log lines
    "started_at" : None,
}

# ══════════════════════════════════════════════════════════════════
# STRATEGY CONSTANTS  (identical to original)
# ══════════════════════════════════════════════════════════════════
ENTRY_TF  = "5m"
TREND_TF  = "1h"
LOOKBACK  = "60 days ago UTC"

LONDON_START = (13, 30); LONDON_END = (17, 30)
NY_START     = (18, 30); NY_END     = (23,  0)

OPT_SWING_LB  = [10, 20, 30]
OPT_RR        = [1.5, 2.0, 2.5]
OPT_MIN_RISK  = [30.0, 50.0, 100.0]

PAPER_POLL_SEC = 30
PAPER_CANDLES  = 100

# ══════════════════════════════════════════════════════════════════
# SSE BROADCAST
# ══════════════════════════════════════════════════════════════════

def broadcast(event: str, data: dict):
    """Push an SSE event to all connected clients."""
    msg = f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
    dead = []
    for q in _sse_queues:
        try:
            q.put_nowait(msg)
        except queue.Full:
            dead.append(q)
    for q in dead:
        try: _sse_queues.remove(q)
        except ValueError: pass


def log(msg: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _lock:
        _job["logs"].append({"ts": ts, "level": level, "msg": msg})
        if len(_job["logs"]) > 200:
            _job["logs"] = _job["logs"][-200:]
    broadcast("log", {"ts": ts, "level": level, "msg": msg})
    print(line)


def set_status(status: str, message: str = "", progress: int = -1):
    with _lock:
        _job["status"]  = status
        _job["message"] = message
        if progress >= 0:
            _job["progress"] = progress
    broadcast("status", {"status": status, "message": message,
                         "progress": _job["progress"]})


def set_progress(pct: int, msg: str = ""):
    with _lock:
        _job["progress"] = pct
        if msg: _job["message"] = msg
    broadcast("progress", {"progress": pct, "message": msg})


# ══════════════════════════════════════════════════════════════════
# STRATEGY LOGIC  (ported from original, unchanged logic)
# ══════════════════════════════════════════════════════════════════

def fetch_ohlcv(client, symbol, interval, lookback):
    log(f"Fetching {symbol} {interval} …")
    klines = client.get_historical_klines(symbol, interval, lookback)
    if not klines:
        log(f"No data for {symbol} {interval}", "warn")
        return pd.DataFrame()
    df = pd.DataFrame(klines, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp","open","high","low","close","volume"]].reset_index(drop=True)
    log(f"Got {len(df)} candles for {symbol} {interval}")
    return df


def in_session(ts: pd.Timestamp) -> bool:
    h, m = ts.hour, ts.minute
    mins = h * 60 + m
    london = (LONDON_START[0]*60 + LONDON_START[1]) <= mins <= (LONDON_END[0]*60 + LONDON_END[1])
    ny     = (NY_START[0]*60    + NY_START[1])     <= mins <= (NY_END[0]*60    + NY_END[1])
    return london or ny


def get_htf_bias(ts_5m, df_1h):
    past_1h = df_1h[df_1h["timestamp"] < ts_5m]
    if past_1h.empty: return None
    last = past_1h.iloc[-1]
    if   last["close"] > last["open"]: return "BULL"
    elif last["close"] < last["open"]: return "BEAR"
    return None


def detect_signal(i, df_5m, df_1h, params):
    cur = df_5m.iloc[i]
    if not in_session(cur["timestamp"]): return None
    bias = get_htf_bias(cur["timestamp"], df_1h)
    if bias is None: return None
    lb = params["SWING_LOOKBACK"]
    if i < lb: return None

    window  = df_5m.iloc[i - lb: i]
    sw_high = window["high"].max()
    sw_low  = window["low"].min()

    # SHORT
    if bias == "BEAR" and cur["high"] > sw_high and cur["close"] < sw_high:
        entry = cur["close"]
        sl    = cur["high"] + params["SL_BUFFER"]
        risk  = sl - entry
        if params["MIN_RISK_PTS"] <= risk <= params["MAX_RISK_PTS"]:
            return {"direction":"SHORT","entry":round(entry,4),
                    "sl":round(sl,4),"tp":round(entry - risk*params["RR_RATIO"],4),
                    "risk":round(risk,4),"timestamp":cur["timestamp"],
                    "sw_high":round(sw_high,4),"sw_low":round(sw_low,4)}

    # LONG
    elif bias == "BULL" and cur["low"] < sw_low and cur["close"] > sw_low:
        entry = cur["close"]
        sl    = cur["low"] - params["SL_BUFFER"]
        risk  = entry - sl
        if params["MIN_RISK_PTS"] <= risk <= params["MAX_RISK_PTS"]:
            return {"direction":"LONG","entry":round(entry,4),
                    "sl":round(sl,4),"tp":round(entry + risk*params["RR_RATIO"],4),
                    "risk":round(risk,4),"timestamp":cur["timestamp"],
                    "sw_high":round(sw_high,4),"sw_low":round(sw_low,4)}
    return None


def run_backtest(df_5m, df_1h, params):
    capital      = params["INITIAL_CAPITAL"]
    risk_pct     = params["RISK_PER_TRADE"]
    fee          = params["FEE_RATE"]
    timeout      = params["TIMEOUT_CANDLES"]
    rr           = params["RR_RATIO"]
    trades       = []
    equity       = capital
    equity_curve = [capital]
    monthly_pnl  = {}
    total_bars   = len(df_5m)

    for i in range(params["SWING_LOOKBACK"], total_bars - timeout):
        sig = detect_signal(i, df_5m, df_1h, params)
        if sig is None:
            equity_curve.append(equity)
            continue

        result     = "TIMEOUT"
        exit_price = sig["entry"]
        future     = df_5m.iloc[i+1 : i+1+timeout]

        for _, fc in future.iterrows():
            if sig["direction"] == "SHORT":
                if fc["low"]  <= sig["tp"]: result = "WIN";  exit_price = sig["tp"]; break
                if fc["high"] >= sig["sl"]: result = "LOSS"; exit_price = sig["sl"]; break
            else:
                if fc["high"] >= sig["tp"]: result = "WIN";  exit_price = sig["tp"]; break
                if fc["low"]  <= sig["sl"]: result = "LOSS"; exit_price = sig["sl"]; break

        risk_dollars = equity * risk_pct
        if result == "WIN":
            pnl = risk_dollars * rr * (1 - fee)
        elif result == "LOSS":
            pnl = -risk_dollars * (1 + fee)
        else:
            pnl = 0.0

        equity = max(equity + pnl, 0)
        month_key = sig["timestamp"].strftime("%Y-%m")
        monthly_pnl[month_key] = round(monthly_pnl.get(month_key, 0) + pnl, 2)

        trade_rec = {
            "direction" : sig["direction"],
            "entry"     : sig["entry"],
            "sl"        : sig["sl"],
            "tp"        : sig["tp"],
            "risk"      : sig["risk"],
            "timestamp" : str(sig["timestamp"]),
            "exit_price": round(exit_price, 4),
            "result"    : result,
            "pnl"       : round(pnl, 2),
            "equity"    : round(equity, 2),
        }
        trades.append(trade_rec)
        equity_curve.append(equity)

        # broadcast live trade as it's found
        broadcast("trade", trade_rec)

    return {"trades": trades, "equity": equity,
            "equity_curve": equity_curve, "monthly_pnl": monthly_pnl}


def compute_stats(results, initial_capital):
    trades = results["trades"]
    if not trades:
        return {"Total Trades": 0}

    df  = pd.DataFrame(trades)
    wins     = len(df[df["result"] == "WIN"])
    losses   = len(df[df["result"] == "LOSS"])
    timeouts = len(df[df["result"] == "TIMEOUT"])
    total    = len(df)
    win_r    = wins / total

    gross_win = df[df["pnl"] > 0]["pnl"].sum()
    gross_los = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf        = round(gross_win / gross_los, 2) if gross_los > 0 else 99.0

    avg_win = float(df[df["pnl"] > 0]["pnl"].mean()) if wins   else 0.0
    avg_los = float(df[df["pnl"] < 0]["pnl"].mean()) if losses else 0.0

    # Streaks
    outcomes = [1 if r == "WIN" else 0 for r in df["result"]]
    mws = mls = cur = 0
    if outcomes:
        cur = 1
        for j in range(1, len(outcomes)):
            if outcomes[j] == outcomes[j-1]: cur += 1
            else:
                if outcomes[j-1] == 1: mws = max(mws, cur)
                else:                  mls = max(mls, cur)
                cur = 1
        if outcomes[-1] == 1: mws = max(mws, cur)
        else:                  mls = max(mls, cur)

    eq       = pd.Series(results["equity_curve"])
    max_dd   = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)
    tot_ret  = (results["equity"] - initial_capital) / initial_capital * 100
    eq_ret   = eq.pct_change().dropna()
    ann      = np.sqrt(252 * 288)
    sharpe   = float(eq_ret.mean() / eq_ret.std() * ann) if eq_ret.std() > 0 else 0.0

    ev = (win_r * avg_win) - ((1 - win_r) * abs(avg_los))

    # Session breakdown
    if "timestamp" in df.columns:
        df["ts_obj"] = pd.to_datetime(df["timestamp"])
        def is_london(ts):
            try:
                h, m = ts.hour, ts.minute
                mins = h * 60 + m
                return (LONDON_START[0]*60 + LONDON_START[1]) <= mins <= (LONDON_END[0]*60 + LONDON_END[1])
            except: return False
        df["in_london"] = df["ts_obj"].apply(is_london)
        london_wins = int(len(df[df["in_london"] & (df["result"]=="WIN")]))
        ny_wins     = int(len(df[~df["in_london"] & (df["result"]=="WIN")]))
        london_tot  = int(len(df[df["in_london"]]))
        ny_tot      = int(len(df[~df["in_london"]]))
    else:
        london_wins = ny_wins = london_tot = ny_tot = 0

    return {
        "totalTrades"  : total,
        "wins"         : wins,
        "losses"       : losses,
        "timeouts"     : timeouts,
        "winRate"      : round(win_r * 100, 1),
        "profitFactor" : pf,
        "avgWin"       : round(avg_win, 2),
        "avgLoss"      : round(avg_los, 2),
        "bestTrade"    : round(float(df["pnl"].max()), 2),
        "worstTrade"   : round(float(df["pnl"].min()), 2),
        "maxWinStreak" : mws,
        "maxLossStreak": mls,
        "evPerTrade"   : round(ev, 2),
        "londonWR"     : round(london_wins/london_tot*100, 1) if london_tot else None,
        "nyWR"         : round(ny_wins/ny_tot*100, 1)         if ny_tot     else None,
        "totalReturn"  : round(tot_ret, 2),
        "maxDrawdown"  : round(max_dd, 2),
        "sharpeRatio"  : round(sharpe, 2),
        "finalCapital" : round(results["equity"], 2),
        "monthlyPnl"   : results["monthly_pnl"],
        "equityCurve"  : results["equity_curve"],
    }


# ══════════════════════════════════════════════════════════════════
# BACKTEST THREAD
# ══════════════════════════════════════════════════════════════════

def backtest_thread(cfg: dict):
    try:
        set_status("running", "Connecting to Binance …", 2)

        if not BINANCE_AVAILABLE:
            set_status("error", "python-binance not installed"); return

        client = BinanceClient(cfg["apiKey"], cfg["apiSecret"])

        symbols = cfg["symbols"]
        variant = cfg["variant"]
        capital = float(cfg["capital"])
        risk    = float(cfg["riskPct"]) / 100.0

        all_results = []
        total_steps = len(symbols) * (3 if variant in ("all","optimise") else 1)
        step = 0

        for symbol in symbols:
            is_eth = symbol == "ETHUSDT"
            base_params = {
                "SWING_LOOKBACK"  : 20,
                "RR_RATIO"        : 2.0,
                "MIN_RISK_PTS"    : 5.0  if is_eth else 50.0,
                "MAX_RISK_PTS"    : 200.0 if is_eth else 2000.0,
                "SL_BUFFER"       : 1.0  if is_eth else 10.0,
                "TIMEOUT_CANDLES" : 200,
                "INITIAL_CAPITAL" : capital,
                "RISK_PER_TRADE"  : risk,
                "FEE_RATE"        : float(cfg.get("fee", 0.001)),
            }

            log(f"Fetching data for {symbol} …")
            set_progress(int(step/total_steps*80) + 5, f"Fetching {symbol} data …")
            df_5m = fetch_ohlcv(client, symbol, ENTRY_TF, LOOKBACK)
            df_1h = fetch_ohlcv(client, symbol, TREND_TF, LOOKBACK)

            if df_5m.empty or df_1h.empty:
                log(f"No data for {symbol}, skipping.", "warn"); continue

            # Build variants list
            if variant == "default":
                variants = [("Default (RR 2.0, Swing 20)", base_params)]
            elif variant in ("all", "optimise"):
                variants = [
                    ("Default (RR 2.0, Swing 20)", base_params),
                    ("Conservative (RR 1.5, Swing 10)",
                     {**base_params, "RR_RATIO": 1.5, "SWING_LOOKBACK": 10}),
                    ("Aggressive (RR 2.5, Swing 30)",
                     {**base_params, "RR_RATIO": 2.5, "SWING_LOOKBACK": 30}),
                ]
            else:
                variants = [("Default (RR 2.0, Swing 20)", base_params)]

            for label, params in variants:
                log(f"Running backtest: {symbol} — {label}")
                set_progress(int(step/total_steps*80) + 10,
                             f"Backtesting {symbol} {label} …")
                res   = run_backtest(df_5m, df_1h, params)
                stats = compute_stats(res, capital)

                result_obj = {
                    "symbol"  : symbol,
                    "label"   : f"{symbol} {label}",
                    "params"  : {k: v for k, v in params.items()
                                 if k not in ("INITIAL_CAPITAL","RISK_PER_TRADE","FEE_RATE")},
                    "stats"   : stats,
                    "trades"  : res["trades"],
                }
                all_results.append(result_obj)
                log(f"Done: {symbol} {label} — {stats['totalTrades']} trades, "
                    f"WR={stats['winRate']}%, PF={stats['profitFactor']}")

                broadcast("result_ready", result_obj)
                step += 1

        # Optimisation grid
        opt_results = []
        if variant == "optimise" and symbols:
            symbol = symbols[0]
            is_eth = symbol == "ETHUSDT"
            base_p = {
                "MIN_RISK_PTS"    : 5.0  if is_eth else 50.0,
                "MAX_RISK_PTS"    : 200.0 if is_eth else 2000.0,
                "SL_BUFFER"       : 1.0  if is_eth else 10.0,
                "TIMEOUT_CANDLES" : 200,
                "INITIAL_CAPITAL" : capital,
                "RISK_PER_TRADE"  : risk,
                "FEE_RATE"        : float(cfg.get("fee", 0.001)),
            }
            df_5m_o = fetch_ohlcv(client, symbol, ENTRY_TF, LOOKBACK)
            df_1h_o = fetch_ohlcv(client, symbol, TREND_TF, LOOKBACK)
            combos  = list(itertools.product(OPT_SWING_LB, OPT_RR, OPT_MIN_RISK))
            log(f"Running optimisation — {len(combos)} combos …")

            for n, (swing, rr, min_r) in enumerate(combos, 1):
                set_progress(80 + int(n/len(combos)*18),
                             f"Optimising combo {n}/{len(combos)} …")
                p = {**base_p, "SWING_LOOKBACK": swing, "RR_RATIO": rr, "MIN_RISK_PTS": min_r}
                try:
                    res  = run_backtest(df_5m_o, df_1h_o, p)
                    st   = compute_stats(res, capital)
                    opt_results.append({
                        "swingLB" : swing, "rr": rr, "minRisk": min_r,
                        "winRate" : st["winRate"],
                        "pf"      : st["profitFactor"],
                        "sharpe"  : st["sharpeRatio"],
                        "trades"  : st["totalTrades"],
                    })
                except Exception as e:
                    log(f"Combo error: {e}", "warn")

            opt_results.sort(key=lambda x: (x["pf"], x["sharpe"]), reverse=True)
            broadcast("opt_results", {"results": opt_results[:10]})

        final = {
            "runs"       : all_results,
            "optResults" : opt_results,
            "completedAt": datetime.now().isoformat(),
        }

        with _lock:
            _job["results"] = final

        # Persist to disk
        with open(STATE_FILE, "w") as f:
            json.dump(final, f, default=str, indent=2)

        set_status("done", "Backtest complete ✓", 100)
        broadcast("done", final)
        log("Backtest finished successfully.", "success")

    except Exception as e:
        tb = traceback.format_exc()
        log(f"Backtest error: {e}\n{tb}", "error")
        set_status("error", str(e))


# ══════════════════════════════════════════════════════════════════
# PAPER TRADING THREAD
# ══════════════════════════════════════════════════════════════════

def paper_thread(cfg: dict):
    try:
        set_status("running", "Starting paper trading …", 5)

        if not BINANCE_AVAILABLE:
            set_status("error", "python-binance not installed"); return

        client = BinanceClient(cfg["apiKey"], cfg["apiSecret"])

        capital  = float(cfg["capital"])
        risk_pct = float(cfg["riskPct"]) / 100.0
        variant  = cfg.get("variant", "default")
        fee      = float(cfg.get("fee", 0.001))

        # Symbol configs
        symbols = cfg["symbols"]
        def make_params(sym):
            is_eth = sym == "ETHUSDT"
            p = {
                "SWING_LOOKBACK"  : 20,
                "RR_RATIO"        : 2.0,
                "MIN_RISK_PTS"    : 5.0  if is_eth else 50.0,
                "MAX_RISK_PTS"    : 200.0 if is_eth else 2000.0,
                "SL_BUFFER"       : 1.0  if is_eth else 10.0,
                "TIMEOUT_CANDLES" : 200,
                "INITIAL_CAPITAL" : capital,
                "RISK_PER_TRADE"  : risk_pct,
                "FEE_RATE"        : fee,
            }
            if variant == "conservative":
                p["RR_RATIO"] = 1.5; p["SWING_LOOKBACK"] = 10
            elif variant == "aggressive":
                p["RR_RATIO"] = 2.5; p["SWING_LOOKBACK"] = 30
            return p

        params_map = {s: make_params(s) for s in symbols}

        # Load or create paper state
        if os.path.exists(PAPER_STATE_FILE):
            with open(PAPER_STATE_FILE) as f:
                ps = json.load(f)
            # If capital changed meaningfully, start fresh
            if abs(ps.get("capital", 0) - capital) > capital * 0.05:
                log("Capital changed >5%, starting fresh paper session.")
                ps = None
        else:
            ps = None

        if ps is None:
            ps = {
                "capital"       : capital,
                "start_capital" : capital,
                "open_trades"   : {},
                "closed_trades" : [],
                "equity_log"    : [capital],
                "started_at"    : datetime.now().isoformat(),
            }

        with _lock:
            _job["paper_state"] = ps

        log(f"Paper trading started — capital: ${ps['capital']:,.2f}  "
            f"open: {len(ps['open_trades'])}  closed: {len(ps['closed_trades'])}")

        poll = 0
        while True:
            # Check if stop was requested
            with _lock:
                if _job["status"] == "stopping":
                    break

            poll += 1
            now = datetime.now().strftime("%H:%M:%S")
            log(f"Poll #{poll} — capital: ${ps['capital']:,.2f}  "
                f"open: {len(ps['open_trades'])}  closed: {len(ps['closed_trades'])}")

            # 1. Check exits
            _paper_check_exits(ps, client, params_map)

            # 2. Scan for signals
            for sym in symbols:
                _paper_scan_signals(ps, client, sym, params_map[sym])

            # 3. Save state
            with open(PAPER_STATE_FILE, "w") as f:
                json.dump(ps, f, default=str, indent=2)

            with _lock:
                _job["paper_state"] = dict(ps)

            broadcast("paper_update", {
                "capital"     : ps["capital"],
                "openTrades"  : ps["open_trades"],
                "closedTrades": ps["closed_trades"][-20:],
                "equityLog"   : ps["equity_log"][-500:],
            })

            set_progress(50, f"Monitoring — poll #{poll}")

            # Sleep in 1-second chunks so we can check stop flag
            for _ in range(PAPER_POLL_SEC):
                time.sleep(1)
                with _lock:
                    if _job["status"] == "stopping":
                        break

        # Stopped
        with open(PAPER_STATE_FILE, "w") as f:
            json.dump(ps, f, default=str, indent=2)
        set_status("idle", "Paper trading stopped.")
        log("Paper trading stopped. State saved to disk.")

    except Exception as e:
        tb = traceback.format_exc()
        log(f"Paper trade error: {e}\n{tb}", "error")
        set_status("error", str(e))


def _paper_check_exits(ps, client, params_map):
    capital  = ps["capital"]
    to_close = []

    for symbol, trade in ps["open_trades"].items():
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            price  = float(ticker["price"])
        except Exception as e:
            log(f"Price fetch failed for {symbol}: {e}", "warn"); continue

        result     = None
        exit_price = price

        if trade["direction"] == "LONG":
            if price <= trade["sl"]: result = "LOSS"; exit_price = trade["sl"]
            elif price >= trade["tp"]: result = "WIN";  exit_price = trade["tp"]
        else:
            if price >= trade["sl"]: result = "LOSS"; exit_price = trade["sl"]
            elif price <= trade["tp"]: result = "WIN";  exit_price = trade["tp"]

        if result:
            pm      = params_map.get(symbol, {})
            rr      = pm.get("RR_RATIO", 2.0)
            fee     = pm.get("FEE_RATE", 0.001)
            rd      = trade.get("risk_dollars", capital * pm.get("RISK_PER_TRADE", 0.01))
            pnl     = (rd * rr * (1 - fee)) if result == "WIN" else (-rd * (1 + fee))
            capital = max(capital + pnl, 0)

            closed = {**trade, "symbol": symbol,
                      "exit_price": round(exit_price, 4),
                      "result": result, "pnl": round(pnl, 2),
                      "closed_at": datetime.now().isoformat(),
                      "capital_after": round(capital, 2)}
            ps["closed_trades"].append(closed)
            ps["equity_log"].append(capital)
            to_close.append(symbol)

            icon = "WIN ✅" if result == "WIN" else "LOSS ❌"
            log(f"{icon}  {symbol} {trade['direction']}  "
                f"entry={trade['entry']:.4f}  exit={exit_price:.4f}  "
                f"PnL=${pnl:+.2f}  cap=${capital:,.2f}",
                "success" if result == "WIN" else "warn")
            broadcast("paper_trade_closed", closed)

    for sym in to_close:
        del ps["open_trades"][sym]
    ps["capital"] = round(capital, 2)


def _paper_scan_signals(ps, client, symbol, params):
    if symbol in ps["open_trades"]: return

    df_5m = fetch_ohlcv(client, symbol, ENTRY_TF,
                        f"{PAPER_CANDLES + 10} minutes ago UTC")
    df_1h = fetch_ohlcv(client, symbol, TREND_TF, "48 hours ago UTC")

    if df_5m.empty or df_1h.empty: return

    i = len(df_5m) - 2
    if i < params["SWING_LOOKBACK"]: return

    sig = detect_signal(i, df_5m, df_1h, params)
    if sig is None: return

    rd = ps["capital"] * params["RISK_PER_TRADE"]
    trade = {**sig, "symbol": symbol, "risk_dollars": round(rd, 2),
             "opened_at": datetime.now().isoformat(),
             "timestamp": str(sig["timestamp"])}
    ps["open_trades"][symbol] = trade

    log(f"🚀 SIGNAL: {symbol} {sig['direction']}  "
        f"entry={sig['entry']:.4f}  sl={sig['sl']:.4f}  tp={sig['tp']:.4f}  "
        f"risk=${rd:.2f}", "success")
    broadcast("paper_signal", trade)


# ══════════════════════════════════════════════════════════════════
# REST API ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({
            "status"   : _job["status"],
            "type"     : _job["type"],
            "progress" : _job["progress"],
            "message"  : _job["message"],
            "startedAt": _job["started_at"],
            "hasResults": _job["results"] is not None,
            "binanceAvailable": BINANCE_AVAILABLE,
        })


@app.route("/api/logs")
def api_logs():
    with _lock:
        return jsonify(_job["logs"][-100:])


@app.route("/api/results")
def api_results():
    with _lock:
        if _job["results"] is None:
            # Try loading from disk
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE) as f:
                    return jsonify(json.load(f))
            return jsonify({"error": "No results available"}), 404
        return jsonify(_job["results"])


@app.route("/api/paper/state")
def api_paper_state():
    with _lock:
        ps = _job["paper_state"]
    if ps is None:
        if os.path.exists(PAPER_STATE_FILE):
            with open(PAPER_STATE_FILE) as f:
                return jsonify(json.load(f))
        return jsonify({"error": "No paper state"}), 404
    return jsonify(ps)


@app.route("/api/start", methods=["POST"])
def api_start():
    cfg = request.json or {}

    with _lock:
        if _job["status"] == "running":
            return jsonify({"error": "A job is already running"}), 409

        # Validate
        if not cfg.get("apiKey") or not cfg.get("apiSecret"):
            return jsonify({"error": "API key and secret required"}), 400

        mode = cfg.get("mode", "backtest")

        _job["status"]    = "running"
        _job["type"]      = mode
        _job["progress"]  = 0
        _job["message"]   = "Starting …"
        _job["results"]   = None
        _job["logs"]      = []
        _job["started_at"] = datetime.now().isoformat()

    if mode == "paper":
        t = threading.Thread(target=paper_thread, args=(cfg,), daemon=True)
    else:
        t = threading.Thread(target=backtest_thread, args=(cfg,), daemon=True)

    with _lock:
        _job["thread"] = t

    t.start()
    return jsonify({"ok": True, "mode": mode})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with _lock:
        status = _job["status"]
        jtype  = _job["type"]

    if status != "running":
        return jsonify({"error": "No active job"}), 400

    if jtype == "paper":
        with _lock:
            _job["status"] = "stopping"
        log("Stop requested — will stop after current poll.", "warn")
    else:
        # Backtest can't be gracefully stopped mid-run
        # (thread is CPU-bound; we just mark it)
        with _lock:
            _job["status"] = "stopping"
        log("Stop requested — backtest will finish current variant.", "warn")

    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with _lock:
        if _job["status"] == "running":
            return jsonify({"error": "Cannot reset while running"}), 409
        _job.update({
            "type": None, "status": "idle", "progress": 0,
            "message": "", "results": None, "paper_state": None, "logs": [],
        })
    return jsonify({"ok": True})


# ── SSE stream ────────────────────────────────────────────────────
@app.route("/api/stream")
def api_stream():
    q = queue.Queue(maxsize=200)
    _sse_queues.append(q)

    def generate():
        # Send current status immediately on connect
        with _lock:
            snapshot = {
                "status"  : _job["status"],
                "type"    : _job["type"],
                "progress": _job["progress"],
                "message" : _job["message"],
            }
        yield f"event: status\ndata: {json.dumps(snapshot)}\n\n"
        yield "event: ping\ndata: connected\n\n"

        while True:
            try:
                msg = q.get(timeout=25)
                yield msg
            except queue.Empty:
                yield "event: ping\ndata: keep-alive\n\n"

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"]     = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"]        = "keep-alive"
    return resp


# ── Serve static files (the dashboard HTML) ──────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "dashboard.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)

    if not os.path.exists("static/dashboard.html"):
        print("[WARN] static/dashboard.html not found.")
        print("       Place the updated dashboard HTML there, or")
        print("       access the API directly at http://localhost:5000/api/status")

    print("""
╔══════════════════════════════════════════════════════════════╗
║   DAYLIGHT STRATEGY v2 — Backend Server                    ║
║   http://localhost:5000                                    ║
║   Ctrl+C to stop server (active jobs keep state on disk)  ║
╚══════════════════════════════════════════════════════════════╝
""")
    # use_reloader=False is important — reloader spawns a second process
    # which would double the background threads
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False,
            threaded=True)
