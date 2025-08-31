from binance.client import Client
import time, json, math, os

from openai import OpenAI
import pandas as pd
import numpy as np
import math

def get_market_data(client, symbol: str = "ETHUSDC", interval: str = "1h",
                    limit: int = 12, warmup: int = 120, include_ohlcv: bool = False) -> dict:
    """
    Snapshot de marché compact et 'pensable' par un LLM.

    Params
    ------
    client : binance.Client
    symbol : ex. "ETHUSDC"
    interval : ex. "1h"
    limit : nb de barres à retourner (résultats finaux)
    warmup : nb de barres à utiliser pour stabiliser les indicateurs
    include_ohlcv : si True, inclut aussi l'OHLCV brut (pour la UI)

    Retour
    ------
    {
      "symbol": ..., "tf": interval, "fees_bps": 7,
      "features": {
        "ohlcv_stats": { ... },          # synthèse H1 (1h/3h/12h)
        "hourly_snapshots": [ ... ],     # 12 barres enrichies (close/EMA/RSI/MACD/ATR)
        "stats": {
          "ema20"/"ema50"/"rsi14": {last,min,max,mean,slope,(above50_cnt)},
          "macd": {last,signal_last,hist_last,hist_mean},
          "atr_pct": {last,mean},
          "h1": {dist_ema50_pct, hh, ll, consec_hist_pos, hist_slope_3}
        },
        "ohlcv": [...]  # optionnel si include_ohlcv=True
      }
    }
    """


    # ------------------------------
    # Helpers locaux
    # ------------------------------
    def _series_stats(vals, rsi=False) -> dict:
        a = np.array([v for v in vals if v == v], dtype=float)  # drop NaN
        if a.size == 0:
            return {}
        x = np.arange(a.size, dtype=float)
        slope = float(np.polyfit(x, a, 1)[0]) if a.size >= 2 else 0.0
        out = {
            "last": round(a[-1], 2),
            "min": round(a.min(), 2),
            "max": round(a.max(), 2),
            "mean": round(a.mean(), 2),
            "slope": round(slope, 4),
        }
        if rsi:
            out["above50_cnt"] = int((a > 50).sum())
        return out

    def _pack_h1(df_in: pd.DataFrame, n: int = 12) -> list[dict]:
        x = df_in.tail(min(n, len(df_in))).copy()
        x["above_ema"] = (x["c"] > x["ema20"]) & (x["ema20"] > x["ema50"])
        x["rsi_trend"] = (
            x["rsi14"].diff().rolling(3)
            .apply(lambda s: 1 if (s > 0).sum() >= 2 else 0)
            .map({1: "up", 0: "down"})
            .fillna("flat")
        )
        # Nettoyage/arrondis
        out = x[[
            "t", "c", "ema20", "ema50", "rsi14",
            "macd_line", "macd_signal", "macd_hist",
            "atr_pct", "above_ema", "rsi_trend"
        ]].rename(columns={"t": "time", "c": "close", "macd_line": "macd", "macd_signal": "macd_sig"}).copy()
        out["time"] = out["time"].astype(int)
        for col, nd in [("close", 2), ("ema20", 2), ("ema50", 2), ("rsi14", 2), ("macd", 3), ("macd_sig", 3), ("macd_hist", 3)]:
            out[col] = out[col].round(nd)
        return out.to_dict("records")

    def _h1_summary(df_in: pd.DataFrame) -> dict:
        if df_in.empty:
            return {"dist_ema50_pct": None, "hh": None, "ll": None, "consec_hist_pos": 0, "hist_slope_3": None}

        close = df_in["c"].values
        ema50 = df_in["ema50"].values
        hist  = (df_in["macd_line"] - df_in["macd_signal"]).tolist()

        dist = round(100 * (close[-1] - ema50[-1]) / ema50[-1], 2) if ema50[-1] == ema50[-1] else None
        hh   = round(float(df_in["h"].max()), 2)
        ll   = round(float(df_in["l"].min()), 2)

        # <<< remplace l'ancienne ligne par ces deux-là >>>
        consec_pos   = _streak_pos(hist)
        hist_slope_3 = round(pd.Series(hist).diff().tail(3).sum(), 3) if len(hist) >= 3 else None

        return {"dist_ema50_pct": dist, "hh": hh, "ll": ll,
                "consec_hist_pos": int(consec_pos), "hist_slope_3": hist_slope_3}
    
    def _streak_pos(arr, eps=1e-9):
        cnt = 0
        for v in reversed(arr):
            if v is None or (isinstance(v, float) and math.isnan(v)) or v <= eps:
                break
            cnt += 1
        return cnt

    def _ohlcv_stats(df_in: pd.DataFrame, lookback: int = 12) -> dict:
        x = df_in.tail(min(lookback, len(df_in))).copy()
        if x.empty:
            return {}
        # variations %
        def pct(a, b):
            return None if (a != a or b != b or b == 0) else round(100 * (a / b - 1), 2)
        chg_1h = pct(x["c"].iloc[-1], x["o"].iloc[-1]) if len(x) >= 1 else None
        chg_3h = pct(x["c"].iloc[-1], x["o"].iloc[-3]) if len(x) >= 3 else None
        chg_12h = pct(x["c"].iloc[-1], x["o"].iloc[0]) if len(x) >= 12 else None
        # range & volumes
        high_12 = round(float(x["h"].max()), 2)
        low_12 = round(float(x["l"].min()), 2)
        avg_range = round(float((x["h"] - x["l"]).mean()), 2)
        vol_sum = round(float(x["v"].sum()), 4)
        vol_avg = round(float(x["v"].mean()), 4)
        # barres vertes/rouges
        green = int((x["c"] > x["o"]).sum())
        red   = int((x["c"] < x["o"]).sum())

        lg = (x["c"] > x["o"]).astype(int)
        lr = (x["c"] < x["o"]).astype(int)

        longest_green = int(lg.groupby((lg==0).cumsum()).cumcount().add(1).max() or 0)
        longest_red   = int(lr.groupby((lr==0).cumsum()).cumcount().add(1).max() or 0)
        return {
            "change_1h_pct": chg_1h, "change_3h_pct": chg_3h, "change_12h_pct": chg_12h,
            "high_12h": high_12, "low_12h": low_12, "avg_range": avg_range,
            "vol_sum": vol_sum, "vol_avg": vol_avg,
            "green_bars": green, "red_bars": red,
            "longest_green_streak": longest_green, "longest_red_streak": longest_red
        }

    # ------------------------------
    # 1) Chargement avec warm-up
    # ------------------------------
    lookback = max(limit, warmup)
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    if not klines:
        return {"symbol": symbol, "tf": interval, "fees_bps": 7,
                "features": {"ohlcv_stats": {}, "hourly_snapshots": [], "stats": {}, **({"ohlcv": []} if include_ohlcv else {})}}

    df_full = pd.DataFrame(klines, columns=["t","o","h","l","c","v","ct","qv","n","tbv","tbq","i"])[["t","o","h","l","c","v"]].astype(float)
    df_full["t"] = (df_full["t"] // 1000).astype(int)

    # ------------------------------
    # 2) Indicateurs (sur warm-up)
    # ------------------------------
    df_full["ema20"] = df_full["c"].ewm(span=20, adjust=False).mean()
    df_full["ema50"] = df_full["c"].ewm(span=50, adjust=False).mean()

    d = df_full["c"].diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_full["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df_full["c"].ewm(span=12, adjust=False).mean()
    ema26 = df_full["c"].ewm(span=26, adjust=False).mean()
    df_full["macd_line"] = ema12 - ema26
    df_full["macd_signal"] = df_full["macd_line"].ewm(span=9, adjust=False).mean()
    df_full["macd_hist"] = df_full["macd_line"] - df_full["macd_signal"]

    prev_c = df_full["c"].shift(1)
    tr = pd.concat(
        [(df_full["h"] - df_full["l"]).abs(),
         (df_full["h"] - prev_c).abs(),
         (df_full["l"] - prev_c).abs()],
        axis=1
    ).max(axis=1)
    df_full["atr_pct"] = (tr.ewm(span=14, adjust=False).mean() / df_full["c"] * 100).round(2)

    # ------------------------------
    # 3) Tronquer proprement à `limit`
    # ------------------------------
    df = df_full.tail(limit).copy()
    n = len(df)

    # ------------------------------
    # 4) Blocs 'features'
    # ------------------------------
    # 4.a) Stats indicateurs
    ema20_s = df["ema20"].tolist()
    ema50_s = df["ema50"].tolist()
    rsi_s   = df["rsi14"].tolist()
    macd_s  = df["macd_line"].tolist()
    sig_s   = df["macd_signal"].tolist()
    hist_s  = df["macd_hist"].tolist()
    atr_s   = df["atr_pct"].tolist()

    stats = {
        "ema20":  _series_stats(ema20_s),
        "ema50":  _series_stats(ema50_s),
        "rsi14":  _series_stats(rsi_s, rsi=True),
        "macd": {
            "last":        (round(macd_s[-1], 3) if macd_s else None),
            "signal_last": (round(sig_s[-1], 3) if sig_s else None),
            "hist_last":   (round(hist_s[-1], 3) if hist_s else None),
            "hist_mean":   (round(float(pd.Series(hist_s).mean()), 3) if hist_s else None),
        },
        "atr_pct": {
            "last": (atr_s[-1] if atr_s else None),
            "mean": (round(float(pd.Series(atr_s).mean()), 2) if atr_s else None),
        },
        "h1": _h1_summary(df),
    }

    # 4.b) Snapshots H1 (12 barres enrichies)
    hourly_snapshots = _pack_h1(df, n=12)

    # 4.c) Synthèse OHLCV (au lieu des lignes brutes)
    ohlcv_stats = _ohlcv_stats(df, lookback=min(12, n))

    # 4.d) Optionnel: OHLCV brut pour l'UI
    features = {
        "ohlcv_stats": ohlcv_stats,
        "hourly_snapshots": hourly_snapshots,
        "stats": stats,
    }
    if include_ohlcv:
        features["ohlcv"] = df.apply(
            lambda r: {
                "t": int(r["t"]),
                "o": round(float(r["o"]), 2),
                "h": round(float(r["h"]), 2),
                "l": round(float(r["l"]), 2),
                "c": round(float(r["c"]), 2),
                "v": round(float(r["v"]), 4),
            }, axis=1
        ).tolist()

    # ------------------------------
    # 5) Assemblage final
    # ------------------------------
    return {
        "symbol": symbol,
        "tf": interval,
        "fees_bps": 7,
        "features": features
    }


def compute_performance(recent_decisions, capital_current):
    """
    Calcule les performances des décisions récentes de trading.

    recent_decisions : list (liste des décisions récentes)
    capital_current : float (capital actuel)
    """

    # filtre: uniquement les trades clos avec une info de PnL
    closed = [t for t in recent_decisions if t.get("outcome", {}).get("closed")]
    n = len(closed)
    if n == 0:
        return {
            "lookback_trades": 0,
            "winrate": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "last_24h_pnl_pct": 0.0,
        }

    # winrate
    wins = [t for t in closed if t["outcome"].get("pnl_r", 0.0) > 0 or t["outcome"].get("pnl_quote", 0.0) > 0]
    winrate = round(len(wins) / n, 3)

    # expectancy_r (si pas de pnl_r, approx à partir de pnl_quote et risque inconnu -> 0)
    pnl_r_list = [t["outcome"].get("pnl_r") for t in closed if t["outcome"].get("pnl_r") is not None]
    expectancy_r = round(sum(pnl_r_list) / len(pnl_r_list), 3) if pnl_r_list else 0.0

    # profit factor
    gains = 0.0
    losses = 0.0
    for t in closed:
        pnlq = t["outcome"].get("pnl_quote")
        if pnlq is None and t["outcome"].get("pnl_r") is not None:
            # si tu veux, tu peux approx PnL quote = pnl_r * (risk_in_quote) si tu logges ce dernier
            pnlq = 0.0
        if pnlq is None: 
            continue
        if pnlq >= 0:
            gains += pnlq
        else:
            losses += -pnlq
    profit_factor = round(gains / losses, 3) if losses > 0 else (round(gains,3) if gains>0 else 0.0)

    # last_24h_pnl_pct
    now = int(time.time())
    pnl_24h = 0.0
    for t in closed:
        if t.get("ts") and t["ts"] >= now - 24*3600:
            pnlq = t["outcome"].get("pnl_quote", 0.0)
            pnl_24h += pnlq if pnlq is not None else 0.0
    last_24h_pnl_pct = round(100.0 * pnl_24h / capital_current, 3) if capital_current > 0 else 0.0

    return {
        "lookback_trades": n,
        "winrate": winrate,
        "expectancy_r": expectancy_r,
        "profit_factor": profit_factor,
        "last_24h_pnl_pct": last_24h_pnl_pct,
    }

def atr_wilder(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def safe_pct_slope(series, lag=5):
    s = pd.Series(series).dropna()
    if len(s) <= lag: return 0.0
    last, prev = s.iloc[-1], s.iloc[-1-lag]
    denom = abs(prev) if abs(prev) > 1e-12 else 1e-12
    return round(100.0 * (last - prev) / denom, 3)

def pct_return_n(closes, n):
    s = pd.Series(closes).dropna()
    if len(s) <= n: return 0.0
    prev = s.iloc[-1-n]
    denom = prev if abs(prev) > 1e-12 else 1e-12
    return round(100.0 * (s.iloc[-1] / denom - 1.0), 3)

def rank_last_in_window(s, window=30):
    s = pd.Series(s).dropna()
    if len(s) == 0: return 0.0
    w = s.tail(window)
    mn, mx = w.min(), w.max()
    if mx - mn < 1e-12: return 0.5
    return round(float((w.iloc[-1] - mn) / (mx - mn)), 3)

def market_state_from_metrics(ema30d_slope_pct: float, above_ema12m: bool) -> str:
    if ema30d_slope_pct > 1.0 and above_ema12m:
        return "bullish"
    if ema30d_slope_pct < -1.0 and not above_ema12m:
        return "bearish"
    return "range"

def build_df(ohlcv_list):
    df = pd.DataFrame(ohlcv_list)
    for k in ("o","h","l","c","v"):
        if k in df.columns:
            df[k] = df[k].astype(float)
    return df

def compute_regime(client):
    # --- D1 & M1 pour ETH et BTC ---
    eth_d1 = build_df(get_market_data(client, "ETHUSDC", "1d", 60)["features"]["ohlcv"])
    btc_d1 = build_df(get_market_data(client, "BTCUSDC", "1d", 60)["features"]["ohlcv"])
    eth_m1 = build_df(get_market_data(client, "ETHUSDC", "1M", 36)["features"]["ohlcv"])
    btc_m1 = build_df(get_market_data(client, "BTCUSDC", "1M", 36)["features"]["ohlcv"])

    # --- métriques D1 (par actif) ---
    atr_eth = atr_wilder(eth_d1["h"], eth_d1["l"], eth_d1["c"])
    atr_btc = atr_wilder(btc_d1["h"], btc_d1["l"], btc_d1["c"])
    atr_rank_eth = rank_last_in_window(atr_eth, window=30)
    atr_rank_btc = rank_last_in_window(atr_btc, window=30)

    ret5d_eth = pct_return_n(eth_d1["c"], 5)
    ret5d_btc = pct_return_n(btc_d1["c"], 5)

    ema30d_eth = eth_d1["c"].ewm(span=30, adjust=False).mean()
    ema30d_btc = btc_d1["c"].ewm(span=30, adjust=False).mean()
    slope30d_eth = safe_pct_slope(ema30d_eth, lag=5)
    slope30d_btc = safe_pct_slope(ema30d_btc, lag=5)

    # --- métriques M1 (par actif) ---
    c_eth_m, c_btc_m = eth_m1["c"], btc_m1["c"]
    ret3m_eth, ret6m_eth, ret12m_eth = pct_return_n(c_eth_m, 3), pct_return_n(c_eth_m, 6), pct_return_n(c_eth_m, 12)
    ret3m_btc, ret6m_btc, ret12m_btc = pct_return_n(c_btc_m, 3), pct_return_n(c_btc_m, 6), pct_return_n(c_btc_m, 12)

    ema12m_eth = c_eth_m.ewm(span=12, adjust=False).mean()
    ema12m_btc = c_btc_m.ewm(span=12, adjust=False).mean()
    slope12m_eth = safe_pct_slope(ema12m_eth, lag=3)
    slope12m_btc = safe_pct_slope(ema12m_btc, lag=3)
    above12m_eth = bool(c_eth_m.iloc[-1] > ema12m_eth.iloc[-1])
    above12m_btc = bool(c_btc_m.iloc[-1] > ema12m_btc.iloc[-1])

    # --- context (min, max, mean, volume) ---
    ctx_eth = {
        "min_30d": round(float(eth_d1["c"].tail(30).min()), 2),
        "max_30d": round(float(eth_d1["c"].tail(30).max()), 2),
        "mean_30d": round(float(eth_d1["c"].tail(30).mean()), 2),
        "avg_volume_30d": round(float(eth_d1["v"].tail(30).mean()), 2),
    }
    ctx_btc = {
        "min_30d": round(float(btc_d1["c"].tail(30).min()), 2),
        "max_30d": round(float(btc_d1["c"].tail(30).max()), 2),
        "mean_30d": round(float(btc_d1["c"].tail(30).mean()), 2),
        "avg_volume_30d": round(float(btc_d1["v"].tail(30).mean()), 2),
    }

    # --- per_asset ---
    per_asset = {
        "ETHUSDC": {
            "market_state": market_state_from_metrics(slope30d_eth, above12m_eth),
            "daily": {
                "ret_5d_pct": round(ret5d_eth, 3),
                "ema30d_slope_pct": round(slope30d_eth, 3),
                "atr_rank_30d": round(atr_rank_eth, 3)
            },
            "monthly": {
                "ret_3m_pct": round(ret3m_eth, 3),
                "ret_6m_pct": round(ret6m_eth, 3),
                "ret_12m_pct": round(ret12m_eth, 3),
                "above_ema12m": above12m_eth,
                "ema12m_slope_pct": round(slope12m_eth, 3)
            },
            "context": ctx_eth
        },
        "BTCUSDC": {
            "market_state": market_state_from_metrics(slope30d_btc, above12m_btc),
            "daily": {
                "ret_5d_pct": round(ret5d_btc, 3),
                "ema30d_slope_pct": round(slope30d_btc, 3),
                "atr_rank_30d": round(atr_rank_btc, 3)
            },
            "monthly": {
                "ret_3m_pct": round(ret3m_btc, 3),
                "ret_6m_pct": round(ret6m_btc, 3),
                "ret_12m_pct": round(ret12m_btc, 3),
                "above_ema12m": above12m_btc,
                "ema12m_slope_pct": round(slope12m_btc, 3)
            },
            "context": ctx_btc
        }
    }

    # --- overview (moyenne simple ETH/BTC) ---
    overview = {
        "market_state": "range",
        "volatility_rank_30d": round((atr_rank_eth + atr_rank_btc) / 2, 3),
        "daily": {
            "ret_5d_pct": round((ret5d_eth + ret5d_btc) / 2, 3),
            "ema30d_slope_pct": round((slope30d_eth + slope30d_btc) / 2, 3),
            "atr_rank_30d": round((atr_rank_eth + atr_rank_btc) / 2, 3)
        },
        "monthly": {
            "ret_3m_pct": round((ret3m_eth + ret3m_btc) / 2, 3),
            "ret_6m_pct": round((ret6m_eth + ret6m_btc) / 2, 3),
            "ret_12m_pct": round((ret12m_eth + ret12m_btc) / 2, 3),
            "above_ema12m": (above12m_eth or above12m_btc),
            "ema12m_slope_pct": round((slope12m_eth + slope12m_btc) / 2, 3)
        }
    }
    overview["market_state"] = market_state_from_metrics(
        overview["daily"]["ema30d_slope_pct"],
        overview["monthly"]["above_ema12m"]
    )

    # --- relative strength ETH vs BTC ---
    eth_h1 = build_df(get_market_data(client, "ETHUSDC", "1h", 30)["features"]["ohlcv"])
    btc_h1 = build_df(get_market_data(client, "BTCUSDC", "1h", 30)["features"]["ohlcv"])
    rs_h1 = (eth_h1["c"] / btc_h1["c"]).ewm(span=10, adjust=False).mean()
    h1_slope = safe_pct_slope(rs_h1, lag=5)

    rs_d1 = (eth_d1["c"] / btc_d1["c"]).ewm(span=10, adjust=False).mean()
    d1_slope = safe_pct_slope(rs_d1, lag=5)

    if d1_slope > 0.5:
        rel_state = "eth_outperform"
    elif d1_slope < -0.5:
        rel_state = "btc_outperform"
    else:
        rel_state = "neutral"

    relative_strength = {
        "pair": "ETHBTC",
        "state": rel_state,
        "h1_slope_pct": round(h1_slope, 3),
        "d1_slope_pct": round(d1_slope, 3)
    }

    regime = {
        "overview": overview,
        "per_asset": per_asset,
        "relative_strength": relative_strength
    }
    return regime

def build_constraints():
    return {
        "max_exposure_pct": 0.25,   # exposition max par actif (25% du capital)
        "max_rebalance_pct": 0.30,  # rotation max du capital par tour
        "stable_symbol": "USDC",
        "symbols": {
            "ETHUSDC": {"lot_size": 0.0001, "min_notional": 5.0},
            "BTCUSDC": {"lot_size": 0.00001, "min_notional": 5.0}
        }
    }

def update_memory_with_decision(memory: dict, 
                                symbol: str, 
                                tf: str, 
                                decision_dict: dict,
                                price_usdc: float, 
                                qty_quote: float = None, 
                                note: str = "", 
                                keep_last:int = 20):
    
    # recent_decisions append
    rec = {
        "ts": int(time.time()),
        "symbol": symbol,
        "tf": tf,
        "decision": decision_dict.get("decision"),
        "confidence": decision_dict.get("confidence"),
        "entry": decision_dict.get("entry"),
        "entry_price": price_usdc,
        "sl": decision_dict.get("sl", 0.0),
        "tp": decision_dict.get("tp", 0.0),
        "qty_quote": decision_dict.get("qty_quote") if qty_quote is None else qty_quote,
        "risk_check": decision_dict.get("risk_check", "ok"),
        "reason": decision_dict.get("reason", ""),
        "next_steps": decision_dict.get("next_steps", ""),
        "time_sleep_s": int(decision_dict.get("time_sleep_s", 0)),
        "outcome": {"closed": False},   # fermé plus tard, quand le trade est clôturé
        "note": note
    }
    memory.setdefault("recent_decisions", []).append(rec)
    # garder seulement les N derniers
    if len(memory["recent_decisions"]) > keep_last:
        memory["recent_decisions"] = memory["recent_decisions"][-keep_last:]

    # recoller les contraintes si pas présentes
    memory.setdefault("constraints", build_constraints())

    # recalcul performance basique (si tu as déjà la fonction compute_performance)
    try:
        perf = compute_performance(memory.get("recent_decisions", []), memory["capital"]["current"])
        memory["performance"] = perf
    except Exception:
        pass

    return memory

def get_decision(client_openai, payload, prompt_id='pmpt_68b048b3ba7881959eedbbed01c83b720d61d2621e7df6fb'):
    resp = client_openai.responses.create(
        prompt={"id": prompt_id},
        input=json.dumps(payload)
    )
    out_text = resp.output_text
    try:
        return json.loads(out_text)   # JSON -> dict Python
    except Exception:
        return {"error": "invalid JSON", "raw": out_text}
    
def get_capital_and_balances(client: Client, INITIAL_CAPITAL_USDC=5000):
    # Prix spot USDC
    px_eth = float(client.get_symbol_ticker(symbol="ETHUSDC")["price"])
    px_btc = float(client.get_symbol_ticker(symbol="BTCUSDC")["price"])

    # Soldes libres
    eth = float(client.get_asset_balance(asset="ETH")["free"])
    btc = float(client.get_asset_balance(asset="BTC")["free"])
    usdc = float(client.get_asset_balance(asset="USDC")["free"])

    # Valorisation en USDC
    val_eth = eth * px_eth
    val_btc = btc * px_btc
    total  = usdc + val_eth + val_btc

    capital = {
        "initial": INITIAL_CAPITAL_USDC,
        "current": round(total, 2),
        "max_dd_7d": 0.0,               # à calculer ailleurs si besoin
        "risk_per_trade_cap": 0.10,
        "halt_if_below_50pct": True,
        "halt_triggered": (total < 0.5 * INITIAL_CAPITAL_USDC)
    }

    balances = {
        "USDC": {"amount": round(usdc, 6), "value_usdc": round(usdc, 2)},
        "ETH":  {"amount": round(eth, 8),  "value_usdc": round(val_eth, 2), "px": round(px_eth, 2)},
        "BTC":  {"amount": round(btc, 8),  "value_usdc": round(val_btc, 2), "px": round(px_btc, 2)}
    }

    return capital, balances

def execute_trade(client, decision, memory):
    pair = decision.get("pair")
    asset = decision.get("asset")
    tf = "1h"

    if decision.get("decision") == "HOLD":
        memory = update_memory_with_decision(
        memory=memory,
        symbol="USDCUSDT",
        tf=tf,
        decision_dict=decision,
        price_usdc=float(client.get_symbol_ticker(symbol="USDCUSDT")["price"]),
        qty_quote=0,
        note="pre-trade decision HOLD -  Prix USDC",
        keep_last=20
        )
        return memory


    # Prix/tailles
    px = float(client.get_symbol_ticker(symbol=pair)["price"])
    qty_base = float(decision.get("qty_base", 0.0))
    qty_quote = qty_base * px

    # Contraintes
    c = memory["constraints"]
    sc = c["symbols"][pair]
    lot_size = float(sc["lot_size"])
    min_notional = float(sc.get("min_notional", c.get("min_notional", 5.0)))
    max_reb = float(c["max_rebalance_pct"])
    cap = float(memory["capital"]["current"])
    stable = c["stable_symbol"]
    halt = bool(memory["capital"].get("halt_triggered", False))

    # Pre-trade log
    memory = update_memory_with_decision(
        memory=memory,
        symbol=pair,
        tf=tf,
        decision_dict=decision,
        price_usdc=px,
        qty_quote=qty_quote,
        note="pre-trade decision",
        keep_last=20
    )

    # Conditions communes
    ok_rebalance = (qty_quote <= max_reb * cap if cap > 0 else False)
    ok_min = qty_quote >= min_notional
    ok_lot = qty_base >= lot_size
    side = decision.get("decision", "HOLD")

    def post_refresh_and_perf():
        capital, balances = get_capital_and_balances(client, INITIAL_CAPITAL_USDC=memory["capital"]["initial"])
        memory["capital"] = capital
        memory["balances"] = balances
        memory["performance"] = compute_performance(memory["recent_decisions"], capital["current"])
        memory["capital"]["halt_triggered"] = bool(capital["current"] < 0.5 * capital["initial"])

    # HOLD ou HALT
    if side == "HOLD" or halt or decision.get("risk_check") != "ok":
        memory["recent_decisions"][-1]["execution"] = {
            "ts_exec": int(time.time()),
            "side": "SKIPPED",
            "pair": pair,
            "qty_base": 0.0,
            "px_exec": px,
            "reason": "HOLD/halt/risk_check"
        }
        post_refresh_and_perf()
        return memory

    # Vérifs taille
    if not (ok_rebalance and ok_min and ok_lot):
        memory["recent_decisions"][-1]["execution"] = {
            "ts_exec": int(time.time()),
            "side": "SKIPPED",
            "pair": pair,
            "qty_base": qty_base,
            "px_exec": px,
            "reason": "size/min_notional/rebalance"
        }
        post_refresh_and_perf()
        return memory

    # Arrondi lot size avant envoi
    qty_base = math.floor(qty_base / lot_size) * lot_size
    if qty_base <= 0:
        memory["recent_decisions"][-1]["execution"] = {
            "ts_exec": int(time.time()),
            "side": "SKIPPED",
            "pair": pair,
            "qty_base": 0.0,
            "px_exec": px,
            "reason": "rounded_to_zero"
        }
        post_refresh_and_perf()
        return memory

    # SELL
    if side == "SELL":
        free_base = float(client.get_asset_balance(asset=asset)["free"])
        qty_base = min(qty_base, math.floor(free_base / lot_size) * lot_size)
        if qty_base <= 0:
            memory["recent_decisions"][-1]["execution"] = {
                "ts_exec": int(time.time()),
                "side": "SKIPPED",
                "pair": pair,
                "qty_base": 0.0,
                "px_exec": px,
                "reason": "no_base_available"
            }
            post_refresh_and_perf()
            return memory
        
        print("Tentative Sell : ", qty_base, asset, "au prix", px, "USDC")

        order = client.order_market_sell(symbol=pair, quantity=float(qty_base))
        print("Order exécuté:", side,order)
        memory["recent_decisions"][-1]["execution"] = {
            "ts_exec": int(time.time()),
            "side": "SELL",
            "pair": pair,
            "qty_base": qty_base,
            "px_exec": px,
            "order_id": order.get("orderId"),
            "executed_qty": float(order.get("executedQty", 0.0)),
            "cummulative_quote_qty": float(order.get("cummulativeQuoteQty", 0.0))
        }
        post_refresh_and_perf()
        return memory

    # BUY
    if side == "BUY":
        free_stable = float(client.get_asset_balance(asset=stable)["free"])
        max_buy_base = math.floor((free_stable / px) / lot_size) * lot_size
        qty_base = min(qty_base, max_buy_base)

        if qty_base <= 0:
            memory["recent_decisions"][-1]["execution"] = {
                "ts_exec": int(time.time()),
                "side": "SKIPPED",
                "pair": pair,
                "qty_base": 0.0,
                "px_exec": px,
                "reason": "no_stable_available"
            }
            post_refresh_and_perf()
            return memory

        print("Tentative Buy : ", qty_base, asset, "au prix", px, "USDC")

        order = client.order_market_buy(symbol=pair, quantity=float(qty_base))
        print("Order exécuté:", side, order)
        memory["recent_decisions"][-1]["execution"] = {
            "ts_exec": int(time.time()),
            "side": "BUY",
            "pair": pair,
            "qty_base": qty_base,
            "px_exec": px,
            "order_id": order.get("orderId"),
            "executed_qty": float(order.get("executedQty", 0.0)),
            "cummulative_quote_qty": float(order.get("cummulativeQuoteQty", 0.0))
        }
        post_refresh_and_perf()
        return memory

    # Cas inconnu
    memory["recent_decisions"][-1]["execution"] = {
        "ts_exec": int(time.time()),
        "side": "SKIPPED",
        "pair": pair,
        "qty_base": 0.0,
        "px_exec": px,
        "reason": f"unknown_side_{side}"
    }
    post_refresh_and_perf()
    return memory

def get_future_decision(client, client_openai):
    with open("memory.json", "r") as f:
        memory = json.load(f)

    market_data = {
        "ETHUSDC": get_market_data(client, "ETHUSDC"),
        "BTCUSDC": get_market_data(client, "BTCUSDC")
    }

    # 1) Capital + soldes par actif
    capital, balances = get_capital_and_balances(client, INITIAL_CAPITAL_USDC=capital_initial)

    # 2) Règle de halt (50%)
    capital["halt_triggered"] = bool(capital["current"] < 0.5 * capital["initial"])

    # 3) Regime (ETH & BTC, overview + per_asset + relative_strength)
    regime = compute_regime(client)

    # 4) Historique de décisions (préserve si déjà présent)
    recent_decisions = memory.get("recent_decisions", [])

    # 5) Performance (historique + métriques actuelles)
    performance = compute_performance(recent_decisions, capital["current"])


    # 6) Mise à jour de la mémoire
    memory["capital"] = capital
    memory["balances"] = balances
    memory["recent_decisions"] = recent_decisions
    memory["performance"] = performance
    memory["regime"] = regime

    # 7) Préparation du payload
    payload = {
        "market_data": market_data,
        "memory": memory
    }

    decision = get_decision(client_openai, payload)

    return decision

def compute_max_dd_7d(capital_history, current_capital, now_ts=None):
    import time
    now = now_ts or int(time.time())
    last7 = [p["capital"] for p in capital_history if p.get("timestamp",0) >= now-7*86400]
    peak = max(last7+[current_capital]) if last7 else current_capital
    return round(100.0*((current_capital/peak) - 1.0), 3)

def snapshot_capital(memory, current_capital, now_ts=None):
    import time
    now = now_ts or int(time.time())
    hist = memory.setdefault("capital_history", [])
    # anti-doublon: 1 point max / heure
    if not hist or now//3600 != hist[-1]["timestamp"]//3600:
        hist.append({"timestamp": now, "capital": float(current_capital)})
    return hist

def binance_time_offset_ms(client):
    import time
    srv = client.get_server_time()["serverTime"]
    offset = int(srv) - int(time.time() * 1000)
    # Log + garde douce (pas de sleep agressif dans Airflow)
    print(f"[TimeSync] server-local offset = {offset} ms")
    return offset