import time, json, math
import pandas as pd, numpy as np, math
import pandas as pd
import numpy as np

from decimal import Decimal, getcontext

def get_market_data(client,
                    symbol: str = "ETHUSDC",
                    interval: str = "1h",
                    limit: int = 12,
                    warmup: int = 120,
                    include_ohlcv: bool = False,
                    balances: dict | None = None,
                    memory: dict | None = None) -> dict:
    """
    Snapshot de marché 'pensable' par un LLM, avec indicateurs H1 et,
    si balances+memory sont fournis, un bloc de sizing orienté paliers (5/10/25/50/75 %).
    - BUY = % de l'USDC libre (budget quote), borné par max_rebalance et headroom d'expo.
    - SELL = % de la base libre (ETH/BTC), **non plafonné** par max_rebalance (dé-risquage prioritaire).
    Les paliers sont validés APRES arrondi au lot_size et contrôle min_notional.
    """
    
    getcontext().prec = 28

    # -------------------------- helpers génériques --------------------------
    def _series_stats(vals, rsi=False) -> dict:
        a = np.array([v for v in vals if v == v], dtype=float)
        if a.size == 0:
            return {"last": None, "min": None, "max": None, "mean": None, "slope": None, **({"above50_cnt": None} if rsi else {})}
        x = np.arange(a.size, dtype=float)
        slope = float(np.polyfit(x, a, 1)[0]) if a.size >= 2 else 0.0
        out = {"last": round(a[-1], 2), "min": round(a.min(), 2), "max": round(a.max(), 2),
               "mean": round(a.mean(), 2), "slope": round(slope, 4)}
        if rsi: out["above50_cnt"] = int((a > 50).sum())
        return out

    def _streak_pos(arr, eps=1e-9):
        cnt = 0
        for v in reversed(arr):
            if v is None or (isinstance(v, float) and math.isnan(v)) or v <= eps: break
            cnt += 1
        return cnt

    def _pack_h1(df_in: pd.DataFrame, n: int = 12) -> list[dict]:
        if df_in.empty: return []
        x = df_in.tail(min(n, len(df_in))).copy()
        x["above_ema"] = (x["c"] > x["ema20"]) & (x["ema20"] > x["ema50"])
        x["rsi_trend"] = (x["rsi14"].diff().rolling(3)
                          .apply(lambda s: 1 if (s > 0).sum() >= 2 else 0)
                          .map({1: "up", 0: "down"}).fillna("flat"))
        out = x[["t","c","ema20","ema50","rsi14","macd_line","macd_signal","macd_hist","atr_pct","above_ema","rsi_trend"]] \
               .rename(columns={"t":"time","c":"close","macd_line":"macd","macd_signal":"macd_sig"}).copy()
        out["time"] = out["time"].astype(int)
        for col, nd in [("close",2),("ema20",2),("ema50",2),("rsi14",2),("macd",3),("macd_sig",3),("macd_hist",3),("atr_pct",2)]:
            out[col] = out[col].round(nd)
        out["above_ema"] = out["above_ema"].astype(bool)
        out["rsi_trend"] = out["rsi_trend"].astype(str)
        return out.to_dict("records")

    def _h1_summary(df_in: pd.DataFrame) -> dict:
        if df_in.empty:
            return {"dist_ema50_pct": None, "hh": None, "ll": None, "consec_hist_pos": 0, "hist_slope_3": None}
        close = df_in["c"].values
        ema50 = df_in["ema50"].values
        hist  = (df_in["macd_line"] - df_in["macd_signal"]).tolist()
        dist = None if (ema50[-1] != ema50[-1] or ema50[-1] == 0) else round(100*(close[-1]-ema50[-1])/ema50[-1], 2)
        hh, ll = round(float(df_in["h"].max()),2), round(float(df_in["l"].min()),2)
        consec_pos = _streak_pos(hist)
        hist_slope_3 = round(pd.Series(hist).diff().tail(3).sum(), 3) if len(hist)>=3 else None
        return {"dist_ema50_pct": dist, "hh": hh, "ll": ll, "consec_hist_pos": int(consec_pos), "hist_slope_3": hist_slope_3}

    def _ohlcv_stats(df_in: pd.DataFrame, lookback: int = 12) -> dict:
        x = df_in.tail(min(lookback, len(df_in))).copy()
        if x.empty: return {}
        def pct(a,b): return None if (a!=a or b!=b or b==0) else round(100*(a/b-1),2)
        chg_1h  = pct(x["c"].iloc[-1], x["o"].iloc[-1]) if len(x)>=1  else None
        chg_3h  = pct(x["c"].iloc[-1], x["o"].iloc[-3]) if len(x)>=3  else None
        chg_12h = pct(x["c"].iloc[-1], x["o"].iloc[0])  if len(x)>=12 else None
        high_12, low_12 = round(float(x["h"].max()),2), round(float(x["l"].min()),2)
        avg_range = round(float((x["h"]-x["l"]).mean()),2)
        vol_sum, vol_avg = round(float(x["v"].sum()),4), round(float(x["v"].mean()),4)
        lg, lr = (x["c"]>x["o"]).astype(int), (x["c"]<x["o"]).astype(int)
        longest_green = int(lg.groupby((lg==0).cumsum()).cumcount().add(1).max() or 0)
        longest_red   = int(lr.groupby((lr==0).cumsum()).cumcount().add(1).max() or 0)
        return {"change_1h_pct": chg_1h, "change_3h_pct": chg_3h, "change_12h_pct": chg_12h,
                "high_12h": high_12, "low_12h": low_12, "avg_range": avg_range,
                "vol_sum": vol_sum, "vol_avg": vol_avg, "green_bars": int(lg.sum()), "red_bars": int(lr.sum()),
                "longest_green_streak": longest_green, "longest_red_streak": longest_red}

    # -------------------------- 1) chargement + warmup --------------------------
    lookback = max(limit, warmup)
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    if not klines:
        base_out = {"symbol": symbol, "tf": interval, "fees_bps": 7,
                    "features": {"ohlcv_stats": {}, "hourly_snapshots": [], "stats": {}}}
        if include_ohlcv: base_out["features"]["ohlcv"] = []
        return base_out

    df_full = pd.DataFrame(klines, columns=["t","o","h","l","c","v","ct","qv","n","tbv","tbq","i"])[["t","o","h","l","c","v"]].astype(float)
    df_full["t"] = (df_full["t"] // 1000).astype(int)

    # -------------------------- 2) indicateurs --------------------------
    df_full["ema20"] = df_full["c"].ewm(span=20, adjust=False).mean()
    df_full["ema50"] = df_full["c"].ewm(span=50, adjust=False).mean()
    d = df_full["c"].diff(); gain = d.clip(lower=0); loss = -d.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean(); avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss; df_full["rsi14"] = 100 - (100/(1+rs))
    ema12 = df_full["c"].ewm(span=12, adjust=False).mean(); ema26 = df_full["c"].ewm(span=26, adjust=False).mean()
    df_full["macd_line"] = ema12 - ema26; df_full["macd_signal"] = df_full["macd_line"].ewm(span=9, adjust=False).mean()
    df_full["macd_hist"] = df_full["macd_line"] - df_full["macd_signal"]
    prev_c = df_full["c"].shift(1)
    tr = pd.concat([(df_full["h"]-df_full["l"]).abs(), (df_full["h"]-prev_c).abs(), (df_full["l"]-prev_c).abs()], axis=1).max(axis=1)
    df_full["atr_pct"] = (tr.ewm(span=14, adjust=False).mean()/df_full["c"]*100)

    # -------------------------- 3) tronquage H1 --------------------------
    df = df_full.tail(limit).copy()

    # -------------------------- 4) features techniques --------------------------
    ema20_s, ema50_s = df["ema20"].tolist(), df["ema50"].tolist()
    rsi_s = df["rsi14"].tolist()
    macd_s, sig_s, hist_s = df["macd_line"].tolist(), df["macd_signal"].tolist(), df["macd_hist"].tolist()
    atr_s = df["atr_pct"].tolist()
    stats = {
        "ema20": _series_stats(ema20_s),
        "ema50": _series_stats(ema50_s),
        "rsi14": _series_stats(rsi_s, rsi=True),
        "macd": {"last": (round(macd_s[-1],3) if macd_s else None),
                 "signal_last": (round(sig_s[-1],3) if sig_s else None),
                 "hist_last": (round(hist_s[-1],3) if hist_s else None),
                 "hist_mean": (round(float(pd.Series(hist_s).mean()),3) if hist_s else None)},
        "atr_pct": {"last": (round(atr_s[-1],2) if atr_s else None),
                    "mean": (round(float(pd.Series(atr_s).mean()),2) if atr_s else None)},
        "h1": _h1_summary(df),
    }
    hourly_snapshots = _pack_h1(df, n=12)
    ohlcv_stats = _ohlcv_stats(df, lookback=min(12, len(df)))
    features: dict = {"ohlcv_stats": ohlcv_stats, "hourly_snapshots": hourly_snapshots, "stats": stats}
    if include_ohlcv:
        features["ohlcv"] = df.apply(lambda r: {"t": int(r["t"]),
                                                "o": round(float(r["o"]),2),
                                                "h": round(float(r["h"]),2),
                                                "l": round(float(r["l"]),2),
                                                "c": round(float(r["c"]),2),
                                                "v": round(float(r["v"]),4)}, axis=1).tolist()

    # -------------------------- 5) enrichissement position/sizing (optionnel) --------------------------
    if balances is None or memory is None:
        return {"symbol": symbol, "tf": interval, "fees_bps": 7, "features": features}

    # Prix & base
    try: px = float(client.get_symbol_ticker(symbol=symbol)["price"])
    except Exception: px = float(df["c"].iloc[-1]) if not df.empty else 0.0
    base = symbol.replace("USDC", "")  # "ETH" ou "BTC"

    # Contraintes (pair > global > défauts)
    constraints = memory.get("constraints", {})
    symbols_c = constraints.get("symbols", {})
    sym_c = symbols_c.get(symbol, {})
    lot_size = float(sym_c.get("lot_size", constraints.get("lot_size_"+base.lower(), 0.0))) or float({"ETH":0.0001,"BTC":0.00001}.get(base,0.0001))
    min_notional = float(sym_c.get("min_notional", constraints.get("min_notional_usdc", 5.0)))
    max_rebalance_pct = float(constraints.get("max_rebalance_pct", 0.30))
    max_exposure_pct = float(constraints.get("max_exposure_pct", 0.25))

    # Soldes & capital
    usdc_free = float(balances.get("USDC", {}).get("amount", 0.0))
    base_free = float(balances.get(base, {}).get("amount", 0.0))
    cap_cur   = float(memory.get("capital", {}).get("current", 0.0))

    # Exposition actuelle & headroom (budget d'expo restant en USDC pour CET actif)
    val_base_usdc = base_free * px
    expo_cap_usdc = max_exposure_pct * cap_cur
    headroom_usdc = max(0.0, expo_cap_usdc - val_base_usdc)
    exposure_pct  = (val_base_usdc / cap_cur) if cap_cur > 0 else 0.0

    # Limite de rebalance (quote) par tour pour BUY (SELL non borné par cap)
    cap_rebalance_usdc = max_rebalance_pct * cap_cur

    # Paliers BUY : budget borné par rebalance & headroom & cash, validation après arrondi
    buy_levels = []
    for lvl in (5,10,25,50,75,100):
        reason = []
        budget_plan = usdc_free * (lvl/100.0)
        budget_cap  = min(budget_plan, cap_rebalance_usdc, headroom_usdc, usdc_free)
        qty_plan    = (budget_cap / px) if px > 0 else 0.0
        qty_base = float((Decimal(str(qty_plan)) // Decimal(str(lot_size))) * Decimal(str(lot_size)))

        notional    = qty_base * px
        feasible    = True
        if budget_cap <= 0: feasible=False; reason.append("no_budget")
        if qty_base   <= 0: feasible=False; reason.append("rounded_to_zero")
        if notional < min_notional: feasible=False; reason.append("below_min_notional")
        buy_levels.append({
            "pct": lvl,
            "budget_plan_usdc": round(budget_plan, 2),
            "budget_used_usdc": round(budget_cap, 2),
            "qty_base": round(qty_base, 8),
            "notional_usdc": round(notional, 2),
            "feasible": bool(feasible),
            "why_not": ",".join(reason) if (not feasible and reason) else ""
        })

    # Paliers SELL : dé-risquage prioritaire → **aucun cap de rebalance** appliqué ici
    sell_levels = []
    for lvl in (5,10,25,50,75,100):
        reason = []
        qty_plan = base_free * (lvl/100.0)
        qty_base = float((Decimal(str(qty_plan)) // Decimal(str(lot_size))) * Decimal(str(lot_size)))

        notional = qty_base * px
        feasible = True
        if qty_base <= 0: feasible=False; reason.append("rounded_to_zero")
        if notional < min_notional: feasible=False; reason.append("below_min_notional")
        sell_levels.append({
            "pct": lvl,
            "qty_base": round(qty_base, 8),
            "notional_usdc": round(notional, 2),
            "feasible": bool(feasible),
            "why_not": ",".join(reason) if (not feasible and reason) else ""
        })

    # Position state
    pos_state = {
        "status": "long_spot" if base_free > 0 else "flat",
        "size_base": round(base_free, 8),
        "size_quote": round(val_base_usdc, 2),
        "px": round(px, 4),
        "lot_size": float(lot_size),
        "min_notional": float(min_notional),
        "exposure_pct": round(exposure_pct, 4)
    }

    # Ticket récapitulatif (bornes)
    max_buy_usdc = max(0.0, min(cap_rebalance_usdc, headroom_usdc, usdc_free))
    max_buy_qty  = float((Decimal(str(max_buy_usdc/px)) // Decimal(str(lot_size))) * Decimal(str(lot_size))) if px>0 else 0.0
    max_sell_qty = float((Decimal(str(base_free))      // Decimal(str(lot_size))) * Decimal(str(lot_size))) if (base_free*px)>=min_notional else 0.0

    trade_ticket = {
        "max_buy_usdc": round(max_buy_usdc, 2),
        "max_buy_qty_base": round(max_buy_qty, 8),
        "max_sell_qty_base": round(max_sell_qty, 8)
    }

    # Bloc sizing orienté paliers
    features["sizing"] = {
        "mode": "paliers",
        "fees_bps": 7,
        "usdc_free": round(usdc_free, 2),
        "base_free": round(base_free, 8),
        "cap_current_usdc": round(cap_cur, 2),
        "px_est": round(px, 6),
        "constraints": {
            "lot_size": float(lot_size),
            "min_notional_usdc": float(min_notional),
            "max_rebalance_pct": float(max_rebalance_pct),
            "max_exposure_pct": float(max_exposure_pct),
            "headroom_usdc": round(headroom_usdc, 2)
        },
        "paliers_buy": buy_levels,
        "paliers_sell": sell_levels,
        "trade_ticket": trade_ticket
    }
    features["position_state"] = pos_state

    # -------------------------- 6) retour --------------------------
    return {"symbol": symbol, "tf": interval, "fees_bps": 7, "features": features}



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
    """
    Calcule le régime de marché (overview, per_asset, relative_strength) pour ETH/BTC.
    Robuste aux données manquantes et corrige la lecture d'ohlcv via include_ohlcv=True.
    """

    # --- helpers sûrs ---
    def _ohlcv_df(sym: str, tf: str, n: int) -> pd.DataFrame:
        try:
            md = get_market_data(client, sym, tf, limit=n, warmup=max(n, 3), include_ohlcv=True)
            ohlcv = md.get("features", {}).get("ohlcv", [])
            df = build_df(ohlcv) if ohlcv else pd.DataFrame(columns=["t","o","h","l","c","v"])
        except Exception:
            df = pd.DataFrame(columns=["t","o","h","l","c","v"])
        # types + garde
        for col in ["t","o","h","l","c","v"]:
            if col not in df.columns:
                df[col] = []
        return df.dropna()

    def _safe_bool(x):
        try:
            return bool(x)
        except Exception:
            return False

    def _safe_round(x, nd=3):
        try:
            return round(float(x), nd)
        except Exception:
            return None

    def _daily_metrics(df_d1: pd.DataFrame):
        if df_d1.empty:
            return {"atr": None, "atr_rank": None, "ret5d": None, "ema30d_slope": None}
        atr = atr_wilder(df_d1["h"], df_d1["l"], df_d1["c"])
        atr_rank = rank_last_in_window(atr, window=30)
        ret5d = pct_return_n(df_d1["c"], 5)
        ema30d = df_d1["c"].ewm(span=30, adjust=False).mean()
        slope30d = safe_pct_slope(ema30d, lag=5)
        return {"atr": atr, "atr_rank": atr_rank, "ret5d": ret5d, "ema30d_slope": slope30d}

    def _monthly_metrics(df_m1: pd.DataFrame):
        if df_m1.empty:
            return {"ret3m": None, "ret6m": None, "ret12m": None, "ema12m_slope": None, "above_ema12m": False}
        c = df_m1["c"]
        ret3m, ret6m, ret12m = pct_return_n(c, 3), pct_return_n(c, 6), pct_return_n(c, 12)
        ema12m = c.ewm(span=12, adjust=False).mean()
        slope12m = safe_pct_slope(ema12m, lag=3)
        above = _safe_bool(c.iloc[-1] > ema12m.iloc[-1]) if len(c) and len(ema12m) else False
        return {"ret3m": ret3m, "ret6m": ret6m, "ret12m": ret12m, "ema12m_slope": slope12m, "above_ema12m": above}

    def _ctx_30d(df_d1: pd.DataFrame):
        if df_d1.empty:
            return {"min_30d": None, "max_30d": None, "mean_30d": None, "avg_volume_30d": None}
        tail = df_d1.tail(30)
        return {
            "min_30d": _safe_round(tail["c"].min(), 2),
            "max_30d": _safe_round(tail["c"].max(), 2),
            "mean_30d": _safe_round(tail["c"].mean(), 2),
            "avg_volume_30d": _safe_round(tail["v"].mean(), 2),
        }

    # --- 1) Récupération D1 / M1 ---
    eth_d1 = _ohlcv_df("ETHUSDC", "1d", 60)
    btc_d1 = _ohlcv_df("BTCUSDC", "1d", 60)
    eth_m1 = _ohlcv_df("ETHUSDC", "1M", 36)
    btc_m1 = _ohlcv_df("BTCUSDC", "1M", 36)

    # --- 2) Métriques par actif ---
    dm_eth = _daily_metrics(eth_d1)
    dm_btc = _daily_metrics(btc_d1)
    mm_eth = _monthly_metrics(eth_m1)
    mm_btc = _monthly_metrics(btc_m1)

    ctx_eth = _ctx_30d(eth_d1)
    ctx_btc = _ctx_30d(btc_d1)

    # --- 3) per_asset ---
    per_asset = {
        "ETHUSDC": {
            "market_state": market_state_from_metrics(dm_eth["ema30d_slope"] or 0.0, mm_eth["above_ema12m"]),
            "daily": {
                "ret_5d_pct": _safe_round(dm_eth["ret5d"], 3),
                "ema30d_slope_pct": _safe_round(dm_eth["ema30d_slope"], 3),
                "atr_rank_30d": _safe_round(dm_eth["atr_rank"], 3)
            },
            "monthly": {
                "ret_3m_pct": _safe_round(mm_eth["ret3m"], 3),
                "ret_6m_pct": _safe_round(mm_eth["ret6m"], 3),
                "ret_12m_pct": _safe_round(mm_eth["ret12m"], 3),
                "above_ema12m": _safe_bool(mm_eth["above_ema12m"]),
                "ema12m_slope_pct": _safe_round(mm_eth["ema12m_slope"], 3)
            },
            "context": ctx_eth
        },
        "BTCUSDC": {
            "market_state": market_state_from_metrics(dm_btc["ema30d_slope"] or 0.0, mm_btc["above_ema12m"]),
            "daily": {
                "ret_5d_pct": _safe_round(dm_btc["ret5d"], 3),
                "ema30d_slope_pct": _safe_round(dm_btc["ema30d_slope"], 3),
                "atr_rank_30d": _safe_round(dm_btc["atr_rank"], 3)
            },
            "monthly": {
                "ret_3m_pct": _safe_round(mm_btc["ret3m"], 3),
                "ret_6m_pct": _safe_round(mm_btc["ret6m"], 3),
                "ret_12m_pct": _safe_round(mm_btc["ret12m"], 3),
                "above_ema12m": _safe_bool(mm_btc["above_ema12m"]),
                "ema12m_slope_pct": _safe_round(mm_btc["ema12m_slope"], 3)
            },
            "context": ctx_btc
        }
    }

    # --- 4) overview (moyenne simple ETH/BTC) ---
    def _avg(a, b):
        try:
            return (float(a) + float(b)) / 2.0
        except Exception:
            # si l'un manque → retourner l'autre, sinon None
            if a is not None and a == a:
                return float(a)
            if b is not None and b == b:
                return float(b)
            return None

    vol_rank_30d = _avg(dm_eth["atr_rank"], dm_btc["atr_rank"])
    daily_ret5 = _avg(dm_eth["ret5d"], dm_btc["ret5d"])
    daily_slope = _avg(dm_eth["ema30d_slope"], dm_btc["ema30d_slope"])
    m_ret3 = _avg(mm_eth["ret3m"], mm_btc["ret3m"])
    m_ret6 = _avg(mm_eth["ret6m"], mm_btc["ret6m"])
    m_ret12 = _avg(mm_eth["ret12m"], mm_btc["ret12m"])
    m_slope12 = _avg(mm_eth["ema12m_slope"], mm_btc["ema12m_slope"])
    m_above = _safe_bool(mm_eth["above_ema12m"] or mm_btc["above_ema12m"])

    overview = {
        "market_state": "range",  # sera recalculé juste après
        "volatility_rank_30d": _safe_round(vol_rank_30d, 3),
        "daily": {
            "ret_5d_pct": _safe_round(daily_ret5, 3),
            "ema30d_slope_pct": _safe_round(daily_slope, 3),
            "atr_rank_30d": _safe_round(vol_rank_30d, 3)
        },
        "monthly": {
            "ret_3m_pct": _safe_round(m_ret3, 3),
            "ret_6m_pct": _safe_round(m_ret6, 3),
            "ret_12m_pct": _safe_round(m_ret12, 3),
            "above_ema12m": m_above,
            "ema12m_slope_pct": _safe_round(m_slope12, 3)
        }
    }
    overview["market_state"] = market_state_from_metrics(
        overview["daily"]["ema30d_slope_pct"] or 0.0,
        overview["monthly"]["above_ema12m"]
    )

    # --- 5) relative strength ETH vs BTC (H1 et D1, alignement index) ---
    eth_h1 = _ohlcv_df("ETHUSDC", "1h", 60)
    btc_h1 = _ohlcv_df("BTCUSDC", "1h", 60)

    def _rs_slope(df_a: pd.DataFrame, df_b: pd.DataFrame, span: int, lag: int):
        if df_a.empty or df_b.empty:
            return None
        s = pd.DataFrame({"a": df_a["c"].values, "b": df_b["c"].values}).dropna()
        if s.empty:
            return None
        rs = (s["a"] / s["b"]).ewm(span=span, adjust=False).mean()
        return safe_pct_slope(rs, lag=lag)

    h1_slope = _rs_slope(eth_h1, btc_h1, span=10, lag=5)
    d1_slope = _rs_slope(eth_d1, btc_d1, span=10, lag=5)

    if d1_slope is None:
        rel_state = "neutral"
    elif d1_slope > 0.5:
        rel_state = "eth_outperform"
    elif d1_slope < -0.5:
        rel_state = "btc_outperform"
    else:
        rel_state = "neutral"

    relative_strength = {
        "pair": "ETHBTC",
        "state": rel_state,
        "h1_slope_pct": _safe_round(h1_slope, 3),
        "d1_slope_pct": _safe_round(d1_slope, 3)
    }

    # --- 6) retour ---
    regime = {
        "overview": overview,
        "per_asset": per_asset,
        "relative_strength": relative_strength
    }
    return regime



def update_memory_with_decision(memory: dict,
                                symbol: str,
                                tf: str,
                                decision_dict: dict,
                                price_usdc: float,
                                qty_quote: float | None = None,
                                note: str = "",
                                keep_last: int = 20) -> dict:
    """
    Ajoute une entrée dans recent_decisions et maintient les champs dérivés (constraints, performance).
    - Enregistre les infos de décision + sizing (mode 'paliers' avec size_pct si présent).
    - 'execution' sera complété plus tard (par execute_trade).
    """

    # --- sécurités / defaults ---
    if not isinstance(memory, dict):
        memory = {}
        memory.setdefault("recent_decisions", [])
        memory.setdefault("capital", {"current": 0.0, "initial": 0.0})

    # --- extraction sûre des champs de décision ---
    def _f(x, nd=None):
        try:
            v = float(x)
            return round(v, nd) if (nd is not None) else v
        except Exception:
            return 0.0 if nd is not None else None

    decision     = (decision_dict or {}).get("decision", "HOLD")
    asset        = (decision_dict or {}).get("asset")
    pair         = (decision_dict or {}).get("pair", symbol)
    confidence   = _f((decision_dict or {}).get("confidence"), 3)
    entry        = (decision_dict or {}).get("entry", "market")
    sl           = _f((decision_dict or {}).get("sl"), 6)
    tp           = _f((decision_dict or {}).get("tp"), 6)
    rcheck       = (decision_dict or {}).get("risk_check", "ok")
    reason       = (decision_dict or {}).get("reason", "")
    next_steps   = (decision_dict or {}).get("next_steps", "")
    sleep_s      = int((decision_dict or {}).get("time_sleep_s", 0) or 0)
    size_pct     = (decision_dict or {}).get("size_pct")  # attendu ∈ {5,10,25,50,75} ou None
    sizing_mode  = (decision_dict or {}).get("sizing_mode", "paliers" if size_pct is not None else "legacy")

    # qty_quote priorise l'override param puis la valeur de la décision
    q_quote = qty_quote
    if q_quote is None:
        q_quote = _f((decision_dict or {}).get("qty_quote"), 2)

    # --- construction de l'enregistrement ---
    rec = {
        "ts": int(time.time()),
        "symbol": symbol,            # ex: "ETHUSDC" (celui du contexte d'appel)
        "pair": pair,                # ex: "ETHUSDC" (source décision)
        "asset": asset,              # "ETH" | "BTC" | "USDC"
        "tf": tf,
        "decision": decision,        # "BUY" | "SELL" | "HOLD"
        "confidence": confidence,
        "entry": entry,              # "market" | "limit"
        "entry_price": _f(price_usdc, 6),
        "sl": sl,
        "tp": tp,
        "qty_quote": q_quote,        # informatif (la quantité finale est recalculée côté exécution)
        "risk_check": rcheck,        # "ok" | "too_high"
        "reason": reason,
        "next_steps": next_steps,
        "time_sleep_s": sleep_s,
        # Sizing (logique à paliers)
        "sizing_mode": sizing_mode,  # "paliers" ou "legacy"
        "size_pct": int(size_pct) if isinstance(size_pct, (int, float)) else None,
        "palier_initial": int(size_pct) if isinstance(size_pct, (int, float)) else None,
        "palier_effectif": None,     # rempli par execute_trade (après escalade/dé-escalade)
        # État d'issue (complété à l'exécution ou à la clôture)
        "outcome": {"closed": False},
        "note": note
    }
    rec["next_step"] = decision_dict.get("next_step", "")


    # --- append + rétention ---
    memory["recent_decisions"].append(rec)
    try:
        keep_last = int(keep_last)
    except Exception:
        keep_last = 20
    if keep_last > 0 and len(memory["recent_decisions"]) > keep_last:
        memory["recent_decisions"] = memory["recent_decisions"][-keep_last:]

    # --- recalcul performance basique (tolérant aux erreurs) ---
    try:
        perf = compute_performance(memory.get("recent_decisions", []),
                                   float(memory.get("capital", {}).get("current", 0.0)))
        memory["performance"] = perf
    except Exception:
        # on ne casse pas la mise à jour mémoire si la perf est indisponible
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
    
def get_capital_and_balances(client, INITIAL_CAPITAL_USDC: float = 5000.0):
    """
    Récupère le capital total en USDC et les soldes par actif (USDC, ETH, BTC).
    Calcule la valorisation des actifs, l'état du halt, et renvoie capital + balances.
    Compatible avec la logique de sizing par paliers.
    """
    # --- Prix spot USDC ---
    try:
        px_eth = float(client.get_symbol_ticker(symbol="ETHUSDC")["price"])
    except Exception:
        px_eth = 0.0
    try:
        px_btc = float(client.get_symbol_ticker(symbol="BTCUSDC")["price"])
    except Exception:
        px_btc = 0.0

    # --- Soldes libres ---
    try:
        eth = float(client.get_asset_balance(asset="ETH")["free"])
    except Exception:
        eth = 0.0
    try:
        btc = float(client.get_asset_balance(asset="BTC")["free"])
    except Exception:
        btc = 0.0
    try:
        usdc = float(client.get_asset_balance(asset="USDC")["free"])
    except Exception:
        usdc = 0.0

    # --- Valorisation en USDC ---
    val_eth = eth * px_eth
    val_btc = btc * px_btc
    total = usdc + val_eth + val_btc

    # --- Bloc capital ---
    capital = {
        "initial": float(INITIAL_CAPITAL_USDC),
        "current": round(total, 2),
        "max_dd_7d": 0.0,  # recalculé ailleurs via compute_max_dd_7d
        "halt_if_below_50pct": True,
        "halt_triggered": bool(total < 0.5 * INITIAL_CAPITAL_USDC)
    }

    # --- Soldes détaillés ---
    balances = {
        "USDC": {
            "amount": round(usdc, 6),
            "value_usdc": round(usdc, 2),
            "px": 1.0
        },
        "ETH": {
            "amount": round(eth, 8),
            "value_usdc": round(val_eth, 2),
            "px": round(px_eth, 2)
        },
        "BTC": {
            "amount": round(btc, 8),
            "value_usdc": round(val_btc, 2),
            "px": round(px_btc, 2)
        }
    }

    return capital, balances

def execute_trade(client, decision, memory):

    # --- Extraction & garde-fous globaux ---
    pair   = decision.get("pair")
    asset  = decision.get("asset")
    side   = decision.get("decision", "HOLD")
    tf     = decision.get("tf", "1h")
    px     = float(client.get_symbol_ticker(symbol=pair)["price"]) if pair else 1.0

    c          = memory.get("constraints", {})
    sym_c      = c.get("symbols", {}).get(pair, {})
    lot_size   = float(sym_c.get("lot_size", 0.0)) or 0.000001
    min_not    = float(sym_c.get("min_notional", c.get("min_notional", 5.0)))
    max_reb    = float(c.get("max_rebalance_pct", 0.30))
    cap        = float(memory.get("capital", {}).get("current", 0.0))
    stable     = c.get("stable_symbol", "USDC")
    halt       = bool(memory.get("capital", {}).get("halt_triggered", False))
    size_pct   = int(decision.get("size_pct", 0))  # attendu ∈ {5,10,25,50,75}

    # --- Log pré-trade ---
    memory = update_memory_with_decision(
        memory=memory, symbol=pair if side != "HOLD" else "USDCUSDT", tf=tf,
        decision_dict=decision,
        price_usdc=px if side != "HOLD" else float(client.get_symbol_ticker(symbol="USDCUSDT")["price"]),
        qty_quote=0.0, note="pre-trade decision", keep_last=20
    )

    # --- Cas HOLD / HALT / risk_check KO / palier manquant ---
    if side == "HOLD" or halt or decision.get("risk_check") not in ("ok", None) or size_pct not in (5,10,25,50,75,100):
        memory["recent_decisions"][-1]["execution"] = {
            "ts_exec": int(time.time()), "side": "SKIPPED", "pair": pair, "qty_base": 0.0,
            "px_exec": px, "reason": "HOLD/halt/risk_check/size_pct"
        }
        # refresh
        capital, balances = get_capital_and_balances(client, INITIAL_CAPITAL_USDC=memory["capital"]["initial"])
        memory["capital"], memory["balances"] = capital, balances
        memory["performance"] = compute_performance(memory["recent_decisions"], capital["current"])
        memory["capital"]["halt_triggered"] = bool(capital["current"] < 0.5 * capital["initial"])
        return memory

    # --- Fonctions utilitaires locales ---
    def rebalance_ok(notional_quote: float) -> bool:
        return notional_quote <= max_reb * cap if cap > 0 else False

    def round_to_lot(q: float) -> float:
        return math.floor(q / lot_size) * lot_size

    def refresh_post_exec():
        capital, balances = get_capital_and_balances(client, INITIAL_CAPITAL_USDC=memory["capital"]["initial"])
        memory["capital"], memory["balances"] = capital, balances
        memory["performance"] = compute_performance(memory["recent_decisions"], capital["current"])
        memory["capital"]["halt_triggered"] = bool(capital["current"] < 0.5 * capital["initial"])

    # --- SELL : % de base libre, dé-escalade si notional < min_not ---
    if side == "SELL":
        free_base = float(client.get_asset_balance(asset=asset)["free"])
        for palier in ([size_pct] + [p for p in (100,75,50,25,10,5) if p < size_pct]):
            qty_plan = free_base * (palier / 100.0)
            qty_base = float((Decimal(str(qty_base)) // Decimal(str(lot_size))) * Decimal(str(lot_size)))
            notional = qty_base * px
            if qty_base > 0 and notional >= min_not:
                # Exécution SELL
                order = client.order_market_sell(symbol=pair, quantity=float(qty_base))
                memory["recent_decisions"][-1]["execution"] = {
                    "ts_exec": int(time.time()), "side": "SELL", "pair": pair, "qty_base": qty_base,
                    "px_exec": px, "order_id": order.get("orderId"),
                    "executed_qty": float(order.get("executedQty", 0.0)),
                    "cummulative_quote_qty": float(order.get("cummulativeQuoteQty", 0.0)),
                    "palier_initial": size_pct, "palier_effectif": palier
                }
                refresh_post_exec(); return memory
        # Rien de faisable
        memory["recent_decisions"][-1]["execution"] = {
            "ts_exec": int(time.time()), "side": "SKIPPED", "pair": pair, "qty_base": 0.0,
            "px_exec": px, "reason": "min_notional/lot_size after de-escalation",
            "palier_initial": size_pct, "palier_effectif": 0
        }
        refresh_post_exec(); return memory

    # --- BUY : % de USDC libre, escalade si notional < min_not ---
    if side == "BUY":
        usdc_free = float(client.get_asset_balance(asset=stable)["free"])
        # budget max autorisé par rebalance (cap) et cash dispo
        budget_cap = min(max_reb * cap, usdc_free) if cap > 0 else usdc_free
        # escalade : palier initial puis supérieurs
        for palier in ([size_pct] + [p for p in (5,10,25,50,75,100) if p > size_pct]):
            budget_plan = usdc_free * (palier / 100.0)
            budget_use  = min(budget_plan, budget_cap)
            qty_plan    = budget_use / px
            qty_base    = round_to_lot(qty_plan)
            notional    = qty_base * px
            if qty_base > 0 and notional >= min_not and rebalance_ok(notional):
                # Exécution BUY
                order = client.order_market_buy(symbol=pair, quantity=float(qty_base))
                memory["recent_decisions"][-1]["execution"] = {
                    "ts_exec": int(time.time()), "side": "BUY", "pair": pair, "qty_base": qty_base,
                    "px_exec": px, "order_id": order.get("orderId"),
                    "executed_qty": float(order.get("executedQty", 0.0)),
                    "cummulative_quote_qty": float(order.get("cummulativeQuoteQty", 0.0)),
                    "palier_initial": size_pct, "palier_effectif": palier
                }
                refresh_post_exec(); return memory

        # Rien de faisable (soit min_not, soit lot, soit rebalance)
        memory["recent_decisions"][-1]["execution"] = {
            "ts_exec": int(time.time()), "side": "SKIPPED", "pair": pair, "qty_base": 0.0,
            "px_exec": px, "reason": "min_notional/lot_size/rebalance after escalation",
            "palier_initial": size_pct, "palier_effectif": 0
        }
        refresh_post_exec(); return memory

    # --- Côté inconnu ---
    memory["recent_decisions"][-1]["execution"] = {
        "ts_exec": int(time.time()), "side": "SKIPPED", "pair": pair, "qty_base": 0.0,
        "px_exec": px, "reason": f"unknown_side_{side}"
    }
    refresh_post_exec(); return memory

def get_future_decision(client, client_openai, memory):
    memory.setdefault("capital", {})
    memory.setdefault("recent_decisions", [])
    memory.setdefault("capital_history", [])

    # 1) Capital + soldes (initial depuis mémoire si présent, sinon fallback)
    initial_cap = float(memory["capital"].get("initial", 5000.0))
    capital, balances = get_capital_and_balances(client, INITIAL_CAPITAL_USDC=initial_cap)

    # 2) MAJ halt + historique capital + DD 7j
    capital["halt_triggered"] = bool(capital["current"] < 0.5 * capital["initial"])
    snapshot_capital(memory, capital["current"])
    capital["max_dd_7d"] = compute_max_dd_7d(memory.get("capital_history", []), capital["current"])

    # 3) Régime (ETH/BTC)
    regime = compute_regime(client)

    # 4) Performance (à partir de l’historique conservé)
    recent_decisions = memory.get("recent_decisions", [])
    performance = compute_performance(recent_decisions, capital["current"])

    # 5) État mémoire mis à jour (servira aussi au sizing des market_data)
    memory["capital"] = capital
    memory["balances"] = balances
    memory["recent_decisions"] = recent_decisions
    memory["performance"] = performance
    memory["regime"] = regime

    # 6) Market data (avec features de sizing/paliers)
    md_eth = get_market_data(client, "ETHUSDC", balances=balances, memory=memory)
    md_btc = get_market_data(client, "BTCUSDC", balances=balances, memory=memory)
    market_data = {"ETHUSDC": md_eth, "BTCUSDC": md_btc}

    # 7) Payload LLM et décision
    payload = {"market_data": market_data, "memory": memory}
    decision = get_decision(client_openai, payload)

    return decision

def compute_max_dd_7d(capital_history, current_capital, now_ts=None):

    now = now_ts or int(time.time())
    last7 = [p["capital"] for p in capital_history if p.get("timestamp",0) >= now-7*86400]
    peak = max(last7+[current_capital]) if last7 else current_capital
    return round(100.0*((current_capital/peak) - 1.0), 3)

def snapshot_capital(memory, current_capital, now_ts=None):

    now = now_ts or int(time.time())
    hist = memory.setdefault("capital_history", [])
    # anti-doublon: 1 point max / heure
    if not hist or now//3600 != hist[-1]["timestamp"]//3600:
        hist.append({"timestamp": now, "capital": float(current_capital)})
    return hist

def binance_time_offset_ms(client):

    srv = client.get_server_time()["serverTime"]
    offset = int(srv) - int(time.time() * 1000)
    # Log + garde douce (pas de sleep agressif dans Airflow)
    print(f"[TimeSync] server-local offset = {offset} ms")
    return offset