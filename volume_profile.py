from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pandas as pd
import requests


def _norm_symbol(symbol: str) -> str:
    s = (symbol or "").strip()
    if len(s) == 6 and s.isdigit():
        return s
    raise ValueError("股票代码格式不正确，请使用 6 位数字代码，如 600519")


def _secid(code: str) -> str:
    # 1=沪市, 0=深市/创业板
    return f"1.{code}" if code.startswith(("5", "6", "9")) else f"0.{code}"


def _build_profile(df: pd.DataFrame, price_col: str, volume_col: str, amount_col: str) -> dict:
    vol_map = defaultdict(float)
    amt_map = defaultdict(float)
    total_volume = 0.0
    total_amount = 0.0

    for _, r in df.iterrows():
        price = float(pd.to_numeric(r.get(price_col), errors="coerce") or 0)
        vol = float(pd.to_numeric(r.get(volume_col), errors="coerce") or 0)
        amt = float(pd.to_numeric(r.get(amount_col), errors="coerce") or 0)
        if price <= 0 or vol <= 0:
            continue

        p = round(price, 2)
        vol_map[p] += vol
        trade_amt = amt if amt > 0 else price * vol
        amt_map[p] += trade_amt
        total_volume += vol
        total_amount += trade_amt

    rows = []
    for p in sorted(vol_map.keys()):
        v = vol_map[p]
        a = amt_map[p]
        rows.append(
            {
                "price": p,
                "volume": int(round(v)),
                "amount": round(a, 2),
                "volume_pct": round(v / total_volume * 100, 4) if total_volume else 0,
                "amount_pct": round(a / total_amount * 100, 4) if total_amount else 0,
            }
        )

    return {
        "step": 0.01,
        "total_volume": int(round(total_volume)),
        "total_amount": round(total_amount, 2),
        "poc_by_volume": max(vol_map, key=vol_map.get) if vol_map else None,
        "poc_by_amount": max(amt_map, key=amt_map.get) if amt_map else None,
        "profile": rows,
    }


def _fetch_eastmoney_trends(code: str) -> pd.DataFrame:
    # 返回近5日 1分钟数据，字段格式："YYYY-MM-DD HH:MM,open,close,high,low,volume,amount,avg"
    url = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "ndays": 5,
        "iscr": 0,
        "secid": _secid(code),
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    j = r.json()
    data = (j.get("data") or {}).get("trends") or []
    if not data:
        return pd.DataFrame()

    rows = []
    for line in data:
        parts = str(line).split(",")
        if len(parts) < 7:
            continue
        t, _open, close, _high, _low, vol, amt = parts[:7]
        rows.append({"time": t, "price": close, "volume": vol, "amount": amt})
    return pd.DataFrame(rows)


def get_volume_profile(symbol: str, days: int = 1) -> dict:
    code = _norm_symbol(symbol)
    days = max(1, min(int(days), 30))

    df = _fetch_eastmoney_trends(code)
    if df.empty:
        raise RuntimeError("未获取到行情分钟数据")

    df["date"] = df["time"].astype(str).str[:10]
    dates = sorted(df["date"].dropna().unique())
    if not dates:
        raise RuntimeError("分钟数据缺少交易日期")

    use_dates = dates[-days:]
    dfx = df[df["date"].isin(use_dates)].copy()

    out = _build_profile(dfx, "price", "volume", "amount")
    out.update(
        {
            "symbol": code,
            "source": "eastmoney_1m",
            "range_days": len(use_dates),
            "start_date": use_dates[0],
            "end_date": use_dates[-1],
            "note": "分钟级聚合，按最近N个交易日统计",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return out
