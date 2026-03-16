from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
            "Referer": "https://quote.eastmoney.com/",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return s


def _parse_rows(lines) -> pd.DataFrame:
    rows = []
    for line in lines or []:
        parts = str(line).split(",")
        if len(parts) < 7:
            continue
        t, _open, close, _high, _low, vol, amt = parts[:7]
        rows.append({"time": t, "price": close, "volume": vol, "amount": amt})
    return pd.DataFrame(rows)


def _fetch_eastmoney_trends(code: str) -> pd.DataFrame:
    s = _session()

    # 主接口：trends2（通常仅近5日）
    url1 = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
    p1 = {
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "ndays": 5,
        "iscr": 0,
        "secid": _secid(code),
    }
    try:
        r1 = s.get(url1, params=p1, timeout=15)
        r1.raise_for_status()
        j1 = r1.json()
        df = _parse_rows((j1.get("data") or {}).get("trends") or [])
        if not df.empty:
            return df
    except Exception:
        pass

    # 备用接口：kline 分钟线（部分标的也可能只有近几日）
    url2 = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    p2 = {
        "secid": _secid(code),
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 1,
        "fqt": 0,
        "lmt": 20000,
        "end": "20500000",
    }
    r2 = s.get(url2, params=p2, timeout=15)
    r2.raise_for_status()
    j2 = r2.json()
    klines = ((j2.get("data") or {}).get("klines")) or []
    return _parse_rows(klines)


def _fetch_daily_kline(code: str, days: int = 120) -> pd.DataFrame:
    s = _session()
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": _secid(code),
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101,  # 日线
        "fqt": 1,
        "lmt": max(1, min(days, 240)),
        "end": "20500000",
    }
    r = s.get(url, params=params, timeout=15)
    r.raise_for_status()
    kl = ((r.json().get("data") or {}).get("klines")) or []

    rows = []
    for line in kl:
        p = str(line).split(",")
        if len(p) < 6:
            continue
        # 日线近似：按收盘价聚合整日成交量/额（用于补足长周期）
        rows.append({"time": p[0], "price": p[2], "volume": p[5], "amount": p[6] if len(p) > 6 else 0})
    return pd.DataFrame(rows)


def get_volume_profile(symbol: str, days: int = 1) -> dict:
    code = _norm_symbol(symbol)
    days = max(1, min(int(days), 120))

    df_1m = _fetch_eastmoney_trends(code)
    minute_dates = []
    use_dates = []

    if not df_1m.empty:
        df_1m["date"] = df_1m["time"].astype(str).str[:10]
        minute_dates = sorted(df_1m["date"].dropna().unique())

    # 如果分钟数据不足（常见仅5天），自动回退到日线，支持到60/120天
    if len(minute_dates) >= days:
        use_dates = minute_dates[-days:]
        dfx = df_1m[df_1m["date"].isin(use_dates)].copy()
        source = "eastmoney_1m"
        note = "分钟级聚合，按最近N个交易日统计"
    else:
        df_day = _fetch_daily_kline(code, days=max(days, 120))
        if df_day.empty:
            raise RuntimeError("未获取到可用行情数据")
        df_day["date"] = df_day["time"].astype(str).str[:10]
        day_dates = sorted(df_day["date"].dropna().unique())
        use_dates = day_dates[-days:]
        dfx = df_day[df_day["date"].isin(use_dates)].copy()
        source = "eastmoney_1d_fallback"
        note = "分钟数据不足，已切换为日线近似分布（按日收盘价聚合成交量/额）"

    out = _build_profile(dfx, "price", "volume", "amount")
    out.update(
        {
            "symbol": code,
            "source": source,
            "range_days": len(use_dates),
            "start_date": use_dates[0],
            "end_date": use_dates[-1],
            "note": note,
            "minute_days_available": len(minute_dates),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return out
