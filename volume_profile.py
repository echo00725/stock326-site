from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import akshare as ak
import pandas as pd


def _norm_symbol(symbol: str) -> str:
    s = (symbol or "").strip()
    if len(s) == 6 and s.isdigit():
        return s
    raise ValueError("股票代码格式不正确，请使用 6 位数字代码，如 600519")


def _market_prefix(code: str) -> str:
    return "sh" if code.startswith(("5", "6", "9")) else "sz"


def _to_num(x) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return 0.0 if pd.isna(v) else float(v)


def _build_profile(df: pd.DataFrame, price_col: str, volume_col: str, amount_col: str) -> dict:
    vol_map = defaultdict(float)
    amt_map = defaultdict(float)
    total_volume = 0.0
    total_amount = 0.0

    for _, r in df.iterrows():
        price = _to_num(r.get(price_col))
        vol = _to_num(r.get(volume_col))
        amt = _to_num(r.get(amount_col))
        if price <= 0 or vol <= 0:
            continue

        bucket = round(price, 2)  # 固定步长 0.01
        vol_map[bucket] += vol

        # 有 amount 就优先用；没有就 price*volume 近似
        trade_amount = amt if amt > 0 else price * vol
        amt_map[bucket] += trade_amount

        total_volume += vol
        total_amount += trade_amount

    levels = sorted(vol_map.keys())
    rows = []
    for p in levels:
        v = vol_map[p]
        a = amt_map[p]
        rows.append(
            {
                "price": p,
                "volume": int(round(v)),
                "amount": round(a, 2),
                "volume_pct": round((v / total_volume * 100), 4) if total_volume else 0,
                "amount_pct": round((a / total_amount * 100), 4) if total_amount else 0,
            }
        )

    poc_by_volume = max(vol_map, key=vol_map.get) if vol_map else None
    poc_by_amount = max(amt_map, key=amt_map.get) if amt_map else None

    return {
        "step": 0.01,
        "total_volume": int(round(total_volume)),
        "total_amount": round(total_amount, 2),
        "poc_by_volume": poc_by_volume,
        "poc_by_amount": poc_by_amount,
        "profile": rows,
    }


def get_volume_profile(symbol: str, days: int = 1) -> dict:
    code = _norm_symbol(symbol)
    days = max(1, min(int(days), 60))

    mk = _market_prefix(code)
    # 使用新浪1分钟，通常可拿近5交易日左右；用于 1/2 日最稳
    df = ak.stock_zh_a_minute(symbol=f"{mk}{code}", period="1", adjust="")
    if df.empty or not {"day", "close", "volume"}.issubset(df.columns):
        raise RuntimeError("未获取到分钟数据")

    df = df.copy()
    df["date"] = df["day"].astype(str).str[:10]
    trading_dates = sorted(df["date"].dropna().unique())
    if not trading_dates:
        raise RuntimeError("分钟数据缺少交易日期")

    use_dates = trading_dates[-days:]
    dfx = df[df["date"].isin(use_dates)].copy()

    # 新浪接口 amount 有时为空，统一兜底
    if "amount" not in dfx.columns:
        dfx["amount"] = 0

    profile = _build_profile(dfx, price_col="close", volume_col="volume", amount_col="amount")
    profile.update(
        {
            "symbol": code,
            "source": "sina_1m",
            "range_days": len(use_dates),
            "start_date": use_dates[0],
            "end_date": use_dates[-1],
            "note": "分钟级聚合，按最近N个交易日统计",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return profile
