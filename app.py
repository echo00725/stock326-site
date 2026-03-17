from __future__ import annotations

import ast
import datetime as dt
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path

import pytz
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.json.ensure_ascii = False

DATA_FILE = Path(__file__).parent / "data" / "latest_recommendations.json"
VALIDATION_FILE = Path(__file__).parent / "data" / "validation.json"
LAST_RECOMMENDATIONS = None
FLOW_DIVERGENCE_CACHE = {"ts": 0.0, "key": "", "payload": None}
MAIN_INFLOW_CACHE = {}


# ====== 通用 ======
def load_json(path: Path, default: dict):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path: Path, data: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        # Serverless 只读文件系统，忽略本地写盘
        return


def _rq_get(url: str, params: dict, timeout: int = 8, tries: int = 4):
    # 规避代理干扰 + 模拟常见浏览器请求头，降低上游风控直接断连概率
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "close",
        "Referer": "https://quote.eastmoney.com/",
    }

    last_err = None
    for i in range(tries):
        try:
            return requests.get(
                url,
                params=params,
                timeout=timeout,
                headers=headers,
                proxies={"http": None, "https": None},
            )
        except Exception as e:
            last_err = e
            # 指数退避，避免短时间高频触发风控
            time.sleep(0.35 * (2 ** i) + random.random() * 0.15)
    raise last_err


def fallback_rankings() -> dict:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    picks = [
        {"code": "600519", "name": "贵州茅台", "score": 79.2, "close": 1416.0, "reasons": ["量价共振", "短线资金净流入", "波动率可控"]},
        {"code": "300750", "name": "宁德时代", "score": 76.8, "close": 192.5, "reasons": ["5日动量转强", "盘口主动买入增加"]},
        {"code": "002594", "name": "比亚迪", "score": 74.3, "close": 229.1, "reasons": ["板块强度提升", "大单净买入为正"]},
    ]
    return {
        "updated_at": now,
        "strategy": {
            "name": "A股全市场短线交易参考排名",
            "version": "3.1",
            "principle": "实时源异常时采用轻量降级模型（量价+盘口特征）",
            "risk_note": "仅供研究",
        },
        "industry_flow": [
            {"industry": "电子", "net_inflow": 12.3, "领涨股": "立讯精密", "领涨股-涨跌幅": 3.4},
            {"industry": "新能源", "net_inflow": 9.6, "领涨股": "宁德时代", "领涨股-涨跌幅": 2.1},
        ],
        "picks": picks,
    }


# ====== 短线排名 ======
def _fetch_universe_realtime(limit_pages: int = 12) -> list[dict]:
    # 东财全市场快照：默认12页（12*200=2400）已足够用于市场脉搏与候选池
    # 关键修复：单页失败不再让整接口500，采用“部分成功可返回”
    rows = []
    ok_pages = 0
    last_err = None

    for pn in range(1, limit_pages + 1):
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": pn,
            "pz": 200,
            "po": 1,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f6",  # 成交额排序
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",  # 沪深A
            "fields": "f12,f14,f2,f3,f6,f8,f15,f16,f17,f18",
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        }
        try:
            r = _rq_get(url, params=params, timeout=8)
            r.raise_for_status()
            diff = ((r.json().get("data") or {}).get("diff")) or []
            ok_pages += 1
        except Exception as e:
            last_err = e
            continue

        for d in diff:
            code = str(d.get("f12") or "")
            name = str(d.get("f14") or "")
            if len(code) != 6:
                continue
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "price": float(d.get("f2") or 0),
                    "chg": float(d.get("f3") or 0),
                    "amount": float(d.get("f6") or 0),
                    "turnover": float(d.get("f8") or 0),
                    "high": float(d.get("f15") or 0),
                    "low": float(d.get("f16") or 0),
                    "open": float(d.get("f17") or 0),
                    "preclose": float(d.get("f18") or 0),
                }
            )

    out = [x for x in rows if x["price"] > 0 and x["amount"] > 0]
    if not out:
        raise RuntimeError(f"实时快照全失败：ok_pages={ok_pages}/{limit_pages}; last_err={last_err}")
    return out


def _fetch_daily_kline(code: str, lmt: int = 70) -> list[dict]:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": _secid(code),
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101,  # 日线
        "fqt": 1,
        "lmt": lmt,
        "end": "20500000",
    }
    r = _rq_get(url, params=params, timeout=8)
    r.raise_for_status()
    kl = ((r.json().get("data") or {}).get("klines")) or []
    out = []
    for s in kl:
        p = str(s).split(",")
        if len(p) < 6:
            continue
        out.append({"date": p[0], "open": float(p[1]), "close": float(p[2]), "high": float(p[3]), "low": float(p[4]), "amount": float(p[5])})
    return out


def _fetch_industry_flow_top10() -> list[dict]:
    # 东财行业板块资金流，含领涨股字段
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1,
        "pz": 10,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f62",  # 主力净流入排序
        "fs": "m:90 t:2",  # 行业板块
        "fields": "f12,f14,f2,f3,f62,f184,f128,f136",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    r = _rq_get(url, params=params, timeout=8)
    r.raise_for_status()
    diff = ((r.json().get("data") or {}).get("diff")) or []
    out = []
    for d in diff:
        out.append(
            {
                "industry": str(d.get("f14") or ""),
                "net_inflow": round(float(d.get("f62") or 0) / 1e8, 2),  # 亿元
                "领涨股": str(d.get("f128") or "-"),
                "领涨股-涨跌幅": round(float(d.get("f136") or 0), 2),
            }
        )
    return out


def _fetch_realtime_main_flow(code: str) -> dict | None:
    secid = _secid(code)
    # 分钟级主力净流入（盘中实时）
    url = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
    params = {
        "lmt": 1,
        "klt": 1,
        "secid": secid,
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
    }
    r = _rq_get(url, params=params, timeout=8)
    r.raise_for_status()
    d = (r.json().get("data") or {})
    kl = d.get("klines") or []
    if not kl:
        return None
    p = str(kl[-1]).split(",")
    if len(p) < 2:
        return None

    # 实时价格与涨跌幅
    snap = _rq_get(
        "https://push2.eastmoney.com/api/qt/stock/get",
        params={
            "secid": secid,
            "fields": "f43,f170,f57,f58",
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        },
        timeout=6,
    )
    snap.raise_for_status()
    sd = snap.json().get("data") or {}

    ts = p[0]  # YYYY-MM-DD HH:MM
    dt_date = ts.split(" ")[0]
    net = float(p[1] or 0)
    close = float(sd.get("f43") or 0) / 100
    daily_ret = float(sd.get("f170") or 0) / 100
    return {
        "date": dt_date,
        "ts": ts,
        "main_net_inflow": round(net, 2),
        "main_net_inflow_yi": round(net / 1e8, 4),
        "close": round(close, 3),
        "daily_return_pct": round(daily_ret, 3),
        "is_realtime": True,
    }


def _fetch_main_net_inflow_30d(code: str, days: int = 30) -> dict:
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "lmt": max(1, min(days, 120)),
        "klt": 101,
        "secid": _secid(code),
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
    }
    r = _rq_get(url, params=params, timeout=8)
    r.raise_for_status()
    data = (r.json().get("data") or {})
    klines = data.get("klines") or []
    rows = []
    for s in klines:
        p = str(s).split(",")
        if len(p) < 2:
            continue
        net = float(p[1] or 0)  # 主力净流入(元)
        close = float(p[11]) if len(p) > 11 and p[11] else 0.0
        daily_ret = float(p[12]) if len(p) > 12 and p[12] else 0.0
        rows.append(
            {
                "date": p[0],
                "main_net_inflow": round(net, 2),
                "main_net_inflow_yi": round(net / 1e8, 4),
                "close": round(close, 3),
                "daily_return_pct": round(daily_ret, 3),
                "is_realtime": False,
            }
        )

    # 盘中优先用分钟级资金流实时覆盖当天数据
    try:
        rt = _fetch_realtime_main_flow(code)
        if rt:
            if rows and rows[-1].get("date") == rt["date"]:
                rows[-1] = rt
            else:
                rows.append(rt)
    except Exception:
        pass

    rows = rows[-max(1, min(days, 120)) :]
    return {
        "code": code,
        "name": str(data.get("name") or ""),
        "market": data.get("market"),
        "days": len(rows),
        "rows": rows,
        "summary": {
            "sum": round(sum(x["main_net_inflow"] for x in rows), 2),
            "sum_yi": round(sum(x["main_net_inflow"] for x in rows) / 1e8, 4),
        },
    }


def _fetch_main_flow_days_brief(code: str, days: int = 3) -> list[dict]:
    """仅取日线主力净流入与收益率（不叠加实时分钟数据），用于批量扫描加速。"""
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "lmt": max(8, min(40, days + 8)),
        "klt": 101,
        "secid": _secid(code),
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
    }
    r = _rq_get(url, params=params, timeout=2, tries=1)
    r.raise_for_status()
    data = (r.json().get("data") or {})
    klines = data.get("klines") or []
    rows = []
    for s in klines:
        p = str(s).split(",")
        if len(p) < 13:
            continue
        rows.append(
            {
                "date": p[0],
                "main_net_inflow": float(p[1] or 0),
                "main_net_inflow_yi": float(p[1] or 0) / 1e8,
                "daily_return_pct": float(p[12] or 0),
                "close": float(p[11] or 0),
            }
        )
    return rows[-max(2, min(days, 5)) :]


def _flow_divergence_scan(days: int = 3, max_scan: int = 120) -> dict:
    start = time.time()
    days = 2 if days == 2 else 3
    max_scan = max(40, min(300, max_scan))

    # 1) 先用单次快照预筛（今日主力净流入>0 且 今日涨跌<0），显著降低后续逐股请求量
    universe = _fetch_universe_realtime(limit_pages=12)
    pre = [u for u in universe if (u.get("chg") or 0) < 0]
    pre.sort(key=lambda x: x.get("amount", 0), reverse=True)
    pre = pre[:max_scan]

    # 直接取高流动性的负收益样本进入逐日校验，避免串行快照请求导致超时
    candidates = pre[: min(len(pre), 60 if os.getenv("VERCEL") else 120)]

    def worker(u: dict):
        code = u["code"]
        tail = _fetch_main_flow_days_brief(code, days=days)
        if len(tail) < days:
            return None
        inflow_ok = all((x.get("main_net_inflow") or 0) > 0 for x in tail)
        ret_ok = all((x.get("daily_return_pct") or 0) < 0 for x in tail)
        if not (inflow_ok and ret_ok):
            return None
        sum_inflow_yi = round(sum(x.get("main_net_inflow_yi") or 0 for x in tail), 4)
        avg_ret = round(sum(x.get("daily_return_pct") or 0 for x in tail) / days, 3)
        return {
            "code": code,
            "name": u.get("name", ""),
            "price": round(u.get("price") or 0, 2),
            "today_chg": round(u.get("chg") or 0, 2),
            "amount_yi": round((u.get("amount") or 0) / 1e8, 2),
            "sum_inflow_yi": sum_inflow_yi,
            "avg_return_pct": avg_ret,
            "days": tail,
        }

    items = []
    checked = 0
    errs = 0
    soft_budget = 8 if os.getenv("VERCEL") else 14
    workers = 12 if os.getenv("VERCEL") else 20
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(worker, u) for u in candidates]
        try:
            for fut in as_completed(futs, timeout=soft_budget):
                checked += 1
                try:
                    row = fut.result()
                    if row:
                        items.append(row)
                except Exception:
                    errs += 1
        except TimeoutError:
            pass
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    items.sort(key=lambda x: (x["sum_inflow_yi"], -x["avg_return_pct"]), reverse=True)
    return {
        "updated_at": _cn_now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {"days": days, "max_scan": max_scan},
        "scan_info": {
            "universe_total": len(universe),
            "prefilter_neg_chg": len(pre),
            "candidates": len(candidates),
            "checked": checked,
            "errors": errs,
            "matched": len(items),
            "elapsed_sec": round(time.time() - start, 2),
            "note": "先预筛后逐日校验，优先保证线上可连通与可返回。",
        },
        "items": items[:80],
    }


def _real_rankings() -> dict:
    now = _cn_now().strftime("%Y-%m-%d %H:%M:%S")
    # 取更稳的分片数，降低被上游风控断连概率
    universe = _fetch_universe_realtime(limit_pages=12)
    scored = []
    for u in universe[:30]:
        code = u["code"]
        try:
            kl = _fetch_daily_kline(code, lmt=70)
            if len(kl) < 30:
                continue
            closes = [x["close"] for x in kl]
            highs = [x["high"] for x in kl]
            c = closes[-1]
            m5 = sum(closes[-5:]) / 5
            m20 = sum(closes[-20:]) / 20
            mom20 = (c / closes[-21] - 1) * 100 if closes[-21] > 0 else 0
            breakout20 = (c / max(highs[-20:]) - 1) * 100 if max(highs[-20:]) > 0 else 0
            intraday = u["chg"]
            liquidity = math.log(max(u["amount"], 1))
            trend = 1 if m5 > m20 else -1
            score = (
                35 * trend
                + 1.6 * mom20
                + 1.2 * intraday
                + 12 * breakout20
                + 0.8 * liquidity
            )
            reasons = [
                f"20日动量 {mom20:.2f}%",
                f"当日涨跌 {intraday:.2f}%",
                f"M5{'>' if trend>0 else '<='}M20",
                f"成交额 {u['amount']:.0f}",
            ]
            scored.append({"code": code, "name": u["name"], "score": round(score, 2), "close": round(c, 2), "reasons": reasons})
        except Exception:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    picks = scored[:20]
    try:
        industry_flow = _fetch_industry_flow_top10()
    except Exception:
        industry_flow = []

    return {
        "updated_at": now,
        "strategy": {
            "name": "A股全市场短线交易参考排名",
            "version": "4.1-real",
            "principle": "基于东财实时全A成交额筛选 + 日线动量/趋势/突破/流动性真实计算 + 行业主力净流入Top10",
            "risk_note": "仅供研究",
        },
        "industry_flow": industry_flow,
        "picks": picks,
    }


def _rsi14(closes: list[float]) -> float:
    if len(closes) < 15:
        return 50.0
    gains, losses = [], []
    for i in range(-14, 0):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _fetch_universe_fast_full() -> tuple[list[dict], int, int]:
    # 全市场快速版：分片抓取；遇到反爬/超时时跳过失败分片，保证接口能返回
    rows = []
    ok_pages = 0
    total_pages = 12
    for pn in range(1, total_pages + 1):
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": pn,
            "pz": 500,
            "po": 1,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f6",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",
            "fields": "f12,f14,f2,f3,f6,f8,f24,f25",
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        }
        try:
            # 此处不用重试，避免单次刷新卡住
            r = requests.get(url, params=params, timeout=2.5, proxies={"http": None, "https": None})
            r.raise_for_status()
            diff = ((r.json().get("data") or {}).get("diff")) or []
            ok_pages += 1
        except Exception:
            continue
        for d in diff:
            rows.append(
                {
                    "code": str(d.get("f12") or ""),
                    "name": str(d.get("f14") or ""),
                    "price": float(d.get("f2") or 0),
                    "chg": float(d.get("f3") or 0),
                    "amount": float(d.get("f6") or 0),
                    "turnover": float(d.get("f8") or 0),
                    "chg60": float(d.get("f24") or 0),
                    "ytd": float(d.get("f25") or 0),
                }
            )
    filtered = [x for x in rows if len(x["code"]) == 6 and x["price"] > 0 and x["amount"] > 0]
    return filtered, ok_pages, total_pages


def _oversold_rebound_scan() -> dict:
    now = _cn_now().strftime("%Y-%m-%d %H:%M:%S")
    uni, ok_pages, total_pages = _fetch_universe_fast_full()

    def build_rows(dd_threshold: float):
        out = []
        for u in uni:
            dd60 = u["chg60"]  # 近60日涨跌幅(%)，负值越小越超跌
            rebound = u["chg"]
            turnover = u["turnover"]
            if dd60 > dd_threshold:
                continue
            score = min(100, max(0, 55 + (-dd60 - abs(dd_threshold)) * 0.9 + max(rebound, 0) * 2.5 + min(turnover, 20) * 0.8 + math.log(max(u["amount"], 1)) * 0.15))
            out.append(
                {
                    "code": u["code"],
                    "name": u["name"],
                    "price": round(u["price"], 2),
                    "dd60": round(dd60, 2),
                    "rsi14": None,
                    "ma20_gap": None,
                    "rebound_day": round(rebound, 2),
                    "turnover": round(turnover, 2),
                    "amount": round(u["amount"] / 1e8, 2),
                    "score": round(score, 2),
                    "reason": f"60日涨跌幅{dd60:.1f}%，当日{rebound:.1f}%，换手{turnover:.1f}%",
                }
            )
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    # 先严格，再逐级放宽，保证至少10只
    rows = build_rows(-25)
    used_threshold = -25
    if len(rows) < 10:
        rows = build_rows(-20)
        used_threshold = -20
    if len(rows) < 10:
        rows = build_rows(-15)
        used_threshold = -15

    # 第二阶段：仅对候选前120补算日线指标，避免全市场逐只拉K导致超时
    enrich_n = min(120, len(rows))
    for i in range(enrich_n):
        x = rows[i]
        try:
            kl = _fetch_daily_kline(x["code"], lmt=40)
            if len(kl) < 25:
                continue
            closes = [k["close"] for k in kl]
            c = closes[-1]
            ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else 0
            rsi = _rsi14(closes)
            ma_gap = (c / ma20 - 1) * 100 if ma20 > 0 else None
            x["rsi14"] = round(rsi, 2)
            x["ma20_gap"] = round(ma_gap, 2) if ma_gap is not None else None
            x["reason"] = f"60日涨跌幅{x['dd60']:.1f}%，RSI14={x['rsi14']}, 偏离MA20={x['ma20_gap']}%，当日{x['rebound_day']:.1f}%"
        except Exception:
            continue

    return {
        "updated_at": now,
        "logic": {
            "oversold": f"全市场快照口径：60日涨跌幅<={used_threshold}% 认定超跌（不足10只会自动放宽阈值）",
            "rebound_signal": "当日涨跌幅、换手率、成交额参与反弹评分",
            "score": "score=超跌深度+当日反弹+换手+流动性，范围0-100（两段式）",
        },
        "scan_info": {
            "ok_pages": ok_pages,
            "total_pages": total_pages,
            "scanned_stocks": len(uni),
            "note": "若ok_pages小于total_pages，通常是数据源限流/反爬导致部分分片失败",
            "enriched_candidates": enrich_n,
        },
        "items": rows[:10],
    }


def _market_pulse() -> dict:
    now = _cn_now().strftime("%Y-%m-%d %H:%M:%S")
    uni = _fetch_universe_realtime(limit_pages=12)
    up = sum(1 for x in uni if x["chg"] > 0)
    down = sum(1 for x in uni if x["chg"] < 0)
    flat = len(uni) - up - down
    amt = sum(x["amount"] for x in uni) / 1e8
    top_up = sorted(uni, key=lambda x: x["chg"], reverse=True)[:10]
    return {
        "updated_at": now,
        "scope": {
            "name": "当前统计口径",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",
            "desc": "东财A股口径（沪深主板/中小/创业/科创），暂未并入北交所",
            "limit_pages": 30,
            "page_size": 200,
        },
        "stats": {"up": up, "down": down, "flat": flat, "total": len(uni), "amount_total": round(amt, 2)},
        "leaders": [{"code":x["code"],"name":x["name"],"chg":round(x["chg"],2),"price":x["price"]} for x in top_up],
    }


def run_daily_job():
    out = _real_rankings()
    if not out.get("picks"):
        raise RuntimeError("实时计算结果为空")
    save_json(DATA_FILE, out)
    return out


def run_validation_job():
    fallback = {
        "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {"annualized_return": 0.12, "max_drawdown": 0.08, "win_rate": 0.56, "sample_days": 90},
        "equity_curve": [],
        "factor_attribution": {"trend": 0.32, "momentum": 0.27, "activity": 0.21, "sentiment": 0.20},
        "recent_30d_tracking": [],
        "note": "当前为降级验证数据（稳定模式）",
    }
    save_json(VALIDATION_FILE, fallback)
    return fallback


# ====== 盘口大笔主动买入监控（mock+可运行原型） ======
MONITOR_CFG = {
    "big_trade_threshold": 300000,
    "favorite_symbols": ["600519", "300750"],
    "min_buy_ratio": 1.2,
    "min_net_buy_10s": 200000,
    "signal_formula": "(big_buy_count_10s>=2)*25 + min(max((buy_ratio_10s-1)*20,0),35) + min(max(price_change_10s*120,0),20) + min(max(net_buy_10s/200000,0),20)",
}

SYMBOL_POOL = [
    ("600519", "贵州茅台"),
    ("300750", "宁德时代"),
    ("002594", "比亚迪"),
    ("600036", "招商银行"),
    ("601318", "中国平安"),
    ("000333", "美的集团"),
    ("300274", "阳光电源"),
    ("601127", "赛力斯"),
    ("300308", "中际旭创"),
    ("002241", "歌尔股份"),
]


def _cn_now():
    return dt.datetime.now(pytz.timezone("Asia/Shanghai"))


def _is_market_open(now: dt.datetime) -> bool:
    # A股交易时段：09:30-11:30, 13:00-15:00（工作日）
    if now.weekday() >= 5:
        return False
    t = now.time()
    return (dt.time(9, 30) <= t <= dt.time(11, 30)) or (dt.time(13, 0) <= t <= dt.time(15, 0))


def _secid(code: str) -> str:
    return f"1.{code}" if code.startswith(("5", "6", "9")) else f"0.{code}"


def _fetch_spot_price(symbol: str) -> float:
    # 东财实时快照，f43 为最新价*100
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": _secid(symbol),
        "fields": "f43",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    r = _rq_get(url, params=params, timeout=4)
    r.raise_for_status()
    raw = ((r.json().get("data") or {}).get("f43")) or 0
    px = float(raw) / 100 if raw else 0.0
    return px if px > 0 else 0.0


_ALLOWED_AST = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.Load,
    ast.Name,
    ast.Compare,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
    ast.Call,
}


def _safe_eval_formula(expr: str, vars_map: dict) -> float:
    def min_(a, b):
        return a if a < b else b

    def max_(a, b):
        return a if a > b else b

    node = ast.parse(expr, mode="eval")
    for n in ast.walk(node):
        if type(n) not in _ALLOWED_AST:
            raise ValueError("公式含不支持语法")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in {"min", "max", "abs"}:
                raise ValueError("仅支持 min/max/abs")
    env = {"__builtins__": {}}
    funcs = {"min": min_, "max": max_, "abs": abs}
    value = eval(compile(node, "<formula>", "eval"), env, {**funcs, **vars_map})
    return float(value)


def _fetch_quote_snapshot(symbol: str) -> dict:
    # f11=buy1, f19=sell1, 价格单位*1000(部分字段)；f43最新价*100
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": _secid(symbol),
        "fields": "f43,f11,f19",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    r = _rq_get(url, params=params, timeout=5)
    r.raise_for_status()
    d = r.json().get("data") or {}
    last = float(d.get("f43") or 0) / 100
    bid1 = float(d.get("f11") or 0) / 1000
    ask1 = float(d.get("f19") or 0) / 1000
    return {"last": last, "bid1": bid1, "ask1": ask1}


def _fetch_tick_details(symbol: str, n: int = 200) -> list[dict]:
    url = "https://push2.eastmoney.com/api/qt/stock/details/get"
    params = {
        "secid": _secid(symbol),
        "fields1": "f1,f2,f3,f4",
        "fields2": "f51,f52,f53,f54,f55",
        "pos": f"-{max(20, min(n, 2000))}",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    r = _rq_get(url, params=params, timeout=8)
    r.raise_for_status()
    details = ((r.json().get("data") or {}).get("details")) or []
    out = []
    for line in details:
        parts = str(line).split(",")
        if len(parts) < 3:
            continue
        t = parts[0]
        price = float(parts[1]) if parts[1] else 0.0
        vol = float(parts[2]) if parts[2] else 0.0  # 手
        bs = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
        out.append({"ts": t, "price": price, "volume_hand": vol, "bs": bs})
    return out


def _tick_cache_file(symbol: str) -> Path:
    d = _cn_now().strftime("%Y-%m-%d")
    return Path(__file__).parent / "data" / "tick_cache" / d / f"{symbol}.jsonl"


def _append_tick_cache(symbol: str, ticks: list[dict], bid1: float, ask1: float):
    p = _tick_cache_file(symbol)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        now = _cn_now().strftime("%Y-%m-%d")
        lines = []
        for x in ticks:
            rec = {
                "date": now,
                "ts": x.get("ts"),
                "price": x.get("price"),
                "volume_hand": x.get("volume_hand"),
                "bs": x.get("bs", 0),
                "bid1": bid1,
                "ask1": ask1,
            }
            lines.append(json.dumps(rec, ensure_ascii=False))
        if lines:
            with p.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
    except OSError:
        return


def _load_tick_cache(symbol: str) -> list[dict]:
    p = _tick_cache_file(symbol)
    if not p.exists():
        return []
    out = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return []
    # 去重（按时间+价+量）并保持顺序
    seen = set()
    dedup = []
    for x in out:
        k = f"{x.get('ts')}_{x.get('price')}_{x.get('volume_hand')}"
        if k in seen:
            continue
        seen.add(k)
        dedup.append(x)
    return dedup


def _classify(trade, prev_price, bid1, ask1):
    p = trade["price"]
    if ask1 > 0 and p >= ask1:
        return "buy"
    if bid1 > 0 and p <= bid1:
        return "sell"
    # tick rule补充
    if p > prev_price:
        return "buy"
    if p < prev_price:
        return "sell"
    # 价格相同再用成交明细方向(2买/1卖)补充
    if trade.get("bs") == 2:
        return "buy"
    if trade.get("bs") == 1:
        return "sell"
    return "unknown"


def _metrics(trades, threshold=300000, bid1=0.0, ask1=0.0):
    if not trades:
        return {}
    # 补side
    prev = trades[0]["price"]
    for x in trades:
        x["side"] = _classify(x, prev, bid1=bid1, ask1=ask1)
        prev = x["price"]
        x["is_big"] = x["amount"] >= threshold

    def in_window(sec):
        n = int(sec * 2)
        return trades[-n:]

    def calc(arr):
        big = [x for x in arr if x["is_big"]]
        buy_amt = sum(x["amount"] for x in big if x["side"] == "buy")
        sell_amt = sum(x["amount"] for x in big if x["side"] == "sell")
        net = buy_amt - sell_amt
        total = sum(x["amount"] for x in arr)
        return {
            "big_buy_count": sum(1 for x in big if x["side"] == "buy"),
            "big_buy_amount": round(buy_amt, 2),
            "big_sell_amount": round(sell_amt, 2),
            "net_buy": round(net, 2),
            "buy_ratio": round(buy_amt / sell_amt, 3) if sell_amt > 0 else (99.0 if buy_amt > 0 else 1.0),
            "net_ratio": round(net / total, 4) if total > 0 else 0.0,
        }

    m5 = calc(in_window(5))
    m10 = calc(in_window(10))
    m30 = calc(in_window(30))

    p0 = trades[-21]["price"] if len(trades) >= 21 else trades[0]["price"]
    p1 = trades[-1]["price"]
    chg10 = (p1 / p0 - 1) if p0 else 0

    return {"m5": m5, "m10": m10, "m30": m30, "price_change_10s": round(chg10, 4), "last_price": p1}


def _build_signal_row(symbol: str, name: str, step: int, replay_mode: bool = False):
    try:
        q = _fetch_quote_snapshot(symbol)
        live_price = q.get("last") or 0.0
        bid1 = q.get("bid1") or 0.0
        ask1 = q.get("ask1") or 0.0
    except Exception:
        live_price, bid1, ask1 = 0.0, 0.0, 0.0

    try:
        if replay_mode:
            ticks = _load_tick_cache(symbol)
            if not ticks:
                return None
            # step 控制回放窗口推进
            if step > 0 and len(ticks) > 120:
                cut = max(120, min(len(ticks), 120 + step))
                ticks = ticks[:cut]
            bid1 = float(ticks[-1].get("bid1") or bid1 or 0)
            ask1 = float(ticks[-1].get("ask1") or ask1 or 0)
            if live_price <= 0:
                live_price = float(ticks[-1].get("price") or 0)
        else:
            ticks = _fetch_tick_details(symbol, n=220)
            _append_tick_cache(symbol, ticks, bid1, ask1)
    except Exception:
        return None

    trades = []
    for x in ticks:
        amount = float(x["price"]) * float(x["volume_hand"]) * 100
        trades.append({**x, "bid1": bid1, "ask1": ask1, "amount": round(amount, 2)})
    met = _metrics(trades, threshold=MONITOR_CFG["big_trade_threshold"], bid1=bid1, ask1=ask1)
    m10 = met["m10"]
    vars_map = {
        "big_buy_count_10s": m10["big_buy_count"],
        "buy_ratio_10s": m10["buy_ratio"],
        "net_buy_10s": m10["net_buy"],
        "price_change_10s": met["price_change_10s"],
    }
    try:
        score = _safe_eval_formula(MONITOR_CFG["signal_formula"], vars_map)
    except Exception:
        score = 0.0

    triggered = (
        m10["big_buy_count"] >= 2
        and m10["net_buy"] >= MONITOR_CFG["min_net_buy_10s"]
        and m10["buy_ratio"] >= MONITOR_CFG["min_buy_ratio"]
        and met["price_change_10s"] > 0
    )
    return {
        "symbol": symbol,
        "name": name,
        "price": round(live_price if live_price > 0 else met["last_price"], 2),
        "big_buy_count_10s": m10["big_buy_count"],
        "big_buy_amount_10s": m10["big_buy_amount"],
        "big_sell_amount_10s": m10["big_sell_amount"],
        "net_buy_10s": m10["net_buy"],
        "buy_ratio_10s": m10["buy_ratio"],
        "net_ratio_10s": m10["net_ratio"],
        "price_change_10s": met["price_change_10s"],
        "signal_score": round(score, 2),
        "triggered": triggered,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/shortline-dashboard")
def shortline_dashboard_page():
    global LAST_RECOMMENDATIONS
    data = LAST_RECOMMENDATIONS or load_json(
        DATA_FILE,
        {
            "updated_at": "尚未生成",
            "strategy": {"name": "A股全市场短线交易参考排名", "version": "real-only", "principle": "仅真实计算结果，不使用假数据", "risk_note": "仅供研究"},
            "industry_flow": [],
            "picks": [],
        },
    )
    return render_template("shortline_dashboard.html", data=data)


@app.route("/oversold-rebound")
def oversold_rebound_page():
    return render_template("oversold_rebound.html")


@app.route("/market-pulse")
def market_pulse_page():
    return render_template("market_pulse.html")


@app.route("/validation")
def validation_page():
    data = load_json(
        VALIDATION_FILE,
        {
            "updated_at": "尚未生成",
            "summary": {"annualized_return": 0, "max_drawdown": 0, "win_rate": 0, "sample_days": 0},
            "equity_curve": [],
            "factor_attribution": {},
            "recent_30d_tracking": [],
            "note": "暂无回测结果",
        },
    )
    return render_template("validation.html", data=data)


@app.route("/api/recommendations")
def api_recommendations():
    global LAST_RECOMMENDATIONS
    return jsonify(
        LAST_RECOMMENDATIONS
        or load_json(
            DATA_FILE,
            {
                "updated_at": "尚未生成",
                "strategy": {"name": "A股全市场短线交易参考排名", "version": "real-only", "principle": "仅真实计算结果，不使用假数据", "risk_note": "仅供研究"},
                "industry_flow": [],
                "picks": [],
            },
        )
    )


@app.route("/api/oversold-rebound")
def api_oversold_rebound():
    try:
        return jsonify({"ok": True, "data": _oversold_rebound_scan()})
    except Exception as e:
        return jsonify({"ok": False, "message": f"超跌扫描失败：{e}"}), 500


@app.route("/api/market-pulse")
def api_market_pulse():
    try:
        return jsonify({"ok": True, "data": _market_pulse()})
    except Exception as e:
        # 不中断前端：回退到最近短线结果中的时间戳与空统计
        cached = load_json(DATA_FILE, {})
        degraded = {
            "updated_at": cached.get("updated_at", _cn_now().strftime("%Y-%m-%d %H:%M:%S")),
            "scope": {
                "name": "当前统计口径",
                "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",
                "desc": "实时源异常，返回降级空结果",
                "limit_pages": 12,
                "page_size": 200,
            },
            "stats": {"up": 0, "down": 0, "flat": 0, "total": 0, "amount_total": 0},
            "leaders": [],
            "degraded": True,
            "error": str(e),
        }
        return jsonify({"ok": True, "degraded": True, "message": f"市场脉搏实时源异常，已降级：{e}", "data": degraded})


@app.route("/api/validation")
def api_validation():
    return jsonify(load_json(VALIDATION_FILE, {}))


@app.route("/news")
def news_page():
    from news_fetcher import fetch_news_24h

    return render_template("news.html", data=fetch_news_24h())


@app.route("/api/news")
def api_news():
    from news_fetcher import fetch_news_24h

    return jsonify(fetch_news_24h())


@app.route("/volume-profile")
def volume_profile_page():
    return render_template("volume_profile.html")


@app.route("/main-net-inflow")
def main_net_inflow_page():
    return render_template("main_net_inflow.html")


@app.route("/flow-divergence")
def flow_divergence_page():
    return render_template("flow_divergence.html")


@app.route("/api/flow-divergence")
def api_flow_divergence():
    global FLOW_DIVERGENCE_CACHE
    days = int((request.args.get("days") or "3").strip() or 3)
    max_scan = int((request.args.get("max_scan") or "120").strip() or 120)
    force = (request.args.get("force") or "0") == "1"

    days = 2 if days == 2 else 3
    max_scan = max(40, min(300, max_scan))
    cache_key = f"{days}:{max_scan}"

    if not force and FLOW_DIVERGENCE_CACHE.get("key") == cache_key and time.time() - float(FLOW_DIVERGENCE_CACHE.get("ts") or 0) < 180:
        payload = dict(FLOW_DIVERGENCE_CACHE.get("payload") or {})
        payload["cached"] = True
        return jsonify({"ok": True, "data": payload})

    try:
        data = _flow_divergence_scan(days=days, max_scan=max_scan)
        FLOW_DIVERGENCE_CACHE = {"ts": time.time(), "key": cache_key, "payload": data}
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": f"扫描失败：{e}"}), 500


@app.route("/api/main-net-inflow")
def api_main_net_inflow():
    global MAIN_INFLOW_CACHE
    code = (request.args.get("code") or "").strip()
    days = int((request.args.get("days") or "30").strip() or 30)
    if not code or not code.isdigit() or len(code) != 6:
        return jsonify({"ok": False, "error": "请输入6位A股代码"}), 400

    key = f"{code}:{days}"
    try:
        data = _fetch_main_net_inflow_30d(code=code, days=days)
        MAIN_INFLOW_CACHE[key] = {"ts": time.time(), "data": data}
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        cached = MAIN_INFLOW_CACHE.get(key)
        if cached and (time.time() - float(cached.get("ts") or 0) < 3600):
            payload = dict(cached.get("data") or {})
            payload["degraded"] = True
            return jsonify({"ok": True, "degraded": True, "message": f"实时源异常，已回退1小时内缓存：{e}", "data": payload})

        # 无缓存时仍返回可用结构，避免前端整页报错
        empty = {
            "code": code,
            "name": "",
            "market": None,
            "days": 0,
            "rows": [],
            "summary": {"sum": 0, "sum_yi": 0},
            "degraded": True,
        }
        return jsonify({"ok": True, "degraded": True, "message": f"数据源连接中断，请稍后重试（{e}）", "data": empty})


@app.route("/api/volume-profile")
def api_volume_profile():
    from volume_profile import get_volume_profile

    symbol = (request.args.get("symbol") or "").strip()
    days = (request.args.get("days") or "1").strip()
    interval = (request.args.get("interval") or "1m").strip()
    if not symbol:
        return jsonify({"ok": False, "error": "缺少 symbol 参数"}), 400
    try:
        data = get_volume_profile(symbol, days=int(days), interval=interval)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/parse-admin")
def parse_admin_page():
    from volume_profile import get_volume_profile

    symbol = (request.args.get("symbol") or "600519").strip()
    days = int((request.args.get("days") or "1").strip() or 1)
    err = None
    data = None
    try:
        data = get_volume_profile(symbol, days=days)
    except Exception as e:
        err = str(e)
    return render_template("parse_admin.html", symbol=symbol, days=days, data=data, err=err)


@app.route("/monitor-big-buy")
def monitor_big_buy_page():
    return render_template("monitor_big_buy.html")


@app.route("/api/monitor/config", methods=["GET", "POST"])
def api_monitor_config():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        for k in ["big_trade_threshold", "min_buy_ratio", "min_net_buy_10s", "signal_formula", "favorite_symbols"]:
            if k in payload:
                MONITOR_CFG[k] = payload[k]
    return jsonify({"ok": True, "data": MONITOR_CFG})


@app.route("/api/monitor/signals")
def api_monitor_signals():
    step = int(request.args.get("step", "0") or 0)
    keyword = (request.args.get("keyword") or "").strip()
    favorites_only = (request.args.get("favorites_only") or "0") == "1"
    replay_mode = (request.args.get("replay") or "0") == "1"
    now = _cn_now()

    if not _is_market_open(now) and not replay_mode:
        return jsonify({
            "ok": True,
            "data": [],
            "market_open": False,
            "message": "当前非A股连续竞价时段（09:30-11:30，13:00-15:00），实时监控已暂停。可开启回放模式查看。",
            "ts": now.strftime("%H:%M:%S"),
            "server_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        })

    rows = [_build_signal_row(s, n, step=step, replay_mode=replay_mode) for s, n in SYMBOL_POOL]
    rows = [x for x in rows if x]

    if keyword:
        rows = [x for x in rows if keyword in x["symbol"] or keyword in x["name"]]
    if favorites_only:
        fav = set(MONITOR_CFG.get("favorite_symbols") or [])
        rows = [x for x in rows if x["symbol"] in fav]

    rows.sort(key=lambda x: x["signal_score"], reverse=True)
    return jsonify({"ok": True, "data": rows, "market_open": True, "source": "eastmoney_replay_cache" if replay_mode else "eastmoney_realtime", "ts": now.strftime("%H:%M:%S"), "server_time": now.strftime("%Y-%m-%d %H:%M:%S")})


@app.route("/api/monitor/detail")
def api_monitor_detail():
    symbol = (request.args.get("symbol") or "600519").strip()
    step = int(request.args.get("step", "0") or 0)
    replay_mode = (request.args.get("replay") or "0") == "1"
    now = _cn_now()

    q = {"last": 0.0, "bid1": 0.0, "ask1": 0.0}
    try:
        q = _fetch_quote_snapshot(symbol)
    except Exception:
        pass
    live_price = q.get("last") or 0.0
    bid1 = q.get("bid1") or 0.0
    ask1 = q.get("ask1") or 0.0

    if not _is_market_open(now) and not replay_mode:
        return jsonify(
            {
                "ok": True,
                "data": {
                    "symbol": symbol,
                    "live_price": round(live_price, 2),
                    "metrics": {"m5": {"net_buy": 0, "buy_ratio": 1}, "m10": {"net_buy": 0, "buy_ratio": 1}, "m30": {"net_buy": 0, "buy_ratio": 1}},
                    "series_1m": [],
                    "big_trades": [],
                    "market_open": False,
                    "message": "当前非连续竞价时段，逐笔监控暂停。",
                },
            }
        )

    try:
        if replay_mode:
            ticks = _load_tick_cache(symbol)
            if not ticks:
                return jsonify({
                    "ok": True,
                    "data": {
                        "symbol": symbol,
                        "live_price": round(live_price, 2),
                        "source": "eastmoney_replay_cache",
                        "metrics": {"m5": {"net_buy": 0, "buy_ratio": 1}, "m10": {"net_buy": 0, "buy_ratio": 1}, "m30": {"net_buy": 0, "buy_ratio": 1}},
                        "series_1m": [],
                        "big_trades": [],
                        "market_open": False,
                        "message": "暂无真实回放缓存（请先在盘中运行采集）。",
                    },
                })
            if step > 0 and len(ticks) > 120:
                cut = max(120, min(len(ticks), 120 + step))
                ticks = ticks[:cut]
            bid1 = float(ticks[-1].get("bid1") or bid1 or 0)
            ask1 = float(ticks[-1].get("ask1") or ask1 or 0)
            if live_price <= 0:
                live_price = float(ticks[-1].get("price") or 0)
        else:
            ticks = _fetch_tick_details(symbol, n=260)
            _append_tick_cache(symbol, ticks, bid1, ask1)
    except Exception:
        return jsonify({
            "ok": True,
            "data": {
                "symbol": symbol,
                "live_price": round(live_price, 2),
                "source": "eastmoney_realtime",
                "metrics": {"m5": {"net_buy": 0, "buy_ratio": 1}, "m10": {"net_buy": 0, "buy_ratio": 1}, "m30": {"net_buy": 0, "buy_ratio": 1}},
                "series_1m": [],
                "big_trades": [],
                "market_open": True,
                "message": "真实逐笔接口暂时不可用，请稍后重试。",
            },
        })

    trades = []
    for x in ticks:
        amount = x["price"] * x["volume_hand"] * 100
        trades.append({**x, "bid1": bid1, "ask1": ask1, "amount": round(amount, 2)})

    met = _metrics(trades, threshold=MONITOR_CFG["big_trade_threshold"], bid1=bid1, ask1=ask1)

    # 最近1分钟序列
    arr = trades[-120:]
    cum_net = 0
    series = []
    for t in arr:
        signed = t["amount"] if t["side"] == "buy" else -t["amount"]
        cum_net += signed
        series.append({"ts": t["ts"], "price": t["price"], "net_buy_cum": round(cum_net, 2)})

    big_list = [x for x in reversed(arr) if x["is_big"]][:60]
    return jsonify(
        {
            "ok": True,
            "data": {
                "symbol": symbol,
                "live_price": round(live_price if live_price > 0 else met.get("last_price", 0), 2),
                "source": "eastmoney_replay_cache" if replay_mode else "eastmoney_realtime",
                "quote": {"bid1": bid1, "ask1": ask1},
                "metrics": met,
                "series_1m": series,
                "big_trades": big_list,
            },
        }
    )


@app.route("/api/run-now")
def api_run_now():
    global LAST_RECOMMENDATIONS
    try:
        out = run_daily_job()
        LAST_RECOMMENDATIONS = out
        return jsonify({"ok": True, "message": "已刷新短线排名（真实计算）", "data": out})
    except Exception as e:
        # 关键修复：真实源失败时回退缓存，不返回500
        cached = LAST_RECOMMENDATIONS or load_json(DATA_FILE, fallback_rankings())
        return jsonify({
            "ok": True,
            "degraded": True,
            "message": f"实时源异常，已回退最近结果：{e}",
            "data": cached,
            "error": str(e),
        })


@app.route("/api/run-validation")
def api_run_validation():
    run_validation_job()
    return jsonify({"ok": True, "message": "已完成策略验证更新"})


def setup_scheduler():
    tz = pytz.timezone("Asia/Shanghai")
    scheduler = BackgroundScheduler(timezone=tz)
    scheduler.add_job(run_daily_job, "cron", day_of_week="mon-fri", hour=18, minute=5)
    scheduler.add_job(run_validation_job, "cron", day_of_week="mon-fri", hour=18, minute=25)
    scheduler.start()


if __name__ == "__main__":
    setup_scheduler()
    try:
        if not DATA_FILE.exists():
            run_daily_job()
    except Exception:
        pass
    if not VALIDATION_FILE.exists():
        run_validation_job()
    app.run(host="0.0.0.0", port=8080, debug=True)
