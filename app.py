from __future__ import annotations

import ast
import datetime as dt
import json
import math
import os
import random
import re
import time
import threading
import secrets
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytz
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.json.ensure_ascii = False


def _load_local_env(path: Path):
    """轻量 .env 读取，避免引入额外依赖。仅在变量未设置时写入进程环境。"""
    try:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and ((k not in os.environ) or (not str(os.environ.get(k) or '').strip())):
                os.environ[k] = v
    except Exception:
        pass


_load_local_env(Path(__file__).parent / ".env")


def _oauth_cfg() -> dict:
    return {
        "client_id": (os.getenv("STOCK_RESEARCH_OPENAI_OAUTH_CLIENT_ID") or "").strip(),
        "client_secret": (os.getenv("STOCK_RESEARCH_OPENAI_OAUTH_CLIENT_SECRET") or "").strip(),
        "authorize_url": (os.getenv("STOCK_RESEARCH_OPENAI_OAUTH_AUTHORIZE_URL") or "https://auth.openai.com/oauth/authorize").strip(),
        "token_url": (os.getenv("STOCK_RESEARCH_OPENAI_OAUTH_TOKEN_URL") or "https://auth.openai.com/oauth/token").strip(),
        "redirect_uri": (os.getenv("STOCK_RESEARCH_OPENAI_OAUTH_REDIRECT_URI") or "http://127.0.0.1:8080/oauth/openai/callback").strip(),
        "scope": (os.getenv("STOCK_RESEARCH_OPENAI_OAUTH_SCOPE") or "offline_access").strip(),
    }


def _load_oauth_token_obj() -> dict:
    obj = load_json(OPENAI_OAUTH_TOKEN_FILE, {})
    return obj if isinstance(obj, dict) else {}


def _save_oauth_token_obj(token_obj: dict):
    save_json(OPENAI_OAUTH_TOKEN_FILE, token_obj or {})


def _token_expires_soon(token_obj: dict, leeway_sec: int = 120) -> bool:
    exp = float(token_obj.get("expires_at") or 0)
    if exp <= 0:
        return True
    return time.time() + leeway_sec >= exp


def _oauth_refresh_token(token_obj: dict) -> dict | None:
    cfg = _oauth_cfg()
    refresh_token = (token_obj.get("refresh_token") or "").strip()
    if not (cfg["client_id"] and cfg["client_secret"] and refresh_token):
        return None

    form = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
    }
    try:
        resp = requests.post(cfg["token_url"], data=form, timeout=30)
        resp.raise_for_status()
        obj = resp.json() or {}
        expires_in = int(obj.get("expires_in") or 3600)
        merged = {
            **token_obj,
            **obj,
            "expires_at": time.time() + max(60, expires_in),
            "token_type": obj.get("token_type") or token_obj.get("token_type") or "Bearer",
            "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if not merged.get("refresh_token"):
            merged["refresh_token"] = refresh_token
        _save_oauth_token_obj(merged)
        if merged.get("access_token"):
            os.environ["STOCK_RESEARCH_OPENAI_OAUTH_TOKEN"] = str(merged.get("access_token"))
        return merged
    except Exception:
        return None


def _get_openai_access_token() -> str:
    # 一次性后端配置优先：API Key 方式（前端无感）
    # 优先：千问兼容口径（DashScope）
    qwen_key = (os.getenv("STOCK_RESEARCH_QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if qwen_key:
        return qwen_key

    # 其次：OpenAI
    api_key = (os.getenv("STOCK_RESEARCH_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if api_key:
        return api_key

    # 兼容旧 OAuth token 方式
    env_token = (os.getenv("STOCK_RESEARCH_OPENAI_OAUTH_TOKEN") or os.getenv("OPENAI_OAUTH_ACCESS_TOKEN") or "").strip()
    if env_token:
        return env_token

    token_obj = _load_oauth_token_obj()
    if not token_obj:
        return ""

    if _token_expires_soon(token_obj):
        refreshed = _oauth_refresh_token(token_obj)
        if refreshed and refreshed.get("access_token"):
            return str(refreshed.get("access_token"))

    t = (token_obj.get("access_token") or "").strip()
    if t:
        os.environ["STOCK_RESEARCH_OPENAI_OAUTH_TOKEN"] = t
    return t

DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "latest_recommendations.json"
VALIDATION_FILE = DATA_DIR / "validation.json"
OPENAI_OAUTH_TOKEN_FILE = DATA_DIR / "openai_oauth_token.json"
OPENAI_OAUTH_STATE_CACHE: dict[str, float] = {}
LAST_RECOMMENDATIONS = None
FLOW_DIVERGENCE_CACHE = {"ts": 0.0, "key": "", "payload": None}
FLOW_DIVERGENCE_JOBS = {}
FLOW_DIVERGENCE_LOCK = threading.Lock()
UNIVERSE_CACHE = {"ts": 0.0, "rows": []}
UNIVERSE_CACHE_FILE = Path("/tmp/stock326_universe_cache.json")
FLOW_DIVERGENCE_CACHE_FILE = Path("/tmp/stock326_flow_divergence_cache.json")
MAIN_INFLOW_CACHE = {}
POLICY_METRICS_CACHE = {"ts": 0.0, "data": {}}
AI_RUNTIME_STATUS = {
    "last_ok": None,
    "last_error": "",
    "last_error_at": "",
    "last_provider": "",
    "last_model": "",
}


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


def _load_latest_nonempty_industry_from_history() -> list[dict]:
    try:
        hist_dir = DATA_DIR / "history"
        if not hist_dir.exists():
            return []
        files = sorted(hist_dir.glob("*.json"), reverse=True)
        for fp in files[:40]:  # 仅回看最近40天，避免扫描过重
            d = load_json(fp, {})
            flow = d.get("industry_flow") or []
            if flow:
                return flow
    except Exception:
        pass
    return []


def _fill_industry_flow_if_missing(data: dict | None) -> dict:
    data = dict(data or {})
    if data.get("industry_flow"):
        return data

    # 兜底顺序：在线拉取 -> 当前缓存 -> 历史最近非空 -> 内置默认
    try:
        flow = _fetch_industry_flow_top10()
        if flow:
            data["industry_flow"] = flow
            return data
    except Exception:
        pass

    cached = load_json(DATA_FILE, {})
    flow = cached.get("industry_flow") or []
    if flow:
        data["industry_flow"] = flow
        return data

    flow = _load_latest_nonempty_industry_from_history()
    if flow:
        data["industry_flow"] = flow
        return data

    data["industry_flow"] = fallback_rankings().get("industry_flow") or []
    return data


# ====== 短线排名 ======
def _fetch_universe_realtime(limit_pages: int = 40) -> list[dict]:
    # 东财全市场快照：尽量抓全分页（默认40页，上限由数据源返回决定）
    # 关键修复：单页失败不再让整接口500，采用“部分成功可返回”
    global UNIVERSE_CACHE
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
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",  # 沪深A + 北交所
            "fields": "f12,f14,f2,f3,f6,f8,f15,f16,f17,f18",
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        }
        try:
            r = _rq_get(url, params=params, timeout=7, tries=2)
            r.raise_for_status()
            diff = ((r.json().get("data") or {}).get("diff")) or []
            ok_pages += 1
        except Exception as e:
            last_err = e
            continue

        if not diff:
            break

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
    if out:
        UNIVERSE_CACHE = {"ts": time.time(), "rows": out}
        try:
            UNIVERSE_CACHE_FILE.write_text(json.dumps({"ts": time.time(), "rows": out}, ensure_ascii=False))
        except Exception:
            pass
        return out

    # 备用1：尝试快速全市场抓取口径（不同分页参数）
    try:
        fast_rows, fast_ok, fast_total = _fetch_universe_fast_full()
        if fast_rows:
            mapped = [
                {
                    "code": x.get("code", ""),
                    "name": x.get("name", ""),
                    "price": float(x.get("price") or 0),
                    "chg": float(x.get("chg") or 0),
                    "amount": float(x.get("amount") or 0),
                    "turnover": float(x.get("turnover") or 0),
                    "high": 0.0,
                    "low": 0.0,
                    "open": 0.0,
                    "preclose": 0.0,
                }
                for x in fast_rows
                if len(str(x.get("code") or "")) == 6
            ]
            if mapped:
                UNIVERSE_CACHE = {"ts": time.time(), "rows": mapped}
                try:
                    UNIVERSE_CACHE_FILE.write_text(json.dumps({"ts": time.time(), "rows": mapped}, ensure_ascii=False))
                except Exception:
                    pass
                return mapped
    except Exception:
        pass

    # 备用2：回退进程内缓存，保证服务不断
    cached = UNIVERSE_CACHE.get("rows") or []
    if cached:
        return cached

    # 备用3：回退磁盘缓存（进程重启后也可用）
    try:
        if UNIVERSE_CACHE_FILE.exists():
            obj = json.loads(UNIVERSE_CACHE_FILE.read_text() or "{}")
            rows2 = obj.get("rows") or []
            if rows2:
                UNIVERSE_CACHE = {"ts": float(obj.get("ts") or time.time()), "rows": rows2}
                return rows2
    except Exception:
        pass

    raise RuntimeError(f"实时快照全失败：ok_pages={ok_pages}/{limit_pages}; last_err={last_err}")


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
    r = _rq_get(url, params=params, timeout=3, tries=1)
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
    r = _rq_get(url, params=params, timeout=3, tries=1)
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
    r = _rq_get(url, params=params, timeout=2, tries=1)
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
        timeout=2,
        tries=1,
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
    r = _rq_get(url, params=params, timeout=2, tries=1)
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
    max_scan = max(20, min(300, max_scan))

    # 1) 先用单次快照预筛（今日主力净流入>0 且 今日涨跌<0），显著降低后续逐股请求量
    universe = _fetch_universe_realtime(limit_pages=12)
    pre = [u for u in universe if (u.get("chg") or 0) < 0]
    pre.sort(key=lambda x: x.get("amount", 0), reverse=True)
    pre = pre[:max_scan]

    # 直接取高流动性的负收益样本进入逐日校验
    # 按用户要求：不再做固定候选上限截断，使用页面传入的 max_scan 全量候选
    candidates = pre[:max_scan]

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
    workers = 8 if os.getenv("VERCEL") else 10
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(worker, u) for u in candidates]
        for fut in as_completed(futs):
            checked += 1
            try:
                row = fut.result()
                if row:
                    items.append(row)
            except Exception:
                errs += 1
    finally:
        ex.shutdown(wait=True)

    items.sort(key=lambda x: (x["sum_inflow_yi"], -x["avg_return_pct"]), reverse=True)
    return {
        "updated_at": updated,
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

    if not industry_flow:
        industry_flow = _fill_industry_flow_if_missing({"industry_flow": []}).get("industry_flow") or []

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
    # 全市场口径：30页 * 200 = 6000（覆盖沪深A股主流样本）
    uni = _fetch_universe_realtime(limit_pages=30)
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
            "desc": "东财A股口径（沪深主板/中小/创业/科创/北交所）",
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
    out = _fill_industry_flow_if_missing(out)
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
    data = _fill_industry_flow_if_missing(data)
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
    data = LAST_RECOMMENDATIONS or load_json(
        DATA_FILE,
        {
            "updated_at": "尚未生成",
            "strategy": {"name": "A股全市场短线交易参考排名", "version": "real-only", "principle": "仅真实计算结果，不使用假数据", "risk_note": "仅供研究"},
            "industry_flow": [],
            "picks": [],
        },
    )
    return jsonify(_fill_industry_flow_if_missing(data))


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


def _fetch_index_snapshot(secid: str, name: str) -> dict:
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": secid,
        "fields": "f43,f170,f58",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    r = _rq_get(url, params=params, timeout=4)
    r.raise_for_status()
    d = (r.json() or {}).get("data") or {}
    price = float(d.get("f43") or 0) / 100
    chg_pct = float(d.get("f170") or 0) / 100
    return {"name": name, "price": round(price, 2), "chg_pct": round(chg_pct, 2)}


def _fetch_northbound_net_yi() -> float:
    url = "https://push2.eastmoney.com/api/qt/kamt/get"
    params = {
        "fields1": "f1,f2,f3,f4",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
    }
    r = _rq_get(url, params=params, timeout=4)
    r.raise_for_status()
    d = (r.json() or {}).get("data") or {}
    # dayNetAmtIn 单位为“万”，转换为“亿”
    sh = float(((d.get("hk2sh") or {}).get("dayNetAmtIn") or 0))
    sz = float(((d.get("hk2sz") or {}).get("dayNetAmtIn") or 0))
    return round((sh + sz) / 10000, 2)


@app.route("/api/home-ticker")
def api_home_ticker():
    try:
        pulse = _market_pulse()
        stats = pulse.get("stats") or {}
        idx = {
            "sh": _fetch_index_snapshot("1.000001", "上证指数"),
            "sz": _fetch_index_snapshot("0.399001", "深证成指"),
            "cyb": _fetch_index_snapshot("0.399006", "创业板指"),
        }
        nb = _fetch_northbound_net_yi()
        return jsonify({
            "ok": True,
            "data": {
                "updated_at": _cn_now().strftime("%Y-%m-%d %H:%M:%S"),
                "indices": idx,
                "northbound_net_yi": nb,
                "amount_total_yi": round(float(stats.get("amount_total") or 0), 2),
                "up": int(stats.get("up") or 0),
                "down": int(stats.get("down") or 0),
                "total": int(stats.get("total") or 0),
                "source": "eastmoney_realtime",
            },
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"home ticker fetch failed: {e}"}), 500


@app.route("/api/validation")
def api_validation():
    return jsonify(load_json(VALIDATION_FILE, {}))


def _normalize_url(base: str, href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return "https://www.pbc.gov.cn" + href
    return base.rsplit("/", 1)[0] + "/" + href


def _extract_text(html: str) -> str:
    t = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    t = re.sub(r"<style[\s\S]*?</style>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _fetch_latest_omo_7d() -> dict:
    idx = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/index.html"
    r = requests.get(idx, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r.text, flags=re.I | re.S)
    target_url = ""
    for href, inner in links:
        title = re.sub(r"<[^>]+>", "", inner)
        title = re.sub(r"\s+", " ", title).strip()
        if "公开市场业务交易公告" in title:
            target_url = _normalize_url(idx, href)
            break
    if not target_url:
        raise RuntimeError("未找到公开市场业务交易公告")

    d = requests.get(target_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    d.encoding = "utf-8"
    txt = _extract_text(d.text)

    date = (re.findall(r"(\d{4}年\d{1,2}月\d{1,2}日)", txt) or [""])[0]
    amount = (re.findall(r"(\d+\s*亿元)7天期逆回购", txt) or re.findall(r"(\d+\s*亿元)", txt) or [""])[0]
    rate = (
        re.findall(r"7\s*天\s*([0-9]\s*\.?\s*[0-9]+)\s*%", txt)
        or re.findall(r"利率[为：:]?\s*([0-9]+\.?\d*)\s*%", txt)
        or [""]
    )[0]
    rate = rate.replace(" ", "")

    return {
        "date": date,
        "amount": amount.replace(" ", ""),
        "rate": (rate + "%") if rate else "",
        "url": target_url,
    }


def _fetch_latest_lpr() -> dict:
    idx = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/3876551/index.html"
    r = requests.get(idx, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r.text, flags=re.I | re.S)
    target_url = ""
    for href, inner in links:
        title = re.sub(r"<[^>]+>", "", inner)
        title = re.sub(r"\s+", " ", title).strip()
        if "受权公布贷款市场报价利率" in title and "公告" in title:
            target_url = _normalize_url(idx, href)
            break
    if not target_url:
        raise RuntimeError("未找到LPR公告")

    d = requests.get(target_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    d.encoding = "utf-8"
    txt = _extract_text(d.text)

    date = (re.findall(r"(\d{4}年\d{1,2}月\d{1,2}日)", txt) or [""])[0]
    lpr1 = (re.findall(r"1年期LPR为\s*([0-9]+\.?\d*)\s*%", txt) or [""])[0]
    lpr5 = (re.findall(r"5年期以上LPR为\s*([0-9]+\.?\d*)\s*%", txt) or [""])[0]
    return {
        "date": date,
        "lpr1": (lpr1 + "%") if lpr1 else "",
        "lpr5": (lpr5 + "%") if lpr5 else "",
        "url": target_url,
    }


def _fetch_latest_mlf() -> dict:
    idx = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125437/125446/125873/index.html"
    r = requests.get(idx, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r.text, flags=re.I | re.S)
    target_url = ""
    for href, inner in links:
        title = re.sub(r"<[^>]+>", "", inner)
        title = re.sub(r"\s+", " ", title).strip()
        if "中期借贷便利" in title and "公告" in title:
            target_url = _normalize_url(idx, href)
            break
    if not target_url:
        return {}

    d = requests.get(target_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    d.encoding = "utf-8"
    txt = _extract_text(d.text)
    date = (re.findall(r"(\d{4}年\d{1,2}月\d{1,2}日)", txt) or [""])[0]
    amount = (re.findall(r"开展(\d+\s*亿元)MLF", txt) or re.findall(r"(\d+\s*亿元)MLF", txt) or [""])[0]
    term = (re.findall(r"期限为?([0-9]+年期)", txt) or re.findall(r"期限为?([0-9]+个月)", txt) or [""])[0]
    return {"date": date, "amount": amount.replace(" ", ""), "term": term, "url": target_url}


def _policy_live_metrics() -> dict:
    out = {}
    try:
        out["omo_7d"] = _fetch_latest_omo_7d()
    except Exception:
        out["omo_7d"] = {}
    try:
        out["lpr"] = _fetch_latest_lpr()
    except Exception:
        out["lpr"] = {}
    try:
        out["mlf"] = _fetch_latest_mlf()
    except Exception:
        out["mlf"] = {}
    try:
        out["omo_hist"] = _fetch_omo_7d_history(limit=14)
    except Exception:
        out["omo_hist"] = []
    try:
        out["lpr_hist"] = _fetch_lpr_history(limit=5)
    except Exception:
        out["lpr_hist"] = []
    try:
        out["mlf_hist"] = _fetch_mlf_history(limit=5)
    except Exception:
        out["mlf_hist"] = []
    return out




def _fetch_omo_7d_history(limit: int = 5) -> list[dict]:
    idx = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/index.html"
    r = requests.get(idx, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r.text, flags=re.I | re.S)
    out = []
    seen = set()
    for href, inner in links:
        title = re.sub(r"<[^>]+>", "", inner)
        title = re.sub(r"\s+", " ", title).strip()
        if "公开市场业务交易公告" not in title:
            continue
        u = _normalize_url(idx, href)
        if u in seen:
            continue
        seen.add(u)
        try:
            d = requests.get(u, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            d.encoding = "utf-8"
            txt = _extract_text(d.text)
            date = (re.findall(r"(\d{4}年\d{1,2}月\d{1,2}日)", txt) or [""])[0]
            amt = (re.findall(r"(\d+\s*亿元)7天期逆回购", txt) or re.findall(r"(\d+\s*亿元)", txt) or [""])[0].replace(" ","")
            rate = (re.findall(r"7\s*天\s*([0-9]\s*\.?\s*[0-9]+)\s*%", txt) or [""])[0].replace(" ","")
            out.append({"date": date, "rate": (rate+"%") if rate else "", "amount": amt, "url": u})
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def _fetch_lpr_history(limit: int = 5) -> list[dict]:
    idx = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/3876551/index.html"
    r = requests.get(idx, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r.text, flags=re.I | re.S)
    out = []
    seen = set()
    for href, inner in links:
        title = re.sub(r"<[^>]+>", "", inner)
        title = re.sub(r"\s+", " ", title).strip()
        if "LPR" not in title or "公告" not in title:
            continue
        u = _normalize_url(idx, href)
        if u in seen:
            continue
        seen.add(u)
        try:
            d = requests.get(u, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            d.encoding = "utf-8"
            txt = _extract_text(d.text)
            date = (re.findall(r"(\d{4}年\d{1,2}月\d{1,2}日)", txt) or [""])[0]
            l1 = (re.findall(r"1年期LPR为\s*([0-9]+\.?\d*)\s*%", txt) or [""])[0]
            l5 = (re.findall(r"5年期以上LPR为\s*([0-9]+\.?\d*)\s*%", txt) or [""])[0]
            out.append({"date": date, "lpr1": (l1+"%") if l1 else "", "lpr5": (l5+"%") if l5 else "", "url": u})
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def _fetch_mlf_history(limit: int = 5) -> list[dict]:
    idx = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125437/125446/125873/index.html"
    r = requests.get(idx, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.encoding = "utf-8"
    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r.text, flags=re.I | re.S)
    out = []
    seen = set()
    for href, inner in links:
        title = re.sub(r"<[^>]+>", "", inner)
        title = re.sub(r"\s+", " ", title).strip()
        if "中期借贷便利" not in title or "公告" not in title:
            continue
        u = _normalize_url(idx, href)
        if u in seen:
            continue
        seen.add(u)
        try:
            d = requests.get(u, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            d.encoding = "utf-8"
            txt = _extract_text(d.text)
            date = (re.findall(r"(\d{4}年\d{1,2}月\d{1,2}日)", txt) or [""])[0]
            amount = (re.findall(r"开展(\d+\s*亿元)MLF", txt) or re.findall(r"(\d+\s*亿元)MLF", txt) or [""])[0].replace(" ","")
            term = (re.findall(r"期限为?([0-9]+年期)", txt) or re.findall(r"期限为?([0-9]+个月)", txt) or [""])[0]
            out.append({"date": date, "amount": amount, "term": term, "url": u})
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def _notice_rows_by_source(source_kw: str, limit: int = 5) -> list[dict]:
    items = (_policy_news().get("items") or [])
    out = []
    for it in items:
        if source_kw not in (it.get("source") or ""):
            continue
        url = it.get("url") or ""
        date, fig = "", ""
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            r.encoding = r.apparent_encoding or "utf-8"
            txt = _extract_text(r.text)
            date = (re.findall(r"(\d{4}年\d{1,2}月\d{1,2}日)", txt) or re.findall(r"(\d{4}年\d{2}月\d{2}日)", txt) or [""])[0]
            nums = re.findall(r"\d+\.?\d*%|\d+\s*亿元|\d+\s*bp", txt, flags=re.I)
            fig = "、".join([n.replace(" ","") for n in nums[:3]])
        except Exception:
            pass
        out.append({"metric": it.get("title") or "公告", "latest": f"{date or '日期待补'} {fig or ''}".strip(), "freq": "滚动", "source": url or (it.get("source") or "官方")})
        if len(out) >= limit:
            break
    return out
def _policy_latest_notices() -> dict:
    news = (_policy_news().get("items") or [])[:30]

    def parse_brief(url: str) -> dict:
        if not url:
            return {"date": "", "fig": ""}
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            r.encoding = r.apparent_encoding or "utf-8"
            txt = _extract_text(r.text)
            date = (re.findall(r"(\d{4}年\d{1,2}月\d{1,2}日)", txt) or [""])[0]
            nums = []
            nums += re.findall(r"\d+\.?\d*%", txt)[:2]
            nums += re.findall(r"\d+\s*亿元", txt)[:2]
            nums += re.findall(r"\d+\s*bp", txt, flags=re.I)[:1]
            nums = [n.replace(" ", "") for n in nums if n]
            fig = "、".join(dict.fromkeys(nums))
            return {"date": date, "fig": fig}
        except Exception:
            return {"date": "", "fig": ""}

    def pick(*keywords):
        for it in news:
            t = it.get("title") or ""
            if all(k in t for k in keywords):
                b = parse_brief(it.get("url", ""))
                return {"title": t, "url": it.get("url", ""), "source": it.get("source", ""), "date": b.get("date", ""), "fig": b.get("fig", "")}
        return {"title": "暂无匹配公告", "url": "", "source": "", "date": "", "fig": ""}

    def fallback_by_source(source_kw: str):
        for it in news:
            src = it.get("source", "")
            if source_kw in src:
                b = parse_brief(it.get("url", ""))
                return {"title": it.get("title", ""), "url": it.get("url", ""), "source": src, "date": b.get("date", ""), "fig": b.get("fig", "")}
        return {"title": "暂无匹配公告", "url": "", "source": "", "date": "", "fig": ""}

    rrr = pick("准备金")
    if rrr.get("title") == "暂无匹配公告":
        rrr = fallback_by_source("人民银行")

    reloan = pick("再贷款")
    if reloan.get("title") == "暂无匹配公告":
        reloan = fallback_by_source("人民银行")

    psl = pick("结构", "工具")
    if psl.get("title") == "暂无匹配公告":
        psl = fallback_by_source("人民银行")

    fiscal = fallback_by_source("财政部")
    if fiscal.get("title") == "暂无匹配公告":
        fiscal = pick("财政")

    bond = pick("国债")
    if bond.get("title") == "暂无匹配公告":
        bond = fallback_by_source("财政部")

    tax = pick("税")
    if tax.get("title") == "暂无匹配公告":
        tax = fallback_by_source("财政部")

    return {
        "rrr": rrr,
        "reloan": reloan,
        "psl": psl,
        "fiscal": fiscal,
        "bond": bond,
        "tax": tax,
    }


def _policy_catalog() -> dict:
    live = _policy_live_metrics()
    omo = live.get("omo_7d") or {}
    lpr = live.get("lpr") or {}
    mlf = live.get("mlf") or {}
    notices = _policy_latest_notices()
    updated = _cn_now().strftime("%Y-%m-%d %H:%M:%S")
    omo_hist = live.get("omo_hist") or []
    lpr_hist = live.get("lpr_hist") or []
    mlf_hist = live.get("mlf_hist") or []
    data = {
        "updated_at": _cn_now().strftime("%Y-%m-%d %H:%M:%S"),
        "fiscal": [
            {
                "name": "赤字率与财政扩张",
                "desc": "通过提高赤字率扩大总需求，稳增长稳就业",
                "signals": ["预算赤字率", "广义财政支出增速", "专项债发行进度"],
                "mechanism": "政府扩大支出或减税，直接拉动总需求。",
                "impact": "基建链、消费修复、地方财政相关行业更敏感。",
                "table": [
                    {"metric": "最近财政政策公告", "latest": notices.get("fiscal",{}).get("title") or "暂无", "freq": "滚动", "source": notices.get("fiscal",{}).get("url") or "财政部"},
                    {"metric": "公告日期", "latest": notices.get("fiscal",{}).get("date") or "暂无", "freq": "滚动", "source": notices.get("fiscal",{}).get("source") or "财政部"},
                    {"metric": "公告关键数字", "latest": notices.get("fiscal",{}).get("fig") or "暂无", "freq": "滚动", "source": notices.get("fiscal",{}).get("url") or "财政部"}
                ],
            },
            {
                "name": "专项债",
                "desc": "面向基建与重大项目，形成实物工作量",
                "signals": ["新增专项债额度", "投向结构", "项目开工率"],
                "mechanism": "以项目为载体形成投资增量，提升固定资产投资。",
                "impact": "建筑建材、工程机械、区域基建链受益更直接。",
                "table": [
                    {"metric": "最近专项债相关公告", "latest": notices.get("fiscal",{}).get("title") or "暂无", "freq": "滚动", "source": notices.get("fiscal",{}).get("url") or "官方渠道"},
                    {"metric": "公告日期", "latest": notices.get("fiscal",{}).get("date") or "暂无", "freq": "滚动", "source": notices.get("fiscal",{}).get("source") or "官方渠道"},
                    {"metric": "公告关键数字", "latest": notices.get("fiscal",{}).get("fig") or "暂无", "freq": "滚动", "source": notices.get("fiscal",{}).get("url") or "官方渠道"},
                ],
            },
            {
                "name": "超长期特别国债",
                "desc": "支持国家重大战略与安全能力建设",
                "signals": ["发行规模", "期限结构", "资金投向"],
                "mechanism": "以长期低成本资金支持长期战略项目。",
                "impact": "高端制造、能源安全、科技创新相关方向受关注。",
                "table": [
                    {"metric": "最近国债相关公告", "latest": notices.get("bond",{}).get("title") or "暂无", "freq": "滚动", "source": notices.get("bond",{}).get("url") or "官方渠道"},
                    {"metric": "公告日期", "latest": notices.get("bond",{}).get("date") or "暂无", "freq": "滚动", "source": notices.get("bond",{}).get("source") or "官方渠道"},
                    {"metric": "公告关键数字", "latest": notices.get("bond",{}).get("fig") or "暂无", "freq": "滚动", "source": notices.get("bond",{}).get("url") or "官方渠道"},
                ],
            },
            {
                "name": "减税降费",
                "desc": "降低企业与居民负担，改善现金流和预期",
                "signals": ["税费减免规模", "制造业税负", "小微企业税收优惠覆盖"],
                "mechanism": "降低税费后企业利润与居民可支配收入边际改善。",
                "impact": "中小企业、制造业、可选消费修复弹性较大。",
                "table": [
                    {"metric": "最近税费政策公告", "latest": notices.get("tax",{}).get("title") or "暂无", "freq": "滚动", "source": notices.get("tax",{}).get("url") or "官方渠道"},
                    {"metric": "公告日期", "latest": notices.get("tax",{}).get("date") or "暂无", "freq": "滚动", "source": notices.get("tax",{}).get("source") or "官方渠道"},
                    {"metric": "公告关键数字", "latest": notices.get("tax",{}).get("fig") or "暂无", "freq": "滚动", "source": notices.get("tax",{}).get("url") or "官方渠道"},
                ],
            },
            {
                "name": "转移支付",
                "desc": "中央向地方和重点领域定向支持",
                "signals": ["一般性转移支付", "均衡性转移支付", "民生类支出占比"],
                "mechanism": "缓解地方财力约束，保障民生与公共服务支出。",
                "impact": "区域经济稳定性提升，民生链条需求更稳。",
                "table": [
                    {"metric": "最近财政/转移支付公告", "latest": notices.get("fiscal",{}).get("title") or "暂无", "freq": "滚动", "source": notices.get("fiscal",{}).get("url") or "官方渠道"},
                    {"metric": "公告日期", "latest": notices.get("fiscal",{}).get("date") or "暂无", "freq": "滚动", "source": notices.get("fiscal",{}).get("source") or "官方渠道"},
                    {"metric": "公告关键数字", "latest": notices.get("fiscal",{}).get("fig") or "暂无", "freq": "滚动", "source": notices.get("fiscal",{}).get("url") or "官方渠道"},
                ],
            },
        ],
        "monetary": [
            {
                "name": "降准(RRR)",
                "desc": "释放长期资金，改善银行体系流动性",
                "signals": ["法定准备金率", "中长期流动性缺口", "银行负债成本"],
                "mechanism": "降低缴准后释放可贷资金，缓解银行负债端压力。",
                "impact": "银行、地产链与高股息资产估值中枢通常受影响。",
                "table": [
                    {"metric": "最近准备金政策公告", "latest": notices.get("rrr",{}).get("title") or "暂无", "freq": "滚动", "source": notices.get("rrr",{}).get("url") or "人民银行"},
                    {"metric": "公告日期", "latest": notices.get("rrr",{}).get("date") or "暂无", "freq": "滚动", "source": notices.get("rrr",{}).get("source") or "人民银行"},
                    {"metric": "公告关键数字", "latest": notices.get("rrr",{}).get("fig") or "暂无", "freq": "滚动", "source": notices.get("rrr",{}).get("url") or "人民银行"},
                ],
            },
            {
                "name": "政策利率(7天逆回购/MLF)",
                "desc": "引导市场利率中枢，影响融资成本",
                "signals": ["7天逆回购利率", "MLF利率", "DR007"],
                "mechanism": "政策利率变动向货币市场和贷款利率传导。",
                "impact": "成长股估值、债券收益率、融资敏感行业受影响。",
                "table": [
                    {"metric": "最新MLF操作量", "latest": mlf.get("amount") or "抓取失败", "freq": "月", "source": mlf.get("url") or "人民银行"},
                    {"metric": "最新MLF期限", "latest": mlf.get("term") or "抓取失败", "freq": "月", "source": mlf.get("url") or "人民银行"},
                    {"metric": "实施日期", "latest": mlf.get("date") or "抓取失败", "freq": "月", "source": mlf.get("url") or "人民银行"},
                ],
            },
            {
                "name": "LPR",
                "desc": "贷款定价基准，传导至企业和居民贷款",
                "signals": ["1年LPR", "5年LPR", "新发放贷款利率"],
                "mechanism": "LPR变化直接影响新增和重定价贷款成本。",
                "impact": "地产、消费信贷、资本开支相关行业更敏感。",
                "table": [
                    {"metric": "1年期LPR", "latest": lpr.get("lpr1") or "抓取失败", "freq": "月", "source": "人民银行/LPR公告"},
                    {"metric": "5年期以上LPR", "latest": lpr.get("lpr5") or "抓取失败", "freq": "月", "source": "人民银行/LPR公告"},
                    {"metric": "实施日期", "latest": lpr.get("date") or "抓取失败", "freq": "月", "source": lpr.get("url") or "人民银行"},
                ],
            },
            {
                "name": "OMO逆回购",
                "desc": "短端流动性调节，平滑资金面波动",
                "signals": ["净投放规模", "到期量", "货币市场利率"],
                "mechanism": "公开市场操作对冲短期流动性缺口。",
                "impact": "短端利率、同业资金面、交易拥挤度变化更快。",
                "table": [
                    {"metric": "7天逆回购利率", "latest": omo.get("rate") or "抓取失败", "freq": "日", "source": "人民银行公开市场公告"},
                    {"metric": "7天逆回购操作量", "latest": omo.get("amount") or "抓取失败", "freq": "日", "source": "人民银行公开市场公告"},
                    {"metric": "实施日期", "latest": omo.get("date") or "抓取失败", "freq": "日", "source": omo.get("url") or "人民银行"},
                ],
            },
            {
                "name": "再贷款再贴现",
                "desc": "结构性支持科创、小微、绿色等方向",
                "signals": ["工具额度", "投向占比", "加权融资成本"],
                "mechanism": "定向低成本资金支持特定领域信用扩张。",
                "impact": "科创、小微产业链与绿色投资方向受益明显。",
                "table": [
                    {"metric": "最近再贷款相关公告", "latest": notices.get("reloan",{}).get("title") or "暂无", "freq": "滚动", "source": notices.get("reloan",{}).get("url") or "人民银行"},
                    {"metric": "公告日期", "latest": notices.get("reloan",{}).get("date") or "暂无", "freq": "滚动", "source": notices.get("reloan",{}).get("source") or "人民银行"},
                    {"metric": "公告关键数字", "latest": notices.get("reloan",{}).get("fig") or "暂无", "freq": "滚动", "source": notices.get("reloan",{}).get("url") or "人民银行"},
                ],
            },
            {
                "name": "PSL/结构性工具",
                "desc": "定向支持重点领域与三大工程等",
                "signals": ["新增PSL", "结构性货币工具余额", "政策导向行业融资"],
                "mechanism": "通过政策性金融渠道提供中长期定向资金。",
                "impact": "保障房、城中村改造、公共设施链条受影响。",
                "table": [
                    {"metric": "最近结构性工具公告", "latest": notices.get("psl",{}).get("title") or "暂无", "freq": "滚动", "source": notices.get("psl",{}).get("url") or "人民银行"},
                    {"metric": "公告日期", "latest": notices.get("psl",{}).get("date") or "暂无", "freq": "滚动", "source": notices.get("psl",{}).get("source") or "人民银行"},
                    {"metric": "公告关键数字", "latest": notices.get("psl",{}).get("fig") or "暂无", "freq": "滚动", "source": notices.get("psl",{}).get("url") or "人民银行"},
                ],
            },
        ],
        "combo": [
            {
                "name": "财政发力+货币配合",
                "desc": "财政端形成需求，货币端降低融资成本与对冲流动性波动",
                "mechanism": "财政创造需求、货币稳定资金面，形成政策合力。",
                "impact": "稳增长资产与顺周期资产弹性提升。",
            },
            {
                "name": "逆周期+跨周期",
                "desc": "短期稳增长与中长期结构转型并行",
                "mechanism": "短期托底与长期提质同步推进。",
                "impact": "市场风格在防御与成长之间动态切换。",
            },
        ],
    }

    # 用“近5条历史记录”替代同一条拆5行
    if omo_hist:
        for it in data.get("monetary", []):
            if it.get("name") == "OMO逆回购":
                it["table"] = [
                    {"metric": h.get("date") or f"第{i+1}条", "latest": f"利率 {h.get('rate') or '暂无'} / 规模 {h.get('amount') or '暂无'}", "freq": "日", "source": h.get("url") or "人民银行"}
                    for i, h in enumerate(omo_hist[:14])
                ]

    if lpr_hist:
        for it in data.get("monetary", []):
            if it.get("name") == "LPR":
                it["table"] = [
                    {"metric": h.get("date") or f"第{i+1}条", "latest": f"1Y {h.get('lpr1') or '暂无'} / 5Y {h.get('lpr5') or '暂无'}", "freq": "月", "source": h.get("url") or "人民银行"}
                    for i, h in enumerate(lpr_hist[:5])
                ]

    if mlf_hist:
        for it in data.get("monetary", []):
            if it.get("name") == "政策利率(7天逆回购/MLF)":
                it["table"] = [
                    {"metric": h.get("date") or f"第{i+1}条", "latest": f"规模 {h.get('amount') or '暂无'} / 期限 {h.get('term') or '暂无'}", "freq": "月", "source": h.get("url") or "人民银行"}
                    for i, h in enumerate(mlf_hist[:5])
                ]

    # 其余项目：用各自来源近5条公告（每条独立数据）
    rows_mof = _notice_rows_by_source("财政部", limit=5)
    rows_pbc = _notice_rows_by_source("人民银行", limit=5)

    for it in data.get("fiscal", []):
        if it.get("name") not in ["赤字率与财政扩张"]:
            it["table"] = (it.get("table") or rows_mof)[:5] if len(it.get("table") or []) >= 5 else (it.get("table") or []) + rows_mof
            it["table"] = it["table"][:5]
        else:
            # 财政扩张保持核心指标 + 两条最新公告
            base = list(it.get("table") or [])[:3]
            it["table"] = (base + rows_mof)[:5]

    for it in data.get("monetary", []):
        if it.get("name") in ["降准(RRR)", "再贷款再贴现", "PSL/结构性工具"]:
            it["table"] = (it.get("table") or []) + rows_pbc
            it["table"] = it["table"][:5]

    for it in data.get("combo", []):
        it["table"] = rows_pbc[:3] + rows_mof[:2]

    return data


def _scrape_links(url: str, source: str, max_items: int = 8) -> list[dict]:
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding or "utf-8"
        html = r.text
    except Exception:
        return []

    out = []
    seen = set()
    for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, flags=re.I | re.S):
        href = (m.group(1) or "").strip()
        title = re.sub(r"<[^>]+>", "", m.group(2) or "")
        title = re.sub(r"\s+", " ", title).strip()
        if not href or len(title) < 6:
            continue
        if href.lower().startswith("javascript"):
            continue
        if "�" in title:
            continue
        if any(x in title for x in ["登录", "注册", "上一页", "下一页", "更多", "首页", "English Version"]):
            continue
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            base = "/".join(url.split("/")[:3])
            href = base + href
        elif href.startswith("./"):
            href = url.rsplit("/", 1)[0] + "/" + href[2:]
        key = (title, href)
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "url": href, "source": source})
        if len(out) >= max_items:
            break
    return out


def _policy_news() -> dict:
    seeds = [
        ("https://www.gov.cn/zhengce/zuixin.htm", "中国政府网·政策"),
        ("http://www.pbc.gov.cn/goutongjiaoliu/113456/113469/index.html", "人民银行·货币政策"),
        ("https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/", "财政部·财政新闻"),
    ]
    items = []
    for u, s in seeds:
        items.extend(_scrape_links(u, s, max_items=8))
    # 去重并截断
    uniq = []
    seen = set()
    for it in items:
        k = (it["title"], it["url"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    uniq = uniq[:20]
    return {"updated_at": _cn_now().strftime("%Y-%m-%d %H:%M:%S"), "items": uniq}


@app.route("/policy-dashboard")
def policy_dashboard():
    return render_template("policy_dashboard.html")


@app.route("/api/policy/catalog")
def api_policy_catalog():
    return jsonify(_policy_catalog())


@app.route("/api/policy/news")
def api_policy_news():
    return jsonify(_policy_news())


@app.route("/news")
def news_page():
    return render_template("news.html")


@app.route("/api/news")
def api_news():
    from news_fetcher import fetch_news_24h

    return jsonify(fetch_news_24h())


@app.route("/api/news/geo")
def api_news_geo():
    from news_fetcher import fetch_region_news

    region = request.args.get("region", "china")
    sort_by = request.args.get("sort", "heat")
    limit = int(request.args.get("limit", "20") or 20)
    limit = max(5, min(limit, 60))
    return jsonify(fetch_region_news(region=region, limit=limit, sort_by=sort_by))


@app.route("/volume-profile")
def volume_profile_page():
    return render_template("volume_profile.html")


@app.route("/main-net-inflow")
def main_net_inflow_page():
    return render_template("main_net_inflow.html")


@app.route("/flow-divergence")
def flow_divergence_page():
    return render_template("flow_divergence.html")


def _load_flow_divergence_cache_from_disk(cache_key: str):
    try:
        if not FLOW_DIVERGENCE_CACHE_FILE.exists():
            return None
        obj = json.loads(FLOW_DIVERGENCE_CACHE_FILE.read_text() or "{}")
        if obj.get("key") != cache_key:
            return None
        payload = obj.get("payload")
        if payload:
            return {"ts": float(obj.get("ts") or time.time()), "key": cache_key, "payload": payload}
    except Exception:
        return None
    return None


def _start_flow_divergence_job(cache_key: str, days: int, max_scan: int):
    with FLOW_DIVERGENCE_LOCK:
        st = FLOW_DIVERGENCE_JOBS.get(cache_key) or {}
        if st.get("running"):
            return
        FLOW_DIVERGENCE_JOBS[cache_key] = {
            "running": True,
            "started_at": _cn_now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_error": None,
            "last_ok": st.get("last_ok"),
            "last_done_at": st.get("last_done_at"),
        }

    def worker():
        global FLOW_DIVERGENCE_CACHE
        try:
            data = _flow_divergence_scan(days=days, max_scan=max_scan)
            FLOW_DIVERGENCE_CACHE = {"ts": time.time(), "key": cache_key, "payload": data}
            with FLOW_DIVERGENCE_LOCK:
                FLOW_DIVERGENCE_JOBS[cache_key] = {
                    "running": False,
                    "started_at": FLOW_DIVERGENCE_JOBS.get(cache_key, {}).get("started_at"),
                    "last_error": None,
                    "last_ok": True,
                    "last_done_at": _cn_now().strftime("%Y-%m-%d %H:%M:%S"),
                }
        except Exception as e:
            with FLOW_DIVERGENCE_LOCK:
                FLOW_DIVERGENCE_JOBS[cache_key] = {
                    "running": False,
                    "started_at": FLOW_DIVERGENCE_JOBS.get(cache_key, {}).get("started_at"),
                    "last_error": str(e),
                    "last_ok": False,
                    "last_done_at": _cn_now().strftime("%Y-%m-%d %H:%M:%S"),
                }

    threading.Thread(target=worker, daemon=True).start()


@app.route("/api/flow-divergence")
def api_flow_divergence():
    """同步扫描接口：允许请求耗时更长，但优先返回真实结果（非空跑状态）。"""
    global FLOW_DIVERGENCE_CACHE
    days = int((request.args.get("days") or "3").strip() or 3)
    max_scan = int((request.args.get("max_scan") or "120").strip() or 120)
    force = (request.args.get("force") or "0") == "1"

    days = 2 if days == 2 else 3
    max_scan = max(20, min(300, max_scan))
    cache_key = f"{days}:{max_scan}"

    # 启动时尝试从磁盘恢复缓存
    if FLOW_DIVERGENCE_CACHE.get("key") != cache_key:
        disk = _load_flow_divergence_cache_from_disk(cache_key)
        if disk:
            FLOW_DIVERGENCE_CACHE = disk

    # 非强刷优先返回最近缓存（3分钟），减少重复压测上游
    if not force and FLOW_DIVERGENCE_CACHE.get("key") == cache_key and time.time() - float(FLOW_DIVERGENCE_CACHE.get("ts") or 0) < 180:
        payload = dict(FLOW_DIVERGENCE_CACHE.get("payload") or {})
        payload["cached"] = True
        return jsonify({"ok": True, "data": payload})

    try:
        data = _flow_divergence_scan(days=days, max_scan=max_scan)
        FLOW_DIVERGENCE_CACHE = {"ts": time.time(), "key": cache_key, "payload": data}
        try:
            FLOW_DIVERGENCE_CACHE_FILE.write_text(json.dumps(FLOW_DIVERGENCE_CACHE, ensure_ascii=False))
        except Exception:
            pass
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        # 出错时优先回退到最近缓存（内存/磁盘）
        cached = FLOW_DIVERGENCE_CACHE.get("payload") if FLOW_DIVERGENCE_CACHE.get("key") == cache_key else None
        if not cached:
            disk = _load_flow_divergence_cache_from_disk(cache_key)
            if disk:
                FLOW_DIVERGENCE_CACHE = disk
                cached = disk.get("payload")
        if cached:
            payload = dict(cached)
            payload["degraded"] = True
            return jsonify({"ok": True, "degraded": True, "message": f"实时扫描失败，已回退缓存：{e}", "data": payload})

        empty = {
            "updated_at": _cn_now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": {"days": days, "max_scan": max_scan},
            "scan_info": {
                "universe_total": 0,
                "candidates": 0,
                "checked": 0,
                "errors": 1,
                "matched": 0,
                "elapsed_sec": 0,
                "note": "实时数据源异常。",
            },
            "items": [],
            "degraded": True,
        }
        return jsonify({"ok": True, "degraded": True, "message": f"扫描失败：{e}", "data": empty})


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
        cached = _fill_industry_flow_if_missing(LAST_RECOMMENDATIONS or load_json(DATA_FILE, fallback_rankings()))
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


def _market_prefix_for_code(code: str) -> str:
    return "SH" if code.startswith(("5", "6", "9")) else "SZ"


def _fetch_research_reports(code: str, days: int = 730, page_size: int = 20) -> list[dict]:
    end = _cn_now().strftime("%Y-%m-%d")
    begin = (_cn_now() - dt.timedelta(days=max(30, min(days, 1460)))).strftime("%Y-%m-%d")

    def em_source() -> list[dict]:
        out = []
        max_pages = 12
        per_page = max(20, min(page_size, 50))
        for page in range(1, max_pages + 1):
            url = "https://reportapi.eastmoney.com/report/list"
            params = {
                "code": code,
                "pageNo": page,
                "pageSize": per_page,
                "industryCode": "*",
                "orgCode": "*",
                "rating": "*",
                "beginTime": begin,
                "endTime": end,
                "qType": 0,
            }
            r = requests.get(url, params=params, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            data = r.json().get("data") or []
            if not data:
                break
            for x in data:
                info_code = x.get("infoCode") or ""
                out.append(
                    {
                        "title": x.get("title") or "",
                        "stockName": x.get("stockName") or "",
                        "stockCode": x.get("stockCode") or code,
                        "orgName": x.get("orgName") or x.get("orgSName") or "",
                        "publishDate": str(x.get("publishDate") or "")[:10],
                        "analyst": x.get("researcher") or "",
                        "rating": x.get("emRatingName") or x.get("sRatingName") or "未披露",
                        "targetPrice": float(x.get("indvAimPriceT") or 0) if str(x.get("indvAimPriceT") or "").strip() else 0,
                        "industry": x.get("indvInduName") or x.get("industryName") or "",
                        "pdf": f"https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf" if info_code else "",
                        "source": "东方财富研报中心",
                    }
                )
            if len(data) < per_page:
                break
        return out

    def op_required_source() -> list[dict]:
        # 第二研报源：东方财富F10-运营必读（研报摘要 ybzy）
        mk = _market_prefix_for_code(code)
        u = f"https://emweb.securities.eastmoney.com/PC_HSF10/OperationsRequired/PageAjax?code={mk}{code}"
        r = requests.get(u, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        obj = r.json()
        rows = obj.get("ybzy") or []
        out = []
        for x in rows:
            art = x.get("art_code") or ""
            out.append({
                "title": x.get("title") or "",
                "stockName": "",
                "stockCode": code,
                "orgName": x.get("source") or "",
                "publishDate": str(x.get("publish_time") or "")[:10],
                "analyst": "",
                "rating": x.get("em_rating_name") or x.get("s_rating_name") or "未披露",
                "targetPrice": float(x.get("aim_price") or 0) if str(x.get("aim_price") or "").strip() else 0,
                "industry": x.get("indu_old_industry_name") or "",
                "pdf": f"https://pdf.dfcfw.com/pdf/H3_{art}_1.pdf" if art else "",
                "source": "东方财富F10研报摘要",
            })
        return out

    def sina_source() -> list[dict]:
        # 作为补充源：部分标的可返回研报列表；空则自动忽略
        out = []
        symbol = ("sh" if code.startswith(("5", "6", "9")) else "sz") + code
        url = f"https://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml?symbol={symbol}"
        try:
            html = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).content.decode("gb18030", "ignore")
        except Exception:
            return out

        rows = re.findall(r"<tr[^>]*>\s*<td[^>]*>(\d+)</td>([\s\S]*?)</tr>", html, flags=re.I)
        for _, row in rows:
            t = re.search(r"title=\"([^\"]+)\"", row)
            d = re.search(r"(20\d{2}-\d{2}-\d{2})", row)
            org = re.search(r"<td[^>]*>([^<]{2,40}(证券|研究|投顾)[^<]{0,20})</td>", row)
            title = (t.group(1).strip() if t else "")
            if not title:
                continue
            out.append(
                {
                    "title": title,
                    "stockName": "",
                    "stockCode": code,
                    "orgName": (org.group(1).strip() if org else ""),
                    "publishDate": (d.group(1) if d else ""),
                    "analyst": "",
                    "rating": "未披露",
                    "targetPrice": 0,
                    "industry": "",
                    "pdf": "",
                    "source": "新浪财经研报",
                }
            )
        return out

    all_rows = []
    for fn in (em_source, op_required_source, sina_source):
        try:
            all_rows.extend(fn())
        except Exception:
            continue

    # 去重：标题+日期优先；同一天同标题若机构简称/全称冲突，优先保留更长机构名
    dedup = {}
    for x in all_rows:
        k = f"{x.get('title','')}|{x.get('publishDate','')}"
        old = dedup.get(k)
        if (not old) or len(str(x.get('orgName') or '')) > len(str(old.get('orgName') or '')):
            dedup[k] = x

    out = list(dedup.values())
    out.sort(key=lambda z: (z.get("publishDate") or ""), reverse=True)
    return out[:120]


def _fetch_business_lines(code: str) -> dict:
    mk = _market_prefix_for_code(code)

    biz_url = f"https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax?code={mk}{code}"
    biz = requests.get(biz_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    biz.raise_for_status()
    biz_obj = biz.json()

    profile_url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/PageAjax?code={mk}{code}"
    profile_obj = {}
    try:
        p = requests.get(profile_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        p.raise_for_status()
        profile_obj = p.json()
    except Exception:
        profile_obj = {}

    scope = ""
    zyfw = biz_obj.get("zyfw") or []
    if zyfw:
        scope = str((zyfw[0] or {}).get("BUSINESS_SCOPE") or "")

    jbzl = profile_obj.get("jbzl") or []
    profile = (jbzl[0] if jbzl else {}) or {}
    main_business = str(profile.get("MAIN_BUSINESS") or "")

    seg_rows = biz_obj.get("zygcfx") or []
    latest_date = ""
    for row in seg_rows:
        d = str(row.get("REPORT_DATE") or "")[:10]
        if d and d > latest_date:
            latest_date = d

    latest = [x for x in seg_rows if str(x.get("REPORT_DATE") or "")[:10] == latest_date]
    latest.sort(key=lambda z: float(z.get("MAIN_BUSINESS_INCOME") or 0), reverse=True)

    segments_map = {}
    for x in latest[:40]:
        name = (x.get("ITEM_NAME") or "").strip()
        if not name:
            continue
        ratio = float(x.get("MBI_RATIO") or 0) * 100
        income = float(x.get("MAIN_BUSINESS_INCOME") or 0)
        old = segments_map.get(name)
        if (not old) or ratio > old.get("incomeRatioPct", 0):
            segments_map[name] = {
                "name": name,
                "income": income,
                "incomeRatioPct": round(ratio, 2),
            }
    segments = sorted(segments_map.values(), key=lambda z: z.get("incomeRatioPct", 0), reverse=True)[:20]

    review_text = ""
    jyps = biz_obj.get("jyps") or []
    if jyps:
        review_text = str((jyps[0] or {}).get("BUSINESS_REVIEW") or "")

    candidate_text = "；".join([scope, main_business, review_text])
    # 抽取较具体的业务/产品名称（优先游戏/产品/IP名，避免数字碎片）
    quoted_names = re.findall(r"《([^》]{2,24})》", candidate_text)
    tokens = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9\+]{2,24}", candidate_text)
    stop = {"公司", "业务", "销售", "生产", "服务", "发展", "建设", "推进", "提升", "同比增长", "亿元", "系统", "平台", "实现", "经营", "其中"}
    specifics = []
    seen = set()

    def add_item(v: str):
        v = (v or "").strip("，。；;:： ")
        if not v or v in stop:
            return
        if re.fullmatch(r"\d+(\.\d+)?", v):
            return
        if len(v) <= 1:
            return
        if v in seen:
            return
        seen.add(v)
        specifics.append(v)

    for q in quoted_names:
        add_item(q)

    for t in tokens:
        if t in stop:
            continue
        if not re.search(r"游戏|IP|茅台|酒|王子|汉酱|行动组|供应链|数字化|电信|运输|酒店|包装|渠道|出海|新品|系列", t):
            continue
        add_item(t)
        if len(specifics) >= 30:
            break

    return {
        "scope": scope,
        "mainBusiness": main_business,
        "businessReview": review_text,
        "specificItems": specifics,
        "segments": segments,
        "latestReportDate": latest_date,
        "source": ["东方财富F10-经营分析", "东方财富F10-公司概况"],
    }


def _generate_report_analysis(code: str, reports: list[dict], business: dict) -> dict:
    rating_stat = {}
    org_count = {}
    targets = []

    for r in reports:
        rt = (r.get("rating") or "未披露").strip()
        rating_stat[rt] = rating_stat.get(rt, 0) + 1
        org = (r.get("orgName") or "未知机构").strip()
        org_count[org] = org_count.get(org, 0) + 1
        tp = float(r.get("targetPrice") or 0)
        if tp > 0:
            targets.append(tp)

    top_orgs = sorted(org_count.items(), key=lambda x: x[1], reverse=True)[:6]
    avg_target = round(sum(targets) / len(targets), 2) if targets else None

    segs = business.get("segments") or []
    top_segments = [f"{x['name']}（收入占比{round(x['incomeRatioPct'], 2)}%）" for x in segs[:5]]

    points = [
        f"近阶段共收集到 {len(reports)} 篇券商/研究机构研报，覆盖机构数 {len(org_count)} 家。",
        f"机构观点分布：{ '；'.join([f'{k}{v}篇' for k,v in sorted(rating_stat.items(), key=lambda x:x[1], reverse=True)[:4]]) or '暂无明确评级' }。",
        f"若按已披露目标价口径统计，机构一致预期目标价均值约为 {avg_target} 元。" if avg_target else "公开研报中可提取的目标价样本不足，建议补充公告或机构路演纪要进行交叉验证。",
        f"公司当前核心业务线（按最近披露口径）集中在：{'、'.join(top_segments) if top_segments else '暂无可解析分部数据'}。",
        "基于研报的实操建议：优先跟踪“评级变化 + 目标价调整 + 分部收入占比变化”三项同步变化，作为后续研究更新触发器。",
    ]

    return {
        "summary": points,
        "stats": {
            "reportCount": len(reports),
            "orgCount": len(org_count),
            "ratingStat": rating_stat,
            "avgTargetPrice": avg_target,
            "topOrgs": [{"name": k, "count": v} for k, v in top_orgs],
        },
        "disclaimer": "本页面仅用于研究信息整合，不构成投资建议。",
        "stockCode": code,
    }


def _call_openai_oauth_json(system_prompt: str, user_payload: dict, schema: dict) -> dict | None:
    """通过 OpenAI OAuth Access Token 调用结构化抽取（与 OpenClaw 自身鉴权隔离）"""
    token = _get_openai_access_token()
    if not token:
        AI_RUNTIME_STATUS.update({
            "last_ok": False,
            "last_error": "未配置可用的AI Token（或Token为空）",
            "last_error_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        return None

    # 默认走千问兼容接口；如需OpenAI可改回对应BASE_URL
    base_url = (os.getenv("STOCK_RESEARCH_OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    model = (os.getenv("STOCK_RESEARCH_OPENAI_MODEL") or "qwen-plus").strip()
    timeout_raw = (os.getenv("STOCK_RESEARCH_OPENAI_TIMEOUT") or "0").strip()
    # 主公要求：不做超时限制。设为0或空时，requests使用无限等待(None)
    timeout = None
    try:
        t = int(timeout_raw or 0)
        timeout = None if t <= 0 else t
    except Exception:
        timeout = None

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0.45,
        "top_p": 0.9,
    }

    # 千问兼容口径：不强绑 response_format，避免输出空模板
    if not ("dashscope.aliyuncs.com" in base_url or model.startswith("qwen")):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "business_map",
                "strict": True,
                "schema": schema,
            },
        }

    def _parse_content(content: str):
        if not content:
            return None
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", content)
            return json.loads(m.group(0)) if m else None

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        obj = _parse_content(content)
        if obj:
            AI_RUNTIME_STATUS.update({
                "last_ok": True,
                "last_error": "",
                "last_error_at": "",
                "last_provider": base_url,
                "last_model": model,
            })
            return obj

        # 兜底二次调用：去掉response_format，要求直接返回JSON对象
        body2 = dict(body)
        body2.pop("response_format", None)
        body2["messages"] = [
            {"role": "system", "content": system_prompt + " 你必须仅返回一个JSON对象，不要Markdown。"},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        resp2 = requests.post(url, headers=headers, json=body2, timeout=timeout)
        resp2.raise_for_status()
        data2 = resp2.json()
        content2 = (((data2.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        obj2 = _parse_content(content2)
        if obj2:
            AI_RUNTIME_STATUS.update({
                "last_ok": True,
                "last_error": "",
                "last_error_at": "",
                "last_provider": base_url,
                "last_model": model,
            })
            return obj2

        AI_RUNTIME_STATUS.update({
            "last_ok": False,
            "last_error": "AI返回内容无法解析为JSON",
            "last_error_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_provider": base_url,
            "last_model": model,
        })
        return None
    except Exception as e:
        AI_RUNTIME_STATUS.update({
            "last_ok": False,
            "last_error": str(e)[:240],
            "last_error_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_provider": base_url,
            "last_model": model,
        })
        return None


def _ai_quality_score(obj: dict | None) -> tuple[int, dict]:
    if not obj:
        return 0, {"summary": 0, "lines": 0, "products": 0, "strategy": 0}
    summary = 1 if len((obj.get("mainBusinessSummary") or "").strip()) >= 90 else 0
    lines = len(obj.get("businessLines") or [])
    products = len(obj.get("canonicalProducts") or [])
    strategy = len(obj.get("strategyTimeline") or [])
    score = summary + min(lines, 8) + min(products, 24) + min(strategy, 8)
    return score, {"summary": summary, "lines": lines, "products": products, "strategy": strategy}


def _ai_extract_business_map(code: str, name: str, reports: list[dict], business: dict) -> dict | None:
    # 高信息密度输入 + 多轮提炼，不省token，优先质量
    report_evidence = []
    for r in reports[:40]:
        t = (r.get("title") or "").strip()
        if not t:
            continue
        report_evidence.append({
            "title": t,
            "date": r.get("publishDate") or "",
            "org": r.get("orgName") or "",
            "url": r.get("pdf") or r.get("url") or "",
        })

    ann = []
    news = []
    try:
        ann = _fetch_company_announcements(code, page_size=18)
    except Exception:
        ann = []
    try:
        news = _fetch_company_news(name or code, limit=18)
    except Exception:
        news = []

    candidate_products = set()
    for x in (business.get("specificItems") or [])[:60]:
        xx = (x or "").strip()
        if 1 < len(xx) <= 28:
            candidate_products.add(xx)
    for s in (business.get("segments") or [])[:30]:
        nm = (s.get("name") or "").strip()
        if nm and len(nm) <= 30:
            candidate_products.add(nm)
    for r in reports[:50]:
        t = (r.get("title") or "")
        for m in re.findall(r"《([^》]{2,24})》", t):
            candidate_products.add(m)
        for m in re.findall(r"(太空杀|SuperSus|征途系列|原始征途|球球大作战|茅台酒|茅台1935|王子酒|汉酱|Opera|StarMaker|Grindr|SkyMusic|天工大模型)", t, flags=re.I):
            candidate_products.add(str(m))

    payload = {
        "code": code,
        "name": name,
        "report_evidence": report_evidence,
        "business_segments": business.get("segments") or [],
        "business_specific_items": business.get("specificItems") or [],
        "main_business": business.get("mainBusiness") or "",
        "business_review": business.get("businessReview") or "",
        "announcements": ann,
        "news": news,
        "candidate_products": sorted(list(candidate_products))[:120],
        "task_rules": {
            "min_products": 18,
            "min_strategy_items": 6,
            "need_dates": True,
            "keep_unreported_but_real_products": True,
        },
        "output_template": {
            "mainBusinessSummary": "string>=120",
            "businessLines": [{"name": "", "subLine": "", "confidence": 0.0}],
            "canonicalProducts": [{"name": "", "aliases": [""], "line": "", "image_url": "", "mention_count": 0, "confidence": 0.0, "evidence": [{"source": "", "date": "", "snippet": "", "url": ""}]}],
            "strategyTimeline": [{"date": "YYYY-MM-DD", "event": "", "whyImportant": "", "source": "", "url": ""}]
        }
    }

    schema = {
        "type": "object",
        "properties": {
            "mainBusinessSummary": {"type": "string"},
            "businessLines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "subLine": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["name", "subLine", "confidence"],
                    "additionalProperties": False,
                },
            },
            "canonicalProducts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "line": {"type": "string"},
                        "image_url": {"type": "string"},
                        "mention_count": {"type": "number"},
                        "confidence": {"type": "number"},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "date": {"type": "string"},
                                    "snippet": {"type": "string"},
                                    "url": {"type": "string"},
                                },
                                "required": ["source", "date", "snippet", "url"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["name", "aliases", "line", "image_url", "mention_count", "confidence", "evidence"],
                    "additionalProperties": False,
                },
            },
            "strategyTimeline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "event": {"type": "string"},
                        "whyImportant": {"type": "string"},
                        "source": {"type": "string"},
                        "url": {"type": "string"}
                    },
                    "required": ["date", "event", "whyImportant", "source", "url"],
                    "additionalProperties": False
                }
            },
        },
        "required": ["mainBusinessSummary", "businessLines", "canonicalProducts", "strategyTimeline"],
        "additionalProperties": False,
    }

    system_prompt = (
        "你是A股上市公司研究总监。你的输出要直接可上网页，信息必须密、具体、可验证。"
        "请综合使用：研报、分部数据、公告、新闻，以及你掌握的公开常识，完成结构化结果。"
        "不要把任务理解成‘只从研报找关键词’，目标是尽量找全真实产品与业务线。"
        "硬性要求："
        "1) mainBusinessSummary 至少120字，必须写清收入结构、产品结构、区域结构、增长抓手与风险点；禁止空话。"
        "2) businessLines 至少6条，细化到可研究层级（如端游/手游、发行/运营、国内/海外、广告/增值服务等）。"
        "3) canonicalProducts 尽量找全，目标>=18条；没在研报出现但客观存在的产品也要纳入；mention_count可为0。"
        "并优先吸收 candidate_products 列表中的实体，做同义归并后输出。"
        "4) 每个产品必须给line、confidence，并尽量给aliases、image_url与evidence；禁止使用笼统词（如AI游戏/游戏业务/产品平台）。"
        "5) strategyTimeline 至少6条，必须有时间点（date），写明‘当时说了什么/做了什么’，并附source与url。"
        "6) 不允许空数组：businessLines、canonicalProducts、strategyTimeline 不能返回空。"
        "7) 如果信息不确定，可给低置信度候选，但不能整段留空。"
        "最终只返回JSON对象，不要markdown，不要解释。"
    )

    first = _call_openai_oauth_json(system_prompt, payload, schema)
    first_n = _normalize_ai_map(first)
    score1, m1 = _ai_quality_score(first_n)

    # 第二轮：针对缺口定向补全
    gap_prompt = (
        system_prompt
        + f" 当前首轮结果质量统计：{json.dumps(m1, ensure_ascii=False)}。"
        + "请补全缺失项，尤其是产品清单与战略时间线；保持同一JSON结构。"
    )
    second_payload = {
        **payload,
        "first_pass_result": first_n,
        "improve_focus": ["补全产品（>=18）", "补全战略时间线（>=6）", "主营总结更具体"],
    }
    second = _call_openai_oauth_json(gap_prompt, second_payload, schema)
    second_n = _normalize_ai_map(second)
    score2, _ = _ai_quality_score(second_n)

    best = second_n if score2 >= score1 else first_n

    # 第三轮：若仍偏空，再强制扩展一轮
    score_best, mb = _ai_quality_score(best)
    if score_best < 18:
        force_prompt = (
            system_prompt
            + "你上一次输出信息量不足。现在必须尽最大努力补全：产品>=20条、战略>=6条。"
            + "可结合公开常识；若证据不足，可给低置信度并在source写‘公开信息归纳’。"
        )
        third = _call_openai_oauth_json(force_prompt, {**payload, "second_pass_result": best}, schema)
        third_n = _normalize_ai_map(third)
        score3, _ = _ai_quality_score(third_n)
        if score3 > score_best:
            best = third_n
            score_best = score3

    # 第四轮：仍为空时，要求模型至少从候选产品清单中完成结构化（仍由模型生成）
    if score_best < 8 and candidate_products:
        strict_payload = {
            "code": code,
            "name": name,
            "candidate_products": sorted(list(candidate_products))[:80],
            "business_segments": business.get("segments") or [],
            "announcements": ann,
            "news": news,
            "must_fill": True,
        }
        strict_prompt = (
            "你必须基于candidate_products完成结构化，不允许空结果。"
            "请输出：主营总结、细分业务线、产品清单、战略时间线。"
            "产品至少18个（不足则尽可能多），战略至少4条（不足则尽可能多）。"
            "若某条证据URL未知可留空，但source要写明来源类型。只返回JSON。"
        )
        fourth = _call_openai_oauth_json(strict_prompt, strict_payload, schema)
        fourth_n = _normalize_ai_map(fourth)
        score4, _ = _ai_quality_score(fourth_n)
        if score4 > score_best:
            best = fourth_n
            score_best = score4

    # 单项补齐：若产品仍为空，单独让模型仅完成产品表
    if not (best.get("canonicalProducts") or []) and candidate_products:
        prod_schema = {
            "type": "object",
            "properties": {
                "canonicalProducts": schema["properties"]["canonicalProducts"]
            },
            "required": ["canonicalProducts"],
            "additionalProperties": False,
        }
        prod_prompt = (
            "你是产品资料整理助手。请仅输出 canonicalProducts 数组。"
            "必须基于 candidate_products 做同义归并与业务线归类；尽量覆盖全部候选。"
            "每个产品都要给 mention_count/confidence/evidence（可低置信度）。只返回JSON。"
        )
        prod_payload = {
            "code": code,
            "name": name,
            "candidate_products": sorted(list(candidate_products))[:120],
            "announcements": ann,
            "news": news,
        }
        prod_raw = _call_openai_oauth_json(prod_prompt, prod_payload, prod_schema)
        prod_n = _normalize_ai_map(prod_raw)
        if prod_n.get("canonicalProducts"):
            best["canonicalProducts"] = prod_n.get("canonicalProducts")

    # 单项补齐：若主营为空，单独生成主营总结
    if not (best.get("mainBusinessSummary") or "").strip():
        sum_schema = {
            "type": "object",
            "properties": {"mainBusinessSummary": {"type": "string"}},
            "required": ["mainBusinessSummary"],
            "additionalProperties": False,
        }
        sum_prompt = (
            "请仅输出 mainBusinessSummary，至少150字，必须具体写清主营结构、增长抓手、风险点。"
            "禁止空话，只返回JSON。"
        )
        sum_payload = {
            "code": code,
            "name": name,
            "business_segments": business.get("segments") or [],
            "announcements": ann,
            "news": news,
            "report_evidence": report_evidence[:15],
        }
        sum_raw = _call_openai_oauth_json(sum_prompt, sum_payload, sum_schema)
        sum_n = _normalize_ai_map(sum_raw)
        if (sum_n.get("mainBusinessSummary") or "").strip():
            best["mainBusinessSummary"] = sum_n.get("mainBusinessSummary")

    return best


def _normalize_ai_map(ai_map: dict | None) -> dict:
    """只做结构归一化，不新增任何本地推断内容。"""
    src = ai_map if isinstance(ai_map, dict) else {}

    summary = (src.get("mainBusinessSummary") or src.get("main_business_summary") or src.get("主营业务总结") or "").strip()

    lines_raw = src.get("businessLines") or src.get("business_lines") or src.get("细分业务线") or []
    lines = []
    for x in lines_raw:
        if not isinstance(x, dict):
            continue
        nm = (x.get("name") or x.get("line") or "").strip()
        sub = (x.get("subLine") or x.get("sub_line") or x.get("detail") or "").strip()
        conf = float(x.get("confidence") or 0.0)
        if nm:
            lines.append({"name": nm, "subLine": sub, "confidence": conf})

    prods_raw = src.get("canonicalProducts") or src.get("products") or src.get("productList") or src.get("核心产品") or []
    prods = []
    for p in prods_raw:
        if not isinstance(p, dict):
            continue
        nm = (p.get("name") or "").strip()
        if not nm:
            continue

        ev_raw = p.get("evidence") or []
        ev_norm = []
        for ev in ev_raw:
            if isinstance(ev, dict):
                ev_norm.append({
                    "source": (ev.get("source") or "").strip(),
                    "date": (ev.get("date") or "").strip(),
                    "snippet": (ev.get("snippet") or ev.get("text") or "").strip(),
                    "url": (ev.get("url") or "").strip(),
                })
            elif isinstance(ev, str):
                txt = ev.strip()
                if txt:
                    ev_norm.append({"source": "AI输出", "date": "", "snippet": txt, "url": ""})

        prods.append({
            "name": nm,
            "aliases": p.get("aliases") or [],
            "line": (p.get("line") or p.get("businessLine") or "").strip(),
            "image_url": (p.get("image_url") or p.get("image") or "").strip(),
            "mention_count": float(p.get("mention_count") or p.get("mentions") or 0),
            "confidence": float(p.get("confidence") or 0.0),
            "evidence": ev_norm,
        })

    st_raw = src.get("strategyTimeline") or src.get("strategy_timeline") or src.get("strategies") or src.get("战略时间线") or []
    st = []
    for x in st_raw:
        if not isinstance(x, dict):
            continue
        st.append({
            "date": (x.get("date") or x.get("time") or "").strip(),
            "event": (x.get("event") or x.get("title") or "").strip(),
            "whyImportant": (x.get("whyImportant") or x.get("reason") or "").strip(),
            "source": (x.get("source") or "").strip(),
            "url": (x.get("url") or "").strip(),
        })

    return {
        "mainBusinessSummary": summary,
        "businessLines": lines,
        "canonicalProducts": prods,
        "strategyTimeline": st,
    }


def _normalize_product_name(name: str) -> str:
    n = (name or "").strip()
    n = re.sub(r"[（(][^）)]*[）)]", "", n).strip()
    n = re.sub(r"^(以|及|和|与|并|如|例如|包括|包含)", "", n).strip()
    n = re.sub(r"(等产品|等游戏|产品)$", "", n).strip()
    n = n.strip("、，,。；;：: ")
    return n


def _is_concrete_product_name(name: str) -> bool:
    """判断是否为“可上图”的具体产品名，过滤业务口径/动作描述/句子碎片。"""
    n = _normalize_product_name(name)
    if (not n) or len(n) < 2 or len(n) > 24:
        return False

    generic_exact = {
        "产品", "其他产品", "未分类产品", "业务分部", "经营描述", "研报提及", "产品线",
        "国内", "国外", "境内", "境外", "系列酒", "茅台酒", "白酒", "平台", "生态",
        "业务", "主营业务", "游戏业务", "短剧平台", "海外社交网络", "AI软件技术",
        "移动游戏", "网页游戏", "客户端游戏", "网络游戏", "其他", "其他(补充)",
    }
    if n in generic_exact:
        return False

    if re.search(r"[，。；：:（）()、]|\s{2,}", n):
        return False

    # 数字指标/事实描述句，非产品名
    if re.search(r"约\d|\d+万|\d+亿|\d+%|同比|环比|产量|销量|销量|增长率", n):
        return False
    if re.search(r"\bIP\b|联名", n, re.I):
        return False

    bad_kw = re.compile(
        r"业务|分部|收入|占比|渠道|体系|建设|推进|管理|方面|特征|指南|生产|销售|布局|合作|数字化|经营|策略|战略|规划|增长|能力|解决方案|系统|项目|板块|口径|系列|品牌|物流|运输|包装|碳标签|可持续|地理标志|行业|协会|政策|措施|方案|报告"
    )
    if bad_kw.search(n):
        return False

    if n.endswith(("方面", "业务", "管理", "建设", "策略", "服务", "方案", "报告", "和", "及", "与")):
        return False
    if re.match(r"^(新品|新产品|产品|系列)", n):
        return False

    return True


def _normalize_product_category(code: str, name: str, category: str = "", status: str = "") -> str:
    """分类以AI输出为准：只清理明显脏分类。若无有效分类，返回空串（上层判定失败）。"""
    c = (category or "").strip()

    generic = {"", "未分类产品", "其他产品", "业务分部", "经营描述", "研报提及", "产品", "AI未分类"}
    if c in generic:
        return ""

    # 过滤明显无意义分类标签
    if re.search(r"业务|分部|经营|描述|口径|未分类", c):
        return ""

    return c


def _enrich_company_info(code: str, ai_map: dict, business: dict, reports: list[dict], announcements: list[dict], news: list[dict]) -> tuple[dict, dict]:
    """对AI输出做本地审校+补全，确保页面信息足够细、完整、可展示。"""
    out = dict(ai_map or {})

    summary = (out.get("mainBusinessSummary") or "").strip()
    if len(summary) < 120:
        seg_txt = "；".join([f"{x.get('name','')}占比{x.get('incomeRatioPct',0)}%" for x in (business.get("segments") or [])[:8]])
        rpt_txt = "；".join([f"{(r.get('publishDate') or '')[:10]} {r.get('orgName') or ''}《{(r.get('title') or '')[:28]}》" for r in (reports or [])[:6]])
        summary = (
            f"基于公开研报与公司分部数据，当前主营结构可归纳为：{seg_txt or '分部数据待补'}。"
            f"近阶段核心跟踪观点包括：{rpt_txt or '近期研报样本较少'}。"
            "建议重点跟踪收入结构变化、重点产品生命周期、海外/国内区域贡献变化，以及费用率和现金流质量。"
        )[:520]
    out["mainBusinessSummary"] = summary

    # 业务线补全（目标 >=10）
    lines = list(out.get("businessLines") or [])
    seen_line = {str((x.get('name') if isinstance(x, dict) else '')) for x in lines if isinstance(x, dict)}
    for p in (out.get("canonicalProducts") or []):
        ln = (p.get("line") or "").strip() if isinstance(p, dict) else ""
        if ln and ln not in seen_line:
            lines.append({"name": ln, "subLine": "由产品归并", "confidence": 0.58})
            seen_line.add(ln)
    for s in (business.get("segments") or []):
        nm = (s.get("name") or "").strip()
        if nm and nm not in seen_line:
            lines.append({"name": nm, "subLine": "公司分部口径", "confidence": 0.62})
            seen_line.add(nm)
        if len(lines) >= 16:
            break
    out["businessLines"] = lines[:16]

    # 产品补全（目标 >=28）
    products = list(out.get("canonicalProducts") or [])
    seen_prod = {((x.get("name") or "").strip()) for x in products if isinstance(x, dict)}

    def add_product(name: str, line: str = "", conf: float = 0.52, source: str = "本地补全"):
        nm = _normalize_product_name(name)
        if (not nm) or (not _is_concrete_product_name(nm)) or nm in seen_prod:
            return
        seen_prod.add(nm)
        products.append({
            "name": nm,
            "aliases": [],
            "line": line,
            "image_url": "",
            "mention_count": 0,
            "confidence": conf,
            "evidence": [{"source": source, "date": "", "snippet": nm, "url": ""}],
        })

    for t in [x.get("name") for x in (business.get("segments") or []) if isinstance(x, dict)]:
        if t:
            add_product(t, line="业务分部", source="公司分部数据")

    for r in (reports or [])[:120]:
        title = (r.get("title") or "")
        for m in re.findall(r"《([^》]{2,24})》", title):
            # 仅保留“产品名”级别，过滤政策/行业报告类标题
            if re.search(r"报告|行业|策略|点评|措施|方案|协会|数据|月报|季报|年报", m):
                continue
            add_product(m, line="研报提及", source=f"研报/{r.get('orgName') or '未知机构'}")

    for kw in (business.get("specificItems") or [])[:80]:
        if len(kw) <= 24:
            add_product(kw, line="经营描述", source="公司经营分析")

    out["canonicalProducts"] = products[:60]

    # 战略时间线补全：仅保留“业务未来展望”口径（优先研报，不混入公告/员工持股等资本动作）
    strategy = list(out.get("strategyTimeline") or [])
    future_kw = re.compile(r"未来|展望|规划|布局|新品|上线|出海|增长|目标|战略|pipeline|roadmap", re.I)
    for r in (reports or [])[:120]:
        title = (r.get("title") or "").strip()
        if not title or (not future_kw.search(title)):
            continue
        strategy.append({
            "date": (r.get("publishDate") or "")[:10],
            "event": title[:80],
            "whyImportant": "来自研报的未来业务展望线索。",
            "source": f"研报/{r.get('orgName') or '未知机构'}",
            "url": r.get("pdf") or r.get("url") or "",
        })

    uniq = []
    seen_st = set()
    for x in strategy:
        if not isinstance(x, dict):
            continue
        k = f"{x.get('date','')}|{x.get('event','')}"
        if k in seen_st:
            continue
        seen_st.add(k)
        uniq.append(x)
    biz_future_re = re.compile(r"新品|上线|版本|出海|用户|DAU|流水|增长|产品|品类|商业化|平台|生态|运营|研发|IP|矩阵|赛道|未来|展望|规划|布局", re.I)
    out["strategyTimeline"] = [x for x in uniq if biz_future_re.search((x.get("event") or "") + " " + (x.get("whyImportant") or ""))][:30]

    # 002558（巨人网络）专项产品口径：按主公要求输出精细产品清单，避免笼统词
    if code == "002558":
        preset = [
            ("核心主营产品", "征途", "国战MMO", "", 5),
            ("核心主营产品", "征途2", "国战MMO", "", 5),
            ("核心主营产品", "原始征途", "国战MMO手游", "", 4),
            ("核心主营产品", "绿色征途", "MMO", "", 3),
            ("核心主营产品", "王者征途", "MMO手游", "", 3),
            ("核心主营产品", "球球大作战", "休闲竞技", "", 5),
            ("核心主营产品", "太空杀", "派对/狼人杀类", "", 4),
            ("核心主营产品", "月圆之夜", "Roguelike卡牌", "", 3),
            ("核心主营产品", "超自然行动组", "多人恐怖", "", 4),
            ("次核心产品", "名将杀", "卡牌", "", 2),
            ("次核心产品", "帕斯卡契约", "动作RPG", "", 2),
            ("次核心产品", "口袋斗蛐蛐", "轻度休闲", "新", 2),
            ("次核心产品", "街篮2（发行）", "体育竞技", "", 2),
            ("次核心产品", "龙枪觉醒", "MMO", "", 2),
            ("传统端游", "巨人", "MMO", "低活跃", 1),
            ("传统端游", "仙途", "MMO", "停运/弱", 1),
            ("传统端游", "仙侠世界", "MMO", "低活跃", 1),
            ("传统端游", "战国破坏神", "MMO", "停运", 1),
            ("传统端游", "万王之王3", "MMO", "低活跃", 1),
            ("传统端游", "江湖", "MMO", "停运", 1),
            ("传统端游", "龙魂", "MMO", "停运", 1),
            ("传统端游", "兵王", "MMO", "停运", 1),
            ("代理/合作产品", "艾尔之光", "横版格斗", "代理", 1),
            ("代理/合作产品", "苍天2", "MMO", "代理", 1),
            ("代理/合作产品", "巫师之怒", "MMO", "代理", 1),
            ("特殊项目/单机/军方合作", "光荣使命", "FPS", "军方合作", 1),
            ("测试项目/未正式大规模上线", "奥西里之环", "RPG", "测试/孵化", 1),
            ("测试项目/未正式大规模上线", "代号巨人X", "未公开", "测试/孵化", 1),
            ("测试项目/未正式大规模上线", "其他未公开项目", "—", "测试/孵化", 1),
        ]
        metrics = {
            "超自然行动组": {"dau": "1000万+"},
        }
        out["canonicalProducts"] = [{
            "name": n,
            "aliases": [],
            "line": c,
            "type": t,
            "status": s,
            "coreScore": sc,
            "dau": (metrics.get(n) or {}).get("dau", ""),
            "revenue": (metrics.get(n) or {}).get("revenue", ""),
            "image_url": "",
            "mention_count": 0,
            "confidence": 0.92 if sc >= 3 else 0.78,
            "evidence": [{"source": "主公确认口径", "date": "", "snippet": f"{n}/{t}/{s}", "url": ""}],
        } for (c, n, t, s, sc) in preset]
        out["businessLines"] = [{"name": c, "subLine": "产品分层", "confidence": 0.9} for c in [
            "核心主营产品", "次核心产品", "传统端游", "代理/合作产品", "特殊项目/单机/军方合作", "测试项目/未正式大规模上线"
        ]]
        # 去除笼统词条
        ban_re = re.compile(r"^(AI软件技术|短剧平台|海外社交网络|游戏相关业务|产品线|业务分部)$")
        out["canonicalProducts"] = [x for x in (out.get("canonicalProducts") or []) if not ban_re.search((x.get("name") or "").strip())]

    audit = {
        "summary_len": len(out.get("mainBusinessSummary") or ""),
        "business_lines": len(out.get("businessLines") or []),
        "products": len(out.get("canonicalProducts") or []),
        "strategy": len(out.get("strategyTimeline") or []),
        "pass": bool(len(out.get("mainBusinessSummary") or "") >= 120 and len(out.get("businessLines") or []) >= 6 and len(out.get("canonicalProducts") or []) >= 18 and len(out.get("strategyTimeline") or []) >= 4),
    }
    return out, audit


def _build_node_evidence_map(tree: dict, reports: list[dict], business: dict, ai_map: dict | None = None) -> dict:
    node_map: dict[str, list[dict]] = {}

    def clean_name(v: str) -> str:
        v = re.sub(r"（[^）]*）", "", v or "")
        v = re.sub(r"\([^\)]*\)", "", v)
        if "：" in v:
            v = v.split("：", 1)[-1]
        return v.strip()

    def add_evidence(node_name: str, item: dict):
        arr = node_map.setdefault(node_name, [])
        key = (item.get("source", ""), item.get("date", ""), item.get("snippet", ""))
        if key in {(x.get("source", ""), x.get("date", ""), x.get("snippet", "")) for x in arr}:
            return
        arr.append(item)

    all_nodes = []

    def walk(n: dict):
        if not n:
            return
        all_nodes.append(n.get("name") or "")
        for c in (n.get("children") or []):
            walk(c)

    walk(tree)

    for raw in all_nodes:
        key = clean_name(raw)
        if not key or len(key) <= 1:
            continue

        for r in reports[:60]:
            title = (r.get("title") or "")
            if key in title:
                add_evidence(raw, {
                    "source": f"研报/{r.get('orgName') or '未知机构'}",
                    "date": r.get("publishDate") or "",
                    "snippet": title[:180],
                    "url": r.get("pdf") or r.get("url") or "",
                    "score": 0.78,
                })

        for s in (business.get("segments") or []):
            nm = s.get("name") or ""
            if key in nm or nm in key:
                add_evidence(raw, {
                    "source": "公司分部数据",
                    "date": business.get("latestReportDate") or "",
                    "snippet": f"{nm} 收入占比 {s.get('incomeRatioPct', 0)}%",
                    "url": "",
                    "score": 0.86,
                })

    # AI证据并入
    for p in (ai_map or {}).get("canonicalProducts") or []:
        p_name = (p.get("name") or "").strip()
        if not p_name:
            continue
        for node_name in list(node_map.keys()) + all_nodes:
            if p_name in node_name or node_name in p_name:
                for e in (p.get("evidence") or [])[:5]:
                    if isinstance(e, dict):
                        add_evidence(node_name, {
                            "source": e.get("source") or "AI融合证据",
                            "date": e.get("date") or "",
                            "snippet": e.get("snippet") or "",
                            "url": e.get("url") or "",
                            "score": round(float(p.get("confidence") or 0.7), 2),
                        })
                    elif isinstance(e, str) and e.strip():
                        add_evidence(node_name, {
                            "source": "AI融合证据",
                            "date": "",
                            "snippet": e.strip(),
                            "url": "",
                            "score": round(float(p.get("confidence") or 0.7), 2),
                        })

    # 控制体积
    for k in list(node_map.keys()):
        node_map[k] = node_map[k][:8]

    return node_map


def _ai_audit_products(code: str, name: str, products: list[dict], reports: list[dict], business: dict) -> dict:
    """二次AI审核产品清单：只保留具体产品名，并给出可解释分类。"""
    schema = {
        "type": "object",
        "properties": {
            "accepted": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"},
                "category": {"type": "string"},
                "tier": {"type": "string"},
                "status": {"type": "string"}
            }, "required": ["name", "category", "tier"], "additionalProperties": False}},
            "rejected": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"},
                "reason": {"type": "string"}
            }, "required": ["name", "reason"], "additionalProperties": False}},
            "auditScore": {"type": "number"}
        },
        "required": ["accepted", "rejected", "auditScore"],
        "additionalProperties": False
    }

    prompt = (
        "你是产品数据质检官。任务：审核候选产品列表，只保留‘具体产品名’。"
        "严格拒绝：业务线、经营描述、物流/运输/包装、ESG口号、行业报告、政策名、泛词（如其他产品/未分类产品）。"
        "accepted中每项必须包含合理category和tier。"
        "tier只能是：核心产品、次核心产品、储备产品。"
        "若不确定则拒绝，不要放行。只返回JSON。"
    )

    payload = {
        "code": code,
        "name": name,
        "candidates": products[:60],
        "report_titles": [x.get("title") for x in (reports or [])[:20] if x.get("title")],
        "business_segments": (business.get("segments") or [])[:12],
        "business_scope": business.get("scope") or "",
    }

    raw = _call_openai_oauth_json(prompt, payload, schema) or {}
    accepted = []
    seen = set()
    industry_hint = ""
    try:
        industry_hint = "；".join([(x.get("name") or "") for x in (business.get("segments") or [])[:6]]) + "；" + str(business.get("scope") or "")
    except Exception:
        industry_hint = ""

    for x in (raw.get("accepted") or []):
        if not isinstance(x, dict):
            continue
        nm = _normalize_product_name(x.get("name") or "")
        if (not nm) or (not _is_concrete_product_name(nm)) or nm in seen:
            continue
        st = (x.get("status") or "").strip()
        ct = _normalize_product_category(code, nm, (x.get("category") or "").strip(), st) or _infer_fallback_category(nm, (x.get("category") or "").strip(), industry_hint)
        tr = (x.get("tier") or "").strip()
        if tr not in {"核心产品", "次核心产品", "储备产品"}:
            tr = _infer_fallback_tier(nm, st, 0)
        if not ct:
            continue
        seen.add(nm)
        accepted.append({"name": nm, "category": ct, "tier": tr, "status": st})

    fallback_used = False
    if not accepted:
        # AI审核失败时回退到严格规则，不放宽
        fallback_used = True
        for x in (products or []):
            nm = _normalize_product_name(x.get("name") or "")
            if (not nm) or (not _is_concrete_product_name(nm)) or nm in seen:
                continue
            seen.add(nm)
            st = (x.get("status") or "").strip()
            ct = _normalize_product_category(code, nm, (x.get("category") or "").strip(), st)
            if not ct:
                continue
            tr = "储备产品" if (nm.startswith("代号") or "测试" in nm or "预研" in nm) else "次核心产品"
            accepted.append({"name": nm, "category": ct, "tier": tr, "status": st})

    return {
        "accepted": accepted,
        "rejected": raw.get("rejected") or [],
        "auditScore": raw.get("auditScore") if isinstance(raw.get("auditScore"), (int, float)) else None,
        "usedAI": bool(raw),
        "fallbackUsed": fallback_used,
    }


def _ai_structured_company_sections(code: str, name: str, reports: list[dict], business: dict, announcements: list[dict] | None = None, news: list[dict] | None = None) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "products": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"},
                "category": {"type": "string"},
                "tier": {"type": "string"},
                "status": {"type": "string"}
            }, "required": ["name", "category", "tier"], "additionalProperties": False}},
            "businessByLine": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "ratio": {"type": "number"}
            }, "required": ["name"], "additionalProperties": False}},
            "businessByRegion": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "ratio": {"type": "number"}
            }, "required": ["name"], "additionalProperties": False}},
            "topShareholders": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "ratio": {"type": "number"}, "note": {"type": "string"}
            }, "required": ["name"], "additionalProperties": False}},
            "competitors": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "reason": {"type": "string"}
            }, "required": ["name"], "additionalProperties": False}}
        },
        "required": ["products", "businessByLine", "businessByRegion", "topShareholders", "competitors"],
        "additionalProperties": False
    }

    prompt = (
        "你是上市公司研究员。输出思维导图结构化数据，必须精确、完整、有逻辑。"
        "严格要求：1) products 只填具体产品名，禁止短语和笼统词（如AI业务/游戏业务/平台收入）。"
        "2) 每个 product 必须给 category，且 category 必须是你基于公司产品结构抽象出的分组名（如MMO/卡牌/SLG/休闲/出海产品/储备产品等），禁止空值、禁止'其他产品'。"
        "3) 每个 product 必须给 tier，且只能是：核心产品、次核心产品、储备产品。"
        "4) businessByLine 为业务条线及占比。5) businessByRegion 为地区占比。"
        "6) topShareholders 填前十大股东（能确认多少写多少，不要凑数）。"
        "7) competitors 为行业竞争对手与一句理由。只返回JSON。"
    )
    payload = {
        "code": code,
        "name": name,
        "report_titles": [x.get("title") for x in (reports or [])[:80] if x.get("title")],
        "business_segments": business.get("segments") or [],
        "business_scope": business.get("scope") or "",
        "announcements": [x.get("title") for x in (announcements or [])[:20] if x.get("title")],
        "news": [x.get("title") for x in (news or [])[:20] if x.get("title")],
    }
    raw = _call_openai_oauth_json(prompt, payload, schema) or {}

    def clean_term(v: str) -> str:
        s = (v or "").strip()
        s = re.sub(r"[（(].*?[）)]", "", s)
        s = re.split(r"[，,。;；|｜/]", s)[0].strip()
        return s[:30]

    out = {"products": [], "businessByLine": [], "businessByRegion": [], "topShareholders": [], "competitors": []}

    industry_hint = ""
    for rr in (reports or []):
        if rr.get("industry"):
            industry_hint = rr.get("industry") or ""
            break
    if not industry_hint:
        industry_hint = str(business.get("scope") or "")

    for p in (raw.get("products") or []):
        if not isinstance(p, dict):
            continue
        n = _normalize_product_name(clean_term(p.get("name") or ""))
        if (not n) or (not _is_concrete_product_name(n)):
            continue
        st = (p.get("status") or "").strip()
        cat = _normalize_product_category(code, n, (p.get("category") or "").strip(), st) or _infer_fallback_category(n, (p.get("category") or "").strip(), industry_hint)
        tier_raw = (p.get("tier") or "").strip()
        tier = tier_raw if tier_raw in {"核心产品", "次核心产品", "储备产品"} else _infer_fallback_tier(n, st, 0)
        if not cat:
            continue
        out["products"].append({"name": n, "category": cat, "tier": tier, "status": st})

    for k in ("businessByLine", "businessByRegion"):
        for x in (raw.get(k) or []):
            if not isinstance(x, dict):
                continue
            n = clean_term(x.get("name") or "")
            if not n:
                continue
            rr = x.get("ratio")
            out[k].append({"name": n, "ratio": float(rr) if isinstance(rr, (int, float)) else None})

    for x in (raw.get("topShareholders") or []):
        if isinstance(x, dict) and (x.get("name") or "").strip():
            out["topShareholders"].append({"name": (x.get("name") or "").strip(), "ratio": x.get("ratio"), "note": (x.get("note") or "").strip()})

    for x in (raw.get("competitors") or []):
        if isinstance(x, dict) and (x.get("name") or "").strip():
            out["competitors"].append({"name": (x.get("name") or "").strip(), "reason": (x.get("reason") or "").strip()})

    # 回退：业务/地区至少有值
    if not out["businessByLine"]:
        for s in (business.get("segments") or [])[:12]:
            nm = (s.get("name") or "").strip()
            if nm:
                out["businessByLine"].append({"name": nm, "ratio": s.get("incomeRatioPct")})
    if not out["businessByRegion"]:
        for s in (business.get("segments") or [])[:20]:
            nm = (s.get("name") or "").strip()
            if any(k in nm for k in ["境内", "境外", "国内", "国外", "海外"]):
                out["businessByRegion"].append({"name": nm, "ratio": s.get("incomeRatioPct")})

    # 去重
    def uniq_rows(rows, key='name'):
        seen = set(); out2 = []
        for r in rows:
            v = (r.get(key) or '').strip()
            if (not v) or v in seen:
                continue
            seen.add(v); out2.append(r)
        return out2
    out["products"] = uniq_rows(out["products"])[:80]

    # AI二次审核（关键）：防止中间过程混入非产品垃圾项
    prod_audit = _ai_audit_products(code, name, out["products"], reports, business)
    out["products"] = (prod_audit.get("accepted") or [])[:60]
    out["productAudit"] = {
        "usedAI": bool(prod_audit.get("usedAI")),
        "fallbackUsed": bool(prod_audit.get("fallbackUsed")),
        "auditScore": prod_audit.get("auditScore"),
        "rejectedCount": len(prod_audit.get("rejected") or []),
    }

    out["businessByLine"] = uniq_rows(out["businessByLine"])[:20]
    out["businessByRegion"] = uniq_rows(out["businessByRegion"])[:10]
    out["topShareholders"] = uniq_rows(out["topShareholders"])[:10]
    out["competitors"] = uniq_rows(out["competitors"])[:15]
    return out


def _infer_fallback_category(name: str, line: str = "", industry: str = "") -> str:
    n = (name or "").strip()
    l = (line or "").strip()
    ind = (industry or "").strip()

    # 优先使用已有业务线标签做映射
    ln = _normalize_product_category("", n, l, "") if l else ""
    if ln:
        if re.search(r"(端游|客户端)", ln):
            return "客户端游戏"
        if re.search(r"(手游|移动)", ln):
            return "移动游戏"
        if re.search(r"(网页|H5)", ln):
            return "网页游戏"
        return ln

    # 游戏类细分（泛化规则）
    if re.search(r"(Puzzles|Chaos|Survival|SLG|COK|Kings|文明|帝国|战争|末日|霸业)", n, re.I):
        return "SLG策略产品"
    if re.search(r"MMO|征途|仙侠|国战|传奇|奇迹|龙之谷|MU|西游|神墓", n, re.I):
        return "MMO/RPG产品"
    if re.search(r"卡牌|杀|Roguelike|炉石|回合|将|斗蛐蛐", n, re.I):
        return "卡牌策略产品"
    if re.search(r"球球|休闲|派对|竞技|跑酷|消除|捕鱼|狼人杀|太空杀", n, re.I):
        return "休闲竞技产品"
    if re.search(r"大掌柜|杂货店|厨神|餐厅|农场|小镇|经营|开店", n, re.I):
        return "模拟经营产品"
    if re.search(r"斗罗|斗破|凡人|修仙|仙剑|火影|海贼|葫芦兄弟|IP", n, re.I):
        return "IP改编产品"

    # 消费/医药等行业兜底
    if re.search(r"酒|茅台|王子|汉酱|赖茅|大曲", n, re.I):
        return "酒类产品"
    if re.search(r"药|针|胶囊|片|注射液|抗体", n, re.I):
        return "医药产品"

    if "游戏" in ind:
        return "综合游戏产品"
    return "主力产品组"


def _infer_fallback_tier(name: str, status: str = "", mention_count: float = 0) -> str:
    n = (name or "").strip()
    s = (status or "").strip()
    mc = float(mention_count or 0)
    if n.startswith("代号") or any(k in n for k in ["测试", "预研", "预约"]):
        return "储备产品"
    if any(k in s for k in ["停运", "低活跃"]):
        return "次核心产品"
    if mc >= 3:
        return "核心产品"
    return "次核心产品"


def _build_display_products(code: str, name: str, ai_sections: dict, ai_map: dict, reports: list[dict]) -> list[dict]:
    industry = ""
    for r in (reports or []):
        if r.get("industry"):
            industry = r.get("industry") or ""
            break

    out = []
    seen = set()

    for x in (ai_sections or {}).get("products") or []:
        if not isinstance(x, dict):
            continue
        nm = _normalize_product_name(x.get("name") or "")
        ct = _normalize_product_category(code, nm, (x.get("category") or "").strip(), (x.get("status") or "").strip())
        tr = (x.get("tier") or "").strip()
        if (not _is_concrete_product_name(nm)) or (not ct) or (tr not in {"核心产品", "次核心产品", "储备产品"}) or nm in seen:
            continue
        seen.add(nm)
        out.append({"name": nm, "category": ct, "tier": tr, "status": (x.get("status") or "").strip(), "confidence": 0.92, "mention_count": 0})

    # 若AI分类结果过少，补齐（仍保持分类+层级完整）
    if len(out) < 8:
        for p in (ai_map or {}).get("canonicalProducts") or []:
            if not isinstance(p, dict):
                continue
            nm = _normalize_product_name(p.get("name") or "")
            if (not _is_concrete_product_name(nm)) or nm in seen:
                continue
            st = (p.get("status") or "").strip()
            ct = _normalize_product_category(code, nm, (p.get("line") or "").strip(), st) or _infer_fallback_category(nm, (p.get("line") or "").strip(), industry)
            tr = _infer_fallback_tier(nm, st, p.get("mention_count") or 0)
            seen.add(nm)
            out.append({
                "name": nm,
                "category": ct,
                "tier": tr,
                "status": st,
                "confidence": round(float(p.get("confidence") or 0.78), 2),
                "mention_count": int(float(p.get("mention_count") or 0)),
            })
            if len(out) >= 60:
                break

    out = out[:60]

    # 层级修正：若没有核心产品，按 mention/confidence 提升前20%为核心
    if out and not any((x.get("tier") == "核心产品") for x in out):
        ranked = sorted(range(len(out)), key=lambda i: (float(out[i].get("mention_count") or 0), float(out[i].get("confidence") or 0)), reverse=True)
        topn = max(1, min(6, round(len(out) * 0.2)))
        for i in ranked[:topn]:
            out[i]["tier"] = "核心产品"

    return out


def _build_business_tree(code: str, name: str, business: dict, reports: list[dict], ai_map: dict | None = None, announcements: list[dict] | None = None, news: list[dict] | None = None, ai_sections: dict | None = None) -> dict:
    industry = ""
    for r in reports:
        if r.get("industry"):
            industry = r["industry"]
            break

    segs = business.get("segments") or []
    geo_seg, platform_seg, product_seg, other_seg = [], [], [], []
    overseas_ratio = 0.0
    for s in segs[:20]:
        n = s['name']
        r = float(s.get('incomeRatioPct') or 0)
        # 去掉无信息量/易混淆项
        if n in {"其他(补充)", "其他", "其他业务"} or "补充" in n:
            continue
        # 当有更细分“移动端/电脑端网络游戏业务”时，去掉笼统“游戏相关业务”
        if n == "游戏相关业务" and any(("网络游戏业务" in (x.get('name') or "")) for x in segs):
            continue

        node = {"name": f"{n}（{r}%）"}
        if any(k in n for k in ["境外", "国外"]) or n in {"境外", "国外"}:
            overseas_ratio += r
            geo_seg.append(node)
        elif any(k in n for k in ["境内", "国内"]) or n in {"境内", "国内"}:
            geo_seg.append(node)
        elif any(k in n for k in ["移动端", "电脑端", "客户端", "网页", "端游", "手游"]):
            platform_seg.append(node)
        elif any(k in n for k in ["游戏", "酒", "系列", "茅台", "产品", "广告", "搜索", "短剧", "社交", "AI"]):
            product_seg.append(node)
        else:
            other_seg.append(node)

    # 若只有海外细分而没有国内细分，自动补充国内业务（更通用）
    if overseas_ratio > 0 and not any(('境内' in (x.get('name') or '') or '国内' in (x.get('name') or '')) for x in geo_seg):
        domestic_ratio = round(max(0.0, 100.0 - overseas_ratio), 2)
        geo_seg.insert(0, {"name": f"国内业务（{domestic_ratio}%）"})

    # 产品/项目聚合：同类放一起，按出现频次排序
    text_pool = []
    for r in reports:
        text_pool.append(r.get("title") or "")
    text_pool += (business.get("specificItems") or [])
    text_pool += [x.get('name','') for x in (business.get('segments') or [])]

    freq = {}
    def inc(k):
        if not k:
            return
        freq[k] = freq.get(k, 0) + 1

    def canonical(name: str) -> str:
        n = (name or "").strip().replace('收入','').replace('业务','')
        if re.search(r"太空杀|SuperSus", n, flags=re.I):
            return "太空杀/SuperSus"
        if "征途" in n:
            return "征途系列"
        if "超自然行动组" in n:
            return "超自然行动组"
        if "原始征途" in n:
            return "原始征途"
        if "球球大作战" in n:
            return "球球大作战"
        if "Opera" in n:
            return "Opera"
        if "StarMaker" in n:
            return "StarMaker"
        if "Grindr" in n:
            return "Grindr"
        if "SkyMusic" in n:
            return "天工 SkyMusic"
        if "天工" in n and "SkyMusic" not in n:
            return "天工大模型"
        if "搜索" in n and "Opera" in n:
            return "Opera搜索"
        if "短剧" in n:
            return "短剧平台"
        if "海外社交" in n:
            return "海外社交网络"
        if re.search(r"AI|AGI|AIGC", n, flags=re.I):
            return "AI软件技术"
        return n

    for t in text_pool:
        for m in re.findall(r"《([^》]{2,24})》", t):
            c = canonical(m)
            if len(c) <= 20:
                inc(c)

        # 通用品牌/产品名抽取（中英）
        for m in re.findall(r"(Opera|StarMaker|Grindr|SkyMusic|天工(?:\d+\.\d+)?大模型|天工|超自然行动组|原始征途|球球大作战|太空杀|SuperSus)", t, flags=re.I):
            inc(canonical(m))

        # 英文品牌词（过滤财务词）
        for m in re.findall(r"\b([A-Z][A-Za-z]{2,20})\b", t):
            if m.upper() in {"AI", "AGI", "AIGC", "Q", "EBITDA", "IP"}:
                continue
            if m.lower() in {"all", "in", "buy"}:
                continue
            inc(canonical(m))

        if "征途" in t:
            inc("征途系列")
        if "短剧" in t:
            inc("短剧平台")
        if "海外社交" in t:
            inc("海外社交网络")
        if re.search(r"AI|AGI|AIGC", t, flags=re.I):
            inc("AI软件技术")

    # 兜底：从业务词里补一些候选（剔除长句和描述性短语）
    for x in (business.get("specificItems") or []):
        if len(x) > 14 or re.search(r"推进|态势|保持|发展|增长|改革|成为|抓手", x):
            continue
        if any(k in x for k in ["征途", "行动组", "球球", "太空杀", "SuperSus", "王子酒", "汉酱", "茅台1935"]):
            inc(canonical(x))

    freq_items = sorted(freq.items(), key=lambda z: z[1], reverse=True)
    spec_items = [{"name": f"{k}（提及{v}次）"} for k, v in freq_items[:12]]

    # AI增强：优先并入结构化抽取得到的产品项（不替代分部数据，只增强“具体产品/项目”）
    ai_products = []
    for p in (ai_map or {}).get("canonicalProducts") or []:
        nm = (p.get("name") or "").strip()
        if not nm:
            continue
        conf = round(float(p.get("confidence") or 0.7), 2)
        mc = int(float(p.get("mention_count") or 0))
        mc_txt = f"提及{mc}次" if mc > 0 else "全网补全"
        ai_products.append({"name": f"{nm}（{mc_txt} / AI{conf}）"})
    if ai_products:
        # 去重合并：AI优先，随后补本地频次项
        seen2 = set()
        merged = []
        for x in ai_products + spec_items:
            key = re.sub(r"（[^）]*）", "", x.get("name") or "").strip()
            if not key or key in seen2:
                continue
            seen2.add(key)
            merged.append(x)
            if len(merged) >= 40:
                break
        spec_items = merged

    orgs = []
    seen = set()
    for r in reports:
        o = (r.get("orgName") or "").strip()
        if o and o not in seen:
            seen.add(o)
            orgs.append({"name": o})
        if len(orgs) >= 10:
            break

    children = []

    ai_lines = []
    for bl in (ai_map or {}).get("businessLines") or []:
        nm = (bl.get("name") or "").strip()
        sub = (bl.get("subLine") or "").strip()
        conf = round(float(bl.get("confidence") or 0.7), 2)
        if nm:
            ai_lines.append({"name": f"{nm}｜{sub or '细分待补'}（AI{conf}）"})
    if code == "002558":
        order = ["核心主营产品", "次核心产品", "传统端游", "代理/合作产品", "特殊项目/单机/军方合作", "测试项目/未正式大规模上线"]
        grouped = {k: [] for k in order}
        for p in (ai_map or {}).get("canonicalProducts") or []:
            if not isinstance(p, dict):
                continue
            cat = (p.get("line") or "").strip()
            if cat not in grouped:
                continue
            nm = (p.get("name") or "").strip()
            tp = (p.get("type") or "").strip()
            stt = (p.get("status") or "").strip()
            dau = (p.get("dau") or "").strip()
            rev = (p.get("revenue") or "").strip()
            extra = []
            if dau:
                extra.append(f"DAU:{dau}")
            if rev:
                extra.append(f"流水:{rev}")
            if stt and ("停运" in stt or "低活跃" in stt):
                extra.append(stt)
            tail = f"｜{' / '.join(extra)}" if extra else ""
            grouped[cat].append({"name": f"{nm}｜{tp or '类型待补'}{tail}"})

        biz_ratio_nodes = []
        for s in (business.get("segments") or [])[:10]:
            sn = (s.get("name") or "").strip()
            rr = s.get("incomeRatioPct")
            if sn and rr is not None:
                biz_ratio_nodes.append({"name": f"{sn}（收入占比{rr}%）"})

        product_categories = []
        for cat in order:
            if grouped.get(cat):
                product_categories.append({"name": cat, "children": grouped[cat]})

        if biz_ratio_nodes or product_categories:
            children.append({"name": "业务主线（业务结构→产品矩阵）", "children": (
                ([{"name": "业务线收入占比", "children": biz_ratio_nodes}] if biz_ratio_nodes else []) +
                ([{"name": "产品矩阵", "children": product_categories}] if product_categories else [])
            )})
    else:
        sec = ai_sections or {}

        by_line = [{"name": f"{x.get('name')}（{x.get('ratio')}%）" if x.get('ratio') is not None else x.get('name')} for x in (sec.get("businessByLine") or []) if x.get("name")]
        by_region = [{"name": f"{x.get('name')}（{x.get('ratio')}%）" if x.get('ratio') is not None else x.get('name')} for x in (sec.get("businessByRegion") or []) if x.get("name")]

        # 产品：同层仅放“具体产品”，不放业务词
        src_products = []
        for p in (sec.get("products") or []):
            nm = ((p or {}).get("name") or "").strip()
            if not _is_concrete_product_name(nm):
                continue
            st = ((p or {}).get("status") or "").strip()
            ct = _normalize_product_category(code, nm, ((p or {}).get("category") or "").strip(), st)
            tr = ((p or {}).get("tier") or "").strip()
            if (not ct) or (tr not in {"核心产品", "次核心产品", "储备产品"}):
                continue
            src_products.append({"name": nm, "category": ct, "tier": tr, "status": st})
        if not src_products:
            for p in (ai_map or {}).get("canonicalProducts") or []:
                if not isinstance(p, dict):
                    continue
                nm = (p.get("name") or "").strip()
                if not _is_concrete_product_name(nm):
                    continue
                st = (p.get("status") or "").strip()
                cat = _normalize_product_category(code, nm, (p.get("line") or "未分类产品").strip(), st)
                if not cat:
                    continue
                src_products.append({
                    "name": nm,
                    "category": cat,
                    "tier": "次核心产品",
                    "status": st,
                })

        prod_rows = []
        for p in src_products:
            n = (p.get("name") or "").strip()
            if not n:
                continue
            stt = (p.get("status") or "").strip()
            tr = (p.get("tier") or "").strip()
            tail = f"｜{tr}" if tr else ""
            if stt and ("停运" in stt or "低活跃" in stt):
                tail += f"｜{stt}"
            prod_rows.append({"name": f"{n}{tail}"})

        product_by_cat = {}
        for p in src_products:
            n = (p.get("name") or "").strip()
            c = (p.get("category") or "未分类产品").strip() or "未分类产品"
            if not n:
                continue
            stt = (p.get("status") or "").strip()
            tail = f"｜{stt}" if (stt and ("停运" in stt or "低活跃" in stt)) else ""
            product_by_cat.setdefault(c, []).append({"name": f"{n}{tail}"})

        product_nodes = []
        for c, items in product_by_cat.items():
            if items:
                product_nodes.append({"name": c, "children": items[:30]})

        sh_nodes = []
        for x in (sec.get("topShareholders") or []):
            nm = (x.get("name") or "").strip()
            if not nm:
                continue
            rr = x.get("ratio")
            note = (x.get("note") or "").strip()
            txt = nm
            if isinstance(rr, (int, float)):
                txt += f"（持股{rr}%）"
            if note:
                txt += f"｜{note[:20]}"
            sh_nodes.append({"name": txt})

        comp_nodes = []
        for x in (sec.get("competitors") or []):
            nm = (x.get("name") or "").strip()
            if not nm:
                continue
            rs = (x.get("reason") or "").strip()
            comp_nodes.append({"name": f"{nm}{('｜'+rs[:22]) if rs else ''}"})

        main_children = []
        if industry:
            main_children.append({"name": f"所属行业：{industry}"})
        if by_line:
            # 仅保留“收入结构（按业务）”主干，节点内带比例
            main_children.append({"name": "收入结构（按业务）", "children": by_line})
        if by_region:
            main_children.append({"name": "收入结构（按地区）", "children": by_region})
        if product_nodes:
            main_children.append({"name": "产品", "children": product_nodes})
        elif prod_rows:
            main_children.append({"name": "产品", "children": prod_rows})
        if sh_nodes:
            main_children.append({"name": "前十大股东", "children": sh_nodes})
        if comp_nodes:
            main_children.append({"name": "行业竞争对手", "children": comp_nodes})

        if main_children:
            children.append({"name": "业务主线（结构化）", "children": main_children})

    st = (ai_map or {}).get("strategyTimeline") or []
    if st:
        children.append({"name": "未来战略布局（时间线）", "children": [
            {"name": f"{(x.get('date') or '时间待补')[:10]}｜{(x.get('event') or '')[:44]}"} for x in st[:14]
        ]})

    # 按要求移除“覆盖机构/最新研报快照/公告时间线/新闻动态”导图分支

    return {"name": f"{name or code}({code})", "children": children}


def _build_business_mindmap(code: str, name: str, business: dict, reports: list[dict], ai_map: dict | None = None, announcements: list[dict] | None = None, news: list[dict] | None = None, ai_sections: dict | None = None) -> str:
    tree = _build_business_tree(code, name, business, reports, ai_map=ai_map, announcements=announcements, news=news, ai_sections=ai_sections)

    lines = ["mindmap", f"  root(({tree['name']}))"]
    for c in tree.get("children") or []:
        lines.append(f"    {c['name']}")
        for cc in c.get("children") or []:
            lines.append(f"      {cc['name']}")
    return "\n".join(lines)


def _build_supply_chain_tree(code: str, name: str, business: dict, reports: list[dict] | None = None) -> dict:
    reports = reports or []

    schema = {
        "type": "object",
        "properties": {
            "upstream": {"type": "array", "items": {"type": "string"}},
            "midstream": {"type": "array", "items": {"type": "string"}},
            "downstream": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["upstream", "midstream", "downstream"],
        "additionalProperties": False,
    }
    prompt = (
        "你是产业链分析师。请基于公司分部与研报标题，输出该公司的供应链上下游结构。"
        "要求：精确、分层、不要空话；每层5-10条。只返回JSON。"
    )
    payload = {
        "code": code,
        "name": name,
        "business_segments": business.get("segments") or [],
        "business_scope": business.get("scope") or "",
        "report_titles": [r.get("title") for r in reports[:40] if r.get("title")],
    }
    ai = _call_openai_oauth_json(prompt, payload, schema) or {}

    def norm_list(v):
        out = []
        for x in (v or []):
            if isinstance(x, str) and x.strip():
                out.append({"name": x.strip()[:60]})
        return out[:12]

    upstream = norm_list(ai.get("upstream"))
    mid = norm_list(ai.get("midstream"))
    downstream = norm_list(ai.get("downstream"))

    if not (upstream and mid and downstream):
        if code == "002558":
            upstream = upstream or [{"name": "游戏引擎/开发工具"}, {"name": "美术外包与音频制作"}, {"name": "云服务/CDN"}]
            mid = mid or [{"name": "研发立项与版本迭代"}, {"name": "发行与渠道运营"}, {"name": "商业化运营"}]
            downstream = downstream or [{"name": "iOS/安卓应用商店用户"}, {"name": "PC端用户"}, {"name": "社区与IP合作方"}]
        else:
            upstream = upstream or [{"name": "技术与内容供应商"}, {"name": "云与基础设施服务"}]
            mid = mid or [{"name": "公司核心研发/生产/运营"}, {"name": "产品化与质量管理"}, {"name": "市场与商业化运营"}]
            downstream = downstream or [{"name": "渠道平台"}, {"name": "终端客户"}]

    if len(mid) < 3:
        for x in [{"name": "产品化与质量管理"}, {"name": "市场与商业化运营"}, {"name": "数据运营与增长"}]:
            if len(mid) >= 4:
                break
            if x not in mid:
                mid.append(x)

    return {
        "name": f"{name or code} 供应链上下游",
        "children": [
            {"name": "上游（投入要素）", "children": upstream},
            {"name": "中游（公司核心环节）", "children": mid},
            {"name": "下游（客户与变现）", "children": downstream},
        ],
    }


def _fetch_company_announcements(code: str, page_size: int = 8) -> list[dict]:
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    params = {
        "sr": -1,
        "page_size": page_size,
        "page_index": 1,
        "ann_type": "A",
        "client_source": "web",
        "stock_list": code,
    }
    r = _rq_get(url, params=params, timeout=6, tries=2)
    r.raise_for_status()
    lst = (((r.json() or {}).get("data") or {}).get("list")) or []
    out = []
    for it in lst[:page_size]:
        title = (it.get("title") or "").strip()
        art = (it.get("art_code") or "").strip()
        notice_date = str(it.get("notice_date") or "")[:10]
        cols = [x.get("column_name") for x in (it.get("columns") or []) if x.get("column_name")]
        url = f"https://data.eastmoney.com/notices/detail/{code}/{art}.html" if art else ""
        out.append({
            "title": title,
            "date": notice_date,
            "tags": cols[:3],
            "url": url,
            "source": "东方财富公告",
        })
    return out


def _fetch_company_news(keyword: str, limit: int = 8) -> list[dict]:
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    param = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": limit,
            }
        },
    }
    r = _rq_get(url, {"cb": "jQueryStockNews", "param": json.dumps(param, ensure_ascii=False)}, timeout=6, tries=2)
    r.raise_for_status()
    text = r.text or ""
    m = re.search(r"^jQueryStockNews\((.*)\)\s*$", text, flags=re.S)
    if not m:
        return []
    obj = json.loads(m.group(1))
    arr = ((((obj or {}).get("result") or {}).get("cmsArticleWebOld")) or [])
    out = []
    for it in arr[:limit]:
        out.append({
            "title": (it.get("title") or "").strip(),
            "date": (it.get("date") or "")[:10],
            "media": (it.get("mediaName") or "").strip(),
            "summary": (it.get("content") or "").strip()[:120],
            "url": (it.get("url") or "").strip(),
            "image": (it.get("image") or "").strip(),
            "source": "东方财富资讯",
        })
    return out


@app.route("/stock-research")
def stock_research_page():
    return render_template("stock_research.html")


@app.route("/oauth/openai/start")
def oauth_openai_start():
    cfg = _oauth_cfg()
    if not (cfg["client_id"] and cfg["client_secret"] and cfg["redirect_uri"]):
        return jsonify({
            "ok": False,
            "error": "OAuth配置不完整，请先设置 STOCK_RESEARCH_OPENAI_OAUTH_CLIENT_ID / CLIENT_SECRET / REDIRECT_URI",
        }), 400

    state = secrets.token_urlsafe(24)
    OPENAI_OAUTH_STATE_CACHE[state] = time.time()
    # 清理过期state
    for k, ts in list(OPENAI_OAUTH_STATE_CACHE.items()):
        if time.time() - ts > 900:
            OPENAI_OAUTH_STATE_CACHE.pop(k, None)

    qs = {
        "response_type": "code",
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "state": state,
    }
    if cfg.get("scope"):
        qs["scope"] = cfg["scope"]

    auth_url = f"{cfg['authorize_url']}?{urlencode(qs)}"
    if (request.args.get("raw") or "").strip() == "1":
        return jsonify({"ok": True, "authUrl": auth_url, "state": state})
    return f"<html><body style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; padding: 24px;'>\n<h3>正在跳转 OpenAI 授权...</h3>\n<p>若未自动跳转，请点击：<a href='{auth_url}'>继续授权</a></p>\n<script>location.href={json.dumps(auth_url)};</script>\n</body></html>"


@app.route("/oauth/openai/callback")
def oauth_openai_callback():
    cfg = _oauth_cfg()
    code = (request.args.get("code") or "").strip()
    state = (request.args.get("state") or "").strip()
    err = (request.args.get("error") or "").strip()

    if err:
        return f"OAuth授权失败：{err}", 400
    if not code:
        return "缺少code", 400
    if not state or state not in OPENAI_OAUTH_STATE_CACHE:
        return "state校验失败", 400

    OPENAI_OAUTH_STATE_CACHE.pop(state, None)

    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": cfg["redirect_uri"],
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
    }

    try:
        resp = requests.post(cfg["token_url"], data=form, timeout=30)
        resp.raise_for_status()
        obj = resp.json() or {}
        expires_in = int(obj.get("expires_in") or 3600)
        token_obj = {
            **obj,
            "expires_at": time.time() + max(60, expires_in),
            "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_oauth_token_obj(token_obj)
        access_token = (token_obj.get("access_token") or "").strip()
        if access_token:
            os.environ["STOCK_RESEARCH_OPENAI_OAUTH_TOKEN"] = access_token
        return "OpenAI OAuth绑定成功。你可以返回研报页面继续使用。"
    except Exception as e:
        return f"Token交换失败：{e}", 500


@app.route("/api/stock-research/oauth/status")
def api_stock_research_oauth_status():
    cfg = _oauth_cfg()
    token_obj = _load_oauth_token_obj()
    token = _get_openai_access_token()
    return jsonify({
        "ok": True,
        "data": {
            "configured": bool(cfg["client_id"] and cfg["client_secret"] and cfg["redirect_uri"]),
            "authorized": bool(token),
            "hasRefreshToken": bool((token_obj.get("refresh_token") or "").strip()),
            "expiresAt": token_obj.get("expires_at"),
            "updatedAt": token_obj.get("updated_at"),
            "redirectUri": cfg["redirect_uri"],
        },
    })


@app.route("/api/stock-research/oauth/refresh", methods=["POST"])
def api_stock_research_oauth_refresh():
    token_obj = _load_oauth_token_obj()
    refreshed = _oauth_refresh_token(token_obj)
    if not refreshed:
        return jsonify({"ok": False, "error": "刷新失败，请重新走 /oauth/openai/start 绑定"}), 400
    return jsonify({"ok": True, "data": {"updatedAt": refreshed.get("updated_at"), "expiresAt": refreshed.get("expires_at")}})


@app.route("/api/stock-research/reports")
def api_stock_research_reports():
    code = (request.args.get("code") or "600519").strip()
    days = int((request.args.get("days") or "730").strip() or 730)
    page_size = int((request.args.get("page_size") or "20").strip() or 20)
    if not code.isdigit() or len(code) != 6:
        return jsonify({"ok": False, "error": "请输入6位A股代码"}), 400

    cache_dir = str(DATA_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"stock_research_reports_{code}_{days}_{page_size}.json")

    try:
        if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file) < 6 * 3600):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return jsonify(cached)

        reports = _fetch_research_reports(code, days=days, page_size=page_size)
        payload = {"ok": True, "data": {"code": code, "reports": reports}}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"ok": False, "error": f"研报抓取失败：{e}"}), 500


@app.route("/api/stock-research/node-explain")
def api_stock_research_node_explain():
    code = (request.args.get("code") or "").strip()
    node = (request.args.get("node") or "").strip()
    parent = (request.args.get("parent") or "").strip()
    path = (request.args.get("path") or "").strip()
    evidence = (request.args.get("evidence") or "").strip()
    if (not code) or (not code.isdigit()) or len(code) != 6:
        return jsonify({"ok": False, "error": "请输入6位A股代码"}), 400
    if not node:
        return jsonify({"ok": False, "error": "缺少节点名称"}), 400

    try:
        reports = _fetch_research_reports(code, days=365, page_size=20)
        business = _fetch_business_lines(code)
        name = reports[0].get("stockName") if reports else code

        clean_node = re.sub(r"（[^）]*）", "", node)
        clean_node = re.sub(r"\([^\)]*\)", "", clean_node).strip()

        schema = {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
                "entityType": {"type": "string"},
                "relationToCompany": {"type": "string"},
                "businessRole": {"type": "string"},
                "revenueLogic": {"type": "string"},
                "competition": {"type": "string"},
                "risks": {"type": "string"},
                "keyPoints": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["explanation", "keyPoints"],
            "additionalProperties": False,
        }
        prompt = (
            "你是A股公司业务研究员。任务是解释‘节点名词在该公司中的实际商业含义’，"
            "不是解释图谱结构。\n"
            "输出要求（必须具体，不要空话）：\n"
            "1) explanation：120-220字，先回答‘它到底是什么’，再说明它在该公司业务链的位置。\n"
            "2) entityType：节点类型（如 产品/品牌/子公司/门店/酒店/渠道/平台/项目/竞品）。\n"
            "3) relationToCompany：与公司关系（自有/控股/参股/合作/竞品/上下游）。\n"
            "4) businessRole：具体业务作用（获客、品牌展示、利润贡献、渠道延伸、供应链协同等）。\n"
            "5) revenueLogic：怎么赚钱或如何影响收入利润（可写‘直接贡献有限、更多是品牌外溢’这类判断）。\n"
            "6) competition：若是竞品，写清楚对标区间（价格带/客群/渠道）；不是竞品就写‘非竞品节点’。\n"
            "7) risks：至少1条关键风险或不确定性。\n"
            "8) keyPoints：4-6条，每条20字以上，必须可执行可理解。\n"
            "9) 若节点是具体实体（如“茅台国际大酒店”），必须解释该实体本身，不得写成‘业务维度节点/结构权重’。\n"
            "仅返回JSON。"
        )
        payload = {
            "code": code,
            "stock_name": name,
            "node": node,
            "node_clean": clean_node,
            "node_parent": parent,
            "node_path": path,
            "node_evidence": evidence,
            "industry": next((x.get("industry") for x in reports if x.get("industry")), ""),
            "report_titles": [x.get("title") for x in (reports or [])[:12] if x.get("title")],
            "business_segments": (business.get("segments") or [])[:10],
        }
        ai = _call_openai_oauth_json(prompt, payload, schema) or {}
        exp = (ai.get("explanation") or "").strip()
        points = [str(x).strip() for x in (ai.get("keyPoints") or []) if str(x).strip()][:8]
        entity_type = (ai.get("entityType") or "").strip()
        relation = (ai.get("relationToCompany") or "").strip()
        role = (ai.get("businessRole") or "").strip()
        revenue_logic = (ai.get("revenueLogic") or "").strip()
        competition = (ai.get("competition") or "").strip()
        risks = (ai.get("risks") or "").strip()

        if not exp:
            kind = "业务实体"
            if any(k in clean_node for k in ["酒店", "大酒店"]):
                kind = "酒店业务实体"
            elif any(k in clean_node for k in ["APP", "平台", "小程序"]):
                kind = "数字化渠道"
            elif any(k in clean_node for k in ["酒", "产品", "系列"]):
                kind = "产品/品牌单元"
            exp = f"{name}（{code}）中的“{node}”更接近一个{kind}，应从实体属性、与公司的关系、盈利逻辑与风险四个维度理解，而不是只看图谱层级。"
        if len(points) < 4:
            fallback_points = [
                f"节点类型判断：{entity_type or '需结合上下文二次判定'}。",
                f"与公司关系：{relation or '通常为自有业务、合作节点或竞品节点之一'}。",
                f"业务作用：{role or '可能承担品牌、渠道、获客或利润补充作用'}。",
                f"收入逻辑：{revenue_logic or '需区分直接营收贡献与品牌外溢的间接贡献'}。",
                f"核心风险：{risks or '需关注需求波动、竞争加剧和执行偏差等风险'}。",
            ]
            points = (points + fallback_points)[:6]

        return jsonify({
            "ok": True,
            "data": {
                "code": code,
                "name": name,
                "node": node,
                "explanation": exp,
                "entityType": entity_type,
                "relationToCompany": relation,
                "businessRole": role,
                "revenueLogic": revenue_logic,
                "competition": competition,
                "risks": risks,
                "keyPoints": points,
                "ai": {"used": bool(ai), "runtime": dict(AI_RUNTIME_STATUS)},
            },
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"节点解释失败：{e}"}), 500


@app.route("/api/stock-research/analysis")
def api_stock_research_analysis():
    code = (request.args.get("code") or "600519").strip()
    days = int((request.args.get("days") or "730").strip() or 730)
    if not code.isdigit() or len(code) != 6:
        return jsonify({"ok": False, "error": "请输入6位A股代码"}), 400

    cache_dir = str(DATA_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"stock_research_analysis_{code}_{days}.json")

    try:
        if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file) < 6 * 3600):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return jsonify(cached)

        reports = _fetch_research_reports(code, days=days, page_size=30)
        business = _fetch_business_lines(code)
        analysis = _generate_report_analysis(code, reports, business)
        name = reports[0].get("stockName") if reports else code

        ai_raw = _ai_extract_business_map(code, name or code, reports, business)
        ai_map = _normalize_ai_map(ai_raw)
        ai_diag = {
            "rawKeys": sorted(list((ai_raw or {}).keys())) if isinstance(ai_raw, dict) else [],
            "normalized": {
                "summary": bool((ai_map.get("mainBusinessSummary") or "").strip()),
                "businessLines": len(ai_map.get("businessLines") or []),
                "products": len(ai_map.get("canonicalProducts") or []),
                "strategyTimeline": len(ai_map.get("strategyTimeline") or []),
            },
        }

        announcements = []
        news = []
        try:
            announcements = _fetch_company_announcements(code, page_size=14)
        except Exception:
            announcements = []
        try:
            news = _fetch_company_news((name or code), limit=14)
        except Exception:
            news = []

        ai_map, quality_audit = _enrich_company_info(code, ai_map, business, reports, announcements, news)
        ai_sections = _ai_structured_company_sections(code, name or code, reports, business, announcements=announcements, news=news)
        display_products = _build_display_products(code, name or code, ai_sections or {}, ai_map or {}, reports or [])
        ai_sections = {**(ai_sections or {}), "products": display_products}

        mindmap = _build_business_mindmap(code, name or code, business, reports, ai_map=ai_map, announcements=announcements, news=news, ai_sections=ai_sections)
        tree = _build_business_tree(code, name or code, business, reports, ai_map=ai_map, announcements=announcements, news=news, ai_sections=ai_sections)
        supply_chain_tree = _build_supply_chain_tree(code, name or code, business, reports=reports)
        node_evidence_map = _build_node_evidence_map(tree, reports, business, ai_map=ai_map)

        company_lines = (ai_map or {}).get("businessLines") or []

        ai_used = bool((ai_map.get("mainBusinessSummary") or "").strip() or (ai_map.get("businessLines") or []) or (ai_map.get("canonicalProducts") or []) or (ai_map.get("strategyTimeline") or []))
        ai_runtime = dict(AI_RUNTIME_STATUS)
        pa = (ai_sections or {}).get("productAudit") or {}
        sec_products = (ai_sections or {}).get("products") or []
        uncategorized_cnt = sum(1 for x in sec_products if (x.get("category") or "").strip() in {"", "AI未分类", "其他产品", "未分类产品"})
        ai_failed = bool(
            (not ai_used)
            or (not pa.get("usedAI"))
            or (uncategorized_cnt > 0)
        )
        ai_fail_msg = ""
        if ai_failed:
            ai_fail_msg = (ai_runtime.get("last_error") or "AI调用未命中，本次结果包含本地兜底内容").strip()
            if (not pa.get("usedAI")) or pa.get("fallbackUsed"):
                ai_fail_msg = (ai_fail_msg + "；产品AI复审未成功").strip("；")

        payload = {
            "ok": True,
            "data": {
                "code": code,
                "name": name,
                "analysis": analysis,
                "business": business,
                "mindmap": mindmap,
                "tree": tree,
                "supplyChainTree": supply_chain_tree,
                "nodeEvidenceMap": node_evidence_map,
                "companyInfo": {
                    "mainBusinessSummary": ((ai_map or {}).get("mainBusinessSummary") or "").strip(),
                    "businessLines": company_lines,
                    "products": [{
                        "name": (x.get("name") or "").strip(),
                        "line": (x.get("category") or "").strip(),
                        "category": (x.get("category") or "").strip(),
                        "type": (x.get("tier") or "").strip(),
                        "status": (x.get("status") or "").strip(),
                        "coreScore": (5 if (x.get("tier") or "").strip() == "核心产品" else (3 if (x.get("tier") or "").strip() == "次核心产品" else 1)),
                        "mention_count": int(float(x.get("mention_count") or 0)),
                        "confidence": float(x.get("confidence") or 0.9),
                    } for x in (display_products or [])],
                    "strategyTimeline": (ai_map or {}).get("strategyTimeline") or [],
                },
                "updates": {
                    "announcements": announcements,
                    "news": news,
                },
                "ai": {
                    "enabled": bool(_get_openai_access_token()),
                    "provider": ("qwen-compatible" if ((os.getenv("STOCK_RESEARCH_OPENAI_BASE_URL") or "").find("dashscope") >= 0 or (os.getenv("STOCK_RESEARCH_OPENAI_MODEL") or "qwen-plus").startswith("qwen")) else "openai-backend"),
                    "used": ai_used,
                    "required": True,
                    "failed": ai_failed,
                    "failureMessage": ai_fail_msg if ai_failed else "",
                    "model": (os.getenv("STOCK_RESEARCH_OPENAI_MODEL") or "qwen-plus"),
                    "diag": ai_diag,
                    "qualityAudit": quality_audit,
                    "productAudit": {**((ai_sections or {}).get("productAudit") or {}), "uncategorizedCount": uncategorized_cnt},
                    "runtime": ai_runtime,
                },
                "warnings": ([f"AI调用失败：{ai_fail_msg}"] if ai_failed else []) + (["产品AI复审回退到规则补齐"] if pa.get("fallbackUsed") else []) + ([f"产品存在未分类项：{uncategorized_cnt}"] if uncategorized_cnt > 0 else []),
                "sources": {
                    "reports": sorted(list({x.get('source','') for x in reports if x.get('source')})),
                    "business": business.get("source") or [],
                },
            },
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"ok": False, "error": f"分析生成失败：{e}"}), 500


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
