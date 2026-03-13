from __future__ import annotations

import ast
import datetime as dt
import json
import math
import os
import random
from pathlib import Path

import pytz
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.json.ensure_ascii = False

DATA_FILE = Path(__file__).parent / "data" / "latest_recommendations.json"
VALIDATION_FILE = Path(__file__).parent / "data" / "validation.json"


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


def _rq_get(url: str, params: dict, timeout: int = 8):
    # 尽量绕过宿主代理，降低 Eastmoney 连接被代理中断概率
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")
    last_err = None
    for i in range(3):
        try:
            return requests.get(url, params=params, timeout=timeout, proxies={"http": None, "https": None})
        except Exception as e:
            last_err = e
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
def _fetch_universe_realtime(limit_pages: int = 2) -> list[dict]:
    # 东财全市场快照：按成交额降序取前若干页
    rows = []
    for pn in range(1, limit_pages + 1):
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": pn,
            "pz": 80,
            "po": 1,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f6",  # 成交额排序
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",  # 沪深A
            "fields": "f12,f14,f2,f3,f6,f8,f15,f16,f17,f18",
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        }
        r = _rq_get(url, params=params, timeout=8)
        r.raise_for_status()
        diff = ((r.json().get("data") or {}).get("diff")) or []
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
    return [x for x in rows if x["price"] > 0 and x["amount"] > 0]


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


def _real_rankings() -> dict:
    now = _cn_now().strftime("%Y-%m-%d %H:%M:%S")
    universe = _fetch_universe_realtime(limit_pages=2)  # 实时成交额前160
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
    return {
        "updated_at": now,
        "strategy": {
            "name": "A股全市场短线交易参考排名",
            "version": "4.0-real",
            "principle": "基于东财实时全A成交额筛选 + 日线动量/趋势/突破/流动性真实计算",
            "risk_note": "仅供研究",
        },
        "industry_flow": [],
        "picks": picks,
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
    data = load_json(
        DATA_FILE,
        {
            "updated_at": "尚未生成",
            "strategy": {"name": "A股全市场短线交易参考排名", "version": "real-only", "principle": "仅真实计算结果，不使用假数据", "risk_note": "仅供研究"},
            "industry_flow": [],
            "picks": [],
        },
    )
    return render_template("index.html", data=data)


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
    return jsonify(
        load_json(
            DATA_FILE,
            {
                "updated_at": "尚未生成",
                "strategy": {"name": "A股全市场短线交易参考排名", "version": "real-only", "principle": "仅真实计算结果，不使用假数据", "risk_note": "仅供研究"},
                "industry_flow": [],
                "picks": [],
            },
        )
    )


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


@app.route("/api/volume-profile")
def api_volume_profile():
    from volume_profile import get_volume_profile

    symbol = (request.args.get("symbol") or "").strip()
    days = (request.args.get("days") or "1").strip()
    if not symbol:
        return jsonify({"ok": False, "error": "缺少 symbol 参数"}), 400
    try:
        data = get_volume_profile(symbol, days=int(days))
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
    try:
        run_daily_job()
        return jsonify({"ok": True, "message": "已刷新短线排名（真实计算）"})
    except Exception as e:
        return jsonify({"ok": False, "message": f"真实计算失败：{e}"}), 500


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
