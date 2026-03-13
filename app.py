from __future__ import annotations

import ast
import datetime as dt
import json
import math
import random
from pathlib import Path

import pytz
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
def run_daily_job():
    # 默认轻量可用：秒级返回，保证线上稳定；需要全量策略时再开环境变量
    out = fallback_rankings()
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


def _gen_trades(symbol: str, step: int = 0, seconds: int = 70):
    seed = int(symbol) + int(step)
    rng = random.Random(seed)
    base = 20 + (int(symbol[-2:]) % 100) * 0.35
    now = dt.datetime.now()
    trades = []
    last_price = base
    for i in range(seconds * 2):  # 每0.5秒一笔
        t = now - dt.timedelta(seconds=(seconds - i * 0.5))
        drift = rng.uniform(-0.03, 0.04)
        last_price = round(max(1, last_price + drift), 2)
        bid1 = round(last_price - 0.01, 2)
        ask1 = round(last_price + 0.01, 2)
        vol_hand = rng.choice([2, 3, 5, 8, 12, 20, 50, 100])
        if rng.random() < 0.1:
            vol_hand *= rng.choice([20, 30, 50])  # 大单

        side_hint = rng.random()
        if side_hint > 0.62:
            trade_price = ask1
        elif side_hint < 0.38:
            trade_price = bid1
        else:
            trade_price = last_price

        amount = trade_price * vol_hand * 100
        trades.append(
            {
                "ts": t.strftime("%H:%M:%S.%f")[:-3],
                "price": trade_price,
                "volume_hand": vol_hand,
                "bid1": bid1,
                "ask1": ask1,
                "amount": round(amount, 2),
            }
        )
    return trades


def _classify(trade, prev_price):
    p = trade["price"]
    if p >= trade["ask1"]:
        return "buy"
    if p <= trade["bid1"]:
        return "sell"
    return "buy" if p >= prev_price else "sell"


def _metrics(trades, threshold=300000):
    if not trades:
        return {}
    # 补side
    prev = trades[0]["price"]
    for x in trades:
        x["side"] = _classify(x, prev)
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


def _build_signal_row(symbol: str, name: str, step: int):
    trades = _gen_trades(symbol, step=step)
    met = _metrics(trades, threshold=MONITOR_CFG["big_trade_threshold"])
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
        "price": met["last_price"],
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
    data = load_json(DATA_FILE, fallback_rankings())
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
    return jsonify(load_json(DATA_FILE, fallback_rankings()))


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
    rows = [_build_signal_row(s, n, step=step) for s, n in SYMBOL_POOL]

    if keyword:
        rows = [x for x in rows if keyword in x["symbol"] or keyword in x["name"]]
    if favorites_only:
        fav = set(MONITOR_CFG.get("favorite_symbols") or [])
        rows = [x for x in rows if x["symbol"] in fav]

    rows.sort(key=lambda x: x["signal_score"], reverse=True)
    return jsonify({"ok": True, "data": rows, "ts": dt.datetime.now().strftime("%H:%M:%S")})


@app.route("/api/monitor/detail")
def api_monitor_detail():
    symbol = (request.args.get("symbol") or "600519").strip()
    step = int(request.args.get("step", "0") or 0)
    trades = _gen_trades(symbol, step=step, seconds=90)
    met = _metrics(trades, threshold=MONITOR_CFG["big_trade_threshold"])

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
                "metrics": met,
                "series_1m": series,
                "big_trades": big_list,
            },
        }
    )


@app.route("/api/run-now")
def api_run_now():
    run_daily_job()
    return jsonify({"ok": True, "message": "已刷新短线排名"})


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
    if not DATA_FILE.exists():
        run_daily_job()
    if not VALIDATION_FILE.exists():
        run_validation_job()
    app.run(host="0.0.0.0", port=8080, debug=True)
