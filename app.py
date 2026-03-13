from __future__ import annotations

import json
from pathlib import Path

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)
app.json.ensure_ascii = False
DATA_FILE = Path(__file__).parent / "data" / "latest_recommendations.json"
VALIDATION_FILE = Path(__file__).parent / "data" / "validation.json"


def run_daily_job():
    from strategy import AShareShortTermEngine

    engine = AShareShortTermEngine(top_n=20, candidate_n=600)
    engine.run_daily()


def run_validation_job():
    from strategy import AShareShortTermEngine

    engine = AShareShortTermEngine(top_n=20, candidate_n=600)
    engine.build_validation(recent_days=140)


def load_json(path: Path, default: dict):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


@app.route("/")
def index():
    data = load_json(
        DATA_FILE,
        {
            "updated_at": "尚未生成",
            "strategy": {"name": "A股全市场短线交易参考排名", "version": "3.0", "principle": "待生成", "risk_note": "仅供研究"},
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
    return jsonify(load_json(DATA_FILE, {}))


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
