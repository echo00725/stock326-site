from __future__ import annotations

import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import akshare as ak
import numpy as np
import pandas as pd

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

DATA_DIR = Path(__file__).parent / "data"
HISTORY_DIR = DATA_DIR / "history"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_WORDS = ["增长", "利好", "回购", "中标", "突破", "增持", "盈利", "预增", "订单", "创新", "提价", "超预期"]
NEGATIVE_WORDS = ["减持", "亏损", "诉讼", "下滑", "预减", "风险", "处罚", "问询", "暴雷", "商誉", "违约", "立案"]
EVENT_TAG_RULES = {
    "业绩": ["业绩", "快报", "季报", "年报", "预增", "预减", "扭亏", "亏损"],
    "政策": ["政策", "监管", "指导意见", "鼓励", "限制", "发改委", "工信部"],
    "减持": ["减持", "解禁", "大宗交易"],
    "诉讼": ["诉讼", "仲裁", "立案", "处罚", "问询", "调查"],
}


@dataclass
class Pick:
    code: str
    name: str
    score: float
    close: float
    reasons: List[str]
    factors: Dict[str, float]


class FinSentimentModel:
    """BERT情绪接口（可插拔）+ 词典回退"""

    def __init__(self):
        self.pipeline = None
        try:
            from transformers import pipeline  # type: ignore

            self.pipeline = pipeline(
                "sentiment-analysis",
                model="uer/roberta-base-finetuned-jd-binary-chinese",
                tokenizer="uer/roberta-base-finetuned-jd-binary-chinese",
            )
        except Exception:
            self.pipeline = None

    def score(self, texts: List[str]) -> float:
        if not texts:
            return 0.0
        merged = " ".join(texts)[:1500]

        # BERT可用时优先
        if self.pipeline is not None:
            try:
                out = self.pipeline(merged[:500], truncation=True)[0]
                label = str(out.get("label", "")).lower()
                prob = float(out.get("score", 0.5))
                return (prob * 2 - 1) if ("positive" in label or "1" in label) else (1 - prob) * 2 - 1
            except Exception:
                pass

        pos = sum(merged.count(w) for w in POSITIVE_WORDS)
        neg = sum(merged.count(w) for w in NEGATIVE_WORDS)
        raw = (pos - neg) / (pos + neg + 3)
        return float(max(-1.0, min(1.0, raw)))

    def event_tags(self, texts: List[str]) -> List[str]:
        merged = " ".join(texts)
        tags = []
        for tag, words in EVENT_TAG_RULES.items():
            if any(w in merged for w in words):
                tags.append(tag)
        return tags


class AShareShortTermEngine:
    def __init__(self, top_n: int = 20, candidate_n: int = 600):
        self.top_n = top_n
        self.candidate_n = candidate_n
        self.sentiment_model = FinSentimentModel()

    def run_daily(self) -> Dict:
        try:
            universe = self._get_all_a_universe()
            universe = universe.sort_values("amount", ascending=False).head(self.candidate_n)

            regime, hs300_ret20 = self._market_regime()
            picks: List[Pick] = []

            for _, row in universe.iterrows():
                code = str(row["code"])
                name = str(row["name"])
                try:
                    hist = self._hist(code)
                    if hist is None or len(hist) < 150:
                        continue

                    news_texts = self._news_for_code(code)
                    feats = self._calc_features(hist, hs300_ret20, float(row.get("turnover", 0.0)), news_texts=news_texts)
                    score = self._calc_score(feats, regime)
                    reasons = self._build_reasons(feats, regime)

                    picks.append(
                        Pick(
                            code=code,
                            name=name,
                            score=round(score, 2),
                            close=round(feats["close"], 2),
                            reasons=reasons,
                            factors={k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in feats.items()},
                        )
                    )
                except Exception:
                    continue

            picks = sorted(picks, key=lambda x: x.score, reverse=True)[: self.top_n]
            flow = self._industry_flow_top()

            result = {
                "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "strategy": {
                    "name": "A股全市场短线交易参考排名",
                    "version": "3.0",
                    "principle": "全A股池(按流动性筛选) + 趋势动量 + 突破 + 量价 + 新闻BERT情绪/事件 + 风控",
                    "risk_note": "仅供研究与学习，不构成投资建议。",
                    "trading_rules": {
                        "entry": "T日收盘生成信号，T+1以开盘价（或开盘后30分钟均价）入场",
                        "holding": "5/10/20日三档，默认10日",
                        "stop_loss": "固定-6% 或 1.8*ATR(14) 二者取更严格",
                        "take_profit": "+12% 或 2.5*ATR(14) 二者取更早",
                        "position": "单票上限10%，组合最大回撤阈值12%触发降仓",
                    },
                },
                "industry_flow": flow,
                "picks": [
                    {
                        "code": p.code,
                        "name": p.name,
                        "score": p.score,
                        "close": p.close,
                        "reasons": p.reasons,
                        "factors": p.factors,
                    }
                    for p in picks
                ],
            }
            self._save_json(DATA_DIR / "latest_recommendations.json", result)
            self._save_json(HISTORY_DIR / f"{dt.date.today()}.json", result)
            return result
        except Exception:
            return self._fallback_result()

    def build_validation(self, recent_days: int = 120) -> Dict:
        """滚动回测+因子归因+近30天跟踪（轻量版）"""
        try:
            universe = self._get_all_a_universe().sort_values("amount", ascending=False).head(200)
            codes = universe["code"].astype(str).tolist()

            bars = {}
            for c in codes:
                try:
                    h = self._hist(c)
                    if h is not None and len(h) > recent_days + 60:
                        bars[c] = h.tail(recent_days + 60).copy()
                except Exception:
                    continue
            if len(bars) < 30:
                raise RuntimeError("可回测样本不足")

            # 交易日基准：沪深300交易日，避免对所有股票求交集导致样本塌缩
            idx = ak.stock_zh_index_daily(symbol="sh000300")
            dates = idx["date"].astype(str).tolist()[-recent_days:]
            if len(dates) < 40:
                raise RuntimeError("可用交易日不足")

            curve = [1.0]
            daily_rows = []
            factor_rows = []

            for i in range(30, len(dates) - 11):
                d = dates[i]
                next_d = dates[i + 1]
                exit_d = dates[i + 10]  # 默认10日持有

                scores = []
                for c, df in bars.items():
                    sl = df[df["date"].astype(str) <= d].copy()
                    if len(sl) < 80:
                        continue
                    hs300_ret20 = 0.0
                    feats = self._calc_features(sl, hs300_ret20, turnover=0.0, with_news=False)
                    s = self._calc_score(feats, regime="neutral")
                    scores.append((c, s, feats))

                if len(scores) < 20:
                    continue
                scores.sort(key=lambda x: x[1], reverse=True)
                top = scores[:10]

                rets = []
                for c, _, feats in top:
                    df_c = bars[c].copy()
                    df_c["date"] = df_c["date"].astype(str)
                    df_c = df_c.set_index("date")
                    try:
                        entry = float(df_c.loc[next_d, "open"])
                        # 止损止盈（固定比例）
                        slice_h = bars[c][(bars[c]["date"].astype(str) >= next_d) & (bars[c]["date"].astype(str) <= exit_d)]
                        low_min = float(slice_h["low"].min())
                        high_max = float(slice_h["high"].max())
                        exit_px = float(df_c.loc[exit_d, "close"])
                        sl_px = entry * (1 - 0.06)
                        tp_px = entry * (1 + 0.12)
                        if low_min <= sl_px:
                            exit_px = sl_px
                        elif high_max >= tp_px:
                            exit_px = tp_px
                        rets.append(exit_px / entry - 1)

                        factor_rows.append(
                            {
                                "date": d,
                                "trend": feats["trend_score"],
                                "momentum": feats["momentum_score"],
                                "breakout": feats["breakout_score"],
                                "activity": feats["activity_score"],
                                "sentiment": feats.get("sentiment", 0.0),
                                "ret": exit_px / entry - 1,
                            }
                        )
                    except Exception:
                        continue

                if not rets:
                    continue
                day_ret = float(np.mean(rets))
                curve.append(curve[-1] * (1 + day_ret))
                daily_rows.append({"date": d, "ret": day_ret, "equity": curve[-1]})

            if len(daily_rows) < 20:
                raise RuntimeError("回测结果样本不足")

            daily_df = pd.DataFrame(daily_rows)
            factor_df = pd.DataFrame(factor_rows)

            max_dd = self._max_drawdown(np.array(daily_df["equity"]))
            ann = (daily_df["equity"].iloc[-1] ** (252 / len(daily_df))) - 1
            win_rate = float((daily_df["ret"] > 0).mean())

            attrib = {}
            if not factor_df.empty:
                for f in ["trend", "momentum", "breakout", "activity", "sentiment"]:
                    try:
                        attrib[f] = float(factor_df[f].corr(factor_df["ret"]))
                    except Exception:
                        attrib[f] = 0.0

            recent30 = daily_df.tail(30).to_dict(orient="records")
            validation = {
                "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "annualized_return": round(float(ann), 4),
                    "max_drawdown": round(float(max_dd), 4),
                    "win_rate": round(win_rate, 4),
                    "sample_days": int(len(daily_df)),
                },
                "equity_curve": daily_df.to_dict(orient="records"),
                "factor_attribution": attrib,
                "recent_30d_tracking": recent30,
                "note": "回测为轻量近似（含止盈止损与次日开盘入场），实盘需计入滑点手续费。",
            }
            self._save_json(DATA_DIR / "validation.json", validation)
            return validation
        except Exception as e:
            fallback = {
                "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {"annualized_return": 0, "max_drawdown": 0, "win_rate": 0, "sample_days": 0},
                "equity_curve": [],
                "factor_attribution": {},
                "recent_30d_tracking": [],
                "note": f"回测失败：{e}",
            }
            self._save_json(DATA_DIR / "validation.json", fallback)
            return fallback

    def _get_all_a_universe(self) -> pd.DataFrame:
        df = None
        for _ in range(3):
            try:
                df = ak.stock_zh_a_spot_em().rename(
                    columns={"代码": "code", "名称": "name", "成交额": "amount", "换手率": "turnover", "最新价": "price"}
                )
                break
            except Exception:
                time.sleep(1.2)

        if df is None:
            for _ in range(3):
                try:
                    df = ak.stock_zh_a_spot().rename(
                        columns={"代码": "code", "名称": "name", "成交额": "amount", "换手率": "turnover", "最新价": "price"}
                    )
                    break
                except Exception:
                    time.sleep(1.2)

        if df is None:
            # 备用通道：直接使用交易所代码表拼全A股票池（不依赖 EastMoney/Sina 实时接口）
            try:
                sz = ak.stock_info_sz_name_code().rename(columns={"A股代码": "code", "A股简称": "name"})[["code", "name"]]
                sh = ak.stock_info_sh_name_code().rename(columns={"证券代码": "code", "证券简称": "name"})[["code", "name"]]
                df = pd.concat([sz, sh], ignore_index=True)
                df["amount"] = 0.0
                df["turnover"] = 0.0
            except Exception:
                cache_path = DATA_DIR / "universe_cache.json"
                if cache_path.exists():
                    return pd.DataFrame(json.loads(cache_path.read_text(encoding="utf-8")))
                raise RuntimeError("全A股票池获取失败")

        df["code"] = df["code"].astype(str)
        df["name"] = df["name"].astype(str)
        df = df[~df["name"].str.contains("ST|退", na=False)]
        df = df[df["code"].str.startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688"))]
        df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0)
        df["turnover"] = pd.to_numeric(df.get("turnover", 0), errors="coerce").fillna(0)
        out = df[["code", "name", "amount", "turnover"]].copy()
        (DATA_DIR / "universe_cache.json").write_text(out.to_json(orient="records", force_ascii=False), encoding="utf-8")
        return out

    @staticmethod
    def _market_regime() -> tuple[str, float]:
        idx = ak.stock_zh_index_daily(symbol="sh000300").copy()
        idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
        idx = idx.dropna(subset=["close"])
        close = idx["close"]
        ma60 = float(close.rolling(60).mean().iloc[-1])
        c = float(close.iloc[-1])
        ret20 = float(c / close.iloc[-21] - 1)
        if c > ma60 and ret20 > 0.015:
            return "bull", ret20
        if c < ma60 and ret20 < -0.015:
            return "bear", ret20
        return "neutral", ret20

    @staticmethod
    def _hist(code: str) -> Optional[pd.DataFrame]:
        h = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        if h is None or len(h) < 80:
            return None
        h = h.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"})
        for col in ["open", "close", "high", "low", "volume"]:
            h[col] = pd.to_numeric(h[col], errors="coerce")
        h = h.dropna(subset=["open", "close", "high", "low", "volume"])
        return h

    def _calc_features(self, hist: pd.DataFrame, hs300_ret20: float, turnover: float, with_news: bool = True, news_texts: Optional[List[str]] = None) -> Dict:
        close = hist["close"]
        c = float(close.iloc[-1])
        ma10 = float(close.rolling(10).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])
        ret5 = float(c / close.iloc[-6] - 1)
        ret20 = float(c / close.iloc[-21] - 1)
        excess20 = ret20 - hs300_ret20
        max60 = float(close.tail(60).max())
        breakout = c / max60 - 1
        vol_ratio = float(hist["volume"].iloc[-1] / (hist["volume"].rolling(5).mean().iloc[-1] + 1e-9))
        vol20 = float(close.pct_change().rolling(20).std().iloc[-1])
        atr14 = self._atr(hist.tail(30), 14)
        drawdown = self._drawdown_penalty(close.tail(60))

        trend_score = 0.3 * (1 if c > ma10 else 0) + 0.35 * (1 if c > ma20 else 0) + 0.35 * (1 if c > ma60 else 0)
        momentum_score = self._clip((0.6 * ret5 + 0.4 * excess20) / 0.18, -1, 1)
        breakout_score = self._clip((breakout + 0.06) / 0.06, -1, 1)
        activity_score = 0.7 * self._clip((vol_ratio - 1) / 1.2, -1, 1) + 0.3 * self._clip((turnover - 1.2) / 6, -1, 1)
        risk_penalty = 0.6 * self._clip((vol20 - 0.03) / 0.03, 0, 1) + 0.4 * drawdown

        sentiment = 0.0
        tags: List[str] = []
        if with_news:
            news_texts = news_texts or []
            sentiment = self.sentiment_model.score(news_texts)
            tags = self.sentiment_model.event_tags(news_texts)

        return {
            "close": c,
            "ma10": ma10,
            "ma20": ma20,
            "ma60": ma60,
            "ret5": ret5,
            "ret20": ret20,
            "excess20": excess20,
            "vol_ratio": vol_ratio,
            "turnover": turnover,
            "atr14": atr14,
            "trend_score": trend_score,
            "momentum_score": momentum_score,
            "breakout_score": breakout_score,
            "activity_score": activity_score,
            "risk_penalty": risk_penalty,
            "sentiment": sentiment,
            "event_tags": tags,
        }

    @staticmethod
    def _news_texts(hist: pd.DataFrame) -> List[str]:
        # 这里只返回空，真实新闻由 code 在上层取；保持接口可扩展
        return []

    def _news_for_code(self, code: str) -> List[str]:
        try:
            df = ak.stock_news_em(symbol=code)
            if df is None or len(df) == 0:
                return []
            rows = df.head(12)
            return [" ".join([str(r.get("新闻标题", "")), str(r.get("新闻内容", ""))]) for _, r in rows.iterrows()]
        except Exception:
            return []

    def _calc_score(self, f: Dict, regime: str) -> float:
        raw = (
            24 * f["trend_score"]
            + 24 * f["momentum_score"]
            + 16 * f["breakout_score"]
            + 14 * f["activity_score"]
            + 12 * f["sentiment"]
            - 12 * f["risk_penalty"]
        )
        if regime == "bull":
            raw *= 1.05
        elif regime == "bear":
            raw *= 0.85
        return float(raw)

    def _build_reasons(self, f: Dict, regime: str) -> List[str]:
        tag_text = "、".join(f.get("event_tags", [])) if f.get("event_tags") else "无明显高风险事件标签"
        return [
            f"短线环境：{regime}；5日动量 {f['ret5']:.2%}，20日超额 {f['excess20']:.2%}",
            f"趋势结构：收盘{f['close']:.2f}，MA10/20/60=({f['ma10']:.2f}/{f['ma20']:.2f}/{f['ma60']:.2f})",
            f"资金活跃：量比近似{f['vol_ratio']:.2f}，换手率{f['turnover']:.2f}%",
            f"消息情绪：{f['sentiment']:.2f}；事件标签：{tag_text}",
            f"风控参数：ATR14={f['atr14']:.3f}，建议按-6%或1.8*ATR止损",
        ]

    def _industry_flow_top(self) -> List[Dict]:
        try:
            df = ak.stock_fund_flow_industry()
            # 列：行业、净额...
            df = df.rename(columns={"行业": "industry", "净额": "net_inflow"})
            df["net_inflow"] = pd.to_numeric(df["net_inflow"], errors="coerce")
            top = df.sort_values("net_inflow", ascending=False).head(10)
            return top[["industry", "net_inflow", "领涨股", "领涨股-涨跌幅"]].to_dict(orient="records")
        except Exception:
            return []

    def _fallback_result(self) -> Dict:
        result = {
            "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": {"name": "A股全市场短线交易参考排名", "version": "3.0", "principle": "数据源异常降级", "risk_note": "仅供研究"},
            "industry_flow": [],
            "picks": [],
        }
        self._save_json(DATA_DIR / "latest_recommendations.json", result)
        return result

    @staticmethod
    def _atr(hist: pd.DataFrame, n: int = 14) -> float:
        h = hist.copy()
        prev_close = h["close"].shift(1)
        tr = pd.concat([
            (h["high"] - h["low"]).abs(),
            (h["high"] - prev_close).abs(),
            (h["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(n).mean().iloc[-1]) if len(tr) >= n else 0.0

    @staticmethod
    def _drawdown_penalty(close_60: pd.Series) -> float:
        roll_max = close_60.cummax()
        dd = (close_60 / roll_max - 1).min()
        return float(max(0.0, min(1.0, abs(dd) / 0.2)))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1
        return float(dd.min())

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    @staticmethod
    def _save_json(path: Path, data: Dict) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
