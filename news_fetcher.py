from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List
import time
import re
from urllib.parse import quote_plus
import feedparser

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

CN_SOURCES = [
    {"name": "新华社", "url": "https://www.xinhuanet.com/politics/news_politics.xml"},
    {"name": "中国新闻网", "url": "https://www.chinanews.com.cn/rss/china.xml"},
    {"name": "澎湃新闻", "url": "https://www.thepaper.cn/rss_news.xml"},
    {"name": "财联社", "url": "https://www.cls.cn/rss.xml"},
    {"name": "界面新闻", "url": "https://www.jiemian.com/rss/index.xml"},
]

GLOBAL_SOURCES = [
    {"name": "Reuters", "url": "https://feeds.reuters.com/reuters/worldNews"},
    {"name": "Bloomberg", "url": "https://feeds.bloomberg.com/markets/news.rss"},
    {"name": "CNBC", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"},
    {"name": "BBC", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "Financial Times", "url": "https://www.ft.com/world?format=rss"},
]

_TRANSLATE_CACHE: Dict[str, str] = {}
_NEWS_CACHE: Dict[str, object] = {"ts": 0.0, "data": None}
_REGION_NEWS_CACHE: Dict[str, Dict[str, object]] = {}


def _clean_text(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _to_zh(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if re.search(r"[\u4e00-\u9fff]", text):
        return text
    if text in _TRANSLATE_CACHE:
        return _TRANSLATE_CACHE[text]
    if GoogleTranslator is None:
        return text
    try:
        out = GoogleTranslator(source="auto", target="zh-CN").translate(text)
        _TRANSLATE_CACHE[text] = out
        return out
    except Exception:
        return text


def _compress_sentences(text: str, max_sent: int = 2, max_len: int = 120) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    # 中英文分句
    parts = re.split(r"(?<=[。！？!?\.])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    # 过滤低信息片段
    bad = ["点击", "查看更多", "原标题", "来源", "责任编辑", "广告", "免责声明"]
    scored = []
    for p in parts:
        if any(b in p for b in bad):
            continue
        score = 0
        # 信息密度特征：数字、百分比、金额、时间、专有名词线索
        score += len(re.findall(r"\d+\.?\d*%?", p)) * 2
        score += len(re.findall(r"(亿元|万美元|万亿|bps|基点|同比|环比|季度|年内)", p)) * 2
        score += 1 if len(p) >= 24 else 0
        scored.append((score, p))
    if not scored:
        scored = [(0, parts[0])]
    top = [p for _, p in sorted(scored, key=lambda x: x[0], reverse=True)[:max_sent]]
    out = " ".join(top)
    return out[:max_len]


def _make_abstract(title: str, summary: str, title_zh: str) -> str:
    s = _clean_text(summary)
    core = _compress_sentences(s, max_sent=2, max_len=130)

    # 模板：事件 + 关键信息 + 影响提示
    if core:
        # 去重：避免与标题重复太多
        if core.replace(" ", "")[:20] in (title_zh or "").replace(" ", ""):
            return f"要点：{core}"
        return f"事件：{title_zh}；要点：{core}"

    # 无summary时，至少给结构化提示
    hint = "关注其对市场情绪、相关行业与资产价格的短期影响。"
    return f"事件：{title_zh}。{hint}"


SECTOR_RULES = {
    "AI/算力": ["人工智能", "大模型", "算力", "数据中心", "云计算", "GPU", "芯片"],
    "半导体": ["半导体", "晶圆", "封测", "EDA", "存储", "先进制程"],
    "新能源": ["光伏", "风电", "储能", "锂电", "新能源车", "电池"],
    "金融": ["银行", "券商", "保险", "利率", "降准", "降息", "国债", "流动性"],
    "地产基建": ["地产", "楼市", "基建", "城投", "保障房", "施工"],
    "消费": ["消费", "零售", "白酒", "餐饮", "家电", "旅游"],
    "医药": ["医药", "创新药", "医疗器械", "集采", "医保"],
    "大宗商品": ["原油", "黄金", "煤炭", "钢铁", "有色", "铜", "铝"],
    "军工": ["军工", "国防", "导弹", "航天", "舰艇"],
}

POS_HINT_WORDS = ["超预期", "增长", "上调", "突破", "回购", "增持", "改善", "宽松", "利好", "提振"]
NEG_HINT_WORDS = ["下调", "下滑", "亏损", "诉讼", "处罚", "减持", "收紧", "风险", "违约", "制裁"]

EVENT_RULES = {
    "宏观宽松": ["降息", "降准", "流动性", "宽松", "刺激政策"],
    "宏观收紧": ["加息", "通胀超预期", "收紧", "缩表"],
    "监管风险": ["处罚", "立案", "问询", "反垄断", "监管升级"],
    "业绩超预期": ["业绩超预期", "预增", "上修", "扭亏"],
    "业绩不及预期": ["预减", "亏损", "下修", "不及预期"],
    "地缘冲突": ["冲突", "战争", "袭击", "停火", "谈判"],
    "商品波动": ["油价", "金价", "铜价", "煤价", "天然气"],
}


def _impact_assessment(text: str) -> str:
    p = sum(text.count(w) for w in POS_HINT_WORDS)
    n = sum(text.count(w) for w in NEG_HINT_WORDS)
    if p - n >= 2:
        return "利多"
    if n - p >= 2:
        return "利空"
    return "中性"


def _affected_sectors(text: str) -> List[str]:
    out = []
    for k, words in SECTOR_RULES.items():
        if any(w in text for w in words):
            out.append(k)
    return out[:3]


def _detect_events(text: str) -> List[str]:
    events = []
    for evt, kws in EVENT_RULES.items():
        if any(k in text for k in kws):
            events.append(evt)
    return events[:2]


def _trading_hint(impact: str, sectors: List[str], events: List[str]) -> str:
    sec = "、".join(sectors) if sectors else "相关板块"
    evt = "、".join(events) if events else "一般新闻驱动"

    # 事件驱动式提示（更明确）
    if "宏观宽松" in events:
        return f"交易提示：事件属{evt}，优先看多{sec}中的高弹性方向；若开盘后1小时放量站上早盘高点可跟随，否则不追高。"
    if "宏观收紧" in events or "监管风险" in events:
        return f"交易提示：事件属{evt}，{sec}偏防守；反抽不放量以减仓为主，跌破前低需止损。"
    if "业绩超预期" in events and impact != "利空":
        return f"交易提示：事件属{evt}，可观察{sec}是否出现‘放量突破+回踩不破’，满足再参与。"
    if "业绩不及预期" in events:
        return f"交易提示：事件属{evt}，避免左侧抄底；仅在恐慌后缩量止跌并收复关键均线时小仓试错。"
    if "地缘冲突" in events or "商品波动" in events:
        return f"交易提示：事件属{evt}，优先交易受益链条（如资源/避险）并回避受压链条，盘中以资金流向确认。"

    if impact == "利多":
        return f"交易提示：{evt}偏正向，关注{sec}龙头是否放量领涨；只做强不做弱。"
    if impact == "利空":
        return f"交易提示：{evt}偏负向，{sec}以风控优先；冲高缩量或跌破关键位应执行止损。"
    return f"交易提示：{evt}影响有限，{sec}先观察；等待量价共振与板块共振后再出手。"


def _recent(entries, hours: int = 24, translate_titles: bool = False, max_translate: int = 2) -> List[Dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out = []
    translated_count = 0
    for e in entries:
        dt = None
        if getattr(e, "published_parsed", None):
            dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        if dt and dt < cutoff:
            continue

        title = _clean_text(getattr(e, "title", "(无标题)"))
        summary_raw = _clean_text(getattr(e, "summary", "") or "")
        if translate_titles and translated_count < max_translate:
            title_zh = _to_zh(title)
            translated_count += 1
        else:
            title_zh = title
        abstract = _make_abstract(title, summary_raw, title_zh)
        full_text = f"{title_zh} {summary_raw}"
        impact = _impact_assessment(full_text)
        sectors = _affected_sectors(full_text)
        events = _detect_events(full_text)
        hint = _trading_hint(impact, sectors, events)

        now_utc = datetime.now(timezone.utc)
        recency = 0
        if dt:
            hours = max((now_utc - dt).total_seconds() / 3600, 0)
            recency = max(0, 60 - int(hours * 2))  # 越新分越高
        impact_score = {"利多": 25, "中性": 15, "利空": 10}.get(impact, 12)
        heat_score = int(recency + impact_score + len(events) * 8 + len(sectors) * 4)

        out.append(
            {
                "title": title,
                "title_zh": title_zh,
                "summary": summary_raw[:260],
                "abstract": abstract,
                "impact": impact,
                "sectors": sectors,
                "events": events,
                "trading_hint": hint,
                "link": getattr(e, "link", ""),
                "published": dt.isoformat() if dt else "",
                "published_ts": int(dt.timestamp()) if dt else 0,
                "heat": heat_score,
            }
        )
    return out


REGION_PRESETS = {
    "china": {"name": "中国", "query": "China economy policy markets", "gl": "CN", "hl": "zh-CN", "ceid": "CN:zh-Hans", "keywords": ["china","chinese","beijing","shanghai","中国","北京"]},
    "us": {"name": "美国", "query": "United States economy policy markets", "gl": "US", "hl": "en-US", "ceid": "US:en", "keywords": ["united states","u.s.","america","fed","washington","美国"]},
    "europe": {"name": "欧洲", "query": "Europe economy policy ECB markets", "gl": "GB", "hl": "en-GB", "ceid": "GB:en", "keywords": ["europe","ecb","eu","euro zone","欧洲"]},
    "japan": {"name": "日本", "query": "Japan economy BoJ policy markets", "gl": "JP", "hl": "ja", "ceid": "JP:ja", "keywords": ["japan","boj","tokyo","日本"]},
    "middleeast": {"name": "中东", "query": "Middle East geopolitics oil markets", "gl": "AE", "hl": "en", "ceid": "AE:en", "keywords": ["middle east","saudi","iran","israel","oil","中东"]},
    "india": {"name": "印度", "query": "India economy RBI policy markets", "gl": "IN", "hl": "en-IN", "ceid": "IN:en"},
    "russia": {"name": "俄罗斯", "query": "Russia economy central bank policy", "gl": "RU", "hl": "en", "ceid": "RU:en"},
    "brazil": {"name": "巴西", "query": "Brazil economy central bank policy", "gl": "BR", "hl": "pt-BR", "ceid": "BR:pt-419"},
    "canada": {"name": "加拿大", "query": "Canada economy BoC policy markets", "gl": "CA", "hl": "en-CA", "ceid": "CA:en"},
    "australia": {"name": "澳大利亚", "query": "Australia economy RBA policy markets", "gl": "AU", "hl": "en-AU", "ceid": "AU:en"},
    "korea": {"name": "韩国", "query": "South Korea economy central bank policy", "gl": "KR", "hl": "ko", "ceid": "KR:ko"},
    "singapore": {"name": "新加坡", "query": "Singapore MAS policy economy markets", "gl": "SG", "hl": "en-SG", "ceid": "SG:en"},
    "uk": {"name": "英国", "query": "United Kingdom economy BoE policy markets", "gl": "GB", "hl": "en-GB", "ceid": "GB:en"},
    "germany": {"name": "德国", "query": "Germany economy markets", "gl": "DE", "hl": "de", "ceid": "DE:de"},
    "france": {"name": "法国", "query": "France economy markets", "gl": "FR", "hl": "fr", "ceid": "FR:fr"},
    "italy": {"name": "意大利", "query": "Italy economy markets", "gl": "IT", "hl": "it", "ceid": "IT:it"},
    "spain": {"name": "西班牙", "query": "Spain economy markets", "gl": "ES", "hl": "es", "ceid": "ES:es"},
    "mexico": {"name": "墨西哥", "query": "Mexico economy central bank policy", "gl": "MX", "hl": "es", "ceid": "MX:es-419"},
    "indonesia": {"name": "印尼", "query": "Indonesia economy central bank policy", "gl": "ID", "hl": "id", "ceid": "ID:id"},
    "turkiye": {"name": "土耳其", "query": "Turkey economy central bank policy", "gl": "TR", "hl": "tr", "ceid": "TR:tr"},
    "saudi": {"name": "沙特", "query": "Saudi Arabia economy oil policy", "gl": "SA", "hl": "ar", "ceid": "SA:ar"},
    "uae": {"name": "阿联酋", "query": "UAE economy policy markets", "gl": "AE", "hl": "en", "ceid": "AE:en"},
}


def _google_rss(query: str, hl: str, gl: str, ceid: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"


def _region_cfg(region: str) -> dict:
    key = (region or "china").lower().strip()
    if key in REGION_PRESETS:
        cfg = dict(REGION_PRESETS[key])
        cfg["key"] = key
        cfg.setdefault("keywords", [cfg.get("name", "").lower()])
        return cfg
    # 任意国家名兜底：用 Google RSS 搜索生成
    pretty = region.strip() or "Global"
    return {
        "key": key or "global",
        "name": pretty,
        "query": f"{pretty} economy policy markets",
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
        "keywords": [pretty.lower()],
    }


def fetch_region_news(region: str = "china", limit: int = 20, sort_by: str = "heat") -> Dict:
    cfg = _region_cfg(region)
    key = cfg["key"]
    kws = [k.lower() for k in (cfg.get("keywords") or []) if k]

    now_ts = time.time()
    ck = f"{key}:{limit}:{sort_by}"
    cache = _REGION_NEWS_CACHE.get(ck)
    if cache and (now_ts - float(cache.get("ts", 0.0)) < 180):
        return cache.get("data")  # type: ignore[return-value]

    primary_feed = _google_rss(cfg["query"], cfg["hl"], cfg["gl"], cfg["ceid"])
    fallback_feeds = [
        "https://feeds.reuters.com/reuters/worldNews",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
    ]

    rows = []
    seen = set()

    # 先尽量保留“该国家”主查询结果
    try:
        d0 = feedparser.parse(primary_feed)
        primary_items = _recent(d0.entries, translate_titles=True, max_translate=10)
    except Exception:
        primary_items = []
    for it in primary_items:
        k = (it.get("title") or "", it.get("link") or "")
        if k in seen:
            continue
        seen.add(k)
        rows.append(it)
        if len(rows) >= max(8, limit):
            break

    # 再用全球源补齐
    for feed in fallback_feeds:
        try:
            d = feedparser.parse(feed)
            items = _recent(d.entries, translate_titles=True, max_translate=8)
        except Exception:
            items = []
        for it in items:
            k = (it.get("title") or "", it.get("link") or "")
            if k in seen:
                continue
            seen.add(k)
            rows.append(it)
            if len(rows) >= limit * 2:
                break

    # 地区相关性打分，避免“点国家但新闻几乎不变”
    for it in rows:
        blob = f"{it.get('title','')} {it.get('title_zh','')} {it.get('summary','')}".lower()
        rel = sum(1 for k in kws if k and k in blob)
        it["region_rel"] = rel

    if sort_by == "time":
        rows = sorted(rows, key=lambda x: (x.get("region_rel", 0), x.get("published_ts", 0)), reverse=True)
    elif sort_by == "impact":
        rank = {"利多": 3, "中性": 2, "利空": 1}
        rows = sorted(rows, key=lambda x: (x.get("region_rel", 0), rank.get(x.get("impact", "中性"), 2), x.get("heat", 0)), reverse=True)
    else:
        rows = sorted(rows, key=lambda x: (x.get("region_rel", 0), x.get("heat", 0), x.get("published_ts", 0)), reverse=True)

    rows = rows[:limit]
    data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "region": key,
        "region_name": cfg["name"],
        "items": rows,
    }
    _REGION_NEWS_CACHE[ck] = {"ts": now_ts, "data": data}
    return data


def fetch_news_24h(limit_per_source: int = 8) -> Dict:
    # 180秒缓存，避免每次打开页面都重新抓取+翻译导致卡顿
    now_ts = time.time()
    if _NEWS_CACHE.get("data") and (now_ts - float(_NEWS_CACHE.get("ts", 0.0)) < 180):
        return _NEWS_CACHE["data"]  # type: ignore[return-value]

    def collect(sources, translate_titles: bool = False):
        data = []
        for s in sources:
            try:
                d = feedparser.parse(s["url"])
                items = _recent(d.entries, translate_titles=translate_titles, max_translate=2)[:limit_per_source]
            except Exception:
                items = []
            data.append({"source": s["name"], "url": s["url"], "items": items})
        return data

    cn = collect(CN_SOURCES, translate_titles=False)
    global_ = collect(GLOBAL_SOURCES, translate_titles=True)
    data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cn": cn,
        "global_news": global_,
    }
    _NEWS_CACHE["ts"] = now_ts
    _NEWS_CACHE["data"] = data
    return data
