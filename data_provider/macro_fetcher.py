# -*- coding: utf-8 -*-
"""
MacroFetcher — 宏观与大宗商品行情数据源

与 BaseFetcher 体系的关系：
- 本模块**不**继承 BaseFetcher，也不注册进 DataFetcherManager 的个股故障切换链。
  原因：BaseFetcher 的契约面向个股（stock_code / 股票名称 / 基本面适配器），
  宏观序列（美元指数、国债收益率、原油、贵金属）不属于该语义，
  混入个股优先级链会污染故障切换顺序。
- 但返回的 DataFrame 仍沿用 data_provider.base.STANDARD_COLUMNS 的列约定，
  便于下游复用既有的技术分析与报告代码。

数据源：
1. yfinance（默认，无需配置）—— 行情类序列：贵金属、原油、美元指数、国债收益率
2. FRED（可选，配置 FRED_API_KEY 后启用）—— 官方宏观序列：PCE、CPI、失业率等

遵循"不配置也可运行，配置后增强能力"：未配置 FRED_API_KEY 时，
行情类序列照常工作，仅官方宏观序列不可用。
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base import STANDARD_COLUMNS, DataFetchError

logger = logging.getLogger(__name__)

_FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_EIA_BASE_URL = "https://api.eia.gov/v2/seriesid"
# CFTC Socrata 公共端点：无需 API Key
_CFTC_BASE_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
_HTTP_TIMEOUT = 15
_MAX_RETRIES = 3


@dataclass(frozen=True)
class MacroSeries:
    """单条宏观序列的定义。"""
    key: str            # 内部统一键名，如 'dxy'
    label: str          # 展示名称
    source: str         # 'yfinance' | 'fred'
    symbol: str         # 数据源侧的代码
    unit: str = ''      # 单位，用于报告展示


# === 序列注册表 ===
# 行情类（yfinance，无需 API Key）
_QUOTE_SERIES: Dict[str, MacroSeries] = {
    s.key: s for s in [
        MacroSeries('gold', 'Gold Spot / Futures', 'yfinance', 'GC=F', 'USD/oz'),
        MacroSeries('silver', 'Silver Futures', 'yfinance', 'SI=F', 'USD/oz'),
        MacroSeries('wti', 'WTI Crude', 'yfinance', 'CL=F', 'USD/bbl'),
        MacroSeries('brent', 'Brent Crude', 'yfinance', 'BZ=F', 'USD/bbl'),
        MacroSeries('dxy', 'US Dollar Index', 'yfinance', 'DX-Y.NYB', 'index'),
        MacroSeries('ust5y', 'US 5Y Treasury', 'yfinance', '^FVX', '%'),
        MacroSeries('ust10y', 'US 10Y Treasury', 'yfinance', '^TNX', '%'),
        MacroSeries('ust30y', 'US 30Y Treasury', 'yfinance', '^TYX', '%'),
    ]
}

# 官方宏观类（FRED，需 API Key）
_FRED_SERIES: Dict[str, MacroSeries] = {
    s.key: s for s in [
        MacroSeries('pce_core', 'Core PCE Price Index', 'fred', 'PCEPILFE', 'index'),
        MacroSeries('cpi', 'CPI (All Urban)', 'fred', 'CPIAUCSL', 'index'),
        MacroSeries('cpi_core', 'Core CPI', 'fred', 'CPILFESL', 'index'),
        MacroSeries('unemployment', 'Unemployment Rate', 'fred', 'UNRATE', '%'),
        MacroSeries('fed_funds', 'Fed Funds Rate', 'fred', 'DFF', '%'),
        MacroSeries('real_10y', '10Y Real Yield (TIPS)', 'fred', 'DFII10', '%'),
        MacroSeries('ust2y', 'US 2Y Treasury', 'fred', 'DGS2', '%'),
        MacroSeries('breakeven_10y', '10Y Breakeven Inflation', 'fred', 'T10YIE', '%'),
    ]
}


# 能源库存类（EIA 周度，需 API Key）
# 柴油（distillate）库存是当前最关键的通胀传导渠道：
# 原油回落时柴油仍偏紧会让 PPI 保持高位，进而压制黄金。
_EIA_SERIES: Dict[str, MacroSeries] = {
    s.key: s for s in [
        MacroSeries('eia_crude', 'US Crude Stocks (ex-SPR)', 'eia', 'PET.WCESTUS1.W', 'kbbl'),
        MacroSeries('eia_distillate', 'US Distillate Stocks', 'eia', 'PET.WDISTUS1.W', 'kbbl'),
        MacroSeries('eia_gasoline', 'US Gasoline Stocks', 'eia', 'PET.WGTSTUS1.W', 'kbbl'),
        MacroSeries('eia_spr', 'US SPR Stocks', 'eia', 'PET.WCSSTUS1.W', 'kbbl'),
    ]
}


# CFTC 持仓类（Commitments of Traders，周度，无需 API Key）
# symbol 形如 "<合约代码>:<字段>"：
#   noncomm_net   = 非商业（投机）净持仓 = 多头 - 空头，即市场常说的"管理基金净多"
#   open_interest = 总持仓量
# 合约代码来自 CFTC legacy futures-only 数据集；黄金 088691 已核对，
# 其余代码如需扩展请先在 CFTC 站点确认。
_COT_SERIES: Dict[str, MacroSeries] = {
    s.key: s for s in [
        MacroSeries('cot_gold_net', 'COT Gold Non-Comm Net', 'cftc', '088691:noncomm_net', 'contracts'),
        MacroSeries('cot_gold_oi', 'COT Gold Open Interest', 'cftc', '088691:open_interest', 'contracts'),
        MacroSeries('cot_silver_net', 'COT Silver Non-Comm Net', 'cftc', '084691:noncomm_net', 'contracts'),
    ]
}

SERIES_REGISTRY: Dict[str, MacroSeries] = {
    **_QUOTE_SERIES, **_COT_SERIES, **_FRED_SERIES, **_EIA_SERIES,
}

# 盘面快照的默认序列组合
DEFAULT_SNAPSHOT_KEYS: List[str] = ['gold', 'silver', 'wti', 'brent', 'dxy', 'ust10y', 'ust30y']


class MacroFetcher:
    """宏观与大宗商品数据获取器。"""

    name = "MacroFetcher"

    def __init__(self, fred_api_key: Optional[str] = None,
                 eia_api_key: Optional[str] = None):
        self._fred_api_key = fred_api_key or self._resolve_key('fred_api_key', 'FRED_API_KEY')
        self._eia_api_key = eia_api_key or self._resolve_key('eia_api_key', 'EIA_API_KEY')
        if not self._fred_api_key:
            logger.debug("[Macro] FRED_API_KEY 未配置，官方宏观序列不可用（行情序列不受影响）")
        if not self._eia_api_key:
            logger.debug("[Macro] EIA_API_KEY 未配置，能源库存序列不可用（行情序列不受影响）")

    @staticmethod
    def _resolve_key(config_attr: str, env_var: str) -> Optional[str]:
        """先读 src.config，失败则回退环境变量。配置缺失不应阻断行情序列。"""
        try:
            from src.config import get_config
            config = get_config()
            key = getattr(config, config_attr, None)
            if key:
                return key
        except Exception:  # noqa: BLE001 - 配置缺失不应阻断行情序列
            logger.debug("[Macro] 读取 config 失败，回退到环境变量 %s", env_var, exc_info=True)
        return os.getenv(env_var) or None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def available_series(self) -> List[MacroSeries]:
        """返回当前配置下真正可用的序列。"""
        series = list(_QUOTE_SERIES.values()) + list(_COT_SERIES.values())
        if self._fred_api_key:
            series.extend(_FRED_SERIES.values())
        if self._eia_api_key:
            series.extend(_EIA_SERIES.values())
        return series

    def fetch(self, series_key: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取单条序列的历史数据。

        Args:
            series_key: SERIES_REGISTRY 中的键名，如 'gold' / 'dxy' / 'ust10y'
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'

        Returns:
            DataFrame，列为 STANDARD_COLUMNS。FRED 序列只有 close 有值，
            open/high/low 与 close 对齐，volume/amount 为 0。
        """
        series = SERIES_REGISTRY.get(series_key)
        if series is None:
            raise DataFetchError(
                f"[Macro] 未知序列 '{series_key}'，可用：{', '.join(sorted(SERIES_REGISTRY))}"
            )

        if series.source == 'yfinance':
            raw = self._fetch_yfinance(series, start_date, end_date)
        elif series.source == 'fred':
            raw = self._fetch_fred(series, start_date, end_date)
        elif series.source == 'eia':
            raw = self._fetch_eia(series, start_date, end_date)
        elif series.source == 'cftc':
            raw = self._fetch_cftc(series, start_date, end_date)
        else:  # pragma: no cover - 注册表受控，不应出现
            raise DataFetchError(f"[Macro] 不支持的数据源 '{series.source}'")

        return self._normalize(raw, series)

    def fetch_snapshot(self, series_keys: Optional[List[str]] = None,
                       lookback_days: int = 7) -> Dict[str, Dict[str, float]]:
        """
        获取多条序列的最新快照（最新值 + 日变动）。

        单条序列失败不会中断整体，失败项不出现在结果中并记录日志，
        与 data_provider 既有的降级策略保持一致。

        Returns:
            {series_key: {'value': float, 'change': float, 'change_pct': float,
                          'label': str, 'unit': str, 'date': 'YYYY-MM-DD'}}
        """
        keys = series_keys or DEFAULT_SNAPSHOT_KEYS
        end = datetime.now()
        start = end - timedelta(days=max(lookback_days, 5))
        start_date = start.strftime('%Y-%m-%d')
        end_date = end.strftime('%Y-%m-%d')

        snapshot: Dict[str, Dict[str, float]] = {}
        for key in keys:
            try:
                df = self.fetch(key, start_date, end_date)
            except Exception as exc:  # noqa: BLE001 - 单源失败不拖垮整体
                logger.warning("[Macro] 序列 %s 获取失败：%s", key, exc)
                continue

            if df.empty:
                logger.warning("[Macro] 序列 %s 返回空数据", key)
                continue

            series = SERIES_REGISTRY[key]
            last = df.iloc[-1]
            prev_close = float(df.iloc[-2]['close']) if len(df) >= 2 else float(last['close'])
            value = float(last['close'])
            change = value - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0.0

            snapshot[key] = {
                'label': series.label,
                'unit': series.unit,
                'date': str(last['date']),
                'value': round(value, 4),
                'change': round(change, 4),
                'change_pct': round(change_pct, 4),
            }

        return snapshot

    # ------------------------------------------------------------------
    # 数据源实现
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, series: MacroSeries, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataFetchError("[Macro] 未安装 yfinance，请执行 pip install yfinance") from exc

        # yfinance 的 end 为开区间，补一天以包含 end_date 当天
        end_exclusive = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                df = yf.download(
                    series.symbol,
                    start=start_date,
                    end=end_exclusive,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
                if df is not None and not df.empty:
                    return df
                last_error = DataFetchError(f"[Macro] yfinance 返回空数据：{series.symbol}")
            except Exception as exc:  # noqa: BLE001 - 网络抖动重试
                last_error = exc
                logger.debug("[Macro] yfinance %s 第 %d 次失败：%s", series.symbol, attempt + 1, exc)

        raise DataFetchError(f"[Macro] yfinance 获取 {series.symbol} 失败：{last_error}")

    def _fetch_fred(self, series: MacroSeries, start_date: str, end_date: str) -> pd.DataFrame:
        if not self._fred_api_key:
            raise DataFetchError(
                f"[Macro] 序列 '{series.key}' 需要 FRED_API_KEY，请在 .env 中配置"
            )

        params = {
            'series_id': series.symbol,
            'api_key': self._fred_api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
        }

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(_FRED_BASE_URL, params=params, timeout=_HTTP_TIMEOUT)
                resp.raise_for_status()
                observations = resp.json().get('observations', [])
                if observations:
                    return pd.DataFrame(observations)
                last_error = DataFetchError(f"[Macro] FRED 返回空数据：{series.symbol}")
            except Exception as exc:  # noqa: BLE001 - 网络抖动重试
                last_error = exc
                logger.debug("[Macro] FRED %s 第 %d 次失败：%s", series.symbol, attempt + 1, exc)

        raise DataFetchError(f"[Macro] FRED 获取 {series.symbol} 失败：{last_error}")

    def _fetch_eia(self, series: MacroSeries, start_date: str, end_date: str) -> pd.DataFrame:
        """EIA v2 seriesid 路由，返回周度库存观测。"""
        if not self._eia_api_key:
            raise DataFetchError(
                f"[Macro] 序列 '{series.key}' 需要 EIA_API_KEY，请在 .env 中配置"
            )

        url = f"{_EIA_BASE_URL}/{series.symbol}"
        params = {
            'api_key': self._eia_api_key,
            'start': start_date,
            'end': end_date,
        }

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=_HTTP_TIMEOUT)
                resp.raise_for_status()
                rows = (resp.json().get('response') or {}).get('data', [])
                if rows:
                    return pd.DataFrame(rows)
                last_error = DataFetchError(f"[Macro] EIA 返回空数据：{series.symbol}")
            except Exception as exc:  # noqa: BLE001 - 网络抖动重试
                last_error = exc
                logger.debug("[Macro] EIA %s 第 %d 次失败：%s", series.symbol, attempt + 1, exc)

        raise DataFetchError(f"[Macro] EIA 获取 {series.symbol} 失败：{last_error}")

    def _fetch_cftc(self, series: MacroSeries, start_date: str, end_date: str) -> pd.DataFrame:
        """CFTC COT 周度持仓，Socrata 公共端点，无需 API Key。"""
        try:
            contract_code, _field = series.symbol.split(':', 1)
        except ValueError as exc:
            raise DataFetchError(
                f"[Macro] COT symbol 需形如 '<合约代码>:<字段>'，实际为 '{series.symbol}'"
            ) from exc

        params = {
            'cftc_contract_market_code': contract_code,
            '$where': (
                f"report_date_as_yyyy_mm_dd >= '{start_date}T00:00:00.000'"
                f" AND report_date_as_yyyy_mm_dd <= '{end_date}T23:59:59.999'"
            ),
            '$order': 'report_date_as_yyyy_mm_dd ASC',
            '$limit': 5000,
        }

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(_CFTC_BASE_URL, params=params, timeout=_HTTP_TIMEOUT)
                resp.raise_for_status()
                rows = resp.json()
                if rows:
                    return pd.DataFrame(rows)
                last_error = DataFetchError(f"[Macro] CFTC 返回空数据：{series.symbol}")
            except Exception as exc:  # noqa: BLE001 - 网络抖动重试
                last_error = exc
                logger.debug("[Macro] CFTC %s 第 %d 次失败：%s", series.symbol, attempt + 1, exc)

        raise DataFetchError(f"[Macro] CFTC 获取 {series.symbol} 失败：{last_error}")

    # ------------------------------------------------------------------
    # 标准化
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(df: pd.DataFrame, series: MacroSeries) -> pd.DataFrame:
        """统一为 STANDARD_COLUMNS 列结构。"""
        if df is None or df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        if series.source == 'cftc':
            _code, field = series.symbol.split(':', 1)
            if field == 'noncomm_net':
                longs = pd.to_numeric(df.get('noncomm_positions_long_all'), errors='coerce')
                shorts = pd.to_numeric(df.get('noncomm_positions_short_all'), errors='coerce')
                close = longs - shorts
            else:
                close = pd.to_numeric(df.get('open_interest_all'), errors='coerce')
            out = pd.DataFrame()
            out['date'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd']).dt.strftime('%Y-%m-%d')
            out['open'] = close
            out['high'] = close
            out['low'] = close
            out['close'] = close
            out['volume'] = 0.0
            out['amount'] = 0.0
            out = out.dropna(subset=['close'])
        elif series.source == 'eia':
            out = pd.DataFrame()
            out['date'] = pd.to_datetime(df['period']).dt.strftime('%Y-%m-%d')
            close = pd.to_numeric(df['value'], errors='coerce')
            out['open'] = close
            out['high'] = close
            out['low'] = close
            out['close'] = close
            out['volume'] = 0.0
            out['amount'] = 0.0
            out = out.dropna(subset=['close'])
        elif series.source == 'fred':
            out = pd.DataFrame()
            out['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            # FRED 用 '.' 表示缺失值
            close = pd.to_numeric(df['value'].replace('.', pd.NA), errors='coerce')
            out['open'] = close
            out['high'] = close
            out['low'] = close
            out['close'] = close
            out['volume'] = 0.0
            out['amount'] = 0.0
            out = out.dropna(subset=['close'])
        else:
            frame = df.copy()
            # yfinance 多层列名（单标的下载时偶发）展平
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame = frame.reset_index()

            date_col = 'Date' if 'Date' in frame.columns else frame.columns[0]
            out = pd.DataFrame()
            out['date'] = pd.to_datetime(frame[date_col]).dt.strftime('%Y-%m-%d')
            out['open'] = pd.to_numeric(frame.get('Open'), errors='coerce')
            out['high'] = pd.to_numeric(frame.get('High'), errors='coerce')
            out['low'] = pd.to_numeric(frame.get('Low'), errors='coerce')
            out['close'] = pd.to_numeric(frame.get('Close'), errors='coerce')
            out['volume'] = pd.to_numeric(frame.get('Volume'), errors='coerce').fillna(0.0)
            out['amount'] = 0.0
            out = out.dropna(subset=['close'])

        out = out.sort_values('date').reset_index(drop=True)
        out['pct_chg'] = out['close'].pct_change() * 100
        out['pct_chg'] = out['pct_chg'].fillna(0.0).round(4)

        return out[STANDARD_COLUMNS]
