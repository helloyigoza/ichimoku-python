import polars as pl
import numpy as np
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime, date

# Dışarıya sadece ana fonksiyonu açıyoruz (Tak-Çalıştır uyumluluğu)
__all__ = ['calculate_ichimoku']

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATIONS
# -------------------------------------------------------------------------
class IchiConfig:
    """Sistem genelinde kullanılacak sabit değerler."""
    BULLISH_LEG = 1
    BEARISH_LEG = 0
    HIGHS = 'HIGHS'
    LOWS = 'LOWS'
    SWINGS = 'SWINGS'
    TENKAN = 'TENKAN CROSS'
    KIJUN = 'KIJUN CROSS'
    KUMO = 'KUMO CHANGE'
    WAVE_N = 'WAVE N'
    BULLISH = 'BULLISH'
    BEARISH = 'BEARISH'

@dataclass(slots=True)
class IchiParams:
    """Hesaplama parametrelerini yöneten veri sınıfı."""
    tenkan: int = 9
    kijun: int = 26
    span_b: int = 52
    pivot: int = 10
    cycle_mode: str = IchiConfig.SWINGS

# -------------------------------------------------------------------------
# DOMAIN ENTITIES (DATACLASSES)
# -------------------------------------------------------------------------
@dataclass(slots=True)
class ChartPoint:
    time: Any
    index: int
    price: float

@dataclass(slots=True)
class SwingPoint:
    barTime: Any
    barIndex: int
    priceLevel: float
    leg: int 

@dataclass(slots=True)
class TimeCycleData:
    firstBarTime: Any
    firstBarIndex: int
    lastBarTime: Optional[Any] = None
    lastBarIndex: Optional[int] = None

@dataclass(slots=True)
class WaveData:
    points: List[ChartPoint]
    tag: str
    bias: str 

@dataclass(slots=True)
class PriceTargetData:
    barIndex: int 
    priceLevel: float
    tag: str

# -------------------------------------------------------------------------
# CORE UTILITIES
# -------------------------------------------------------------------------
class TimeStandardizer:
    """Zaman verilerini standartlaştıran yardımcı sınıf."""
    @staticmethod
    def to_iso_format(ts: Any) -> str:
        if isinstance(ts, (datetime, date)):
            return ts.isoformat()
        return str(ts)

    @staticmethod
    def json_default(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (datetime, date)): return TimeStandardizer.to_iso_format(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# -------------------------------------------------------------------------
# POLARS NATIVE ENGINE
# -------------------------------------------------------------------------
class PolarsDataProcessor:
    """Gelen veriyi Polars Dataframe'ine çevirip doğrular ve hazırlar."""
    @staticmethod
    def prepare(data: Any) -> pl.DataFrame:
        if not isinstance(data, pl.DataFrame):
            try:
                df = pl.DataFrame(data)
            except Exception as e:
                raise ValueError(f"Veri Polars DataFrame'ine çevrilemedi: {e}")
        else:
            df = data.clone()

        # Kolon isimlerini standartlaştır
        rename_map = {col: col.lower() for col in df.columns}
        df = df.rename(rename_map)

        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Eksik OHLC kolonları: Gerekli kolonlar {required_cols}")

        if 'time' not in df.columns:
            # Time kolonu yoksa index tabanlı bir dummy kolon oluştur
            df = df.with_row_index("time")
        
        if 'volume' not in df.columns:
            df = df.with_columns(pl.lit(0).alias('volume'))

        return df

class IchimokuCore:
    """Rust tabanlı Polars Expression'ları ile Ichimoku Core hesaplamaları."""
    
    @staticmethod
    def pine_average_expr(high_col: str, low_col: str, length: int) -> pl.Expr:
        """PineScript'teki (highest + lowest) / 2 mantığının native Polars karşılığı."""
        highest = pl.col(high_col).rolling_max(window_size=length)
        lowest = pl.col(low_col).rolling_min(window_size=length)
        return (highest + lowest) / 2.0

    @classmethod
    def apply_cloud_expressions(cls, df: pl.DataFrame, params: IchiParams) -> pl.DataFrame:
        """Vektörel olarak tüm Ichimoku bulut çizgilerini tek seferde (Lazy) hesaplar."""
        # 1. Aşama: Temel çizgiler
        df = df.with_columns([
            cls.pine_average_expr("high", "low", params.tenkan).alias("tenkan_sen"),
            cls.pine_average_expr("high", "low", params.kijun).alias("kijun_sen"),
            cls.pine_average_expr("high", "low", params.span_b).alias("span_b_base")
        ])
        
        # 2. Aşama: Gecikmeli ve ilerlemeli (Shifted) çizgiler
        df = df.with_columns([
            ((pl.col("tenkan_sen") + pl.col("kijun_sen")) / 2.0).shift(params.kijun).alias("senkou_span_a"),
            pl.col("span_b_base").shift(params.kijun).alias("senkou_span_b"),
            pl.col("close").shift(-params.kijun).alias("chikou_span")
        ])
        return df

# -------------------------------------------------------------------------
# ADVANCED THEORY ENGINE
# -------------------------------------------------------------------------
class TheoryEngine:
    """Polars ve Numpy hibrit mimarisiyle Swing, Wave ve Zaman Döngüsü analizleri."""
    
    def __init__(self, df: pl.DataFrame, params: IchiParams):
        self.df = df
        self.params = params

    def _apply_pivot_and_signals(self) -> pl.DataFrame:
        """İleriye dönük taramaları (Forward-Window) Polars reverse() & rolling ile optimize eder."""
        plen = self.params.pivot
        
        # İleriye dönük en yüksek ve en düşük (Numba yerine Rust hızında native vektörel çözüm)
        self.df = self.df.with_columns([
            pl.col("high").reverse().rolling_max(plen).reverse().shift(-1).alias("highest_future"),
            pl.col("low").reverse().rolling_min(plen).reverse().shift(-1).alias("lowest_future")
        ])

        # Trigger ve State oluşturma
        self.df = self.df.with_columns([
            (pl.col("high") > pl.col("highest_future")).alias("is_bear_trigger"),
            (pl.col("low") < pl.col("lowest_future")).alias("is_bull_trigger")
        ])

        self.df = self.df.with_columns(
            pl.when(pl.col("is_bull_trigger")).then(IchiConfig.BULLISH_LEG)
              .when(pl.col("is_bear_trigger")).then(IchiConfig.BEARISH_LEG)
              .otherwise(None)
              .alias("state_signals")
        )

        # Forward fill ve gecikme (Shift) ile mevcut bacak trendi
        self.df = self.df.with_columns(
            pl.col("state_signals").forward_fill().shift(plen).fill_null(IchiConfig.BEARISH_LEG).alias("current_leg")
        )

        # Pivot noktaları ve Kesişimler
        self.df = self.df.with_columns([
            (pl.col("current_leg") != pl.col("current_leg").shift(1)).fill_null(False).alias("new_pivot"),
            ((pl.col("current_leg") == IchiConfig.BULLISH_LEG) & (pl.col("current_leg").shift(1) == IchiConfig.BEARISH_LEG)).alias("pivot_low"),
            ((pl.col("current_leg") == IchiConfig.BEARISH_LEG) & (pl.col("current_leg").shift(1) == IchiConfig.BULLISH_LEG)).alias("pivot_high"),
            (pl.col("senkou_span_a") >= pl.col("senkou_span_b")).alias("kumo_bullish"),
            (pl.col("close") > pl.col("tenkan_sen")).alias("t_cross"),
            (pl.col("close") > pl.col("kijun_sen")).alias("k_cross"),
        ])

        self.df = self.df.with_columns([
            (pl.col("kumo_bullish") != pl.col("kumo_bullish").shift(1)).fill_null(False).alias("kumo_change"),
            (pl.col("t_cross") != pl.col("t_cross").shift(1)).fill_null(False).alias("tenkan_cross"),
            (pl.col("k_cross") != pl.col("k_cross").shift(1)).fill_null(False).alias("kijun_cross"),
        ])

        return self.df

    def generate_theory_data(self) -> Dict[str, Any]:
        """Vektörel altyapıdan alınan sinyalleri numpy array'leri ile sekansiyel olarak işler."""
        self.df = self._apply_pivot_and_signals()
        df_len = len(self.df)
        plen = self.params.pivot

        # Numpy Array dönüşümleri (State Machine için en hızlı yöntem)
        times = self.df["time"].to_numpy()
        highs = self.df["high"].to_numpy()
        lows = self.df["low"].to_numpy()
        
        new_pivot = self.df["new_pivot"].to_numpy()
        pivot_low = self.df["pivot_low"].to_numpy()
        pivot_high = self.df["pivot_high"].to_numpy()
        tenkan_cross = self.df["tenkan_cross"].to_numpy()
        kijun_cross = self.df["kijun_cross"].to_numpy()
        kumo_change = self.df["kumo_change"].to_numpy()

        events_of_interest = set(np.where(new_pivot)[0])
        
        event_map = {
            IchiConfig.HIGHS: pivot_high, IchiConfig.LOWS: pivot_low, 
            IchiConfig.SWINGS: new_pivot, IchiConfig.TENKAN: tenkan_cross, 
            IchiConfig.KIJUN: kijun_cross, IchiConfig.KUMO: kumo_change
        }

        mode = self.params.cycle_mode
        if mode in event_map and mode != IchiConfig.SWINGS:
            events_of_interest.update(np.where(event_map[mode])[0])

        sorted_events = sorted(list(events_of_interest))
        
        swings, waves, targets, cycles, forecasts = [], [], [], [], []

        for i in sorted_events:
            if new_pivot[i]:
                p_idx = i - plen
                if p_idx < 0: continue
                
                p_time = times[p_idx]
                bias = IchiConfig.BULLISH_LEG if pivot_low[i] else IchiConfig.BEARISH_LEG
                price = lows[p_idx] if bias == IchiConfig.BULLISH_LEG else highs[p_idx]
                
                sp = SwingPoint(p_time, p_idx, price, bias)
                swings.append(sp)
                
                # Dalga (Wave) Tespiti
                if len(swings) >= 4:
                    a, b, c, d = swings[-4:]
                    if (d.barIndex - a.barIndex) >= 9:
                        thresh = abs(a.priceLevel - b.priceLevel) * 0.2
                        is_bull_n = (a.priceLevel < b.priceLevel and c.priceLevel > a.priceLevel + thresh and 
                                     c.priceLevel < b.priceLevel - thresh and d.priceLevel > b.priceLevel + thresh)
                        is_bear_n = (a.priceLevel > b.priceLevel and c.priceLevel < a.priceLevel - thresh and 
                                     c.priceLevel > b.priceLevel + thresh and d.priceLevel < b.priceLevel - thresh)
                        
                        if is_bull_n or is_bear_n:
                            wb = IchiConfig.BULLISH if d.leg == IchiConfig.BEARISH_LEG else IchiConfig.BEARISH
                            pts = [ChartPoint(p.barTime, p.barIndex, p.priceLevel) for p in [a,b,c,d]]
                            waves.append(WaveData(pts, IchiConfig.WAVE_N, wb))

                # Fiyat Hedefleri (Price Targets) Tespiti
                if len(swings) >= 3:
                    a, b, c = swings[-3:]
                    if (c.barIndex - a.barIndex) >= 9:
                        thresh = abs(a.priceLevel - b.priceLevel) * 0.2
                        if ((a.priceLevel < b.priceLevel and c.priceLevel > a.priceLevel + thresh and c.priceLevel < b.priceLevel - thresh) or 
                            (a.priceLevel > b.priceLevel and c.priceLevel < a.priceLevel - thresh and c.priceLevel > b.priceLevel + thresh)):
                            
                            targets.clear()
                            if a.priceLevel < b.priceLevel:
                                targets.extend([
                                    PriceTargetData(c.barIndex, b.priceLevel + (b.priceLevel - c.priceLevel), "V"),
                                    PriceTargetData(c.barIndex, b.priceLevel + (b.priceLevel - a.priceLevel), "E"),
                                    PriceTargetData(c.barIndex, c.priceLevel + (b.priceLevel - a.priceLevel), "N")
                                ])
                            else:
                                targets.extend([
                                    PriceTargetData(c.barIndex, b.priceLevel - (c.priceLevel - b.priceLevel), "V"),
                                    PriceTargetData(c.barIndex, b.priceLevel - (a.priceLevel - b.priceLevel), "E"),
                                    PriceTargetData(c.barIndex, c.priceLevel - (a.priceLevel - b.priceLevel), "N")
                                ])

            # Zaman Döngüleri (Time Cycles) Tespiti
            if mode in event_map and event_map[mode][i]:
                p_idx = i - plen if mode in [IchiConfig.HIGHS, IchiConfig.LOWS, IchiConfig.SWINGS] else i
                if p_idx >= 0:
                    if cycles:
                        cycles[-1].lastBarTime = times[p_idx]
                        cycles[-1].lastBarIndex = p_idx
                    if len(cycles) >= 11: 
                        cycles.pop(0)
                    cycles.append(TimeCycleData(times[p_idx], p_idx))

        # Zaman Döngüsü Projeksiyonu (Gelecek Tahminleri)
        if cycles and cycles[-1].lastBarIndex is not None:
            f_idx = cycles[-1].lastBarIndex
            pattern = [9, 17, 26, 33, 42, 51, 65, 76, 129, 172, 200, 257]
            forecasts.append({"time": TimeStandardizer.to_iso_format(times[f_idx]), "index": f_idx, "is_anchor": True})
            
            # Ortalama mum süresini bul (Gelecek zamanı hesaplamak için)
            td_seconds = 86400  # Default 1 gün
            if df_len > 1 and isinstance(times[0], (datetime, np.datetime64)):
                td = times[1] - times[0]
                td_seconds = td.item().total_seconds() if hasattr(td, 'item') else 86400

            for offset in pattern:
                pi = f_idx + offset
                if pi < df_len:
                    pt = times[pi]
                else:
                    # Mevcut verinin dışına taşıyorsa zamanı sentetik olarak ilerlet
                    extra_bars = pi - (df_len - 1)
                    if isinstance(times[-1], (datetime, np.datetime64)):
                        pt = times[-1] + np.timedelta64(int(td_seconds * extra_bars), 's')
                    else:
                        pt = f"Future_Bar_{pi}"
                
                forecasts.append({"time": TimeStandardizer.to_iso_format(pt), "index": pi, "is_anchor": False})

        # Serileri JSON uyumlu listelere dönüştüren yardımcı fonksiyon (Polars)
        def _series_to_json(col_name: str) -> List[Dict[str, Any]]:
            filtered_df = self.df.select(["time", col_name]).drop_nulls()
            if filtered_df.height == 0: return []
            
            # Sadece geçerli değerleri alıp time ile formatlayalım
            t_list = filtered_df["time"].to_list()
            v_list = filtered_df[col_name].to_list()
            
            return [{"time": TimeStandardizer.to_iso_format(t), "value": float(v)} for t, v in zip(t_list, v_list)]

        return {
            "ichimoku_lines": {
                "tenkan_sen": _series_to_json("tenkan_sen"),
                "kijun_sen": _series_to_json("kijun_sen"),
                "senkou_span_a": _series_to_json("senkou_span_a"),
                "senkou_span_b": _series_to_json("senkou_span_b"),
                "chikou_span": _series_to_json("chikou_span"),
            },
            "swing_points": [asdict(sp) for sp in swings],
            "waves": [{"tag": w.tag, "bias": w.bias, "points": [asdict(pt) for pt in w.points]} for w in waves[-10:]],
            "time_cycles": [asdict(tc) for tc in cycles],
            "time_cycles_forecast": forecasts,
            "price_targets": [asdict(pt) for pt in targets]
        }


class IchimokuAnalyzer:
    """Veriyi işleyip LLM veya farklı servislerin tüketebileceği formata getiren adaptör sınıfı."""
    @staticmethod
    def extract_last_value(data: dict, key: str) -> float:
        items = data.get("ichimoku_lines", {}).get(key, [])
        if items and isinstance(items[-1], dict):
            return float(items[-1].get('value'))
        return np.nan

    @classmethod
    def analyze(cls, df: pl.DataFrame, params: IchiParams, theory_data: dict, is_ultra: bool) -> dict:
        th = theory_data
        lp = float(df["close"].tail(1).item())
        
        t = cls.extract_last_value(th, "tenkan_sen")
        k = cls.extract_last_value(th, "kijun_sen")
        sa = cls.extract_last_value(th, "senkou_span_a")
        sb = cls.extract_last_value(th, "senkou_span_b")
        
        # Chikou Span'ın mevcut son değeri (kijun kadar geriden gelen fiyat)
        chikou_shift = params.kijun
        cv = df["close"].item(len(df) - 1 - chikou_shift) if len(df) > chikou_shift else np.nan
        
        is_bull = (sa > sb) if not (np.isnan(sa) or np.isnan(sb)) else False
        
        p_kumo = "UNKNOWN"
        if not any(np.isnan(v) for v in [lp, sa, sb]):
            p_kumo = "ABOVE" if lp > max(sa, sb) else "BELOW" if lp < min(sa, sb) else "INSIDE"
            
        tk_sig = "UNKNOWN"
        ts = [d.get('value') for d in th.get("ichimoku_lines", {}).get("tenkan_sen", [])]
        ks = [d.get('value') for d in th.get("ichimoku_lines", {}).get("kijun_sen", [])]
        
        if len(ts) >= 2 and len(ks) >= 2:
            t1, t0, k1, k0 = ts[-2], ts[-1], ks[-2], ks[-1]
            if t1 <= k1 and t0 > k0: tk_sig = "BULLISH_CROSS"
            elif t1 >= k1 and t0 < k0: tk_sig = "BEARISH_CROSS"
            else: tk_sig = "ABOVE" if t0 > k0 else "BELOW"
            
        strn = "STRONG" if not (np.isnan(t) or np.isnan(k)) and abs(t - k)/lp > 0.01 else "WEAK"
        
        w_type, w_bias = "NONE", "NEUTRAL"
        if waves := th.get("waves"):
            w_type = waves[-1].get("tag", "UNKNOWN")
            w_bias = waves[-1].get("bias", "NEUTRAL")

        s_highs = sum(1 for sp in th.get("swing_points", []) if sp.get("leg") == IchiConfig.BEARISH_LEG)
        s_lows = sum(1 for sp in th.get("swing_points", []) if sp.get("leg") == IchiConfig.BULLISH_LEG)
        
        t_cycles = th.get("time_cycles", [])
        c_dur = [tc["lastBarIndex"] - tc["firstBarIndex"] for tc in t_cycles if tc.get("lastBarIndex")]
        avg_dur = round(sum(c_dur)/len(c_dur)) if c_dur else 0
        
        res = {
            "price": round(lp, 4),
            "ichimoku": {
                "tenkan": round(t, 4) if not np.isnan(t) else None,
                "kijun": round(k, 4) if not np.isnan(k) else None,
                "senkou_a": round(sa, 4) if not np.isnan(sa) else None,
                "senkou_b": round(sb, 4) if not np.isnan(sb) else None,
                "chikou": round(cv, 4) if not np.isnan(cv) else None,
            },
            "signals": {
                "tk_signal": tk_sig, "kumo": "BULLISH" if is_bull else "BEARISH", 
                "price_location": p_kumo, "trend_strength": strn
            },
            "swing_analysis": {"swings": {"highs": s_highs, "lows": s_lows}, "wave": w_type, "bias": w_bias},
            "cycles": {"count": len(t_cycles), "avg_duration_bars": avg_dur}
        }
        
        if pts := th.get("price_targets"):
            res["price_targets"] = [{pt.get("tag"): round(pt.get("priceLevel"), 4)} for pt in pts]
        if fcs := th.get("time_cycles_forecast"):
            res["forecast_points"] = [f.get("time") for f in fcs if not f.get("is_anchor")][:5]
            
        if is_ultra:
            tks = "BUY" if "BULL" in tk_sig else "SELL" if "BEAR" in tk_sig else tk_sig
            out = {
                "p": res["price"], "t": res["ichimoku"]["tenkan"], "k": res["ichimoku"]["kijun"],
                "sa": res["ichimoku"]["senkou_a"], "sb": res["ichimoku"]["senkou_b"],
                "sig_tk": tks, "sig_kumo": "BULL" if is_bull else "BEAR", "sig_price": p_kumo, "strength": strn[0]
            }
            if w_type != "NONE": out["wave"] = w_type
            return out
            
        return res

# -------------------------------------------------------------------------
# EXPOSED API (FACADE PATTERN)
# -------------------------------------------------------------------------
def calculate_ichimoku(
    data: Any, # Beklenen format: pl.DataFrame veya Polars tarafından çözümlenebilir List/Dict vb.
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    rich_output: bool = False,
    ultra_compact: bool = False
) -> Union[Tuple[pl.Series, pl.Series, pl.Series, pl.Series, pl.Series], Dict[str, Any]]:
    """
    Polars destekli, Quant-Grade Ichimoku Sistem Hesaplayıcısı.
    
    Pandas tamamen kaldırılmış olup, Rust tabanlı tembel hesaplama motoruyla 
    yüksek frekanslı (HFT) senaryolarda bile kullanılabilir şekilde tasarlanmıştır.
    """
    try:
        # Veri boş/none kontrolü
        if data is None or (isinstance(data, pl.DataFrame) and data.height == 0) or len(data) == 0:
            if rich_output: return {"error": "Input data is empty", "status": "empty_input"}
            empty_s = pl.Series(dtype=pl.Float64)
            return (empty_s, empty_s, empty_s, empty_s, empty_s)
            
        # 1. Veri Hazırlama
        df = PolarsDataProcessor.prepare(data)
        params = IchiParams(tenkan_period, kijun_period, senkou_span_b_period)

        # 2. Temel Bulut Çizgilerinin Hesaplanması (Polars Vectorized)
        df_cloud = IchimokuCore.apply_cloud_expressions(df, params)

        # 3. Eğer sadece temel indikatör serileri isteniyorsa doğrudan Tuple dön
        if not rich_output:
            return (
                df_cloud["tenkan_sen"], 
                df_cloud["kijun_sen"], 
                df_cloud["senkou_span_a"], 
                df_cloud["senkou_span_b"], 
                df_cloud["chikou_span"]
            )
        
        # 4. Gelişmiş Teori Hesaplamaları
        engine = TheoryEngine(df_cloud, params)
        theory_data = engine.generate_theory_data()
        
        # 5. Formatlama ve Çıktı Üretimi
        return IchimokuAnalyzer.analyze(df_cloud, params, theory_data, ultra_compact)

    except Exception as e:
        logging.error(f"Ichimoku motorunda kritik Polars operasyon hatası: {e}", exc_info=True)
        if rich_output: 
            return {"error": str(e), "status": "calculation_error"}
        
        empty_s = pl.Series(dtype=pl.Float64)
        return (empty_s, empty_s, empty_s, empty_s, empty_s)