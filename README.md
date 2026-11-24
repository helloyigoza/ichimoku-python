# ichimoku-polars

Pandas'a bağımlılığı tamamen kaldırılmış, Polars tabanlı Ichimoku hesaplama motoru.  
A Pandas-free Ichimoku engine built on Polars with swing, wave, and time cycle analysis.

---

## Özellikler / Features

- **Temel çizgiler** — Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span
- **Swing analizi** — Pivot tabanlı yüksek/düşük tespiti
- **Wave N deseni** — Dört noktalı kırılma yapısı tespiti
- **Fiyat hedefleri** — V, E, N projeksiyon seviyeleri
- **Zaman döngüleri** — Geçmiş döngülerden gelecek projeksiyon
- **Üç çıktı modu** — Basit Polars serisi, zengin JSON, ultra-kompakt LLM çıktısı

---

## Kurulum / Installation

```bash
pip install polars numpy
```

---

## Kullanım / Usage

### Basit mod — Polars serileri döner

```python
from ichimoku import calculate_ichimoku

tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df)
```

### Zengin mod — Analiz çıktısı

```python
result = calculate_ichimoku(df, rich_output=True)
# result["signals"], result["swing_analysis"], result["price_targets"] ...
```

### Ultra-kompakt mod — LLM / API entegrasyonu için

```python
result = calculate_ichimoku(df, rich_output=True, ultra_compact=True)
# {"p": 42500.0, "sig_tk": "BUY", "sig_kumo": "BULL", ...}
```

---

## Parametreler / Parameters

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `tenkan_period` | 9 | Tenkan-sen periyodu |
| `kijun_period` | 26 | Kijun-sen periyodu |
| `senkou_span_b_period` | 52 | Senkou Span B periyodu |
| `rich_output` | `False` | `True` → detaylı dict çıktısı |
| `ultra_compact` | `False` | `True` → minimal anahtar-değer çıktısı |

Giriş verisi `pl.DataFrame` ya da Polars'ın çözümleyebildiği `list/dict` formatında olabilir. `open`, `high`, `low`, `close` kolonları zorunludur.

Input can be a `pl.DataFrame` or any list/dict Polars can parse. `open`, `high`, `low`, `close` columns are required.

---

## Mimari / Architecture

```
calculate_ichimoku()       ← Tek dışa açık fonksiyon (Facade)
    ├── PolarsDataProcessor   ← Veri doğrulama ve normalizasyon
    ├── IchimokuCore          ← Vektörel bulut çizgisi hesabı (Rust/Polars)
    ├── TheoryEngine          ← Swing, Wave, Cycle (Polars + NumPy hibrit)
    └── IchimokuAnalyzer      ← Sinyal üretimi ve çıktı formatlama
```

---

## Lisans / License

MIT
