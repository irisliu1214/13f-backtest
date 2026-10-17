# 13F Quarterly Backtest

Reproducible 2016–2026 quarterly backtest comparing three 13F accumulation rankings:

1. **Absolute** — largest positive split-adjusted aggregate share change across matched 13F managers.
2. **Percentage** — largest percentage increase in split-adjusted aggregate shares, excluding zero-base/new aggregate positions.
3. **Percentage+floor** — Percentage ranking subject to a minimum positive absolute share-change floor. The primary floor is configurable; a sensitivity table is also produced.

## Point-in-time rules

- Source holdings are SEC-derived Form 13F rows with a point-in-time `availableAt` timestamp.
- Only original `13F-HR` filings known by the statutory filing cutoff are used.
- Managers must be present in both adjacent quarters.
- Options, debt/preferred/warrants and obvious ETF/fund instruments are excluded.
- Stock-split distortions are detected from 13F implied quarter-end prices and aggregate share ratios.
- Entry is the first trading day after the 13F filing cutoff.
- Return is adjusted-close total price return from entry to the first trading day on/after entry + 90 calendar days.
- SPY excess return = stock 90-day return − SPY return over the same dates.

## Outputs

The workflow writes results under `results/`:

- `quarterly_backtest.csv` — requested quarter/method/winner/entry/90-day/SPY excess table.
- `quarterly_metrics.csv` — richer winner diagnostics.
- `floor_sensitivity.csv` — Percentage+floor winners across several floor thresholds.
- `method_summary.csv` — hit rate / average return / average excess-return comparison.
- `data_quality.csv` — manager counts and quarter-level diagnostics.

The default floor is intentionally parameterized because the exact threshold from the interrupted prior Work session was not recoverable. Change `PRIMARY_FLOOR_SHARES` in the workflow (or environment) without changing the rest of the methodology.
