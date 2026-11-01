# India Electricity Demand Forecasting

Monthly electricity demand forecasts (energy met, in million units) for every Indian state and union territory,
built from Grid-India monthly reports, with an interactive Streamlit dashboard.

> Independent student project. Not an official forecast and not affiliated with Grid-India or POSOCO.

## What it does
- Reads Grid-India (NLDC) Monthly Operational Reports (PDF) and extracts state-wise monthly demand
  (Dec 2022 to Sep 2026, 33 states/UTs plus all-India; January 2023 is missing).
- Forecasts the next 12 months per state with an 80% likely range.
- Tests a temperature "what-if" using monthly weather for one representative city per state.
- Shows everything in a dashboard: forecast, seasonality, backtest accuracy, all-states comparison.

## Results (backtest, Jan 2025 to Sep 2026, using only data available at forecast time)
| Forecast ahead | Typical error per state | Same month last year only |
|---|---|---|
| 1 month | about 6.6% | about 7.2% |
| 6-12 months | about 7-8% | about 7-8% |

- The published forecast averages two methods: last year's value times recent growth, and a gradient-boosting model.
- Beyond about 5 months ahead it is no better than "last year x recent growth".
- Weather helps only when the month's weather is known in advance (error roughly 10-20% lower). Typical weather
  adds nothing for forecasts made months ahead.
- Limits: about 4 years of monthly data, small states are noisy, growth is assumed to continue at its recent pace.

## Run it
```
pip install -r requirements.txt
streamlit run dashboard_india.py
```
The dashboard needs `dashboard_template.html`, `site_data.py`, and the CSV/JSON files in this folder.

## Rebuild from the source PDFs
Download the monthly reports from Grid-India (not included here), put them in one folder, then run in order:
```
python parse_reports.py path/to/pdfs
python build_dataset.py
python get_weather.py            # optional, needs internet
python model_india.py            # backtest at 1-3 months
python backtest_long.py          # backtest at 1-12 months
python model_weather.py          # weather experiment
python sensitivity.py
python forecast_next.py
```
`parse_reports.py` needs poppler (`pdftotext`) installed.

## Files
| File | Purpose |
|---|---|
| `parse_reports.py`, `build_dataset.py` | PDFs to a clean monthly table (`india_state_monthly.csv`) |
| `get_weather.py` | Monthly weather per state from Open-Meteo |
| `model_india.py`, `backtest_long.py` | Forecasting methods and backtests |
| `model_weather.py`, `sensitivity.py` | Weather experiment and temperature sensitivity |
| `forecast_next.py` | 12-month forecast (`forecast_next_months.csv`) |
| `dashboard_india.py`, `dashboard_template.html`, `site_data.py` | Streamlit dashboard |

## Data and credits
- Demand: Grid-India (NLDC) Monthly Operational Reports.
- Weather: Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed CC BY 4.0.
