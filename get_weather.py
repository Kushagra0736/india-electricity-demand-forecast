"""Download daily weather for one representative city per state from Open-Meteo's free
historical archive (ERA5 reanalysis) and aggregate it to monthly features.
Run on your own computer (needs internet):  python get_weather.py
Output: weather_state_monthly.csv  (state, date, avg_temp_c, avg_max_temp_c, hot_days_40c, cooling_deg_days, rain_mm)
Open-Meteo: free for non-commercial use, data licensed CC BY 4.0 -> credit "Weather data by Open-Meteo.com"."""
import os
import re
import time
import pandas as pd
import requests

START, END = '2022-11-01', '2026-09-30'     # a little before/after the demand data
URL = 'https://archive-api.open-meteo.com/v1/archive'

# state names match india_state_monthly.csv; one city per state is only a rough proxy for the whole state
CITIES = {
    'Andhra Pradesh': (16.51, 80.65), 'Arunachal Pradesh': (27.08, 93.61), 'Assam': (26.14, 91.74),
    'Bihar': (25.59, 85.14), 'Chandigarh': (30.73, 76.78), 'Chhattisgarh': (21.25, 81.63),
    'DNH & DD (UT)': (20.27, 73.01), 'Delhi': (28.61, 77.21), 'Goa': (15.50, 73.83),
    'Gujarat': (23.02, 72.57), 'Haryana': (28.46, 77.03), 'Himachal Pradesh': (31.10, 77.17),
    'J&K(UT) and Ladakh(UT)': (34.08, 74.80), 'Jharkhand': (23.34, 85.31), 'Karnataka': (12.97, 77.59),
    'Kerala': (9.93, 76.27), 'Madhya Pradesh': (23.26, 77.41), 'Maharashtra': (19.08, 72.88),
    'Manipur': (24.82, 93.94), 'Meghalaya': (25.57, 91.88), 'Mizoram': (23.73, 92.72),
    'Nagaland': (25.67, 94.11), 'Odisha': (20.30, 85.82), 'Puducherry': (11.93, 79.83),
    'Punjab': (30.90, 75.86), 'Rajasthan': (26.91, 75.79), 'Sikkim': (27.33, 88.61),
    'Tamil Nadu': (13.08, 80.27), 'Telangana': (17.39, 78.49), 'Tripura': (23.83, 91.28),
    'Uttar Pradesh': (26.85, 80.95), 'Uttarakhand': (30.32, 78.03), 'West Bengal': (22.57, 88.36),
}


def fetch_daily(lat, lon, tries=6):
    params = dict(latitude=lat, longitude=lon, start_date=START, end_date=END, timezone='Asia/Kolkata',
                  daily='temperature_2m_mean,temperature_2m_max,precipitation_sum')
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(URL, params=params, timeout=60,
                             headers={'User-Agent': 'student-electricity-project/1.0'})
            if r.status_code == 200:
                return pd.DataFrame(r.json()['daily'])
            last_err = f'HTTP {r.status_code}: {r.text[:150]}'
        except requests.exceptions.RequestException as e:      # connection reset, timeout, SSL, ...
            last_err = repr(e)[:200]
        wait = 10 * (i + 1)
        print(f'   attempt {i + 1}/{tries} failed ({last_err}); retrying in {wait}s')
        time.sleep(wait)
    raise RuntimeError(last_err)


def to_monthly(d, state):
    d = d.rename(columns={'time': 'day'}); d['day'] = pd.to_datetime(d['day'])
    d['date'] = d.day.dt.to_period('M').dt.to_timestamp()
    d['hot40'] = (d.temperature_2m_max >= 40).astype(int)
    d['cdd'] = (d.temperature_2m_mean - 24).clip(lower=0)      # cooling degree days, base 24 C
    m = d.groupby('date').agg(avg_temp_c=('temperature_2m_mean', 'mean'),
                              avg_max_temp_c=('temperature_2m_max', 'mean'),
                              hot_days_40c=('hot40', 'sum'), cooling_deg_days=('cdd', 'sum'),
                              rain_mm=('precipitation_sum', 'sum')).round(2).reset_index()
    m.insert(0, 'state', state)
    return m


def main():
    os.makedirs('weather_cache', exist_ok=True)
    out, failed = [], []
    for i, (state, (lat, lon)) in enumerate(CITIES.items(), 1):
        cache = os.path.join('weather_cache', re.sub(r'[^A-Za-z0-9]+', '_', state) + '.csv')
        if os.path.exists(cache):
            print(f'[{i}/{len(CITIES)}] {state} (cached)')
            daily = pd.read_csv(cache)
        else:
            print(f'[{i}/{len(CITIES)}] {state}')
            try:
                daily = fetch_daily(lat, lon)
            except RuntimeError as e:
                print(f'   GIVING UP on {state}: {e}'); failed.append(state); continue
            daily.to_csv(cache, index=False)
            time.sleep(2)                  # be polite to the free service
        out.append(to_monthly(daily, state))
    if out:
        pd.concat(out).to_csv('weather_state_monthly.csv', index=False)
        print('saved weather_state_monthly.csv with', len(out), 'states')
    if failed:
        print('FAILED states (just run the script again, finished ones are cached):', failed)


if __name__ == '__main__':
    main()