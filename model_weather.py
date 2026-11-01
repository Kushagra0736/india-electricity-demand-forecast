"""Does weather help? Same rolling-origin backtest as model_india.py, with weather features added.
Two scenarios for the target month's weather:
  oracle : the actual weather of that month (what you'd have with a perfect weather forecast)
  clim   : the typical weather for that calendar month in earlier years (all you know months ahead)"""
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import model_india as mi

warnings.filterwarnings('ignore')
OFF = mi.OFF
WV = ['avg_temp_c', 'cooling_deg_days', 'hot_days_40c', 'rain_mm']
wdf = pd.read_csv('weather_state_monthly.csv', parse_dates=['date'])
W = {v: wdf.pivot(index='date', columns='state', values=v).sort_index() for v in WV}


def weather_changes(state, d, scenario):
    """Weather of month d minus weather of the same month a year earlier (by variable)."""
    out = {}
    for v in WV:
        s = W[v][state]
        ly = s.get(d - OFF(years=1), np.nan)
        if scenario == 'oracle':
            cur = s.get(d, np.nan)
        else:   # climatology: same calendar month in years before d's year
            same = [s.get(pd.Timestamp(year=y, month=d.month, day=1), np.nan) for y in range(d.year - 5, d.year)]
            same = [x for x in same if not np.isnan(x)]
            cur = float(np.mean(same)) if same else np.nan
        out['w_' + v] = cur - ly
    return out


def add_weather(panel, scenario):
    wc = pd.DataFrame([weather_changes(s, d, scenario) for s, d in zip(panel.state, panel.date)], index=panel.index)
    p = pd.concat([panel, wc], axis=1)
    for c in wc.columns:
        p[c + '_f'] = p[c].fillna(0)
    return p


WCOLS = ['w_' + v for v in WV]
RIDGE_W = mi.RIDGE_F + [c + '_f' for c in WCOLS]
HGB_W = mi.HGB_F + WCOLS


def main():
    test_months = [d for d in mi.wide.index if d >= pd.Timestamp('2025-01-01')]
    rows = []
    for h in (1, 2, 3):
        base = mi.build_panel(h)
        panels = {sc: add_weather(base, sc) for sc in ('oracle', 'clim')}
        for d in test_months:
            tr0 = base[base.date <= d - OFF(months=h)]
            te = base[base.date == d].copy()
            if len(tr0) < 150 or te.empty:
                continue
            ridge = Ridge(alpha=1.0).fit(tr0[mi.RIDGE_F], tr0.y)
            hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                                min_samples_leaf=15, random_state=0).fit(tr0[mi.HGB_F], tr0.y)
            te['seasonal_naive'] = te.lag12
            te['naive_x_g3'] = te.lag12 * te.g3
            te['ridge'] = te.lag12 * np.exp(ridge.predict(te[mi.RIDGE_F]))
            te['hgb'] = te.lag12 * np.exp(hgb.predict(te[mi.HGB_F]))
            te['blend'] = (te.naive_x_g3 + te.hgb) / 2
            for sc in ('oracle', 'clim'):
                p = panels[sc]
                tr, tw = p[p.date <= d - OFF(months=h)], p[p.date == d]
                r_w = Ridge(alpha=1.0).fit(tr[RIDGE_W], tr.y)
                h_w = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                                    min_samples_leaf=15, random_state=0).fit(tr[HGB_W], tr.y)
                te[f'ridge_{sc}'] = te.lag12 * np.exp(r_w.predict(tw[RIDGE_W].values if False else tw.loc[te.index, RIDGE_W]))
                te[f'hgb_{sc}'] = te.lag12 * np.exp(h_w.predict(tw.loc[te.index, HGB_W]))
                te[f'blend_{sc}'] = (te.naive_x_g3 + te[f'hgb_{sc}']) / 2
            te['horizon'] = h
            rows.append(te)
    res = pd.concat(rows)
    models = ['seasonal_naive', 'naive_x_g3', 'ridge', 'hgb', 'blend',
              'ridge_clim', 'hgb_clim', 'blend_clim', 'ridge_oracle', 'hgb_oracle', 'blend_oracle']

    def table(r):
        t = {}
        for m in models:
            ape = (r[m] - r.actual).abs() / r.actual
            g = r.groupby('date')[[m, 'actual']].sum()
            t[m] = dict(MAPE_state=ape.mean() * 100,
                        WMAPE=(r[m] - r.actual).abs().sum() / r.actual.sum() * 100,
                        AllIndia_err=((g[m] - g.actual).abs() / g.actual).mean() * 100)
        return pd.DataFrame(t).T.round(2)

    for h in (1, 2, 3):
        r = res[res.horizon == h]
        print(f'\n=== {h}-month-ahead ({r.date.nunique()} test months, {len(r)} state-forecasts)')
        print(table(r).sort_values('MAPE_state').to_string())
    res.to_csv('backtest_weather_predictions.csv', index=False)

    # what the weather-aware ridge learned (oracle, last fit, 1-month horizon): effect of weather change on YoY demand growth
    p = add_weather(mi.build_panel(1), 'oracle'); tr = p[p.date <= test_months[-1] - OFF(months=1)]
    rg = Ridge(alpha=1.0).fit(tr[RIDGE_W], tr.y)
    print('\nRidge weather coefficients (change in log YoY demand per unit change vs last year):')
    for c, v in zip(RIDGE_W, rg.coef_):
        if c.startswith('w_'): print(f'  {c[2:-2]:<18} {v:+.5f}')


if __name__ == '__main__':
    main()
