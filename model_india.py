"""Forecast monthly energy (MU) per state from Grid-India monthly reports.
Rolling-origin backtest at horizons of 1, 2 and 3 months ahead. For every test month only data
that existed at the forecast origin is used (features AND training rows)."""
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
df = pd.read_csv('india_state_monthly.csv', parse_dates=['date'])
wide = df.pivot(index='date', columns='state', values='energy_met_mu').sort_index()
states = [s for s in wide.columns if s != 'ALL INDIA']
OFF = pd.DateOffset


def growth_list(series, before):
    """YoY ratios for months strictly before `before`, newest first."""
    out = []
    for d in sorted([d for d in series.dropna().index if d < before], reverse=True):
        p = series.get(d - OFF(years=1), np.nan)
        if not np.isnan(p) and p > 0:
            out.append(series[d] / p)
    return out


def feats(series, natl, d, h):
    before = d - OFF(months=h - 1)            # data up to the month before `before` is known
    g, ng = growth_list(series, before), growth_list(natl, before)
    lag12 = series.get(d - OFF(years=1), np.nan)
    if not g or not ng or np.isnan(lag12):
        return None
    lag24 = series.get(d - OFF(years=2), np.nan)
    m = lambda x, k: float(np.mean(x[:k]))
    return dict(date=d, lag12=lag12,
                g1=g[0], g3=m(g, 3), g6=m(g, 6), ng1=ng[0], ng3=m(ng, 3), ng6=m(ng, 6),
                lyg=np.log(lag12 / lag24) if not np.isnan(lag24) and lag24 > 0 else np.nan,
                msin=np.sin(2 * np.pi * d.month / 12), mcos=np.cos(2 * np.pi * d.month / 12),
                actual=series.get(d, np.nan))


def build_panel(h):
    rows = []
    natl = wide['ALL INDIA']
    for s in states:
        for d in wide.index:
            r = feats(wide[s], natl, d, h)
            if r: r['state'] = s; rows.append(r)
    p = pd.DataFrame(rows).dropna(subset=['actual'])
    p['y'] = np.log(p.actual / p.lag12)
    for c in ['g1', 'g3', 'g6', 'ng1', 'ng3', 'ng6']:
        p['l' + c] = np.log(p[c])
    p['lyg_f'] = p.lyg.fillna(0); p['lyg_miss'] = p.lyg.isna().astype(int)
    return p


RIDGE_F = ['lg1', 'lg3', 'lg6', 'lng1', 'lng3', 'lng6', 'lyg_f', 'lyg_miss', 'msin', 'mcos']
HGB_F = ['lg1', 'lg3', 'lg6', 'lng1', 'lng3', 'lng6', 'lyg', 'msin', 'mcos']
models = ['seasonal_naive', 'naive_x_g1', 'naive_x_g3', 'naive_x_g6', 'naive_x_natl_g3',
          'naive_x_natl_g6', 'ridge', 'hgb', 'blend_g3_hgb']


def main():
    test_months = [d for d in wide.index if d >= pd.Timestamp('2025-01-01')]

    out_rows = []
    for h in (1, 2, 3):
        panel = build_panel(h)
        for d in test_months:
            origin_end = d - OFF(months=h)
            tr = panel[panel.date <= origin_end]
            te = panel[panel.date == d].copy()
            if len(tr) < 150 or te.empty:
                continue
            ridge = Ridge(alpha=1.0).fit(tr[RIDGE_F], tr.y)
            hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                                min_samples_leaf=15, random_state=0).fit(tr[HGB_F], tr.y)
            te['seasonal_naive'] = te.lag12
            te['naive_x_g1'] = te.lag12 * te.g1
            te['naive_x_g3'] = te.lag12 * te.g3
            te['naive_x_g6'] = te.lag12 * te.g6
            te['naive_x_natl_g3'] = te.lag12 * te.ng3
            te['naive_x_natl_g6'] = te.lag12 * te.ng6
            te['ridge'] = te.lag12 * np.exp(ridge.predict(te[RIDGE_F]))
            te['hgb'] = te.lag12 * np.exp(hgb.predict(te[HGB_F]))
            te['blend_g3_hgb'] = (te.naive_x_g3 + te.hgb) / 2
            te['horizon'] = h
            out_rows.append(te)
    res = pd.concat(out_rows)


    def table(r):
        t = {}
        for m in models:
            ape = (r[m] - r.actual).abs() / r.actual
            g = r.groupby('date')[[m, 'actual']].sum()
            t[m] = dict(MAPE_state=ape.mean() * 100,
                        WMAPE=(r[m] - r.actual).abs().sum() / r.actual.sum() * 100,
                        AllIndia_err=((g[m] - g.actual).abs() / g.actual).mean() * 100)
        return pd.DataFrame(t).T.round(2).sort_values('MAPE_state')


    print('Test months:', test_months[0].strftime('%Y-%m'), 'to', test_months[-1].strftime('%Y-%m'))
    for h in (1, 2, 3):
        r = res[res.horizon == h]
        print(f'\n=== {h}-month-ahead  ({r.date.nunique()} test months, {len(r)} state-forecasts)')
        print(table(r).to_string())
    last12 = res[(res.horizon == 1) & (res.date >= pd.Timestamp('2025-10-01'))]
    print(f'\n=== 1-month-ahead, most recent 12 months only ({last12.date.nunique()} months)')
    print(table(last12).to_string())
    res.to_csv('backtest_predictions.csv', index=False)


if __name__ == '__main__':
    main()
