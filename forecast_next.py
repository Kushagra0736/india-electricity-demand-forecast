"""Forecast the next 12 months of energy (MU) for every state and all India.
Method: average of (a) same month last year x recent YoY growth and (b) a gradient-boosting model of the
YoY ratio. 'low'/'high' = 80th-percentile absolute % error seen at that horizon in backtest_long.csv.
Run backtest_long.py first."""
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import model_india as mi

warnings.filterwarnings('ignore')
wide, states, OFF = mi.wide, mi.states, mi.OFF
last = wide.index.max()
bt = pd.read_csv('backtest_long.csv', parse_dates=['date'])
natl = wide['ALL INDIA']
ratio = (wide[states].sum(axis=1) / natl).iloc[-12:].mean()      # states sum to ~97.6% of all-India

rows = []
for h in range(1, 13):
    d = last + OFF(months=h)
    panel = mi.build_panel(h)
    tr = panel[panel.date <= last]
    hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                        min_samples_leaf=15, random_state=0).fit(tr[mi.HGB_F], tr.y)
    b = bt[bt.horizon == h]
    q_state = float((((b.blend - b.actual).abs()) / b.actual).quantile(0.8))
    g = b.groupby('date')[['blend', 'actual']].sum()
    q_all = float((((g.blend - g.actual).abs()) / g.actual).quantile(0.8))
    preds = {}
    for s in states:
        r = mi.feats(wide[s], natl, d, h)
        if r is None:
            continue
        x = pd.DataFrame([r])
        for c in ['g1', 'g3', 'g6', 'ng1', 'ng3', 'ng6']:
            x['l' + c] = np.log(x[c])
        pred = (r['lag12'] * r['g3'] + r['lag12'] * np.exp(hgb.predict(x[mi.HGB_F])[0])) / 2
        preds[s] = pred
        rows.append(dict(state=s, month=d.strftime('%Y-%m'), horizon_months=h,
                         same_month_last_year_mu=r['lag12'], forecast_mu=round(pred),
                         low_mu=round(pred * (1 - q_state)), high_mu=round(pred * (1 + q_state))))
    tot = sum(preds.values()) / ratio
    rows.append(dict(state='ALL INDIA', month=d.strftime('%Y-%m'), horizon_months=h,
                     same_month_last_year_mu=natl.get(d - OFF(years=1)), forecast_mu=round(tot),
                     low_mu=round(tot * (1 - q_all)), high_mu=round(tot * (1 + q_all))))
out = pd.DataFrame(rows)
out['yoy_growth_pct'] = ((out.forecast_mu / out.same_month_last_year_mu - 1) * 100).round(1)
out.to_csv('forecast_next_months.csv', index=False)
print(out.state.nunique(), 'entities x', out.month.nunique(), 'months:', out.month.min(), 'to', out.month.max())
print(out[out.state.isin(['ALL INDIA', 'Delhi'])].pivot(index='month', columns='state', values='forecast_mu').to_string())
