"""Backtest the chosen method (blend of 'last year x recent growth' and gradient boosting) at 1-12 months ahead."""
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import model_india as mi

warnings.filterwarnings('ignore')
OFF = mi.OFF
test_months = [d for d in mi.wide.index if d >= pd.Timestamp('2025-01-01')]
rows = []
for h in range(1, 13):
    panel = mi.build_panel(h)
    for d in test_months:
        tr = panel[panel.date <= d - OFF(months=h)]
        te = panel[panel.date == d].copy()
        if len(tr) < 150 or te.empty:
            continue
        hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                            min_samples_leaf=15, random_state=0).fit(tr[mi.HGB_F], tr.y)
        te['seasonal_naive'] = te.lag12
        te['naive_x_g3'] = te.lag12 * te.g3
        te['hgb'] = te.lag12 * np.exp(hgb.predict(te[mi.HGB_F]))
        te['blend'] = (te.naive_x_g3 + te.hgb) / 2
        te['horizon'] = h
        rows.append(te[['state', 'date', 'horizon', 'actual', 'seasonal_naive', 'naive_x_g3', 'hgb', 'blend']])
res = pd.concat(rows)
res.to_csv('backtest_long.csv', index=False)
for h, r in res.groupby('horizon'):
    out = {}
    for m in ['seasonal_naive', 'blend']:
        out[m] = ((r[m] - r.actual).abs() / r.actual).mean() * 100
    g = r.groupby('date')[['blend', 'actual']].sum()
    q = (((r.blend - r.actual).abs()) / r.actual).quantile(0.8) * 100
    print(f"h={h:2d} months={r.date.nunique():2d}  state MAPE: last-year-only {out['seasonal_naive']:.1f}%  blend {out['blend']:.1f}%  "
          f"| all-India {((g.blend - g.actual).abs() / g.actual).mean() * 100:.1f}% | 80% range +/-{q:.1f}%")
