"""Build india_dashboard.html: injects the pipeline's real data into tpl.html."""
import json, numpy as np, pandas as pd
h = pd.read_csv('india_state_monthly.csv', parse_dates=['date']); fc = pd.read_csv('forecast_next_months.csv')
bt = pd.read_csv('backtest_long.csv', parse_dates=['date']); wx = pd.read_csv('weather_state_monthly.csv', parse_dates=['date'])
sens = json.load(open('sensitivity.json'))
hm = [d.strftime('%Y-%m') for d in pd.date_range(h.date.min(), h.date.max(), freq='MS')]
fm = sorted(fc.month.unique()); bm = sorted(bt[bt.horizon == 1].date.unique())
R = lambda x, n=0: None if pd.isna(x) else round(float(x), n)
wx_all = wx.groupby('date').avg_temp_c.mean()
ents = {}
for s in ['ALL INDIA'] + sorted(x for x in h.state.unique() if x != 'ALL INDIA'):
    hs = h[h.state == s].set_index('date').energy_met_mu
    ts = wx_all if s == 'ALL INDIA' else wx[wx.state == s].set_index('date').avg_temp_c
    f = fc[fc.state == s].sort_values('month')
    acc = []
    for hz in (1, 3, 6, 12):
        b = bt[bt.horizon == hz]
        if s == 'ALL INDIA':
            g = b.groupby('date')[['blend', 'seasonal_naive', 'actual']].sum(); n = len(g)
        else:
            g = b[b.state == s]; n = len(g)
        acc.append([R(((g.blend - g.actual).abs() / g.actual).mean() * 100, 2), R(((g.seasonal_naive - g.actual).abs() / g.actual).mean() * 100, 2), n])
    b1 = bt[bt.horizon == 1]
    g = (b1.groupby('date')[['blend', 'actual']].sum() if s == 'ALL INDIA' else b1[b1.state == s].set_index('date')[['blend', 'actual']]).reindex(bm)
    ents[s] = dict(h=[R(hs.get(pd.Timestamp(m + '-01'))) for m in hm], t=[R(ts.get(pd.Timestamp(m + '-01')), 1) for m in hm],
                   f=f.forecast_mu.tolist(), lo=f.low_mu.tolist(), hi=f.high_mu.tolist(), acc=acc,
                   ba=[R(v) for v in g.actual], bp=[R(v) for v in g.blend],
                   l12=R(hs.tail(12).sum()), n12=R(f.forecast_mu.sum()))
    # ALL INDIA hist/last12 use the all-India row itself
D = dict(hm=hm, fm=fm, bm=[pd.Timestamp(d).strftime('%Y-%m') for d in bm], ents=ents,
         sens=dict(h=dict(p=sens['hot_apr_sep']['pct_per_degC'], lo=sens['hot_apr_sep']['ci_low'], hi=sens['hot_apr_sep']['ci_high'])))
open('india_dashboard.html', 'w').write(open('tpl.html').read().replace('__DATA__', json.dumps(D, separators=(',', ':'))))
print('built', len(open('india_dashboard.html').read()) // 1024, 'KB')
