"""Build the JSON that the dashboard template needs from the pipeline CSVs."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
REG = {'NR': ['Chandigarh', 'Delhi', 'Haryana', 'Himachal Pradesh', 'J&K(UT) and Ladakh(UT)', 'Punjab', 'Rajasthan', 'Uttar Pradesh', 'Uttarakhand'],
       'WR': ['Chhattisgarh', 'DNH & DD (UT)', 'Goa', 'Gujarat', 'Madhya Pradesh', 'Maharashtra'],
       'SR': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Puducherry', 'Tamil Nadu', 'Telangana'],
       'ER': ['Bihar', 'Jharkhand', 'Odisha', 'Sikkim', 'West Bengal'],
       'NER': ['Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Tripura']}
REGION = {s: r for r, ss in REG.items() for s in ss} | {'ALL INDIA': 'India'}
R = lambda x, n=0: None if pd.isna(x) else round(float(x), n)


def build():
    h = pd.read_csv(HERE / 'india_state_monthly.csv', parse_dates=['date']); fc = pd.read_csv(HERE / 'forecast_next_months.csv')
    bt = pd.read_csv(HERE / 'backtest_long.csv', parse_dates=['date']); wx = pd.read_csv(HERE / 'weather_state_monthly.csv', parse_dates=['date'])
    sens = json.load(open(HERE / 'sensitivity.json', encoding='utf-8'))['hot_apr_sep']
    hm = [d.strftime('%Y-%m') for d in pd.date_range(h.date.min(), h.date.max(), freq='MS')]
    fm = sorted(fc.month.unique()); bm = sorted(bt[bt.horizon == 1].date.unique())
    wx_all = wx.groupby('date').avg_temp_c.mean(); ents = {}
    for s in ['ALL INDIA'] + sorted(x for x in h.state.unique() if x != 'ALL INDIA'):
        hs = h[h.state == s].set_index('date').energy_met_mu
        ts = wx_all if s == 'ALL INDIA' else wx[wx.state == s].set_index('date').avg_temp_c
        f = fc[fc.state == s].sort_values('month'); acc = []
        for hz in (1, 3, 6, 12):
            b = bt[bt.horizon == hz]
            g = b.groupby('date')[['blend', 'seasonal_naive', 'actual']].sum() if s == 'ALL INDIA' else b[b.state == s]
            acc.append([R(((g.blend - g.actual).abs() / g.actual).mean() * 100, 2), R(((g.seasonal_naive - g.actual).abs() / g.actual).mean() * 100, 2), len(g)])
        b1 = bt[bt.horizon == 1]
        g = (b1.groupby('date')[['blend', 'actual']].sum() if s == 'ALL INDIA' else b1[b1.state == s].set_index('date')[['blend', 'actual']]).reindex(bm)
        ents[s] = dict(h=[R(hs.get(pd.Timestamp(m + '-01'))) for m in hm], t=[R(ts.get(pd.Timestamp(m + '-01')), 1) for m in hm],
                       f=f.forecast_mu.tolist(), lo=f.low_mu.tolist(), hi=f.high_mu.tolist(), acc=acc,
                       ba=[R(v) for v in g.actual], bp=[R(v) for v in g.blend], l12=R(hs.tail(12).sum()), n12=R(f.forecast_mu.sum()))
    return dict(hm=hm, fm=fm, bm=[pd.Timestamp(d).strftime('%Y-%m') for d in bm], ents=ents, reg=REGION,
                sens=dict(h=dict(p=sens['pct_per_degC'], lo=sens['ci_low'], hi=sens['ci_high'])),
                nstates=len(ents) - 1)


def render():
    return (HERE / 'dashboard_template.html').read_text(encoding='utf-8').replace('__DATA__', json.dumps(build(), separators=(',', ':')))
