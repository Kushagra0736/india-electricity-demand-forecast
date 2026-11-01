"""How much does a warmer/cooler month than last year change demand growth?
Pooled regression (state-months, 1-month-ahead backtest residuals vs temperature change). Writes sensitivity.json."""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')
r = pd.read_csv('backtest_weather_predictions.csv', parse_dates=['date'])
w = pd.read_csv('weather_state_monthly.csv', parse_dates=['date'])
T = w.pivot(index='date', columns='state', values='avg_temp_c')
x = r[r.horizon == 1].copy()
x['dT'] = [T.loc[d, s] - T.loc[d - pd.DateOffset(years=1), s] for s, d in zip(x.state, x.date)]
x['res'] = np.log(x.actual / x.ridge)          # growth the no-weather model did not explain
x['hot'] = x.date.dt.month.isin([4, 5, 6, 7, 8, 9])
out = {}
for name, sub in [('hot_apr_sep', x[x.hot]), ('cool_oct_mar', x[~x.hot])]:
    m = sm.OLS(sub.res, sm.add_constant(sub.dT)).fit(cov_type='cluster', cov_kwds={'groups': sub.date.astype(str)})
    lo, hi = m.conf_int().loc['dT']
    out[name] = dict(pct_per_degC=round(100 * m.params['dT'], 2), ci_low=round(100 * lo, 2),
                     ci_high=round(100 * hi, 2), n=int(len(sub)))
json.dump(out, open('sensitivity.json', 'w'), indent=2); print(out)
