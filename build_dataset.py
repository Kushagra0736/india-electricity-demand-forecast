"""Clean the raw parse into one tidy table: one row per state per month."""
import pandas as pd

raw = pd.read_csv('india_state_monthly_raw.csv')
raw['state'] = raw.state.str.replace(r'^[\s\._]+', '', regex=True).str.strip()
# name variants in older reports (region tag glued to the name, truncated brackets)
raw['state'] = raw.state.replace({'WR Goa': 'Goa', 'NR Punjab': 'Punjab', 'SR Kerala': 'Kerala',
                                  'J&K(UT) and Ladakh(UT': 'J&K(UT) and Ladakh(UT)',
                                  'DNHDDPDCL': 'DNH & DD (UT)'})
# not states: private plants, railways, DVC
drop = {'AMNSIL', 'BALCO', 'DVC', 'Railways_ER ISTS', 'RIL JAMNAGAR', 'Railway_NR ISTS/NFL',
        'Railway_NR ISTS', 'NFL', 'Railway/NFL'}
raw = raw[~raw.state.isin(drop)].copy()

# every month appears in its own report (is_current=1) and in next year's report as "last year" (is_current=0)
key = ['state', 'year', 'month']
cur = raw[raw.is_current == 1].drop_duplicates(key)
old = raw[raw.is_current == 0].drop_duplicates(key)
both = cur.merge(old, on=key, suffixes=('', '_later'))
rev = (both.energy_met_mu - both.energy_met_mu_later).abs()
print(f'months present in two reports: {len(both)} rows; energy_met differs (revised) in {(rev > 0.5).sum()} rows, '
      f'max diff {rev.max():.0f} MU')
# keep the figure from the month's own report; use the later one only to fill months with no report of their own
df = pd.concat([cur, old[~old.set_index(key).index.isin(cur.set_index(key).index)]], ignore_index=True)
df['from_own_report'] = df.is_current
df['date'] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
df = df.sort_values(['state', 'date']).drop(columns=['source', 'is_current'])
df['days_in_month'] = df.date.dt.days_in_month
df['energy_met_mu_per_day'] = (df.energy_met_mu / df.days_in_month).round(2)
df.to_csv('india_state_monthly.csv', index=False)

# validation: states should add up to ~97-98% of ALL INDIA every month
st = df[df.state != 'ALL INDIA'].groupby('date').energy_met_mu.sum()
ai = df[df.state == 'ALL INDIA'].set_index('date').energy_met_mu
ratio = (st / ai)
print(df.shape, df.state.nunique(), 'entities;', df.date.nunique(), 'months:',
      df.date.min().strftime('%Y-%m'), 'to', df.date.max().strftime('%Y-%m'))
print(f'states/all-India ratio: min {ratio.min():.3f}, max {ratio.max():.3f}')
bad = ratio[(ratio < 0.96) | (ratio > 0.99)]
print('months failing the check:', [d.strftime('%Y-%m') for d in bad.index] or 'none')
all_months = pd.date_range(df.date.min(), df.date.max(), freq='MS')
print('missing months:', [d.strftime('%Y-%m') for d in all_months if d not in set(df.date)])
print('entities per month (min/max):', df.groupby('date').state.nunique().min(), df.groupby('date').state.nunique().max())
