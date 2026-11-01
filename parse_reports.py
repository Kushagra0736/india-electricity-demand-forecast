"""Extract state-wise monthly energy and peak demand from Grid-India (NLDC) Monthly Operational Reports.
Usage: python parse_reports.py path/to/pdfs
Each report gives the report month AND the same month of the previous year (is_current flags which)."""
import glob, hashlib, os, re, subprocess, sys
import pandas as pd

UP = sys.argv[1] if len(sys.argv) > 1 else '.'
MONTHS = {'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
          'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7, 'august': 8, 'aug': 8,
          'september': 9, 'sep': 9, 'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
          'december': 12, 'dec': 12}
REGION_TAGS = {'NR', 'WR', 'SR', 'ER', 'NER', 'TOTAL'}
NUM = re.compile(r'-?\d+\.?\d*')
NAME_RE = re.compile(r'Monthly_Report_([A-Za-z]+)_(\d{4})')


def clean(line):
    s = re.sub(r'[^\x20-\x7e]', ' ', line)
    return re.sub(r'\s+', ' ', s).strip()


def parse_section(lines, start_pat, end_pat, n_nums):
    """Return [(name, numbers)]. Handles names printed on the line after the numbers."""
    i0 = next(i for i, l in enumerate(lines) if re.search(start_pat, clean(l)) and 'OF' in clean(l).upper() and i > 150)
    i1 = next(i for i in range(i0 + 1, len(lines)) if re.search(end_pat, clean(lines[i])) and i > 150)
    out, pending = [], None
    for l in lines[i0:i1]:
        s = clean(l)
        if not s:
            continue
        nums = NUM.findall(s)
        text = re.sub(r'-?\d+\.?\d*%?', '', s).replace('|', '').strip()
        text = re.sub(r'[()._/:\-]+$', '', text).strip()
        if len(nums) == n_nums and not re.search(r'[A-Za-z]{3,}.*\d.*[A-Za-z]{3,}', s):
            if text and re.search(r'[A-Za-z]{2,}', text):
                out.append((text, [float(x) for x in nums])); pending = None
            else:
                pending = [float(x) for x in nums]
        elif pending is not None and re.fullmatch(r"[A-Za-z&()\. ,_/]+", s) and s not in REGION_TAGS and len(s) > 3:
            out.append((s, pending)); pending = None
    return [(n, v) for n, v in out if not n.upper().startswith('TOTAL')]


def main():
    files, seen = [], {}
    for f in sorted(glob.glob(os.path.join(UP, 'Monthly_Report_*.pdf'))):
        h = hashlib.md5(open(f, 'rb').read()).hexdigest()
        if h in seen:
            print(f'skip duplicate: {os.path.basename(f)} (same as {os.path.basename(seen[h])})'); continue
        seen[h] = f; files.append(f)
    rows = []
    for f in files:
        base = os.path.basename(f)
        m = NAME_RE.match(base)
        if not m or m.group(1).lower() not in MONTHS:
            print('cannot read month/year from name:', base); continue
        mon, yr = MONTHS[m.group(1).lower()], int(m.group(2))
        txt = subprocess.run(['pdftotext', '-layout', f, '-'], capture_output=True).stdout.decode('utf-8', 'ignore')
        lines = txt.split('\n')
        try:
            energy = parse_section(lines, r'12\. ENERGY COMPARISON', r'13\. (PEAK )?DEMAND COMPARISON', 10)
            peak = parse_section(lines, r'13\. (PEAK )?DEMAND COMPARISON', r'14\. SCHEDULE (AND )?DRAWAL', 8)
        except StopIteration:
            print(f'!! {base}: could not find the comparison tables (layout differs)'); continue
        e, p = dict(energy), dict(peak)
        for name, ev in e.items():
            pv = p.get(name)
            for k, y in enumerate([yr - 1, yr]):
                rows.append(dict(state=name, year=y, month=mon, source=base, is_current=int(k == 1),
                                 energy_req_mu=ev[k], energy_met_mu=ev[5 + k],
                                 peak_demand_mw=pv[k] if pv else None, peak_met_mw=pv[4 + k] if pv else None))
        print(f'{base}: {len(e)} energy rows, {len(p)} peak rows')
    pd.DataFrame(rows).to_csv('india_state_monthly_raw.csv', index=False)
    print('saved', len(rows), 'rows from', len(files), 'unique reports')


if __name__ == '__main__':
    main()
