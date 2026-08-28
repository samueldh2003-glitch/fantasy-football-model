import json, os
from datetime import datetime
import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

YEARS = list(range(2012, 2026))
POSITIONS = ['QB', 'RB', 'WR', 'TE']

def pdf(x):
    return x.to_pandas() if hasattr(x, 'to_pandas') else pd.DataFrame(x)

def num(d, *names):
    for n in names:
        if n in d.columns:
            return pd.to_numeric(d[n], errors='coerce').fillna(0.0)
    return pd.Series(0.0, index=d.index)

def div(a, b):
    return a.divide(b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def fantasy(d, ppr=True):
    y = num(d, 'passing_yards')*.04 + num(d, 'passing_tds')*4 - num(d, 'passing_interceptions')*2
    y += num(d, 'rushing_yards')*.1 + num(d, 'rushing_tds')*6
    y += num(d, 'receiving_yards')*.1 + num(d, 'receiving_tds')*6
    y -= num(d, 'fumbles_lost')*2
    if ppr:
        y += num(d, 'receptions')
    for x in ['passing_2pt_conversions', 'rushing_2pt_conversions', 'receiving_2pt_conversions']:
        y += num(d, x)*2
    return y

print('Loading nflverse player data:', YEARS)
r = pdf(nfl.load_player_stats(seasons=YEARS, summary_level='reg'))
if 'season_type' in r.columns:
    r = r[r['season_type'].astype(str).str.upper().eq('REG')]
r = r[r['position'].isin(POSITIONS)].copy()
pid = 'player_id' if 'player_id' in r.columns else 'gsis_id'
name = 'player_display_name' if 'player_display_name' in r.columns else 'player_name'

if 'team' not in r.columns:
    for c in ['recent_team', 'latest_team', 'team_abbr', 'posteam', 'fantasy_team']:
        if c in r.columns:
            r['team'] = r[c]
            break
if 'team' not in r.columns:
    r['team'] = 'UNK'

keys = [pid, 'season', 'position']
number_cols = r.select_dtypes(include=[np.number]).columns.tolist()
a = r.groupby(keys, as_index=False)[number_cols].sum()
nm = r.groupby(keys)[name].first().reset_index()
tm = (r.groupby(keys + ['team'], as_index=False).size()
      .sort_values('size').drop_duplicates(keys)[keys + ['team']])
a = a.merge(nm, on=keys, how='left').merge(tm, on=keys, how='left')

a['games'] = num(a, 'games', 'games_played')
a['pass_att'] = num(a, 'attempts', 'passing_attempts')
a['pass_yds'] = num(a, 'passing_yards')
a['pass_td'] = num(a, 'passing_tds')
a['ints'] = num(a, 'passing_interceptions')
a['carries'] = num(a, 'carries', 'rushing_attempts')
a['rush_yds'] = num(a, 'rushing_yards')
a['rush_td'] = num(a, 'rushing_tds')
a['targets'] = num(a, 'targets')
a['rec'] = num(a, 'receptions')
a['rec_yds'] = num(a, 'receiving_yards')
a['rec_td'] = num(a, 'receiving_tds')
a['fumbles_lost'] = num(a, 'fumbles_lost')
a['td'] = a['pass_td'] + a['rush_td'] + a['rec_td']
a['touches'] = a['carries'] + a['targets']
a['ppr'] = fantasy(a, True)
a['std'] = fantasy(a, False)
a['ppr_ppg'] = div(a['ppr'], a['games'])
a['std_ppg'] = div(a['std'], a['games'])

r['_pa'] = num(r, 'attempts', 'passing_attempts')
r['_py'] = num(r, 'passing_yards')
r['_ptd'] = num(r, 'passing_tds')
r['_ra'] = num(r, 'carries', 'rushing_attempts')
r['_ry'] = num(r, 'rushing_yards')
r['_rtd'] = num(r, 'rushing_tds')
t = r.groupby(['season', 'team'], as_index=False)[['_pa','_py','_ptd','_ra','_ry','_rtd']].sum()
t = t.rename(columns={'_pa':'team_pass_att','_py':'team_pass_yds','_ptd':'team_pass_td','_ra':'team_rush_att','_ry':'team_rush_yds','_rtd':'team_rush_td'})
a = a.merge(t, on=['season', 'team'], how='left')
a['target_share'] = div(a['targets'], a['team_pass_att'])
a['rush_share'] = div(a['carries'], a['team_rush_att'])
a['yds_touch'] = div(a['rush_yds'] + a['rec_yds'], a['touches'])
a['td_rate'] = div(a['td'], a['touches'])
a['pass_td_rate'] = div(a['pass_td'], a['pass_att'])
a['rush_td_rate'] = div(a['rush_td'], a['carries'])
a['rec_td_rate'] = div(a['rec_td'], a['targets'])

try:
    b = pdf(nfl.load_players())
    bp = 'gsis_id' if 'gsis_id' in b.columns else 'player_id'
    bc = [bp] + [c for c in ['birth_date', 'years_exp'] if c in b.columns]
    b = b[bc].drop_duplicates(bp).rename(columns={bp: pid})
    a = a.merge(b, on=pid, how='left')
except Exception:
    pass

if 'birth_date' in a.columns:
    bd = pd.to_datetime(a['birth_date'], errors='coerce')
    a['age'] = a['season'] - bd.dt.year - ((bd.dt.month*100 + bd.dt.day) > 701).astype(int)
else:
    a['age'] = np.nan
a['age'] = a.groupby('position')['age'].transform(lambda s: s.fillna(s.median())).fillna(26)
if 'years_exp' in a.columns:
    a['experience'] = pd.to_numeric(a['years_exp'], errors='coerce').fillna(0)
else:
    a['experience'] = 0.0

BASE = ['age','experience','games','ppr_ppg','std_ppg','touches','targets','carries','rec','rush_yds','rec_yds','td','td_rate','target_share','rush_share','yds_touch','pass_td_rate','rush_td_rate','rec_td_rate','team_pass_att','team_pass_yds','team_pass_td','team_rush_att','team_rush_yds','team_rush_td']
TREND = ['ppr_ppg','std_ppg','touches','targets','carries','td_rate','target_share','rush_share','yds_touch']
rows = []

for _, g in a.groupby([pid, 'position']):
    g = g.sort_values('season')
    for i in range(len(g)-1):
        cur, nxt = g.iloc[i], g.iloc[i+1]
        if int(nxt['season']) != int(cur['season']) + 1 or float(nxt['games']) < 6:
            continue
        z = {f: cur.get(f, np.nan) for f in BASE}
        h = g[g['season'] <= cur['season']].tail(3)
        for f in TREND:
            z[f+'_3yr'] = h[f].mean()
            z[f+'_trend'] = h[f].iloc[-1] - h[f].iloc[0] if len(h) > 1 else 0
        z.update(player_id=cur[pid], player_name=cur[name], position=cur['position'], season=int(cur['season']), target_ppr=float(nxt['ppr_ppg']), target_std=float(nxt['std_ppg']))
        rows.append(z)

panel = pd.DataFrame(rows)
if panel.empty:
    raise RuntimeError('No valid player-season training rows were created.')
FEATURES = BASE + [c for c in panel.columns if c.endswith('_3yr') or c.endswith('_trend')]
models = {}
validation = []

for pos in POSITIONS:
    d = panel[panel['position'] == pos].copy()
    if len(d) < 40:
        continue
    hold = int(d['season'].max())
    tr, va = d[d['season'] < hold], d[d['season'] == hold]
    med = tr[FEATURES].median(numeric_only=True).fillna(0)
    X = tr[FEATURES].replace([np.inf,-np.inf], np.nan).fillna(med)
    XV = va[FEATURES].replace([np.inf,-np.inf], np.nan).fillna(med)
    base_model = GradientBoostingRegressor(n_estimators=180, max_depth=2, learning_rate=.035, loss='huber', random_state=42)
    base_model.fit(X, tr['target_ppr'])
    validation.append({'position':pos,'holdout_season':hold,'n':int(len(va)),'mae_ppg':round(float(mean_absolute_error(va['target_ppr'], base_model.predict(XV))),3)})
    qs = []
    for q in (.10, .50, .90):
        m = GradientBoostingRegressor(n_estimators=220, max_depth=2, learning_rate=.03, loss='quantile' if q != .50 else 'huber', alpha=q, random_state=42)
        m.fit(X, tr['target_ppr'])
        qs.append(m)
    models[pos] = (qs, med)

latest = a[a['season'] == 2025].copy()
for f in TREND:
    latest[f+'_3yr'] = latest.apply(lambda x: a[(a[pid] == x[pid]) & (a['season'] <= 2025)].tail(3)[f].mean(), axis=1)
    latest[f+'_trend'] = latest.apply(lambda x: (lambda h: h[f].iloc[-1] - h[f].iloc[0] if len(h) > 1 else 0)(a[(a[pid] == x[pid]) & (a['season'] <= 2025)].tail(3)), axis=1)

out = []
for _, p in latest.iterrows():
    if p['position'] not in models:
        continue
    qs, med = models[p['position']]
    x = pd.DataFrame([{f:p.get(f, np.nan) for f in FEATURES}]).replace([np.inf,-np.inf], np.nan).fillna(med).fillna(0)
    v = sorted(float(m.predict(x)[0]) for m in qs)
    lo, mid, hi = v
    gap = max(0, float(p['ppr_ppg'] - p['std_ppg']))
    out.append({'playerId':str(p[pid]),'name':str(p[name]),'position':str(p['position']),'team':str(p['team']),'projectionPPR':round(mid*17,1),'ppgPPR':round(mid,2),'floorPPR':round(lo*17,1),'ceilingPPR':round(hi*17,1),'projectionNonPPR':round(max(0,mid-gap)*17,1),'ppgNonPPR':round(max(0,mid-gap),2),'floorNonPPR':round(max(0,lo-gap)*17,1),'ceilingNonPPR':round(max(0,hi-gap)*17,1),'healthyAssumption':True})

out.sort(key=lambda p: -p['projectionPPR'])
for i, p in enumerate(out, 1):
    p['modelRankPPR'] = i
for i, p in enumerate(sorted(out, key=lambda p: -p['projectionNonPPR']), 1):
    p['modelRankNonPPR'] = i

result = {'schemaVersion':1,'modelVersion':'0.3.2','generatedAt':datetime.utcnow().isoformat(timespec='seconds')+'Z','source':'nflverse','sourceCoverage':'2012-2025','projectionSeason':2026,'healthyAssumption':True,'injuryProbabilityModeled':False,'bettingMarketsUsed':False,'positions':POSITIONS,'note':'Real statistical projections for QB/RB/WR/TE. K and DST intentionally deferred.','validation':validation,'players':out}
os.makedirs('data', exist_ok=True)
with open('data/projections.json','w') as f: json.dump(result, f, indent=2)
with open('data/model_status.json','w') as f: json.dump({'status':'ready','modelVersion':'0.3.2','generatedAt':result['generatedAt'],'players':len(out),'validation':validation}, f, indent=2)
print('Generated', len(out), 'player projections')
print(validation)
