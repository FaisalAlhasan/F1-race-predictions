#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import pandas as pd
import numpy as np
import requests

model_pos  = joblib.load('model_pos.pkl')
results_df = joblib.load('results_df.pkl')

sessions = pd.read_csv('sessions.csv', parse_dates=['date_start','date_end'])
race_sessions = (
    sessions[sessions['session_type']=='Race']
    .sort_values(['year','date_start'])
    .assign(round=lambda d: d.groupby('year').cumcount()+1)
)

hist = race_sessions.merge(
    results_df,
    on=['year','round'],
    how='inner'
)


hist = hist.sort_values(['driver','year','round']).reset_index(drop=True)
hist['driver_form'] = (
    hist.groupby('driver')['position']
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
)


hist['driver_volatility'] = (
    hist.groupby('driver')['position']
        .rolling(5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
).fillna(hist['position'].std())

hist['delta'] = hist['position'] - hist['grid']
track_vol = (
    hist.groupby('location')['delta']
        .std()
        .rename('track_volatility')
)
hist = hist.merge(track_vol, on='location', how='left')

# pick race
year, rnd = 2024, 2

curl = f'http://ergast.com/api/f1/{year}/{rnd}/circuits.json'
cres = requests.get(curl, timeout=5); cres.raise_for_status()
loc = cres.json()['MRData']['CircuitTable']['Circuits'][0]['Location']['locality']

tv = track_vol.get(loc, track_vol.mean())

qurl = f'http://ergast.com/api/f1/{year}/{rnd}/qualifying.json?limit=30'
qres = requests.get(qurl, timeout=5); qres.raise_for_status()
qual = qres.json()['MRData']['RaceTable']['Races'][0]['QualifyingResults']

rows = []
for r in qual:
    drv  = r['Driver']['driverId']
    grid = int(r['position'])
    recent = (
        results_df[results_df['driver']==drv]
        .sort_values(['year','round'])
        .tail(5)
    )
    form = recent['position'].mean() if not recent.empty else hist['position'].mean()
    vol  = recent['position'].std()  if len(recent)>1    else hist['position'].std()
    rows.append({
        'driver':            drv,
        'grid':              grid,
        'driver_form':       form,
        'driver_volatility': vol,
        'track_volatility':  tv
    })

up_df = pd.DataFrame(rows)

up_df['pred_pos'] = model_pos.predict(
    up_df[['grid','driver_form','driver_volatility','track_volatility']]
)

pred_df = (
    up_df.sort_values('pred_pos')
         .reset_index(drop=True)
         .assign(predicted_finish=lambda d: d.index+1)
)

print(f"\nPredicted finishing order for {year} round {rnd} ({loc}):\n")
print(pred_df[['predicted_finish','driver','grid','pred_pos']].to_string(index=False))
