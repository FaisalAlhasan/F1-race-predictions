#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import requests
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("🚀 sales.py starting…", flush=True)

# 1. Load session metadata
print("→ Loading sessions.csv", flush=True)
sessions = pd.read_csv('sessions.csv', parse_dates=['date_start','date_end'])

# 2. Fetch historical race results
records = []
print("→ Beginning to fetch Ergast data…", flush=True)
for year in range(2015, 2024):
    for rnd in range(1, 23):
        print(f"   → Fetching {year} round {rnd}…", flush=True)
        try:
            resp = requests.get(
                f'http://ergast.com/api/f1/{year}/{rnd}/results.json?limit=100',
                timeout=5
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"      ⚠️  error fetching {year}/{rnd}: {e}", flush=True)
            continue

        races = resp.json()['MRData']['RaceTable']['Races']
        if not races:
            continue

        for r in races[0]['Results']:
            records.append({
                'year':        year,
                'round':       rnd,
                'driver':      r['Driver']['driverId'],
                'constructor': r['Constructor']['constructorId'],
                'grid':        int(r['grid']),
                'laps':        int(r['laps']),
                'position':    int(r['position']),  # target #1
                'points':      float(r['points']),  # target #2
            })

        time.sleep(0.2)

print(f"→ Fetched {len(records)} result records", flush=True)
results_df = pd.DataFrame(records)

# 3. Assign rounds to sessions and merge
print("→ Assigning rounds to Race sessions…", flush=True)
race_sessions = sessions[sessions['session_type'] == 'Race'].copy()
race_sessions = race_sessions.sort_values(['year', 'date_start'])
race_sessions['round'] = race_sessions.groupby('year').cumcount() + 1
print(f"   → Found {race_sessions.shape[0]} race sessions across years", flush=True)

print("→ Merging sessions with results…", flush=True)
df = race_sessions.merge(results_df, on=['year', 'round'], how='inner')
print(f"→ Merged into {df.shape[0]} rows for modeling", flush=True)

# 4. Feature engineering
df = df.sort_values(['driver', 'year', 'round']).reset_index(drop=True)

# rolling average finish = driver_form
df['driver_form'] = (
    df.groupby('driver')['position']
      .rolling(window=5, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

# **driver_volatility** = std dev of last 5 finishes
df['driver_volatility'] = (
    df.groupby('driver')['position']
      .rolling(window=5, min_periods=1)
      .std()
      .reset_index(level=0, drop=True)
).fillna(df['position'].std())

# **track_volatility** = std dev of (finish - grid) per circuit
track_vol = (
    df.assign(delta=lambda d: d['position'] - d['grid'])
      .groupby('circuit_short_name')['delta']
      .std()
      .rename('track_volatility')
)
df = df.merge(track_vol, on='circuit_short_name', how='left')

# 5. Prepare features and targets
features = ['grid', 'driver_form', 'driver_volatility', 'track_volatility']
X = df[features].fillna(df[features].mean())
y_pos    = df['position']
y_points = df['points']

# 6. Time‐aware Train/Test split
split = TimeSeriesSplit(n_splits=3)
for train_idx, test_idx in split.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_pos, y_test_pos = y_pos.iloc[train_idx], y_pos.iloc[test_idx]
    y_train_pts, y_test_pts = y_points.iloc[train_idx], y_points.iloc[test_idx]

# 7. Train & evaluate position model
model_pos = RandomForestRegressor(n_estimators=100, random_state=42)
model_pos.fit(X_train, y_train_pos)
y_pred_pos = model_pos.predict(X_test)

mae_pos  = mean_absolute_error(y_test_pos, y_pred_pos)
rmse_pos = np.sqrt(mean_squared_error(y_test_pos, y_pred_pos))
print(f"Finishing-Position → MAE: {mae_pos:.3f}, RMSE: {rmse_pos:.3f}", flush=True)

# 8. Train & evaluate points model
model_pts = RandomForestRegressor(n_estimators=100, random_state=42)
model_pts.fit(X_train, y_train_pts)
y_pred_pts = model_pts.predict(X_test)

mae_pts  = mean_absolute_error(y_test_pts, y_pred_pts)
rmse_pts = np.sqrt(mean_squared_error(y_test_pts, y_pred_pts))
print(f"Points Scored → MAE: {mae_pts:.3f}, RMSE: {rmse_pts:.3f}", flush=True)

# 9. Serialize the trained position model & data
joblib.dump(model_pos, 'model_pos.pkl')
joblib.dump(results_df, 'results_df.pkl')
print("💾 Saved model_pos.pkl and results_df.pkl", flush=True)

print("✅ sales.py completed successfully", flush=True)
