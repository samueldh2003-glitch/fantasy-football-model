# Fantasy Football Model

Mobile-first fantasy football projection app with a statistical projection engine.

## V0.4
- Real 2026 projections displayed in the app from generated model data
- PPR and Non-PPR rankings
- League-size-aware replacement value
- League-value sorting
- Player detail pages with projection, PPG, floor, ceiling and league value
- Validation metrics surfaced in the Method page
- Local league settings persisted on device
- No fabricated K/DST projections

## V0.3 model foundation
- Real historical NFL data from nflverse
- Forward-in-time season-to-season training examples
- Position-specific statistical models for QB/RB/WR/TE
- Player production, opportunity, efficiency, age/experience and team environment features
- TD rate included as a learned predictive feature
- Model-generated 10th/50th/90th percentile outcome ranges
- Healthy/available projection assumption; no injury probability model
- No betting-market inputs
- PPR and Non-PPR outputs
- Automated GitHub Actions model rebuild
- Static projection JSON consumed by the mobile web app

K and D/ST are intentionally **not fabricated**. Their position-specific pipelines will be added after validation rather than filling the app with unsupported numbers.

## Data
The model uses nflverse player-level NFL statistics and player identity/biographical data.

## Next milestones
- Broader forward-chaining backtests and calibration
- Better uncertainty calibration
- K and D/ST models
- Explicit offseason/current-team context
- Weekly projections
- ESPN and Sleeper ADP
- Model-vs-market draft value
- Richer player explanations
- Continued leakage/overfitting controls

## Run locally
```bash
pip install -r requirements.txt
python model/build_model.py
```

The website is static and requires no account or frontend build step.
