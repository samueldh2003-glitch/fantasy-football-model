# Fantasy Football Model

Mobile-first fantasy football projection app with a statistical projection engine.

## V0.3
- Real historical NFL data from nflverse
- Forward-in-time season-to-season training examples
- Position-specific statistical models for QB/RB/WR/TE
- Player production, opportunity, efficiency, age/experience and team environment features
- TD-rate regression through learned features
- Model-generated 10th/50th/90th percentile outcome ranges
- Healthy/available projection assumption; no injury probability model
- No betting-market inputs
- PPR and Non-PPR outputs
- Automated GitHub Actions model rebuild
- Static projection JSON consumed by the mobile web app

K and D/ST are intentionally **not fabricated** in V0.3. Their position-specific pipelines will be added after validation rather than filling the app with unsupported numbers.

## Data
The model uses nflverse player-level NFL statistics. nflverse provides season/week player statistics and player identity data; its player data includes NFL/ESPN IDs and biographical information. See the nflverse documentation for the data dictionary and update schedule.

## V0.4 roadmap
- Validate and improve the model with broader forward-chaining backtests
- Add K and D/ST models
- Add current-team/offseason context more explicitly
- Add weekly projections
- Add ESPN and Sleeper ADP
- Calculate model-vs-market draft value
- Add richer player explanation pages
- Continue reducing leakage and overfitting

## Run locally
```bash
pip install -r requirements.txt
python model/build_model.py
```

The website itself remains static and requires no account or build step.
