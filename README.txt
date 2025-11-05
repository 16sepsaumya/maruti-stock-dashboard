
# StockMarket FastAPI Demo

**Files created** in: /mnt/data/stockapp

- `app.py` – FastAPI server
- `templates/index.html`, `templates/bse.html`, `templates/nifty.html`
- `static/style.css`

**Relies on your uploaded files** (already referenced by absolute paths):
- `/mnt/data/MARUTI.csv`
- `/mnt/data/maruti_lr_model.pkl`
- `/mnt/data/maruti_lstm_model.h5`
- `/mnt/data/maruti_scaler.pkl`

## Run locally
```bash
cd /mnt/data/stockapp
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- Home: http://localhost:8000/
- BSE:  http://localhost:8000/bse
- NIFTY: http://localhost:8000/nifty

## Notes
- The app auto-detects the `date` and `close` columns in `MARUTI.csv`. Expected column names include variants like `Date`/`Timestamp` and `Close`/`Adj Close`.
- LR prediction uses a small feature set (last close, 5‑day return, 5‑day SMA) to be model‑agnostic. If your LR was trained on different features, you can adapt the `predict_next_close_lr` function.
- LSTM prediction uses the last 60 closes with your `maruti_scaler.pkl` and `maruti_lstm_model.h5`. If sequence length differs, update `LSTM_WINDOW` accordingly.
- Charts are generated server‑side as Base64 images.
