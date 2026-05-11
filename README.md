# Daylight Strategy v2 — Backend Server

## What this does
The Python backend runs backtests and paper trading **in the background**.  
**Closing the browser does NOT stop the test.** State is saved to disk.

## Setup (first time)

```bash
pip install flask flask-cors python-binance pandas numpy matplotlib
```

## Run the server

```bash
python server.py
```

Then open your browser at: **http://localhost:5000**

## How it works

| Component | What it does |
|-----------|-------------|
| `server.py` | Flask server — keeps running even if browser is closed |
| `static/dashboard.html` | Web UI — connects to backend via REST + SSE |
| `daylight_state.json` | Backtest results (auto-saved) |
| `paper_trades.json` | Paper trading state (auto-saved, survives restart) |

## API endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `GET /api/status` | GET | Current job status |
| `GET /api/logs` | GET | Last 100 log lines |
| `GET /api/results` | GET | Backtest results |
| `GET /api/paper/state` | GET | Paper trading state |
| `POST /api/start` | POST | Start backtest or paper trade |
| `POST /api/stop` | POST | Stop current job |
| `POST /api/reset` | POST | Clear job state |
| `GET /api/stream` | SSE | Real-time event stream |

## Running in background (so you can close the terminal too)

**Linux/Mac:**
```bash
nohup python server.py &> server.log &
echo "Server PID: $!"
```

**Windows:**
```powershell
Start-Process python -ArgumentList "server.py" -WindowStyle Hidden
```

**As a service (Linux systemd):**
```ini
[Unit]
Description=Daylight Strategy Backend
After=network.target

[Service]
WorkingDirectory=/path/to/daylight_backend
ExecStart=/usr/bin/python3 server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Notes
- API keys are sent from the browser to the local server only, never to the internet
- Paper trading state persists in `paper_trades.json` — resumable after server restart
- Each backtest run appends to `daylight_state.json`
