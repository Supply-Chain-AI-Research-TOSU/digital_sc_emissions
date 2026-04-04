# SCM Forecasting Methods Lab

A live forecasting tools catalog that demonstrates the carbon footprint of ML/AI operations. Built for graduate supply chain management courses at OSU Fisher College of Business.

## Purpose

This tool runs classical forecasting methods (naive, ARIMA, exponential smoothing, etc.) while tracking the carbon emissions of each API request using CodeCarbon. It reveals:
- Per-request emissions with real-world comparisons
- At-scale projections (10k and 500k users)
- Live class-wide aggregate emissions
- Optional parameter optimization (AIC grid search for ARIMA, SSE minimization for exponential smoothing)

## Architecture

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  User       │ ← HTTP →│  FastAPI    │ ← wraps →│ CodeCarbon  │
│  (Browser)  │         │  Backend    │          │ Tracker     │
└─────────────┘         └─────────────┘         └─────────────┘
       │                       │
       │                       ↓
       │              ┌─────────────┐
       └────────────→│  SQLite      │
                     │  Aggregate   │
                     └─────────────┘
```

## Local Development

### Prerequisites

- Python 3.11+
- CodeCarbon installed locally: `pip install -e ~/codecarbon`

### Setup

```bash
cd ~/projects/digital_sc_emissions
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### Run Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8010
```

### Run Frontend

```bash
cd frontend
python -m http.server 8011 --bind 0.0.0.0
```

Open http://localhost:8011 in your browser. Update `API_BASE` in `index.html` if needed (default points to LAN IP on port 8010).

## EC2 Deployment

### EC2 Instance Setup

1. Launch t2.micro (Amazon Linux 2023, us-east-2)
2. Open port 8010 in security group (0.0.0.0/0)
3. SSH into instance

### Deployment Steps

```bash
# Install Python
sudo dnf install python3.11 python3.11-pip -y

# Create venv
python3.11 -m venv ~/forecast_env
source ~/forecast_env/bin/activate

# Install CodeCarbon
pip install codecarbon

# Install project dependencies
pip install fastapi "uvicorn[standard]" statsmodels pmdarima numpy pandas

# Copy backend/ directory from your machine

# Run with start.sh
chmod +x start.sh
./start.sh &
```

Backend will be accessible at `http://<EC2_PUBLIC_IP>:8010`

## GitHub Pages Deployment

1. Copy `frontend/index.html` to your GitHub Pages repo root or `docs/` folder
2. Update `API_BASE` at the top of the `<script>` block:
   ```javascript
   const API_BASE = "http://<YOUR_EC2_IP>:8010";
   ```
3. Commit and push
4. GitHub Pages will automatically deploy

## Post-Session Analysis

After the build session, run the analysis tool to calculate emissions per inference:

```bash
cd tools
python session_analysis.py /path/to/opencode_session_log.json /path/to/emissions_detail.csv
```

Outputs:
- Formatted table to stdout
- JSON summary to `build_session_summary.json`

## Synthetic Dataset

`data/generate_synthetic.py` produces `data/synthetic_demand.csv` with:
- 156 weeks (3 years) of weekly demand
- True DGP: SARIMA(1,1,1)(1,1,0,52)
- Base demand: 500 units/week
- Linear trend: +0.8 units/week
- Seasonal amplitude: 80 units (holiday peak at weeks 48-52)
- Level shift: +120 units starting week 78
- Exogenous: temperature (°F) affecting demand

Run once to generate:
```bash
cd data
python generate_synthetic.py
```

## Files

```
~/projects/digital_sc_emissions/
├── backend/
│   ├── main.py              # FastAPI endpoints
│   ├── forecaster.py        # 10 forecasting methods + optimization
│   ├── emissions_tracker.py # CodeCarbon wrapper
│   ├── aggregate.py         # SQLite session logging
│   ├── requirements.txt     # Python deps
│   └── start.sh             # Startup script (port 8010)
├── frontend/
│   └── index.html           # Single-file SPA
├── data/
│   ├── generate_synthetic.py
│   └── synthetic_demand.csv
├── tools/
│   └── session_analysis.py  # Post-session join
└── README.md
```

## Methods Implemented

- **Baseline**: Naive, Seasonal Naive, Simple Moving Average
- **Exponential Smoothing**: SES, Holt (Linear), Holt-Winters (Triple)
- **ARIMA Family**: ARIMA, SARIMA, ARIMAX (with exog), Auto ARIMA

All exponential smoothing and ARIMA methods support optional parameter optimization.

## Notes

- CodeCarbon 3.x auto-detects region via IP geolocation
- Aggregate database persists in SQLite between restarts
- GPU power read via NVML (development machine only)
- EC2 has no GPU — falls back to CPU TDP estimates
- Frontend includes a methodology dropdown explaining how emissions are measured
