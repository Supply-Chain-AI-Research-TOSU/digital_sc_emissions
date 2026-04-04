# Build Plan: SCM Forecasting Tool with Carbon Emissions Tracking

## Context

You are building a forecasting methods catalog for a graduate supply chain management course
at Fisher College of Business, The Ohio State University. The tool will be used in a live
classroom lesson on AI and sustainability. Students will interact with it, and the instructor
will reveal the carbon footprint of both using the tool and building it afterward.

Do not rush. Build each phase completely before moving to the next. After each phase, verify
the deliverable works before proceeding. If a file already exists, read it before editing.

-----

## Environment

### Build Machine: v-twin

v-twin is a Ubuntu 24.04 LTS virtual machine running under Proxmox on a local homelab server.
It is the primary development environment and runs Ollama serving the LLM used to build this
project. It also runs ComfyUI. All Python work on this machine uses a virtualenv.

|Property         |Value                                                               |
|-----------------|--------------------------------------------------------------------|
|OS               |Ubuntu 24.04 LTS                                                    |
|Hypervisor       |Proxmox (VM guest)                                                  |
|CPU              |AMD Ryzen 9 9900X, 4.4GHz, 12-core                                  |
|System RAM       |64 GB DDR5-6000 (2x 32GB Klevv BOLT V)                              |
|GPU 1            |NVIDIA RTX 3090 Ti 24GB (Founders Edition)                          |
|GPU 2            |NVIDIA RTX 3090 Ti 24GB (Founders Edition)                          |
|Total GPU VRAM   |48 GB                                                               |
|Storage          |240GB SSD (boot) + 4TB NVMe PCIe 4.0 (TEAMGROUP T-FORCE G50)        |
|Ollama models    |Qwen3.5:27b (primary build model), others installed                 |
|Python           |3.11+ (verify with `python3 --version`)                             |
|Active venv      |`~/projects/digital_sc_emissions/venv` (create if it does not exist)|
|CodeCarbon repo  |`~/codecarbon` (already cloned, do not re-clone)                    |
|Working directory|`~/projects/digital_sc_emissions`                                   |

**GPU passthrough note**: v-twin has two RTX 3090 Ti cards. Whether one or both are passed
through to the VM in Proxmox determines what CodeCarbon can see via NVML. Run
`nvidia-smi` inside the VM to confirm GPU visibility. Run `codecarbon detect` after
install to confirm what CodeCarbon reports. If `gpu_power` reads 0.0W consistently,
CodeCarbon is falling back to TDP estimation for the GPU component. Note this in a
comment in `emissions_tracker.py`. The 3090 Ti has a rated TDP of 450W – CodeCarbon
will use this as its fallback estimate if direct NVML sampling is unavailable.

### Virtualenv Setup (do this first, before any other step)

```bash
cd ~/projects/digital_sc_emissions

# Create venv if it does not already exist
python3 -m venv venv

# Activate -- use this in every terminal session before running any Python
source venv/bin/activate

# Verify activation (should show project venv path)
which python

# Install CodeCarbon from the already-cloned repo
pip install -e ~/codecarbon

# Install project dependencies
pip install fastapi "uvicorn[standard]" statsmodels pmdarima numpy pandas
```

All subsequent `python`, `pip`, and `uvicorn` commands in this build plan assume the venv
is activated. Do not use `--break-system-packages`. Do not use sudo for pip. The venv is
the install target for everything.

If the venv already exists and has packages installed, do not recreate it. Run
`pip list | grep -E "fastapi|statsmodels|pmdarima|codecarbon"` to check what is present
and install only what is missing.

### Runtime Target: EC2

|Property      |Value                                                             |
|--------------|------------------------------------------------------------------|
|Instance type |t2.micro (1 vCPU, 1GB RAM)                                        |
|OS            |Amazon Linux 2023                                                 |
|Region        |us-east-2 (Ohio)                                                  |
|Grid intensity|~0.39 kg CO2/kWh (Ohio grid, used by CodeCarbon offline mode)     |
|Python        |3.11 (install via `sudo dnf install python3.11 python3.11-pip -y`)|
|Venv on EC2   |`~/forecast_env` (same name, create fresh on EC2)                 |
|Port          |8010 (open in EC2 security group for 0.0.0.0/0)                   |

EC2 has no GPU. CodeCarbon will use CPU TDP estimation only. RAPL power capping interface
is blocked by the AWS hypervisor, so all measurements are estimates. This is expected and
should be noted in lesson materials.

### Frontend Host

GitHub Pages, static only. Single HTML file with all CSS and JS inline. No build step,
no Node.js, no npm. The file is `frontend/index.html` and is deployed by copying it to
the target GitHub Pages repository.

-----

-----

## Project File Structure

Create exactly this structure. Do not deviate.

```
~/projects/digital_sc_emissions/
├── buildplan.md          # This file (already exists)
├── venv/                 # Python virtualenv (create during setup)
├── backend/
│   ├── main.py               # FastAPI application
│   ├── forecaster.py         # Forecasting engine (statsmodels + pmdarima)
│   ├── emissions_tracker.py  # CodeCarbon per-request wrapper
│   ├── aggregate.py          # SQLite session aggregate counter
│   ├── requirements.txt      # Python dependencies
│   └── start.sh              # EC2 startup script
├── frontend/
│   └── index.html            # Single file: all CSS and JS inline
├── data/
│   └── generate_synthetic.py # Run once to produce synthetic_demand.csv
├── tools/
│   └── session_analysis.py   # Post-build: joins OpenCode logs + CodeCarbon CSV
└── README.md
```

-----

## Phase 1: Backend — Forecasting Engine

**File**: `backend/forecaster.py`

Implement a `run_forecast(method, params, y, exog=None, horizon=12, optimize=False)` function.
`horizon` controls how many periods ahead to forecast. `optimize` enables automatic parameter
selection (minimizing AIC for ARIMA family, SSE for exponential smoothing). `y` is a Python list of
floats (time series observations). `exog` is an optional list of floats (exogenous variable,
same length as `y`). `params` is a dict of method-specific parameters. Return a dict with
keys `forecast` (list of 12 floats), `fitted` (list same length as `y`), `aic` (float or
null), `method_label` (str).

Implement all of the following methods. Use `statsmodels` for all except `auto_arima`.

**Baseline methods** (no params needed):

- `naive`: Last observed value repeated for 12 periods
- `seasonal_naive`: Last full season repeated (default season=52 for weekly data)
- `sma`: Simple moving average. Param: `window` (int, default 4)

**Exponential smoothing methods** (use `statsmodels.tsa.holtwinters`):

- `ses`: Simple Exponential Smoothing. Param: `alpha` (float 0-1, default 0.3)
- `holt`: Holt Linear (Double Exponential Smoothing). Params: `alpha`, `beta`
- `holt_winters`: Holt-Winters Triple Exponential Smoothing. Params: `alpha`, `beta`,
  `gamma`, `seasonal_periods` (int, default 52), `trend` (str: “add” or “mul”, default
  “add”), `seasonal` (str: “add” or “mul”, default “add”)

**ARIMA family** (use `statsmodels.tsa.arima.model.ARIMA`):

- `arima`: Standard ARIMA. Params: `p` (int), `d` (int), `q` (int)
- `sarima`: Seasonal ARIMA. Params: `p`, `d`, `q`, `P`, `D`, `Q`,
  `s` (seasonal period, int, default 52)
- `arimax`: ARIMA with one exogenous variable. Params: `p`, `d`, `q`. Requires `exog`.
  Raise a clear ValueError if `exog` is None.
- `auto_arima`: Use `pmdarima.auto_arima`. Params: `max_p` (int, default 3), `max_q`
  (int, default 3), `seasonal` (bool, default False), `m` (int, default 1 if not
  seasonal). Return fitted model’s AIC.

For all ARIMA methods, generate 12-step ahead forecasts using `get_forecast(12)`. For
Holt-Winters, use `forecast(12)`. For `auto_arima`, use `predict(12)`.

Handle exceptions: if a model fails to converge, return
`{"error": str(e), "method_label": method}` rather than raising.

-----

## Phase 2: Backend — Emissions Tracker Wrapper

**File**: `backend/emissions_tracker.py`

CodeCarbon is already installed into the venv during the setup step above. Verify it is
present before proceeding:

```bash
python -c "from codecarbon import EmissionsTracker; print('CodeCarbon OK')"
```

If that fails, install it: `pip install -e ~/codecarbon`

Implement `track_forecast(fn, *args, **kwargs)`. This function:

1. Instantiates `EmissionsTracker` with these exact settings:
   
   ```python
   EmissionsTracker(
       project_name="scm-forecast-tool-runtime",
       measure_power_secs=1,
       save_to_file=False,
       log_level="error",
   )
   ```
   
   Note: CodeCarbon 3.x removed the `offline` and `country_iso_code` parameters.
   It auto-detects location via IP geolocation.
1. Calls `tracker.start()`
1. Calls `fn(*args, **kwargs)` inside a try/finally
1. Calls `tracker.stop()` in the finally block
1. Returns a dict:
   
   ```python
   {
       "result": fn_result,
       "emissions_kg": float,
       "energy_kwh": float,
       "duration_s": float,
       "cpu_power_w": float,
       "gpu_power_w": float,
       "ram_power_w": float
   }
   ```

If `tracker.stop()` returns None (can happen on EC2), set `emissions_kg` to 0.0 and log a
warning. Do not crash.

-----

## Phase 3: Backend — Aggregate Counter

**File**: `backend/aggregate.py`

Use Python’s built-in `sqlite3` module. No external dependencies.

On import, create (if not exists) a SQLite database at `./aggregate.db` with one table:

```sql
CREATE TABLE IF NOT EXISTS session_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    method TEXT,
    duration_s REAL,
    emissions_kg REAL,
    energy_kwh REAL,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0
)
```

Implement two functions:

`record_request(method, duration_s, emissions_kg, energy_kwh)`: inserts one row.

`get_aggregate()`: returns a dict:

```python
{
    "n_requests": int,
    "total_emissions_g": float,       # kg converted to grams
    "total_energy_kwh": float,
    "avg_emissions_g_per_request": float,
    "method_counts": dict,            # {"arima": 12, "sarima": 5, ...}
    "session_start": str,             # ISO timestamp of first request
    "last_updated": str               # ISO timestamp of most recent request
}
```

-----

## Phase 4: Backend — FastAPI Application

**File**: `backend/main.py`

Use FastAPI with CORS enabled for all origins (class will use GitHub Pages, domain not
known at build time).

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

**POST `/api/forecast`**

Request body (JSON):

```json
{
    "method": "arima",
    "params": {"p": 1, "d": 1, "q": 1},
    "y": [123.4, 145.2, ...],
    "exog": null,
    "horizon": 12,
    "optimize": false
}
```

`horizon` (int, default 12): number of forecast periods. `optimize` (bool, default false):
when true, the backend searches for optimal parameters (AIC grid search for ARIMA family,
`optimized=True` for exponential smoothing).

Steps:

1. Call `track_forecast(run_forecast, method, params, y, exog)`
1. Record to aggregate via `record_request(...)`
1. Compute `emissions_g = emissions_kg * 1000`
1. Compute comparison string via `get_comparison(emissions_g)` (see below)
1. Compute at-scale estimates:
   
   ```python
   scales = {
       "10k_users_daily_g": emissions_g * 10000 * 5,
       "10k_users_annual_kg": emissions_g * 10000 * 5 * 365 / 1000,
       "500k_users_daily_g": emissions_g * 500000 * 5,
       "500k_users_annual_kg": emissions_g * 500000 * 5 * 365 / 1000
   }
   ```
   
   (5 = assumed forecasts per user per day)
1. Return all of: forecast result, emissions data, comparison, scales

**GET `/api/aggregate`**

Returns `get_aggregate()` dict directly. Also compute and include at-scale from the
aggregate’s `avg_emissions_g_per_request`:

```python
avg_g = aggregate["avg_emissions_g_per_request"]
aggregate["scales"] = {
    "10k_users_daily_g": avg_g * 10000 * 5,
    "10k_users_annual_kg": avg_g * 10000 * 5 * 365 / 1000,
    "500k_users_daily_g": avg_g * 500000 * 5,
    "500k_users_annual_kg": avg_g * 500000 * 5 * 365 / 1000
}
```

**GET `/api/health`**

Returns `{"status": "ok", "timestamp": <ISO now>}`. Used by frontend to check connectivity.

**Comparison function** (implement in `main.py`):

```python
COMPARISONS = [
    (0.001,  "less than a single breath"),
    (0.01,   "one second of a smartphone screen"),
    (0.1,    "sending about 10 plain-text emails"),
    (0.5,    "streaming 30 seconds of video"),
    (1.0,    "charging a smartphone for one minute"),
    (5.0,    "boiling a small cup of water"),
    (10.0,   "running a LED bulb for one hour"),
    (50.0,   "driving a car about 200 meters"),
    (200.0,  "a short domestic flight, per passenger"),
]

def get_comparison(grams: float) -> str:
    for threshold, label in COMPARISONS:
        if grams < threshold:
            return label
    return "more than a cross-country flight"
```

**File**: `backend/requirements.txt`

```
fastapi
uvicorn[standard]
statsmodels
pmdarima
numpy
pandas
```

**File**: `backend/start.sh`

```bash
#!/bin/bash
source ~/projects/digital_sc_emissions/venv/bin/activate

cd ~/projects/digital_sc_emissions/backend
uvicorn main:app --host 0.0.0.0 --port 8010 --workers 1
```

Make it executable: `chmod +x backend/start.sh`

-----

## Phase 5: Synthetic Dataset Generator

**File**: `data/generate_synthetic.py`

Generate 3 years of weekly demand data (156 observations) with known structure. Use numpy
and pandas only.

Parameters (hardcoded, not configurable):

- Base demand: 500 units/week
- Linear trend: +0.8 units/week
- Seasonal amplitude: 80 units (period=52, peaks at week 48-52 of each year for holiday)
- Level shift: +120 units starting at week 78 (Year 2, new customer onboarded)
- Exogenous variable: regional temperature (°F), generated as a sine wave
  (mean=55, amplitude=30, period=52) plus N(0,3) noise. Represents a perishable product
  with temperature-sensitive demand.
- Noise: N(0, 25) added to demand
- True DGP: SARIMA(1,1,1)(1,1,0,52) structure underlying the generation process
- Random seed: 42

Output: `data/synthetic_demand.csv` with columns:

```
week, date, demand, temperature_f
```

`date` starts at 2022-01-03 (first Monday of 2022), weekly frequency.

After generating, print summary statistics and the first 10 rows to stdout.

-----

## Phase 6: Frontend

**File**: `frontend/index.html`

Single self-contained HTML file. All CSS and JavaScript inline. No build step. Must work
when opened from GitHub Pages.

**Design direction**: Industrial/utilitarian with high data density. Dark background
(#0f1117), monospace typography for data values, clean sans-serif for labels. Muted teal
(#2dd4bf) as the primary accent. No gradients, no rounded corners except small (2px).
Feels like a professional analytics terminal, not a consumer app. Memorable for being
genuinely functional-looking.

**External CDN dependencies** (load from cdnjs or unpkg, no npm):

- Chart.js 4.x for time series visualization
- Papa Parse for CSV upload/parsing

**Layout** (single page, no routing):

```
┌─────────────────────────────────────────────────────────┐
│  HEADER: "SCM Forecasting Methods Lab" + subtitle       │
│  Fisher College of Business | The Ohio State University │
├──────────────────┬──────────────────────────────────────┤
│  LEFT PANEL      │  RIGHT PANEL                         │
│  ─────────────   │  ──────────                          │
│  1. Data Input   │  Chart: time series + forecast       │
│     CSV upload   │  (Chart.js line chart)               │
│     or use       │                                      │
│     sample data  │  Forecast table: 12 periods          │
│                  │                                      │
│  2. Method       ├──────────────────────────────────────┤
│     Selector     │  EMISSIONS PANEL                     │
│     (dropdown)   │  This request: X.XXXg CO2eq          │
│                  │  ≈ [comparison string]               │
│  3. Parameters   │                                      │
│     (dynamic,    │  AT SCALE:                           │
│     appear based │  10K users/day:  X g/day | X kg/yr   │
│     on method)   │  500K users/day: X g/day | X kg/yr   │
│                  │                                      │
│  4. Exog input   ├──────────────────────────────────────┤
│     (shown only  │  CLASS AGGREGATE (live, polls /api/  │
│     for ARIMAX)  │  aggregate every 15s)                │
│                  │  Requests: 87  |  Total: 0.43g       │
│  [Run Forecast]  │  Avg/request: 0.005g                 │
│                  │  Class at-scale projections          │
│  [Use Sample]    │                                      │
└──────────────────┴──────────────────────────────────────┘
```

**Backend URL**: Read from a `const API_BASE` variable at the top of the script block,
defaulting to `""` (empty string, for local dev). Set it to the EC2 public IP with port
8000 before deploying: `const API_BASE = "http://<EC2_IP>:8010"`.

**Method selector behavior**: When a method is selected from the dropdown, render the
appropriate parameter inputs dynamically. Use sensible defaults (see Phase 1 for defaults).
For `auto_arima`, show `max_p`, `max_q`, `seasonal` checkbox, and `m` (only if seasonal
is checked). For `arimax`, show ARIMA params AND an exogenous variable column selector
(appears after CSV is loaded, listing available numeric columns).

**CSV upload behavior**: Parse with Papa Parse. Accept weekly time series with at minimum
one numeric column for demand. Allow user to select which column is the time series `y`.
If additional numeric columns exist, list them as available exogenous variables for ARIMAX.
On upload, show column headers and first 5 rows in a small preview table.

**Sample data button**: Loads the synthetic dataset (hardcode the first 156 rows of
`synthetic_demand.csv` as a JS array – the generator in Phase 5 will produce it, but
hardcode the values so the frontend works without a file server).

**Chart**: Line chart with two series: “Historical” (actual `y` values) and “Forecast”
(12 future periods, shown with a dashed line). If `fitted` values are returned, show a
third series “Model Fit” in a muted color. X axis: period numbers or dates if available.

**Emissions panel**: On each successful forecast response, update with:

- `emissions_g` formatted to 4 decimal places
- Comparison string from API
- At-scale values formatted with appropriate units (g for daily if < 1000g, else kg)

**Class aggregate panel**: Poll `GET /api/aggregate` every 15 seconds. Display the live
running total. Show a connectivity indicator (green dot if last poll succeeded, red if
failed). If backend is unreachable, show “Backend offline – emissions data unavailable”
without breaking the rest of the UI.

**Error handling**: If the backend returns an error or is unreachable, show a clear
error message in the results panel. Never show a blank panel or a raw JSON error.

**Footer**: “Built with Qwen3.5:27b via OpenCode | Carbon tracked with CodeCarbon”

-----

## Phase 7: Post-Session Analysis Tool

**File**: `tools/session_analysis.py`

This script is run by the instructor after the OpenCode build session to derive
per-inference emissions. It joins the OpenCode session log with the CodeCarbon monitor CSV.

Accept two command-line arguments:

```
python session_analysis.py <opencode_session_log.json> <codecarbon_emissions_detail.csv>
```

OpenCode session log format (inspect the actual file before assuming schema – print the
top-level keys and first message keys to stdout before processing):

Parse messages where `role == "assistant"` and a `tool_calls` or generation event is
present. Extract per-generation windows using available timestamp fields. If the exact
schema differs from expected, print a warning and fall back to session-level totals only.

CodeCarbon detail CSV columns (standard): `timestamp`, `duration`, `emissions`,
`energy_consumed`, `cpu_power`, `gpu_power`, `ram_power`.

For each identified inference window:

1. Find CodeCarbon rows where timestamp falls within [start, end]
1. Sum `emissions` (convert to grams), sum `energy_consumed`, average `gpu_power`
1. Record alongside token counts if available

Output to stdout a formatted table:

```
Per-Inference Breakdown
───────────────────────────────────────────────────────
Call  Duration    Tokens Out   CO2eq (g)   Avg GPU (W)
  1    36.2s          634       0.8234       188.3
  2    28.7s          412       0.6102       191.1
...

Summary
───────────────────────────────────────────────────────
Total calls:              23
Total emissions:          18.4 g CO2eq
Total energy:             0.047 kWh
Avg per call:             0.80 g CO2eq
Avg per 1K output tokens: 1.26 g CO2eq
Idle baseline GPU draw:   14.2 W (avg between calls)
Inference GPU draw:       189.1 W (avg during calls)
```

Also write a JSON summary to `tools/build_session_summary.json` for use in the reveal
slide.

-----

## Phase 8: README

**File**: `README.md`

Write a concise README covering:

1. Project purpose (one paragraph)
1. Architecture diagram (ASCII)
1. Local development setup (install deps, run backend, open frontend)
1. EC2 deployment steps (copy backend/, create venv, run start.sh, open port 8010)
1. GitHub Pages deployment (copy frontend/index.html to repo root or docs/, set API_BASE)
1. How to run the post-session analysis tool
1. Synthetic dataset description (what it represents, the known DGP)

-----

## Build Sequence

Execute phases in this order. Do not skip ahead.

1. **Venv and deps first** (see Environment section above):
   
   ```bash
   cd ~/projects/digital_sc_emissions
   source venv/bin/activate
   python -c "import fastapi, statsmodels, pmdarima, codecarbon; print('All deps OK')"
   ```
   
   If any import fails, install the missing package with `pip install <package>` (venv
   is active, no flags needed). Do not proceed until this check passes.
1. Create directory structure (buildplan.md already exists, do not overwrite it):
   
   ```bash
   cd ~/projects/digital_sc_emissions
   mkdir -p backend frontend data tools
   ```
1. Phase 1: `backend/forecaster.py` – test with:
   
   ```bash
   python -c "from forecaster import run_forecast; print(run_forecast('naive', {}, [100,110,120,130]*10))"
   ```
1. Phase 2: `backend/emissions_tracker.py` – verify CodeCarbon import then test with:
   
   ```bash
   python -c "from emissions_tracker import track_forecast; print('OK')"
   ```
1. Phase 3: `backend/aggregate.py` – test by importing and calling both functions
1. Phase 4: `backend/main.py` + `requirements.txt` + `start.sh` – run with:
   
   ```bash
   uvicorn main:app --reload
   ```
   
   Then test with:
   
   ```bash
   curl -X POST http://localhost:8010/api/forecast \
     -H "Content-Type: application/json" \
     -d '{"method":"arima","params":{"p":1,"d":1,"q":1},"y":[100,110,105,120,115,130,125,140,135,150,145,160],"exog":null}'
   ```
   
   Also test: `curl http://localhost:8010/api/aggregate` and `curl http://localhost:8010/api/health`
1. Phase 5: `data/generate_synthetic.py` – run it, verify `data/synthetic_demand.csv`
   exists with 156 rows and expected columns
1. Phase 6: `frontend/index.html` – open in browser via `python -m http.server 3000`
   from the `frontend/` directory. Verify all panels render, sample data loads, a forecast
   call populates the emissions panel.
1. Phase 7: `tools/session_analysis.py`
1. Phase 8: `README.md`

-----

## Constraints and Non-Negotiables

- Do not use Flask. FastAPI only.
- Do not use React, Vue, or any JS framework requiring a build step. Vanilla JS only.
- Do not use localStorage or sessionStorage in the frontend.
- Do not install any Python package not listed in requirements.txt without noting it.
- The frontend must work as a static file on GitHub Pages with no modifications other
  than setting `API_BASE` to the EC2 IP.
- CodeCarbon 3.x auto-detects region via IP. The `offline` parameter was removed in v3.x.
- The aggregate database must persist between server restarts. SQLite file on disk, not
  in-memory.
- CORS must be open (`allow_origins=["*"]`). This is intentional for a classroom tool.
- If any method fails to converge, return a graceful error – do not raise an unhandled
  500. The frontend must display the error message clearly.
