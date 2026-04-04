from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from forecaster import run_forecast
from emissions_tracker import track_forecast
from aggregate import record_request, get_aggregate

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Scales multiplier – number of forecast requests per user per day
SCALE_MULTIPLIER = 5

# Benchmark comparison thresholds in grams
COMPARISONS = [
    (0.001, "less than a single breath"),
    (0.01, "one second of a smartphone screen"),
    (0.1, "sending about 10 plain-text emails"),
    (0.5, "streaming 30 seconds of video"),
    (1.0, "charging a smartphone for one minute"),
    (5.0, "boiling a small cup of water"),
    (10.0, "running a LED bulb for one hour"),
    (50.0, "driving a car about 200 meters"),
    (200.0, "a short domestic flight, per passenger"),
]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    method: str = Field(..., title="Forecasting method")
    params: Dict[str, Any] = Field({}, title="Parameters for method")
    y: List[float] = Field(..., title="Historical data series")
    exog: Optional[List[float]] = Field(None, title="Optional exogenous series")
    horizon: int = Field(12, ge=1, title="Forecast horizon (number of periods)")
    optimize: bool = Field(False, title="Optimize parameters (minimize AIC/SSE)")


class HealthResponse(BaseModel):
    status: str
    timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_comparison(grams: float) -> str:
    for threshold, label in COMPARISONS:
        if grams < threshold:
            return label
    return "more than a cross-country flight"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/forecast")
def forecast_endpoint(req: ForecastRequest):
    """Run a forecast with emissions tracking."""
    # Validate method
    allowed_methods = {
        "naive",
        "seasonal_naive",
        "sma",
        "ses",
        "holt",
        "holt_winters",
        "arima",
        "sarima",
        "arimax",
        "auto_arima",
    }
    if req.method not in allowed_methods:
        raise HTTPException(status_code=400, detail="Unsupported forecasting method")

    emissions_result = track_forecast(
        run_forecast, req.method, req.params, req.y, req.exog,
        horizon=req.horizon, optimize=req.optimize,
    )

    forecast_result = emissions_result["result"]

    record_request(
        method=req.method,
        duration_s=emissions_result["duration_s"],
        emissions_kg=emissions_result["emissions_kg"],
        energy_kwh=emissions_result["energy_kwh"],
    )

    emissions_g = emissions_result["emissions_kg"] * 1000
    comparison = get_comparison(emissions_g)

    scales = {
        "10k_users_daily_g": emissions_g * 10000 * SCALE_MULTIPLIER,
        "10k_users_annual_kg": emissions_g * 10000 * SCALE_MULTIPLIER * 365 / 1000,
        "500k_users_daily_g": emissions_g * 500000 * SCALE_MULTIPLIER,
        "500k_users_annual_kg": emissions_g * 500000 * SCALE_MULTIPLIER * 365 / 1000,
    }

    return {
        "forecast_result": forecast_result,
        "emissions": {
            "emissions_g": emissions_g,
            "emissions_kg": emissions_result["emissions_kg"],
            "energy_kwh": emissions_result["energy_kwh"],
            "duration_s": emissions_result["duration_s"],
            "cpu_power_w": emissions_result["cpu_power_w"],
            "gpu_power_w": emissions_result["gpu_power_w"],
            "ram_power_w": emissions_result["ram_power_w"],
        },
        "comparison": comparison,
        "scales": scales,
    }


@app.get("/api/aggregate")
def aggregate_endpoint():
    """Return aggregate session statistics."""
    aggregate = get_aggregate()

    if aggregate["n_requests"] > 0:
        avg_g = aggregate["avg_emissions_g_per_request"]
        aggregate["scales"] = {
            "10k_users_daily_g": avg_g * 10000 * SCALE_MULTIPLIER,
            "10k_users_annual_kg": avg_g * 10000 * SCALE_MULTIPLIER * 365 / 1000,
            "500k_users_daily_g": avg_g * 500000 * SCALE_MULTIPLIER,
            "500k_users_annual_kg": avg_g * 500000 * SCALE_MULTIPLIER * 365 / 1000,
        }
    else:
        aggregate["scales"] = {
            "10k_users_daily_g": 0.0,
            "10k_users_annual_kg": 0.0,
            "500k_users_daily_g": 0.0,
            "500k_users_annual_kg": 0.0,
        }

    return aggregate


@app.get("/api/health")
def health_endpoint() -> HealthResponse:
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
