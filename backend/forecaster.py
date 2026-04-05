import gc
import itertools
import threading
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


def run_with_timeout(func, timeout_s, *args, **kwargs):
    """Run a function with a timeout using a thread."""
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return {"error": f"Forecast timed out after {timeout_s} seconds. Try simpler parameters.", "method_label": "timeout"}
    if error[0]:
        return {"error": str(error[0]), "method_label": func.__name__}
    return result[0]


def run_forecast(
    method: str, params: dict, y: list, exog: list = None,
    horizon: int = 12, optimize: bool = False,
) -> dict:
    """Run a forecasting method on time series data.

    Args:
        method: Forecasting method name
        params: Method-specific parameters
        y: Time series observations (list of floats)
        exog: Optional exogenous variable (list of floats, same length as y)
        horizon: Number of periods to forecast (default 12)
        optimize: If True, find best parameters by minimizing AIC/SSE

    Returns:
        Dict with forecast, fitted, aic, method_label, or error
    """
    try:
        y_array = np.array(y)

        if method == "naive":
            return _naive(y_array, horizon)

        elif method == "seasonal_naive":
            season = params.get("season", 52)
            return _seasonal_naive(y_array, season, horizon)

        elif method == "sma":
            window = params.get("window", 4)
            return _sma(y_array, window, horizon)

        elif method == "ses":
            alpha = params.get("alpha", 0.3)
            return _ses(y_array, alpha, horizon, optimize)

        elif method == "holt":
            alpha = params.get("alpha", 0.3)
            beta = params.get("beta", 0.1)
            return _holt(y_array, alpha, beta, horizon, optimize)

        elif method == "holt_winters":
            seasonal_periods = params.get("seasonal_periods", 52)
            trend = params.get("trend", "add")
            seasonal = params.get("seasonal", "add")
            if optimize:
                return _holt_winters_optimized(
                    y_array, seasonal_periods, trend, seasonal, horizon
                )
            alpha = params.get("alpha", 0.3)
            beta = params.get("beta", 0.1)
            gamma = params.get("gamma", 0.1)
            return _holt_winters(
                y_array, alpha, beta, gamma, seasonal_periods, trend, seasonal, horizon
            )

        elif method == "arima":
            if optimize:
                max_p = params.get("max_p", 3)
                max_d = params.get("max_d", 2)
                max_q = params.get("max_q", 3)
                return _arima_optimized(y_array, max_p, max_d, max_q, horizon)
            p = params.get("p", 1)
            d = params.get("d", 1)
            q = params.get("q", 1)
            return _arima(y_array, p, d, q, horizon)

        elif method == "sarima":
            s = params.get("s", 52)
            if optimize:
                s = min(s, 26)  # cap seasonal period to prevent OOM
                max_p = params.get("max_p", 1)
                max_d = params.get("max_d", 1)
                max_q = params.get("max_q", 1)
                max_P = params.get("max_P", 1)
                max_D = params.get("max_D", 1)
                max_Q = params.get("max_Q", 1)
                return run_with_timeout(
                    _sarima_optimized, 120,
                    y_array, max_p, max_d, max_q, max_P, max_D, max_Q, s, horizon
                )
            p = params.get("p", 1)
            d = params.get("d", 1)
            q = params.get("q", 1)
            P = params.get("P", 1)
            D = params.get("D", 1)
            Q = params.get("Q", 1)
            return _sarima(y_array, p, d, q, P, D, Q, s, horizon)

        elif method == "arimax":
            if exog is None:
                raise ValueError(
                    "ARIMAX requires an exogenous variable (exog must not be None)"
                )
            exog_array = np.array(exog)
            if optimize:
                max_p = params.get("max_p", 3)
                max_d = params.get("max_d", 2)
                max_q = params.get("max_q", 3)
                return _arimax_optimized(y_array, exog_array, max_p, max_d, max_q, horizon)
            p = params.get("p", 1)
            d = params.get("d", 1)
            q = params.get("q", 1)
            return _arimax(y_array, exog_array, p, d, q, horizon)

        else:
            return {"error": f"Unknown method: {method}", "method_label": method}

    except Exception as e:
        return {"error": str(e), "method_label": method}


# --- Forecasting helpers ---------------------------------------------------


def _naive(y: np.ndarray, horizon: int) -> dict:
    """Last observed value repeated — only h=1 is meaningful."""
    forecast = [float(y[-1])]
    fitted = np.concatenate([y[0:1], y[:-1]])
    return {
        "forecast": forecast,
        "fitted": fitted.tolist(),
        "aic": None,
        "method_label": "Naive",
        "note": "Naive forecasting repeats the last observed value. All future periods produce the same number, so only h=1 is shown.",
    }


def _seasonal_naive(y: np.ndarray, season: int, horizon: int) -> dict:
    """Last full season repeated."""
    if len(y) < season:
        season = len(y)
    last_season = y[-season:]
    forecast = np.tile(last_season, int(np.ceil(horizon / season)))[:horizon]
    fitted = np.concatenate([y[0:1], y[:-1]])
    return {
        "forecast": forecast.tolist(),
        "fitted": fitted.tolist(),
        "aic": None,
        "method_label": f"Seasonal Naive (season={season})",
    }


def _sma(y: np.ndarray, window: int, horizon: int) -> dict:
    """Simple moving average — only h=1 is meaningful."""
    if window > len(y):
        window = len(y)
    sma_value = float(np.mean(y[-window:]))
    fitted = pd.Series(y).rolling(window=window).mean().fillna(0).values
    return {
        "forecast": [sma_value],
        "fitted": fitted.tolist(),
        "aic": None,
        "method_label": f"SMA (window={window})",
        "note": "Simple Moving Average produces a single point forecast (the average of the last {0} observations). All future periods are identical, so only h=1 is shown.".format(window),
    }


def _ses(y: np.ndarray, alpha: float, horizon: int, optimize: bool = False) -> dict:
    """Simple Exponential Smoothing — only h=1 is meaningful."""
    if optimize:
        model = SimpleExpSmoothing(y).fit(optimized=True)
        opt_alpha = round(float(model.params["smoothing_level"]), 4)
        label = f"SES (α={opt_alpha}, optimized)"
        optimized_params = {"alpha": opt_alpha}
    else:
        model = SimpleExpSmoothing(y).fit(smoothing_level=alpha)
        label = f"SES (α={alpha})"
        optimized_params = None
    forecast = model.forecast(1)
    fitted = model.fittedvalues
    result = {
        "forecast": forecast.tolist(),
        "fitted": fitted.tolist(),
        "aic": float(model.aic),
        "method_label": label,
        "note": "Simple Exponential Smoothing produces a flat forecast — the smoothed level does not change beyond h=1. Only the one-step-ahead forecast is shown.",
    }
    if optimized_params:
        result["optimized_params"] = optimized_params
    return result


def _holt(y: np.ndarray, alpha: float, beta: float, horizon: int, optimize: bool = False) -> dict:
    """Holt Linear (Double Exponential Smoothing)."""
    if optimize:
        model = ExponentialSmoothing(y, trend="add").fit(optimized=True)
        opt_alpha = round(float(model.params["smoothing_level"]), 4)
        opt_beta = round(float(model.params["smoothing_trend"]), 4)
        label = f"Holt (α={opt_alpha}, β={opt_beta}, optimized)"
        optimized_params = {"alpha": opt_alpha, "beta": opt_beta}
    else:
        model = ExponentialSmoothing(y, trend="add").fit(
            smoothing_level=alpha, smoothing_trend=beta
        )
        label = f"Holt (α={alpha}, β={beta})"
        optimized_params = None
    forecast = model.forecast(horizon)
    fitted = model.fittedvalues
    result = {
        "forecast": forecast.tolist(),
        "fitted": fitted.tolist(),
        "aic": float(model.aic),
        "method_label": label,
    }
    if optimized_params:
        result["optimized_params"] = optimized_params
    return result


def _holt_winters(
    y: np.ndarray, alpha: float, beta: float, gamma: float,
    seasonal_periods: int, trend: str, seasonal: str, horizon: int,
) -> dict:
    """Holt-Winters Triple Exponential Smoothing with fixed params."""
    model = ExponentialSmoothing(
        y, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods
    ).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
    forecast = model.forecast(horizon)
    fitted = model.fittedvalues
    return {
        "forecast": forecast.tolist(),
        "fitted": fitted.tolist(),
        "aic": float(model.aic),
        "method_label": f"Holt-Winters ({trend} trend, {seasonal} seasonal, period={seasonal_periods})",
    }


def _holt_winters_optimized(
    y: np.ndarray, seasonal_periods: int, trend: str, seasonal: str, horizon: int,
) -> dict:
    """Holt-Winters with statsmodels-optimized smoothing parameters."""
    model = ExponentialSmoothing(
        y, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods
    ).fit(optimized=True)
    opt_alpha = round(float(model.params["smoothing_level"]), 4)
    opt_beta = round(float(model.params["smoothing_trend"]), 4)
    opt_gamma = round(float(model.params["smoothing_seasonal"]), 4)
    forecast = model.forecast(horizon)
    fitted = model.fittedvalues
    return {
        "forecast": forecast.tolist(),
        "fitted": fitted.tolist(),
        "aic": float(model.aic),
        "method_label": f"Holt-Winters ({trend}/{seasonal}, period={seasonal_periods}, optimized)",
        "optimized_params": {
            "alpha": opt_alpha, "beta": opt_beta, "gamma": opt_gamma,
        },
    }


def _arima(y: np.ndarray, p: int, d: int, q: int, horizon: int) -> dict:
    """Standard ARIMA."""
    model = ARIMA(y, order=(p, d, q)).fit()
    forecast = model.get_forecast(steps=horizon)
    forecast_values = forecast.predicted_mean
    return {
        "forecast": forecast_values.tolist(),
        "fitted": model.fittedvalues.tolist(),
        "aic": float(model.aic),
        "method_label": f"ARIMA({p},{d},{q})",
    }


def _sarima(
    y: np.ndarray, p: int, d: int, q: int, P: int, D: int, Q: int, s: int, horizon: int
) -> dict:
    """Seasonal ARIMA."""
    order = (p, d, q)
    seasonal_order = (P, D, Q, s)
    model = ARIMA(y, order=order, seasonal_order=seasonal_order).fit()
    forecast = model.get_forecast(steps=horizon)
    forecast_values = forecast.predicted_mean
    return {
        "forecast": forecast_values.tolist(),
        "fitted": model.fittedvalues.tolist(),
        "aic": float(model.aic),
        "method_label": f"SARIMA({p},{d},{q})({P},{D},{Q},{s})",
    }


def _arimax(
    y: np.ndarray,
    exog: np.ndarray,
    p: int,
    d: int,
    q: int,
    horizon: int,
) -> dict:
    """ARIMA with exogenous variable."""
    model = ARIMA(y, exog=exog, order=(p, d, q)).fit()
    future_exog = np.tile(exog[-1:], horizon)
    forecast = model.get_forecast(steps=horizon, exog=future_exog)
    forecast_values = forecast.predicted_mean
    return {
        "forecast": forecast_values.tolist(),
        "fitted": model.fittedvalues.tolist(),
        "aic": float(model.aic),
        "method_label": f"ARIMAX({p},{d},{q})",
    }


def _arima_optimized(y: np.ndarray, max_p: int, max_d: int, max_q: int, horizon: int) -> dict:
    """Grid search ARIMA orders, select by lowest AIC."""
    best_aic, best_order, best_model = np.inf, None, None
    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ARIMA(y, order=(p, d, q)).fit()
            if m.aic < best_aic:
                best_aic, best_order, best_model = m.aic, (p, d, q), m
        except Exception:
            continue
    if best_model is None:
        return {"error": "No ARIMA order converged during optimization", "method_label": "ARIMA (optimized)"}
    p, d, q = best_order
    forecast = best_model.get_forecast(steps=horizon)
    return {
        "forecast": forecast.predicted_mean.tolist(),
        "fitted": best_model.fittedvalues.tolist(),
        "aic": float(best_aic),
        "method_label": f"ARIMA({p},{d},{q}) (optimized)",
        "optimized_params": {"p": p, "d": d, "q": q},
    }


def _sarima_optimized(
    y: np.ndarray, max_p: int, max_d: int, max_q: int,
    max_P: int, max_D: int, max_Q: int, s: int, horizon: int,
) -> dict:
    """Grid search SARIMA orders, select by lowest AIC."""
    best_aic, best_orders, best_model = np.inf, None, None
    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        for P, D, Q in itertools.product(range(max_P + 1), range(max_D + 1), range(max_Q + 1)):
            if p == 0 and q == 0 and P == 0 and Q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = ARIMA(y, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit()
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_orders = ((p, d, q), (P, D, Q, s))
                    best_model = m
            except Exception:
                continue
            finally:
                gc.collect()
    if best_model is None:
        return {"error": "No SARIMA order converged during optimization", "method_label": "SARIMA (optimized)"}
    (p, d, q), (P, D, Q, s) = best_orders
    forecast = best_model.get_forecast(steps=horizon)
    return {
        "forecast": forecast.predicted_mean.tolist(),
        "fitted": best_model.fittedvalues.tolist(),
        "aic": float(best_aic),
        "method_label": f"SARIMA({p},{d},{q})({P},{D},{Q},{s}) (optimized)",
        "optimized_params": {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "s": s},
    }


def _arimax_optimized(
    y: np.ndarray, exog: np.ndarray, max_p: int, max_d: int, max_q: int, horizon: int,
) -> dict:
    """Grid search ARIMAX orders, select by lowest AIC."""
    best_aic, best_order, best_model = np.inf, None, None
    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ARIMA(y, exog=exog, order=(p, d, q)).fit()
            if m.aic < best_aic:
                best_aic, best_order, best_model = m.aic, (p, d, q), m
        except Exception:
            continue
    if best_model is None:
        return {"error": "No ARIMAX order converged during optimization", "method_label": "ARIMAX (optimized)"}
    p, d, q = best_order
    future_exog = np.tile(exog[-1:], horizon)
    forecast = best_model.get_forecast(steps=horizon, exog=future_exog)
    return {
        "forecast": forecast.predicted_mean.tolist(),
        "fitted": best_model.fittedvalues.tolist(),
        "aic": float(best_aic),
        "method_label": f"ARIMAX({p},{d},{q}) (optimized)",
        "optimized_params": {"p": p, "d": d, "q": q},
    }


