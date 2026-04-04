import logging
from codecarbon import EmissionsTracker

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def track_forecast(fn, *args, **kwargs):
    """
    Track carbon emissions for a forecasting function call.

    Note: On the v-twin build machine with RTX 3090 Ti GPUs, CodeCarbon should be able
    to read actual GPU power via NVML. If gpu_power reads 0.0W, CodeCarbon falls back to
    TDP estimation (3090 Ti TDP = 450W).

    Args:
        fn: Forecasting function to track
        *args: Positional arguments to pass to fn
        **kwargs: Keyword arguments to pass to fn

    Returns:
        Dict with result and emissions data
    """
    # Note: CodeCarbon 3.x removed offline/country_iso_code params.
    # It auto-detects location via IP geolocation. On EC2 us-east-2 (Ohio)
    # it will resolve correctly. If GPU power reads 0.0W consistently,
    # CodeCarbon falls back to TDP estimation (3090 Ti TDP = 450W).
    tracker = EmissionsTracker(
        project_name="scm-forecast-tool-runtime",
        measure_power_secs=1,
        save_to_file=False,
        log_level="error",
    )

    tracker.start()
    result = None

    try:
        result = fn(*args, **kwargs)
    finally:
        emissions = tracker.stop()

    if emissions is None:
        logger.warning("CodeCarbon returned None for emissions (common on EC2)")
        return {
            "result": result,
            "emissions_kg": 0.0,
            "energy_kwh": 0.0,
            "duration_s": 0.0,
            "cpu_power_w": 0.0,
            "gpu_power_w": 0.0,
            "ram_power_w": 0.0,
        }

    emissions_data = tracker.final_emissions_data
    return {
        "result": result,
        "emissions_kg": emissions_data.emissions,
        "energy_kwh": emissions_data.energy_consumed,
        "duration_s": emissions_data.duration,
        "cpu_power_w": emissions_data.cpu_power,
        "gpu_power_w": emissions_data.gpu_power,
        "ram_power_w": emissions_data.ram_power,
    }
