import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

n_weeks = 156
weeks = np.arange(1, n_weeks + 1)

trend = 0.8 * weeks

seasonal = 80 * np.sin(2 * np.pi * (weeks - 1) / 52 + np.pi / 2)

level_shift = np.where(weeks >= 78, 120, 0)

base = 500
noise = np.random.normal(0, 25, n_weeks)

demand = base + trend + seasonal + level_shift + noise

temperature = (
    55 + 30 * np.sin(2 * np.pi * (weeks - 1) / 52) + np.random.normal(0, 3, n_weeks)
)

start_date = pd.Timestamp("2022-01-03")
dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")

df = pd.DataFrame(
    {"week": weeks, "date": dates, "demand": demand, "temperature_f": temperature}
)

output_path = Path(__file__).parent / "synthetic_demand.csv"
df.to_csv(output_path, index=False)

print("Generated synthetic_demand.csv")
print(f"\nShape: {df.shape}")
print(f"\nSummary Statistics:")
print(df.describe())
print(f"\nFirst 10 rows:")
print(df.head(10).to_string())
