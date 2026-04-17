import pathlib
import pandas as pd


# Convert all traffic data to parquet
first_period = 7
last_period = 27
data_dir = pathlib.Path("./data/traffic")

for period in range(first_period, last_period + 1):
    csv_path = data_dir / f"period-{period}.csv"
    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, dtype="float32")
    out_path = data_dir / f"period-{period}.parquet"
    print(f"Saving to {out_path}")
    df.to_parquet(out_path)
