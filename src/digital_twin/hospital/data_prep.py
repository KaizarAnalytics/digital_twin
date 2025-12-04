import pandas as pd
from pathlib import Path
import os

def load_patients(data_dir: str) -> pd.DataFrame:
    p = Path(data_dir) / "raw" / "patients.csv"
    if not os.path.isdir(p):
        p = Path(data_dir) / "patients.csv"
    df = pd.read_csv(p, parse_dates=["arrival_date","departure_date"])
    return df

def arrivals_per_day(df_patients: pd.DataFrame, service: str) -> pd.DataFrame:
    df = df_patients[df_patients["service"] == service].copy()
    df["arrival_day"] = df["arrival_date"].dt.floor("D")

    s = (
        df.groupby("arrival_day")
          .size()
          .rename("arrivals")
          .sort_index()
    )

    # Ensure continuous date index
    s = s.asfreq("D", fill_value=0)

    return s.to_frame()


def los_values(df_patients: pd.DataFrame, service: str):
    df = df_patients[df_patients["service"]==service].copy()
    los = (df["departure_date"] - df["arrival_date"]).dt.days.astype(float)
    los = los[los>0.0]
    return los.values

def processtime_values(
    df_patients: pd.DataFrame,
    service: str,
) -> pd.Series:
    return los_values(df_patients, service)
