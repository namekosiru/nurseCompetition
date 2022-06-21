import pandas as pd
from glob import glob

def read_data(number: int) -> pd.DataFrame:
    df_acc = pd.read_csv(f"../data/Accelerometer Data/acc_user{number}.csv")
    df_care = pd.read_csv(f"../data/Care Record Data/train{number}.csv")
    return df_acc, df_care

def convert_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_datetime(df[column], format='%Y-%m-%d %H:%M:%S')
    return df

def add_timeLength_timeLengthSeconds(df: pd.DataFrame) -> pd.DataFrame:
    df["time_length"] = df["finish"] - df["start"]
    df["time_length_seconds"] = df["time_length"].map(lambda x: x.total_seconds())
    return df

def read_all_data():
    df_acc = pd.DataFrame()
    df_care = pd.DataFrame()
    for path in glob("../data/Care Record Data/*"):
        tmp = pd.read_csv(path)
        df_acc = pd.concat([df_acc, tmp])
    for path in glob("../data/Care Record Data/*"):
        tmp = pd.read_csv(path)
        df_care = pd.concat([df_care, tmp])
    return df_acc, df_care