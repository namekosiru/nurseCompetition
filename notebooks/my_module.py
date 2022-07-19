from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def read_data(number: int) -> pd.DataFrame:
    df_acc = pd.read_csv(f"../data/Accelerometer Data/acc_user{number}.csv")
    df_care = pd.read_csv(f"../data/Care Record Data/train{number}.csv")
    return df_acc, df_care

def read_all_data():
    df_acc = pd.DataFrame()
    df_care_train = pd.DataFrame()
    df_care_test = pd.DataFrame()

    #加速度データ
    for path in glob("../data/Accelerometer Data/*"):
        tmp = pd.read_csv(path)
        df_acc = pd.concat([df_acc, tmp])

    #訓練ラベルデータ
    for path in glob("../data/Care Record Data/*"):
        tmp = pd.read_csv(path)
        df_care_train = pd.concat([df_care_train, tmp])

    #検証ラベルデータ
    for path in glob("../TestData/**/*"):
        tmp = pd.read_csv(path)
        df_care_test = pd.concat([df_care_test, tmp])
    return df_acc, df_care_train, df_care_test

def convert_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
        任意のカラムをデータタイム型に変換
    """
    for column in columns:
        df[column] = pd.to_datetime(df[column], format='%Y-%m-%d %H:%M:%S')
    return df

def add_timeLength_timeLengthSeconds(df: pd.DataFrame) -> pd.DataFrame:
    """
        start, finishカラムからactivityにかかった時間のカラムを追加する
        time_length: datetime型の差を計算
        time_length_seconds: datetime型の差を秒単位に変換
    """
    df["time_length"] = df["finish"] - df["start"]
    df["time_length_seconds"] = df["time_length"].map(lambda x: x.total_seconds())
    return df


def create_acc_dataframe_label(df_care: pd.DataFrame, df_acc: pd.DataFrame) -> dict:
    """
        df_care, df_accデータフレームからsubjectとactivity_labelに対応するデータを作成
        
    """
    userid = df_care["user_id"].unique()
    seg_label_list = {index:[] for index in userid} # segment's (1 sample's) label list (label mean "activity_type_id", "user_id" etc.).
    seg_list = {index:[] for index in userid} # segment's (1 sample's) accelerometer data list.
    for userid in userid:
        df_care_tmp = df_care[df_care["user_id"] == userid]
        df_acc_tmp = df_acc[df_acc["subject_id"] == userid]
        for index, row in df_care_tmp.iterrows():
            # started_at = df_care_tmp.iloc[i, 6]
            # finished_at = df_care_tmp.iloc[i, 7]
            started_at = row["start"]
            finished_at = row["finish"]
            seg = df_acc_tmp[(df_acc_tmp["datetime"] >=started_at) & (df_acc_tmp["datetime"] <= finished_at)]
            # seg_label = df_care_tmp.loc[i, "activity_type_id"]
            seg_label = row["activity_type_id"]
            if (len(seg)!=0):
                seg_list[userid].append(seg)
                seg_label_list[userid].append(seg_label)
    return seg_list, seg_label_list

def extend_time(df: pd.DataFrame, extend_time: int) -> pd.DataFrame:
    """
        start, finishのカラムの時間を任意の時間(分)だけ前後に拡張する
    """
    import datetime
    df["start_extend"] = df["start"] - datetime.timedelta(minutes=extend_time)
    df["finish_extend"] = df["finish"] + datetime.timedelta(minutes=extend_time)
    return df

def create_frequency_heatmap(df: pd.DataFrame, show_flag: bool = False) -> dict:
    """
        各userの時間-activity_type_idのheatmapを作成する
        show_flag: heatmapの可視化を行うかのflag
    """
    mpl.style.use("seaborn-darkgrid")
    USER_ID: list = df["user_id"].unique()
    users_heatmap: dict = {}
    for user_id in USER_ID:
        heatmap_matrix = np.zeros((28, 24), dtype=int)
        corr = df[df["user_id"] == user_id].groupby(["activity_type_id", "hour"]).\
                                                    count().iloc[:, 0].\
                                                    reset_index()[["activity_type_id", "hour", "id"]].\
                                                    sort_values(["activity_type_id", "hour"])
        for id, hour, count in zip(corr["activity_type_id"], corr["hour"], corr["id"]):
            id = int(id) - 1
            hour = int(hour)
            heatmap_matrix[id, hour] = int(count)
        users_heatmap[user_id] = heatmap_matrix.reshape(-1)

        # TODO: yticksをactivity_type_idに対応させる
        # 各userのheatmapの表示
        if show_flag:
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax = sns.heatmap(heatmap_matrix, annot=True, fmt="d", ax=ax)
            ax.set_title(user_id)
    return users_heatmap

def create_users_filters():
    """
        userごとに予測するactivity_labelが違うためフィルタを作成する
    """
    filter_user_dict = {
        8 : [1,2,3,4,5,7,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24],
        13: [1,2,4,7,8,9,10,11,12,13,14,15,16,19,22,25,27],
        14: [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,19],
        15: [1,2,3,4,5,9,10,11,12,13,14,16,18,19,22],
        25: [1,2,3,4,6,7,8,10,13,14,16,17,18,22,23,24,25,26]
    }
    all_activity_id = np.arange(1, 28)
    all_hours = np.arange(0, 24)

    filter_default = pd.DataFrame(index=all_activity_id, columns=all_hours).fillna(0)
    filter_default.sort_index(inplace=True)

    users_filters = {}

    for user_id, activity_ids in filter_user_dict.items():
        user_filter = filter_default.copy()
        for activity_id in activity_ids:
            user_filter.loc[activity_id, :] = 1
        user_filter.sort_index(inplace=True)
        users_filters[user_id] = user_filter
    return users_filters

def create_y_label(df_care: pd.DataFrame):
    """
        各activity_labelをバイナリで作成する。(正解ラベル)
    """
    feat = df_care.groupby(["activity_type_id", "year-month-date-hour"]).count().\
                    reset_index()[['activity_type_id', 'year-month-date-hour','id']].\
                    rename(columns={"id": "counts"})
    # 頻度を出現のバイナリに変換
    feat["counts"] = feat["counts"].mask(feat.counts > 0, 1)
    df_care_date = df_care.copy()

    activity_type_ids = sorted(list(df_care['activity_type_id'].unique()))
    for activity_id in activity_type_ids:
        df_care_date = pd.merge(df_care_date, feat[feat['activity_type_id'] == activity_id][['year-month-date-hour', 'counts']],
                    on='year-month-date-hour', how="left").rename(columns={"counts": activity_id})
    df_care_date.loc[:, activity_type_ids] = df_care_date.loc[:, activity_type_ids].fillna(0)

    # 日付の重複を削除・year-month-date-hourにソート
    df_care_y = df_care_date[~df_care_date["year-month-date-hour"].duplicated()].loc[:, ["user_id", "year-month-date-hour", *activity_type_ids]].sort_values("year-month-date-hour")

    # activity_labelの欠損値埋め
    for activity_id in np.arange(1, 29):
        if activity_id not in df_care_y.columns:
            df_care_y.loc[:, activity_id] = 0.0

    return df_care_y.reindex(columns=["user_id", "year-month-date-hour", *np.arange(1, 29)])