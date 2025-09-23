import pandas as pd
from io import StringIO
import requests
from dotenv import load_dotenv
import os
from ecbdata import ecbdata

load_dotenv()


def target_building_runups(data, p, v):
    data["Target"] = 0
    index = len(data["Close"])
    for i in range(index - p):
        var = (
            (data["Close"].iloc[i + p] - data["Close"].iloc[i]) / data["Close"].iloc[i]
        ) * 100
        if var >= v:
            data["Target"].iloc[i] = 1
    return data


def add_BOP(data: pd.DataFrame):
    data["Time"] = pd.to_datetime(data["Time"])

    BOP_URL = os.getenv("BOP_URL")
    response = requests.get(BOP_URL)
    bop_data = pd.read_csv(StringIO(response.text))[["TIME_PERIOD", "OBS_VALUE"]]

    bop_data["Year"] = bop_data["TIME_PERIOD"].str[:4].astype(int)
    bop_data["Quarter"] = bop_data["TIME_PERIOD"].str[-2:]

    quarter_publishing_date = {
        "Q1": "-06-25",
        "Q2": "-09-25",
        "Q3": "-12-25",
        "Q4": "-03-25",
    }
    bop_data["AdjustedYear"] = bop_data["Year"] + bop_data["Quarter"].apply(
        lambda q: 1 if q == "Q4" else 0
    )

    bop_data["QuarterDataRelease"] = pd.to_datetime(
        bop_data["AdjustedYear"].astype(str)
        + bop_data["Quarter"].map(quarter_publishing_date)
    )

    bop_data.sort_values("QuarterDataRelease", inplace=True)

    data = pd.merge_asof(
        data.sort_values("Time"),
        bop_data[["QuarterDataRelease", "OBS_VALUE"]],
        left_on="Time",
        right_on="QuarterDataRelease",
        direction="backward",
    ).rename(columns={"OBS_VALUE": "usa_bop"})

    data.drop("QuarterDataRelease", axis=1, inplace=True)
    return data


def add_interest_rates(data: pd.DataFrame):
    fed = os.getenv("FED_URL")
    fed_data = (
        pd.DataFrame(requests.get(fed).json()["observations"])[["date", "value"]]
        .sort_values("date")
        .rename(columns={"value": "fed_rate"})
    )
    data["Time"] = pd.to_datetime(data["Time"]).dt.strftime("%Y%m%d").astype(int)
    fed_data["date"] = (
        pd.to_datetime(fed_data["date"]).dt.strftime("%Y%m%d").astype(int)
    )
    data = pd.merge_asof(
        data.sort_values("Time"),
        fed_data,
        left_on="Time",
        right_on="date",
        direction="backward",
    )
    data["fed_rate"] = data["fed_rate"].astype(float)
    data.drop("date", axis=1, inplace=True)
    ecb_data = (
        ecbdata.get_series("FM.D.U2.EUR.4F.KR.MRR_FR.LEV", start="1999-12-29")
        .rename(columns={"OBS_VALUE": "ecb_rate"})[["ecb_rate", "TIME_PERIOD"]]
        .sort_values("TIME_PERIOD")
    )

    ecb_data["TIME_PERIOD"] = (
        pd.to_datetime(ecb_data["TIME_PERIOD"]).dt.strftime("%Y%m%d").astype(int)
    )

    data = pd.merge_asof(
        data.sort_values("Time"),
        ecb_data,
        left_on="Time",
        right_on="TIME_PERIOD",
        direction="backward",
    )
    data.drop("TIME_PERIOD", axis=1, inplace=True)
    return data


def add_macro(data: pd.DataFrame):
    data = add_BOP(data)
    data = add_interest_rates(data)
    return data
