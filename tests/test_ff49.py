"""Kenneth French data library: 49 industry portfolios and the daily risk-free rate (FF49, RQ3).

Offline on synthetic files in French's CSV layout. The network test runs only with
``CIF_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from cif.data import ff49

INDUSTRY_CSV = (
    "This file was created using the 202608 CRSP database.\r\n"
    "It contains value- and equal-weighted returns for 49 industry portfolios.\r\n"
    "\r\n"
    "Missing data are indicated by -99.99 or -999.\r\n"
    "\r\n"
    "\r\n"
    "  Average Value Weighted Returns -- Daily\r\n"
    ",Agric,Food,Soda\r\n"
    "19260701,   0.56,  -0.07, -99.99\r\n"
    "19260702,   1.00,   0.10,   -999\r\n"
    "19260706,  -0.50,   0.20,   2.00\r\n"
    "\r\n"
    "\r\n"
    "  Average Equal Weighted Returns -- Daily\r\n"
    ",Agric,Food,Soda\r\n"
    "19260701,   0.10,   0.20,   0.30\r\n"
    "19260702,   0.40,   0.50,   0.60\r\n"
    "19260706,   0.70,   0.80,   0.90\r\n"
    "\r\n"
    "Copyright 2026 Eugene F. Fama and Kenneth R. French\r\n"
)

FACTORS_CSV = (
    "This file was created by using the 202608 CRSP database.\r\n"
    "The Tbill return is the simple daily rate that, over the number of trading days\r\n"
    "\r\n"
    ",Mkt-RF,SMB,HML,RF\r\n"
    "19260701,    0.09,   -0.23,   -0.28,    0.01\r\n"
    "19260702,    0.45,   -0.34,   -0.03,    0.01\r\n"
    "19260706,    0.17,    0.29,   -0.38,    0.02\r\n"
    "\r\n"
    "Copyright 2026 Eugene F. Fama and Kenneth R. French\r\n"
)


def _zip(path, member: str, text: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, text)


def test_parse_sections_titles_dates_and_missing_codes():
    tables = ff49.parse_french_csv(INDUSTRY_CSV)
    assert list(tables) == ["Average Value Weighted Returns -- Daily",
                            "Average Equal Weighted Returns -- Daily"]
    vw = tables["Average Value Weighted Returns -- Daily"]
    assert list(vw.columns) == ["Agric", "Food", "Soda"]
    assert list(vw.index) == list(pd.to_datetime(["1926-07-01", "1926-07-02", "1926-07-06"]))
    assert vw.loc["1926-07-01", "Agric"] == pytest.approx(0.56)  # still in percent
    assert np.isnan(vw.loc["1926-07-01", "Soda"]) and np.isnan(vw.loc["1926-07-02", "Soda"])


def test_untitled_table_and_risk_free_in_decimals():
    tables = ff49.parse_french_csv(FACTORS_CSV)
    assert list(tables) == [""]
    rf = ff49.risk_free(tables)
    assert rf.name == "rf"
    np.testing.assert_allclose(rf.to_numpy(), [0.0001, 0.0001, 0.0002])


def test_industry_returns_in_decimals_by_weighting():
    tables = ff49.parse_french_csv(INDUSTRY_CSV)
    vw = ff49.industry_returns(tables, weighting="value", expected_columns=3)
    ew = ff49.industry_returns(tables, weighting="equal", expected_columns=3)
    assert vw.loc["1926-07-06", "Soda"] == pytest.approx(0.02)
    assert ew.loc["1926-07-02", "Food"] == pytest.approx(0.005)
    with pytest.raises(ValueError, match="49"):
        ff49.industry_returns(tables, weighting="value")  # the real file has 49 industries


def test_parser_rejects_unsorted_dates():
    bad = INDUSTRY_CSV.replace("19260706,  -0.50", "19260630,  -0.50")
    with pytest.raises(ValueError, match="increasing"):
        ff49.parse_french_csv(bad)


def test_price_index_keeps_gaps_and_leading_missing():
    r = pd.DataFrame({"A": [0.10, np.nan, 0.10], "B": [np.nan, 0.05, 0.10]},
                     index=pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"]))
    p = ff49.price_index(r)
    np.testing.assert_allclose(p["A"].to_numpy(), [1.10, np.nan, 1.21])
    np.testing.assert_allclose(p["B"].to_numpy(), [np.nan, 1.05, 1.155])
    # the engine's forward-filled returns then reproduce the valid returns exactly
    back = p.ffill().pct_change(fill_method=None)
    assert back.loc["2000-01-05", "A"] == pytest.approx(0.10)


def test_read_zip_takes_the_single_csv_member(tmp_path):
    path = tmp_path / "49_Industry_Portfolios_daily_CSV.zip"
    _zip(path, "49_Industry_Portfolios_Daily.csv", INDUSTRY_CSV)
    assert ff49.read_zip_text(path).startswith("This file was created")


def test_build_dataset_from_raw_zips(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _zip(raw / ff49.INDUSTRY_ZIP, "49_Industry_Portfolios_Daily.csv", INDUSTRY_CSV)
    _zip(raw / ff49.FACTORS_ZIP, "F-F_Research_Data_Factors_daily.csv", FACTORS_CSV)
    out = tmp_path / "out"
    summary = ff49.build_dataset(raw, out, expected_industries=3)
    for name in ("returns_vw.parquet", "returns_ew.parquet", "prices_vw.parquet",
                 "rf_daily.parquet", "factors_daily.parquet"):
        assert (out / name).exists()
    prices = pd.read_parquet(out / "prices_vw.parquet")
    assert prices.loc["1926-07-02", "Agric"] == pytest.approx(1.0056 * 1.01)
    assert summary["vintage"] == "202608 CRSP database"
    assert summary["first_date"] == "1926-07-01" and summary["last_date"] == "1926-07-06"
    assert summary["n_industries"] == 3
    assert summary["first_valid"]["Soda"] == "1926-07-06"


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("CIF_NETWORK_TESTS") != "1", reason="set CIF_NETWORK_TESTS=1")
def test_download_real_files(tmp_path):
    for name in (ff49.INDUSTRY_ZIP, ff49.FACTORS_ZIP):
        meta = ff49.download_zip(name, tmp_path)
        assert (tmp_path / name).stat().st_size == meta["bytes"] > 100_000
    tables = ff49.parse_french_csv(ff49.read_zip_text(tmp_path / ff49.INDUSTRY_ZIP))
    assert ff49.industry_returns(tables).shape[1] == 49


def test_read_zip_rejects_archives_without_exactly_one_csv(tmp_path):
    path = tmp_path / "x.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.csv", "x")
        zf.writestr("b.csv", "y")
    with pytest.raises(ValueError, match="one CSV"):
        ff49.read_zip_text(path)


MONTHLY_CSV = (
    "This file was created using the 202608 CRSP database.\r\n"
    "\r\n"
    ",Mkt-RF,SMB,HML,RF\r\n"
    "192607,    2.89,   -2.55,   -2.39,    0.22\r\n"
    "192608,    2.64,   -1.14,    3.81,    0.25\r\n"
    "\r\n"
    " Annual Factors: January-December \r\n"
    ",Mkt-RF,SMB,HML,RF\r\n"
    "1927,   29.47,   -2.46,   -3.75,    3.12\r\n"
    "\r\n"
    "Copyright 2026 Eugene F. Fama and Kenneth R. French\r\n"
)


def test_monthly_and_annual_tables_are_parsed_as_periods():
    tables = ff49.parse_french_csv(MONTHLY_CSV)
    monthly, annual = tables[""], tables["Annual Factors: January-December"]
    assert list(monthly.index) == list(pd.period_range("1926-07", periods=2, freq="M"))
    assert annual.index[0] == pd.Period("1927", freq="Y")
    assert monthly.loc[pd.Period("1926-08", freq="M"), "RF"] == pytest.approx(0.25)


def test_daily_rf_from_monthly_compounds_to_the_monthly_rate():
    calendar = pd.to_datetime(["1926-07-01", "1926-07-02", "1926-07-06", "1926-08-02", "1926-08-03"])
    rf_m = pd.Series([0.0022, 0.0025], index=pd.period_range("1926-07", periods=2, freq="M"))
    daily = ff49.daily_rf_from_monthly(rf_m, calendar)
    assert list(daily.index) == list(calendar)
    july = (1 + daily.loc["1926-07"]).prod() - 1
    assert july == pytest.approx(0.0022, rel=1e-12)
    assert daily.loc["1926-08-02"] == pytest.approx((1.0025) ** 0.5 - 1, rel=1e-12)
    with pytest.raises(ValueError, match="no monthly rate"):
        ff49.daily_rf_from_monthly(rf_m.iloc[:1], calendar)


def test_build_prefers_the_monthly_risk_free_rate(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _zip(raw / ff49.INDUSTRY_ZIP, "49_Industry_Portfolios_Daily.csv", INDUSTRY_CSV)
    _zip(raw / ff49.FACTORS_ZIP, "F-F_Research_Data_Factors_daily.csv", FACTORS_CSV)
    out = tmp_path / "out"
    summary = ff49.build_dataset(raw, out, expected_industries=3)
    assert summary["rf_source"].startswith("daily factors file")  # no monthly file: rounded rate
    monthly = MONTHLY_CSV.replace("192608", "192609").replace("192607", "192607")
    _zip(raw / ff49.FACTORS_MONTHLY_ZIP, "F-F_Research_Data_Factors.csv", monthly)
    summary = ff49.build_dataset(raw, out, expected_industries=3)
    assert summary["rf_source"].startswith("monthly factors file")
    rf = pd.read_parquet(out / "rf_daily.parquet")["rf"]
    assert (1 + rf).prod() - 1 == pytest.approx(0.0022, rel=1e-12)  # three July days in the panel
