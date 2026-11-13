"""Tests for ``cif.data.riskfree``: daily risk-free rates from FRED DTB3 and CBR RUONIA (WP-E1).

Offline on synthetic fixtures; network tests run only with ``CIF_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cif.data import riskfree as rf

RF = Path(__file__).resolve().parents[1] / "data" / "processed" / "riskfree_v2"


def _built(directory: Path) -> bool:
    """The dataset is built locally: its provenance (versioned in git since 2026-10-05, so present
    in every clone) and every processed file the provenance records."""
    prov = directory / "provenance.json"
    if not prov.exists():
        return False
    recorded = json.loads(prov.read_text(encoding="utf-8")).get("files", {}).get("processed", {})
    return bool(recorded) and all((directory / name).exists() for name in recorded)


needs_rf = pytest.mark.skipif(
    not _built(RF),
    reason="data/processed/riskfree_v2 is not built (scripts/data/build_riskfree.py)",
)
needs_network = pytest.mark.skipif(
    os.environ.get("CIF_NETWORK_TESTS") != "1",
    reason="network test: set CIF_NETWORK_TESTS=1 to run",
)

FRED_CSV = """observation_date,DTB3
2024-01-01,.
2024-01-02,5.35
2024-01-03,5.33
2024-01-04,.
2024-01-05,5.30
"""

# trimmed real answer of DailyInfoWebServ.Ruonia (2024-01-09 .. 2024-01-12)
RUONIA_XML = """<?xml version="1.0" encoding="utf-8"?><soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema"><soap:Body><RuoniaResponse xmlns="http://web.cbr.ru/"><RuoniaResult><xs:schema id="Ruonia" xmlns="" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:msdata="urn:schemas-microsoft-com:xml-msdata"><xs:element name="Ruonia" msdata:IsDataSet="true"/></xs:schema><diffgr:diffgram xmlns:msdata="urn:schemas-microsoft-com:xml-msdata" xmlns:diffgr="urn:schemas-microsoft-com:xml-diffgram-v1"><Ruonia xmlns=""><ro diffgr:id="ro1" msdata:rowOrder="0"><D0>2024-01-11T00:00:00+03:00</D0><ruo>15.2000</ruo><vol>907.9300</vol><DateUpdate>2024-01-12T14:07:52.6+03:00</DateUpdate></ro><ro diffgr:id="ro2" msdata:rowOrder="1"><D0>2024-01-12T00:00:00+03:00</D0><ruo>15.1500</ruo><vol>301.5500</vol><DateUpdate>2024-01-15T14:06:36.203+03:00</DateUpdate></ro><ro diffgr:id="ro3" msdata:rowOrder="2"><D0>2024-01-15T00:00:00+03:00</D0><ruo>15.4200</ruo><vol>424.7000</vol><DateUpdate>2024-01-16T14:05:00.987+03:00</DateUpdate></ro></Ruonia></diffgr:diffgram></RuoniaResult></RuoniaResponse></soap:Body></soap:Envelope>"""


# --------------------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------------------


def test_compound_daily_rate_follows_the_formula():
    assert rf.daily_rate_compound(5.0, 1) == pytest.approx(1.05 ** (1 / 365) - 1)
    assert rf.daily_rate_compound(5.0, 3) == pytest.approx(1.05 ** (3 / 365) - 1)
    # three single days compound to the same as one three-day period
    assert (1 + rf.daily_rate_compound(5.0, 1)) ** 3 == pytest.approx(1 + rf.daily_rate_compound(5.0, 3))


def test_simple_daily_rate_is_act_365():
    assert rf.daily_rate_simple(10.0, 3) == pytest.approx(0.10 * 3 / 365)
    assert rf.daily_rate_simple(0.0, 3) == 0.0


def test_discount_rate_to_bond_equivalent_yield():
    # 13-week bill, discount 5.00 %: price 98.7361, BEY = 365 * 0.05 / (360 - 0.05 * 91)
    assert rf.discount_to_bey(5.0, 91) == pytest.approx(100 * 365 * 0.05 / (360 - 0.05 * 91))
    assert rf.discount_to_bey(5.0, 91) > 5.0  # 365/360 day-count and the price base
    out = rf.discount_to_bey(pd.Series([0.0, 2.0, 5.0]), 91)
    assert out.iloc[0] == 0.0 and out.is_monotonic_increasing


# --------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------


def test_parse_fred_csv_drops_missing_markers():
    s = rf.parse_fred_csv(FRED_CSV)
    assert list(s.index) == pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]).tolist()
    assert s.loc["2024-01-03"] == 5.33 and s.name == "DTB3"


def test_parse_fred_csv_accepts_the_old_header():
    s = rf.parse_fred_csv("DATE,DTB3\n2020-01-02,1.5\n")
    assert s.iloc[0] == 1.5


def test_parse_ruonia_soap():
    df = rf.parse_ruonia_soap(RUONIA_XML)
    assert list(df.columns) == ["y_pct", "volume_bn", "published", "published_assumed"]
    assert df.index.tolist() == pd.to_datetime(["2024-01-11", "2024-01-12", "2024-01-15"]).tolist()
    assert df.loc["2024-01-12", "y_pct"] == pytest.approx(15.15)
    # the fixing of Friday is published on Monday: it is not known before then
    assert df.loc["2024-01-12", "published"] == pd.Timestamp("2024-01-15")


def test_parse_ruonia_soap_without_publication_date_assumes_next_day():
    old = ('<Ruonia xmlns=""><ro><D0>2010-01-11T00:00:00+03:00</D0><ruo>2.9800</ruo><vol>64.5600</vol></ro>'
           "</Ruonia>")
    df = rf.parse_ruonia_soap(f"<root>{old}</root>")
    assert df.loc["2010-01-11", "published"] == pd.Timestamp("2010-01-12")
    assert bool(df.loc["2010-01-11", "published_assumed"])
    real = rf.parse_ruonia_soap(RUONIA_XML)
    assert not real["published_assumed"].any()


def test_parse_ruonia_soap_rejects_a_fault():
    fault = ('<?xml version="1.0"?><soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
             "<soap:Body><soap:Fault><faultstring>boom</faultstring></soap:Fault></soap:Body></soap:Envelope>")
    with pytest.raises(rf.RiskFreeError, match="boom"):
        rf.parse_ruonia_soap(fault)


# --------------------------------------------------------------------------------------
# Alignment to a trading calendar
# --------------------------------------------------------------------------------------


def _obs(**kv):
    return pd.Series({pd.Timestamp(k.replace("_", "-")): v for k, v in kv.items()}, dtype=float)


def test_alignment_uses_the_previous_trading_day_and_calendar_day_count():
    cal = pd.bdate_range("2024-01-02", "2024-01-09")  # Tue..Tue, weekend 6-7
    obs = pd.Series([4.0, 4.1, 4.2, 4.3, 4.4], index=pd.bdate_range("2024-01-02", "2024-01-08"))
    out = rf.daily_rate_on_calendar(obs, cal, compounding="compound")
    assert np.isnan(out["r_daily"].iloc[0])  # nothing is known before the first day
    # Monday 01-08 accrues over the weekend (3 days) at Friday's 4.3 %
    mon = out.loc["2024-01-08"]
    assert mon["days"] == 3 and mon["obs_date"] == pd.Timestamp("2024-01-05") and mon["y_pct"] == 4.3
    assert mon["r_daily"] == pytest.approx(1.043 ** (3 / 365) - 1)
    tue = out.loc["2024-01-09"]
    assert tue["days"] == 1 and tue["y_pct"] == 4.4 and tue["obs_date"] == pd.Timestamp("2024-01-08")


def test_alignment_has_no_look_ahead():
    cal = pd.bdate_range("2024-01-02", "2024-01-05")
    base = pd.Series(1.0, index=cal)
    shocked = base.copy()
    shocked.loc["2024-01-04"] = 99.0  # a spike on the day itself must not reach that day's return
    a = rf.daily_rate_on_calendar(base, cal)
    b = rf.daily_rate_on_calendar(shocked, cal)
    assert a.loc["2024-01-04", "r_daily"] == b.loc["2024-01-04", "r_daily"]
    assert b.loc["2024-01-05", "y_pct"] == 99.0  # it enters the next day


def test_alignment_carries_the_last_value_over_missing_observations():
    # FRED has no value on US holidays; the target market trades on that day
    cal = pd.bdate_range("2024-01-02", "2024-01-08")
    obs = pd.Series([5.0, 5.1, 5.3], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]))
    out = rf.daily_rate_on_calendar(obs, cal)
    assert out.loc["2024-01-05", "y_pct"] == 5.1  # Thursday 01-04 has no fixing: 01-03 stays
    assert out.loc["2024-01-05", "obs_date"] == pd.Timestamp("2024-01-03")


def test_alignment_respects_publication_dates():
    cal = pd.bdate_range("2024-01-10", "2024-01-17")
    d0 = pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15", "2024-01-16"])
    obs = pd.Series([15.0, 15.1, 15.2, 15.3, 15.4], index=d0)
    published = pd.Series(pd.to_datetime(["2024-01-11", "2024-01-12", "2024-01-15", "2024-01-16",
                                          "2024-01-17"]), index=d0)
    out = rf.daily_rate_on_calendar(obs, cal, compounding="simple", available_from=published)
    # Tuesday 01-16 follows Monday 01-15: by Monday's close the fixing of Friday 01-12 is out
    # (published Monday), the fixing of Monday itself is not (published Tuesday)
    tue = out.loc["2024-01-16"]
    assert tue["obs_date"] == pd.Timestamp("2024-01-12") and tue["y_pct"] == 15.2
    assert tue["r_daily"] == pytest.approx(0.152 * 1 / 365)
    naive = rf.daily_rate_on_calendar(obs, cal, compounding="simple")
    assert naive.loc["2024-01-16", "obs_date"] == pd.Timestamp("2024-01-15")


def test_alignment_over_a_long_closure_uses_calendar_days():
    cal = pd.DatetimeIndex(["2022-02-25", "2022-03-24"])  # the 2022 closure
    obs = pd.Series([9.5], index=pd.DatetimeIndex(["2022-02-25"]))
    out = rf.daily_rate_on_calendar(obs, cal, compounding="simple")
    assert out.loc["2022-03-24", "days"] == 27
    assert out.loc["2022-03-24", "r_daily"] == pytest.approx(0.095 * 27 / 365)


def test_alignment_rejects_non_monotone_publication():
    cal = pd.bdate_range("2024-01-02", "2024-01-05")
    obs = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    bad = pd.Series(pd.to_datetime(["2024-01-04", "2024-01-03"]), index=obs.index)
    with pytest.raises(ValueError, match="publication"):
        rf.daily_rate_on_calendar(obs, cal, available_from=bad)


# --------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------


@needs_network
def test_network_fred_dtb3_downloads():
    series, raw = rf.fetch_fred_csv("DTB3")
    # the secondary-market discount rate was slightly negative in March 2020
    assert series.index.max() > pd.Timestamp("2025-01-01") and -1.0 < series.min() < 20
    assert raw.startswith(b"observation_date") or raw.startswith(b"DATE")


@needs_network
def test_network_ruonia_downloads():
    df, raw = rf.fetch_ruonia("2024-01-09", "2024-01-19")
    assert len(df) == 9 and df.loc["2024-01-09", "y_pct"] == pytest.approx(15.59)
    assert b"RuoniaResponse" in raw


# --------------------------------------------------------------------------------------
# Invariants of the built series (skipped when data/processed/riskfree_v2 is absent)
# --------------------------------------------------------------------------------------


@needs_rf
def test_built_series_have_one_nan_and_plausible_daily_rates():
    from cif.data import moex

    us = pd.read_parquet(RF / "us_dtb3.parquet")
    ru = pd.read_parquet(RF / "ru_ruonia.parquet")
    for name, df in (("us", us), ("ru", ru)):
        assert df["r_daily"].isna().sum() == 1 and np.isnan(df["r_daily"].iloc[0]), name
        assert df["r_daily"].dropna().between(-1e-4, 0.02).all(), name
        # the fixing used is never dated after the previous trading day
        prev = df.index.to_series().shift(1)
        assert (df["obs_date"].iloc[1:] <= prev.iloc[1:]).all(), name
    closure = ru.loc["2022-03-24"]
    assert closure["days"] == 27  # 25.02 -> 24.03: calendar days, not one day
    prov = json.loads((RF / "provenance.json").read_text())
    for rel, meta in prov["files"]["processed"].items():
        assert moex.sha256_file(RF / rel) == meta["sha256"], rel
