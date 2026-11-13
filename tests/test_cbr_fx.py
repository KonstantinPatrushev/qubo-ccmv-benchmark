"""Official CBR exchange rates (cif.data.cbr_fx): the XML_dynamic service of cbr.ru.

The rate converts dividends paid in foreign currency (USD for QIWI, RUSAL, Polymetal, …) into
roubles for the Russian market (ANALYSIS_PLAN §3, RU exploratory).
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from cif.data import cbr_fx

_SAMPLE = """<?xml version="1.0" encoding="windows-1251"?>
<ValCurs ID="R01235" DateRange1="01.01.2013" DateRange2="15.01.2013" name="Foreign Currency Market Dynamic">
<Record Date="10.01.2013" Id="R01235"><Nominal>1</Nominal><Value>30,2645</Value></Record>
<Record Date="11.01.2013" Id="R01235"><Nominal>1</Nominal><Value>30,3699</Value></Record>
<Record Date="12.01.2013" Id="R01235"><Nominal>1</Nominal><Value>30,4313</Value></Record>
</ValCurs>
""".encode("cp1251")


def test_parse_dates_comma_decimals_and_currency():
    df = cbr_fx.parse_dynamic(_SAMPLE, "R01235")
    assert list(df.index) == list(pd.to_datetime(["2013-01-10", "2013-01-11", "2013-01-12"]))
    assert df.index.name == "date"
    assert df["rate"].tolist() == pytest.approx([30.2645, 30.3699, 30.4313], abs=0)
    assert df["nominal"].tolist() == [1, 1, 1]


def test_rate_is_per_one_unit_when_the_nominal_is_larger():
    xml = _SAMPLE.replace(b"<Nominal>1</Nominal><Value>30,2645", b"<Nominal>100</Nominal><Value>3026,45")
    df = cbr_fx.parse_dynamic(xml, "R01235")
    assert df["rate"].iloc[0] == pytest.approx(30.2645, rel=1e-12)
    assert df["nominal"].iloc[0] == 100


def test_cyrillic_attributes_in_windows_1251_are_read():
    xml = _SAMPLE.replace(b'name="Foreign Currency Market Dynamic"', "name=\"Динамика курса\"".encode("cp1251"))
    assert len(cbr_fx.parse_dynamic(xml, "R01235")) == 3


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (_SAMPLE.replace(b'ID="R01235"', b'ID="R01239"', 1), "currency"),
        (_SAMPLE.replace(b'Id="R01235"', b'Id="R01239"', 1), "currency"),
        (b'<?xml version="1.0" encoding="windows-1251"?><ValCurs ID="R01235"></ValCurs>', "no records"),
        (_SAMPLE.replace(b'Date="11.01.2013"', b'Date="10.01.2013"'), "duplicate"),
        (_SAMPLE.replace(b"30,3699", b"-1"), "positive"),
        (b"<html><body>error</body></html>", "ValCurs"),
    ],
    ids=["root-id", "record-id", "empty", "duplicate-date", "non-positive", "not-valcurs"],
)
def test_bad_responses_are_rejected(data, match):
    with pytest.raises(ValueError, match=match):
        cbr_fx.parse_dynamic(data, "R01235")


def test_url_of_the_dynamic_service():
    url = cbr_fx.dynamic_url("R01235", pd.Timestamp("2013-01-01"), pd.Timestamp("2025-12-31"))
    assert url == (
        "https://www.cbr.ru/scripts/XML_dynamic.asp?"
        "date_req1=01/01/2013&date_req2=31/12/2025&VAL_NM_RQ=R01235"
    )


def test_build_dataset_from_the_raw_file(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    name = cbr_fx.raw_name("R01235", pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-15"))
    (raw / name).write_bytes(_SAMPLE)
    summary = cbr_fx.build_dataset(raw, out, "R01235", pd.Timestamp("2013-01-01"),
                                   pd.Timestamp("2013-01-15"))
    df = pd.read_parquet(out / "usd_rub.parquet")
    assert df["rate"].tolist() == pytest.approx([30.2645, 30.3699, 30.4313], abs=0)
    assert summary["n_days"] == 3
    assert summary["first_date"] == "2013-01-10" and summary["last_date"] == "2013-01-12"
    assert summary["currency_id"] == "R01235" and summary["units"] == "RUB per 1 USD"
    json.dumps(summary)  # the summary goes into provenance.json


@pytest.mark.skipif(os.environ.get("CIF_NETWORK_TESTS") != "1", reason="set CIF_NETWORK_TESTS=1")
def test_download_real_file(tmp_path):
    meta = cbr_fx.download_dynamic("R01235", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"),
                                   tmp_path)
    df = cbr_fx.parse_dynamic((tmp_path / meta["file"]).read_bytes(), "R01235")
    assert 15 <= len(df) <= 23 and (df["rate"] > 0).all()
