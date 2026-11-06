from futurespace.pipeline.connectors.noaa_3day import parse_forecast

CURRENT = """\
:Product: 3-Day Forecast
:Issued: 2024 Dec 31 1230 UTC
NOAA Kp index breakdown Dec 31-Jan 02 2025

             Dec 31       Jan 01       Jan 02
00-03UT       0.33         5.00 (G1)    2.67
03-06UT       0.67         3.33         2.00
06-09UT       2.33         4.33         2.00
09-12UT       1.33         3.67         2.00
12-15UT       5.00 (G1)    2.67         2.00
15-18UT       6.67 (G3)    2.33         2.00
18-21UT       5.33 (G1)    2.00         2.00
21-00UT       5.00 (G1)    3.00         2.33

Rationale: algo.

B. NOAA Solar Radiation Activity Observation and Forecast
"""

OLD_2022 = """\
Product: 3-Day Forecast
Issued: 2022 Mar 25 0030 UTC
NOAA Kp index breakdown Mar 25-Mar 27 2022

            Mar 25     Mar 26     Mar 27
00-03UT        3          2          3
03-06UT        2          2          2
06-09UT        2          3          2
09-12UT        2          3          2
12-15UT        2          3          2
15-18UT        2          3          2
18-21UT        2          3          2
21-00UT        2          3          4
"""


def test_parses_current_format_and_year_rollover():
    df = parse_forecast(CURRENT)
    # Publicada às 12:30 de 31/12: só os blocos que ainda não tinham começado (a partir das 15h).
    assert str(df["target_time"].min()) == "2024-12-31 15:00:00+00:00"
    jan1 = df[df["target_time"] == "2025-01-01T00:00:00Z"].iloc[0]
    assert (jan1["kp"], jan1["noaa_scale"]) == (5.0, "G1")
    g3 = df[df["target_time"] == "2024-12-31T15:00:00Z"].iloc[0]
    assert (g3["kp"], g3["noaa_scale"]) == (6.67, "G3")
    assert len(df) == 3 + 8 + 8
    assert (df["time"].astype(str) == "2024-12-31 12:30:00+00:00").all()


def test_parses_2022_integer_format():
    df = parse_forecast(OLD_2022)
    assert len(df) == 7 + 8 + 8  # o bloco 00h de 25/03 já tinha começado às 00:30
    last = df[df["target_time"] == "2022-03-27T21:00:00Z"].iloc[0]
    assert last["kp"] == 4.0 and last["noaa_scale"] is None


def test_ignores_files_without_kp_table():
    assert parse_forecast("Issued: 2024 Dec 31 1230 UTC\nnada aqui").empty
