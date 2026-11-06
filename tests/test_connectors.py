import json
from datetime import UTC, datetime

import httpx

from futurespace.pipeline.connectors import omni, swpc
from futurespace.pipeline.http import make_client

START = datetime(2024, 5, 10, tzinfo=UTC)
END = datetime(2024, 5, 11, tzinfo=UTC)


def client_for(routes: dict[str, object]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = routes[request.url.path]
        if isinstance(body, str):
            return httpx.Response(200, text=body)
        return httpx.Response(200, content=json.dumps(body))

    return make_client(transport=httpx.MockTransport(handler))


def test_goes_xray_pivots_bands():
    rows = [
        {"time_tag": "2026-11-06T19:13:00Z", "satellite": 18, "flux": 1e-9, "energy": "0.05-0.4nm"},
        {"time_tag": "2026-11-06T19:13:00Z", "satellite": 18, "flux": 9e-7, "energy": "0.1-0.8nm"},
    ]
    with client_for({"/json/goes/primary/xrays-7-day.json": rows}) as c:
        df = swpc.goes_xray(c, START, END)
    assert df.to_dict("records") == [
        {"time": "2026-11-06T19:13:00Z", "satellite": 18, "flux_long": 9e-7, "flux_short": 1e-9}
    ]


def test_kp_forecast_keeps_only_predictions_stamped_with_fetch_hour():
    rows = [
        {"time_tag": "2026-11-06T00:00:00", "kp": 2.0, "observed": "observed", "noaa_scale": None},
        {"time_tag": "2026-11-08T00:00:00", "kp": 5.0, "observed": "predicted", "noaa_scale": "G1"},
    ]
    with client_for({"/products/noaa-planetary-k-index-forecast.json": rows}) as c:
        df = swpc.kp_forecast(c, START, datetime(2026, 11, 6, 19, 42, tzinfo=UTC))
    assert len(df) == 1
    assert str(df.iloc[0]["time"]) == "2026-11-06 19:00:00+00:00"
    assert df.iloc[0]["noaa_scale"] == "G1"


def test_omni_orders_parameters_and_masks_fill_values():
    info = {
        "parameters": [{"name": "Time"}]
        + [{"name": name, "fill": "99" if name == "KP1800" else "9999.0"} for name in omni.PARAMETERS]
    }
    values = ["9999.0"] * len(omni.PARAMETERS)
    names = list(omni.PARAMETERS)
    values[names.index("KP1800")] = "87"
    values[names.index("V1800")] = "697"
    csv = "2024-05-10T18:30:00.000Z," + ",".join(values) + "\n"
    with client_for({"/hapi/info": info, "/hapi/data": csv}) as c:
        df = omni.omni_hourly(c, START, END)
    row = df.iloc[0]
    assert str(row["time"]) == "2024-05-10 18:00:00+00:00"
    assert row["kp"] == 8.7
    assert row["speed"] == 697
    assert df["dst"].isna().all()
