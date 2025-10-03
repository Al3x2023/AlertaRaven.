import asyncio
from datetime import datetime
import pytest

from models import SensorEventType


@pytest.mark.asyncio
async def test_health_endpoint(test_client):
    resp = await test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["database"] in {"connected", "disconnected"}
    assert "websocket_connections" in data


@pytest.mark.asyncio
async def test_ingest_and_list_sensor_events(test_client, api_key_header):
    # Ingestar un evento sin label para activar la heurística
    payload = {
        "device_id": "dev-int-1",
        "acceleration_magnitude": 26.0,
        "gyroscope_magnitude": 3.0,
        "timestamp": datetime.now().isoformat(),
        "raw_data": {"window": [0.1, 0.2, 0.3]}
    }
    resp = await test_client.post("/api/v1/sensor-events", json=payload, headers=api_key_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["label"] == SensorEventType.PHONE_DROP.value
    event_id = data["event_id"]

    # Listar por device_id
    resp_list = await test_client.get("/api/v1/sensor-events", params={"device_id": "dev-int-1"}, headers=api_key_header)
    assert resp_list.status_code == 200
    items = resp_list.json()["items"]
    assert any(i["event_id"] == event_id for i in items)


@pytest.mark.asyncio
async def test_metrics_and_statistics_endpoints(test_client):
    # Estadísticas
    stats = await test_client.get("/api/statistics")
    assert stats.status_code == 200
    s = stats.json()
    assert "total_alerts" in s
    assert "accident_type_distribution" in s
    assert "status_distribution" in s

    # Resumen de eventos
    summary = await test_client.get("/api/v1/metrics/sensor-events-summary")
    assert summary.status_code == 200
    sm = summary.json()
    assert "total_events" in sm
    assert "by_label" in sm