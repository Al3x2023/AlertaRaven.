import time
from datetime import datetime
import pytest

from models import SensorEvent, SensorEventType
from database import Database


@pytest.mark.asyncio
async def test_db_query_performance(tmp_path):
    """Inserta N eventos y mide tiempo de consulta para asegurar que responde."""
    db_path = tmp_path / "perf_events.db"
    db = Database(str(db_path))
    await db.init_db()

    N = 1000
    # Insertar rápidamente usando save_sensor_event
    for i in range(N):
        ev = SensorEvent(
            device_id=f"dev-perf",
            label=SensorEventType.NORMAL,
            timestamp=datetime.now(),
            acceleration_magnitude=9.8 + (i % 3) * 0.1,
            gyroscope_magnitude=1.0 + (i % 2) * 0.1,
        )
        await db.save_sensor_event(ev)

    start = time.perf_counter()
    rows = await db.get_sensor_events(device_id="dev-perf", limit=2000)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(rows) == N
    # No imponemos umbrales rígidos; solo reportamos
    print(f"DB get_sensor_events: {N} filas en {elapsed_ms:.2f} ms")


@pytest.mark.asyncio
async def test_api_ingest_throughput(test_client, api_key_header):
    """Mide la tasa de ingestión HTTP de eventos de sensores."""
    M = 200
    start = time.perf_counter()
    for i in range(M):
        payload = {
            "device_id": "dev-load",
            "acceleration_magnitude": 18.5 + (i % 5),
            "gyroscope_magnitude": 12.5 + (i % 3),
            "timestamp": datetime.now().isoformat(),
        }
        resp = await test_client.post("/api/v1/sensor-events", json=payload, headers=api_key_header)
        assert resp.status_code == 200
    elapsed = time.perf_counter() - start

    rps = M / elapsed if elapsed > 0 else 0.0
    print(f"API ingest throughput: {rps:.2f} req/s ({M} en {elapsed:.2f}s)")
    # Validación mínima: se insertaron M elementos
    list_resp = await test_client.get("/api/v1/sensor-events", params={"device_id": "dev-load", "limit": 1000}, headers=api_key_header)
    assert list_resp.status_code == 200
    assert list_resp.json()["pagination"]["total"] >= M