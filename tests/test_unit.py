import json
from datetime import datetime
import pytest

from main import classify_event_heuristic
from models import SensorEvent, SensorEventType
from database import Database


@pytest.mark.asyncio
async def test_database_init_and_health(tmp_path):
    db_path = tmp_path / "unit_alertas.db"
    db = Database(str(db_path))
    await db.init_db()
    assert await db.health_check() is True


def test_classify_event_heuristic_cases():
    # Caída del teléfono: alta aceleración, bajo giroscopio
    assert classify_event_heuristic(26.0, 3.0) == SensorEventType.PHONE_DROP
    # Accidente vehículo: alta aceleración y giroscopio
    assert classify_event_heuristic(19.0, 13.0) == SensorEventType.VEHICLE_ACCIDENT
    # Normal por defecto
    assert classify_event_heuristic(10.0, 4.0) == SensorEventType.NORMAL


@pytest.mark.asyncio
async def test_save_and_get_sensor_event(tmp_path):
    db_path = tmp_path / "unit_events.db"
    db = Database(str(db_path))
    await db.init_db()

    event = SensorEvent(
        device_id="device-123",
        label=SensorEventType.NORMAL,
        timestamp=datetime.now(),
        acceleration_magnitude=9.81,
        gyroscope_magnitude=1.23,
        accel_variance=0.5,
        gyro_variance=0.3,
        accel_jerk=0.1,
        predicted_label=None,
        prediction_confidence=None,
        raw_data={"window": [1, 2, 3]},
    )
    event_id = await db.save_sensor_event(event)
    assert event_id == event.event_id

    rows = await db.get_sensor_events(device_id="device-123")
    assert len(rows) == 1
    r = rows[0]
    assert r["device_id"] == "device-123"
    assert r["label"] == SensorEventType.NORMAL.value
    assert r["acceleration_magnitude"] == pytest.approx(9.81)
    assert r["raw_data"] == {"window": [1, 2, 3]}