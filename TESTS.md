# Documentación de Pruebas

## Resumen
- La suite cubre pruebas unitarias, integración, rendimiento y WebSocket.
- Cada prueba usa una base de datos SQLite temporal para aislamiento.

## Cómo Ejecutarlas
- Desde el directorio `api/`:
  - Instala dependencias mínimas: `pip install pytest pytest-asyncio`
  - Ejecuta: `pytest -q`
- Algunos endpoints requieren autenticación con encabezado: `Authorization: Bearer alertaraven_mobile_key_2024`.

## Estructura de Pruebas
- `tests/conftest.py`
  - `event_loop`: bucle de eventos por sesión.
  - `temp_db_path`: path de DB temporal por test.
  - `test_client`: cliente `httpx.AsyncClient` con `ASGITransport(app=app)`. Inicializa la DB con `await db.init_db()` antes de usar la app.
  - `api_key_header`: encabezado de autenticación para endpoints protegidos.

- `tests/test_unit.py`
  - `test_database_init_and_health`: inicializa `Database` y verifica `health_check()`.
  - `test_classify_event_heuristic_cases`: valida umbrales de `classify_event_heuristic`.
  - `test_save_and_get_sensor_event`: inserta y lista eventos de sensores vía `Database`.

- `tests/test_integration_api.py`
  - `test_health_endpoint`: valida `/health` y estadísticas de WebSockets.
  - `test_ingest_and_list_sensor_events`: ingesta en `/api/v1/sensor-events` y listado con filtros.
  - `test_metrics_and_statistics_endpoints`: verifica `/api/statistics` (campos `total_alerts`, `accident_type_distribution`, `status_distribution`) y `/api/v1/metrics/sensor-events-summary`.

- `tests/test_performance.py`
  - `test_db_query_performance`: inserta `N=1000` eventos y mide latencia de consulta DB.
  - `test_api_ingest_throughput`: ingesta `M=200` eventos vía HTTP y calcula throughput, validando que se insertaron.

- `tests/test_websocket.py`
  - `test_websocket_connect`: conecta a `/ws`, verifica mensaje de bienvenida (`type: connection_established`), envía `ping` y cierra.

## Resultados (última ejecución)
- Entorno: Windows, Python 3.13
- Dependencias clave: `fastapi==0.110.0`, `httpx==0.25.2`, `aiosqlite==0.19.0`, `pytest==8.3.3`, `pytest-asyncio==0.23.8`.
- Comando: `pytest -q`
- Resultado: 9 passed, 0 failed (≈13.7s)
- Warnings: `PendingDeprecationWarning` por `python-multipart` import en Starlette.

## Notas de compatibilidad
- La app migró sus eventos de ciclo de vida a `lifespan` en `main.py` para evitar `DeprecationWarning` y manejar correctamente tareas de background (`heartbeat_task`).
- Se eliminó `class Config` vacío en `EmergencyAlert` (Pydantic v2) para evitar `PydanticDeprecatedSince20`.
- Los tests inicializan la base de datos manualmente en el fixture y no dependen del `lifespan` del transporte ASGI.

## Próximas extensiones (opcionales)
- Pruebas de mensajes WebSocket específicos (`new_alert`, `alert_status_change`, `heartbeat`).
- Cobertura de endpoints adicionales: contactos de emergencia, exportación CSV, manejo de alertas por estado.