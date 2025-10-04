# AlertaRavenAPI

## Pruebas

- Requisitos: `pytest`, `pytest-asyncio` (instálalos con `pip install pytest pytest-asyncio`).
- Las pruebas usan una base de datos SQLite temporal por test y el cliente asíncrono de HTTP (`httpx`).

### Ejecutar la suite completa

- Desde el directorio `api/`:

```
pytest -q
```

### Estructura de pruebas

- `tests/conftest.py`: Fixtures para cliente HTTP asíncrono y path de DB temporal.
- `tests/test_unit.py`: Pruebas unitarias de la heurística y CRUD de `Database`.
- `tests/test_integration_api.py`: Pruebas de integración de endpoints (`/health`, `/api/v1/sensor-events`, métricas y estadísticas).
- `tests/test_performance.py`: Pruebas de rendimiento (throughput de ingestión HTTP y consultas DB). No imponen umbrales rígidos; reportan métricas.
- `tests/test_websocket.py`: Prueba de conexión al endpoint WebSocket (`/ws`) y mensaje de bienvenida.

### Notas

- La autenticación de endpoints de entrenamiento usa `Authorization: Bearer alertaraven_mobile_key_2024`.
- En pruebas se inicializa la base de datos manualmente en el fixture para evitar dependencias de `lifespan` del transporte ASGI.

### Resultados (última ejecución)

- Entorno: Windows, Python 3.13
- Dependencias clave: `fastapi==0.110.0`, `httpx==0.25.2`, `aiosqlite==0.19.0`, `pytest==8.3.3`, `pytest-asyncio==0.23.8`.
- Comando: `pytest -q`
- Resultado: 9 passed, 0 failed (≈13.7s)
- Warnings observados:
  - `DeprecationWarning` por `@app.on_event(...)` en FastAPI. Recomendado migrar a manejadores de `lifespan`.
  - `PydanticDeprecatedSince20`: uso de `config` basado en clase. Migrar a `ConfigDict`.