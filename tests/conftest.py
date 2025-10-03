import os
import sys
import asyncio
import tempfile
from pathlib import Path
import pytest

from httpx import AsyncClient
from httpx import ASGITransport
import pytest_asyncio

# Asegurar que el directorio del proyecto (api/) esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar la app y la instancia global de DB
from main import app, db


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def temp_db_path(tmp_path):
    """Proporciona un path de base de datos temporal aislado por test."""
    return os.path.join(tmp_path, "test_alertas.db")


@pytest_asyncio.fixture()
async def test_client(temp_db_path):
    """Cliente HTTPX con lifespan activado y DB apuntando al path temporal."""
    # Redirigir la base de datos global a un archivo temporal por test
    db.db_path = temp_db_path
    # Inicializar DB manualmente para evitar dependencia de lifespan del servidor
    await db.init_db()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture()
def api_key_header():
    return {"Authorization": "Bearer alertaraven_mobile_key_2024"}