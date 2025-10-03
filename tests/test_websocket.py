import pytest

# Prueba de conexión WebSocket usando TestClient (sincrónico)
@pytest.mark.usefixtures("temp_db_path")
def test_websocket_connect(tmp_path):
    from fastapi.testclient import TestClient
    from main import app, db

    # Usar una DB temporal para no tocar el archivo real
    db.db_path = str(tmp_path / "ws_test.db")

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # Mensaje de bienvenida enviado por WebSocketManager.connect
            msg = ws.receive_json()
            assert msg.get("type") == "connection_established"
            # Enviar un texto para liberar el await receive_text del endpoint
            ws.send_text("ping")
            # Cerrar conexión
            ws.close()