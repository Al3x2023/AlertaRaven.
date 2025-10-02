import json
import asyncio
from typing import Dict, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Gestor de conexiones WebSocket para notificaciones en tiempo real"""
    
    def __init__(self):
        # Conexiones activas por tipo de cliente
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "dashboard": set(),  # Conexiones del dashboard web
            "mobile": set(),     # Conexiones de apps móviles
            "admin": set()       # Conexiones de administradores
        }
        
        # Conexiones por device_id para notificaciones específicas
        self.device_connections: Dict[str, WebSocket] = {}
        
        # Lock para operaciones thread-safe
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_type: str = "dashboard", device_id: str = None):
        """Acepta una nueva conexión WebSocket"""
        await websocket.accept()
        
        async with self._lock:
            # Agregar a conexiones por tipo
            if client_type in self.active_connections:
                self.active_connections[client_type].add(websocket)
            else:
                self.active_connections[client_type] = {websocket}
            
            # Si es un dispositivo móvil, guardar por device_id
            if device_id and client_type == "mobile":
                self.device_connections[device_id] = websocket
        
        logger.info(f"Nueva conexión WebSocket: {client_type} (device_id: {device_id})")
        
        # Enviar mensaje de bienvenida
        await self.send_personal_message(websocket, {
            "type": "connection_established",
            "message": "Conectado al sistema de alertas AlertaRaven",
            "client_type": client_type,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def disconnect(self, websocket: WebSocket, client_type: str = "dashboard", device_id: str = None):
        """Desconecta un WebSocket"""
        async with self._lock:
            # Remover de conexiones por tipo
            if client_type in self.active_connections:
                self.active_connections[client_type].discard(websocket)
            
            # Remover de conexiones por device_id
            if device_id and device_id in self.device_connections:
                if self.device_connections[device_id] == websocket:
                    del self.device_connections[device_id]
        
        logger.info(f"Conexión WebSocket desconectada: {client_type} (device_id: {device_id})")
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Envía un mensaje a una conexión específica"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error enviando mensaje personal: {e}")
    
    async def broadcast_to_type(self, client_type: str, message: Dict[str, Any]):
        """Envía un mensaje a todas las conexiones de un tipo específico"""
        if client_type not in self.active_connections:
            return
        
        disconnected = set()
        
        for connection in self.active_connections[client_type].copy():
            # Evitar enviar si la conexión ya no está activa
            if hasattr(connection, "client_state") and connection.client_state != WebSocketState.CONNECTED:
                disconnected.add(connection)
                continue
            try:
                await connection.send_text(json.dumps(message))
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception as e:
                logger.error(f"Error enviando mensaje broadcast: {e}")
                disconnected.add(connection)
        
        # Limpiar conexiones desconectadas
        if disconnected:
            async with self._lock:
                self.active_connections[client_type] -= disconnected
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Envía un mensaje a todas las conexiones activas"""
        for client_type in self.active_connections:
            await self.broadcast_to_type(client_type, message)
    
    async def send_to_device(self, device_id: str, message: Dict[str, Any]):
        """Envía un mensaje a un dispositivo específico"""
        if device_id in self.device_connections:
            await self.send_personal_message(self.device_connections[device_id], message)
        else:
            logger.warning(f"Dispositivo {device_id} no está conectado")
    
    async def notify_new_alert(self, alert_data: Dict[str, Any]):
        """Notifica sobre una nueva alerta de emergencia"""
        message = {
            "type": "new_alert",
            "data": alert_data,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Notificar a dashboards y administradores
        await self.broadcast_to_type("dashboard", message)
        await self.broadcast_to_type("admin", message)
        
        logger.info(f"Notificación de nueva alerta enviada: {alert_data.get('alert_id')}")
    
    async def notify_alert_status_change(self, alert_id: str, old_status: str, new_status: str, device_id: str = None):
        """Notifica sobre cambio de estado de una alerta"""
        message = {
            "type": "alert_status_change",
            "data": {
                "alert_id": alert_id,
                "old_status": old_status,
                "new_status": new_status,
                "device_id": device_id
            },
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Notificar a todos los tipos de cliente
        await self.broadcast_to_all(message)
        
        # Notificar específicamente al dispositivo si está conectado
        if device_id:
            await self.send_to_device(device_id, message)
        
        logger.info(f"Notificación de cambio de estado enviada: {alert_id} ({old_status} -> {new_status})")
    
    async def notify_system_status(self, status: str, message: str):
        """Notifica sobre el estado del sistema"""
        notification = {
            "type": "system_status",
            "data": {
                "status": status,
                "message": message
            },
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await self.broadcast_to_all(notification)
        logger.info(f"Notificación de sistema enviada: {status} - {message}")
    
    async def send_heartbeat(self):
        """Envía heartbeat a todas las conexiones para mantenerlas vivas"""
        message = {
            "type": "heartbeat",
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await self.broadcast_to_all(message)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de conexiones activas"""
        stats = {
            "total_connections": sum(len(connections) for connections in self.active_connections.values()),
            "connections_by_type": {
                client_type: len(connections) 
                for client_type, connections in self.active_connections.items()
            },
            "device_connections": len(self.device_connections),
            "connected_devices": list(self.device_connections.keys())
        }
        
        return stats
    
    async def cleanup_disconnected(self):
        """Limpia conexiones desconectadas"""
        async with self._lock:
            for client_type in self.active_connections:
                disconnected = set()
                for connection in self.active_connections[client_type]:
                    try:
                        # Intentar enviar un ping para verificar la conexión
                        await connection.ping()
                    except:
                        disconnected.add(connection)
                
                # Remover conexiones desconectadas
                self.active_connections[client_type] -= disconnected
            
            # Limpiar device_connections
            disconnected_devices = []
            for device_id, connection in self.device_connections.items():
                try:
                    await connection.ping()
                except:
                    disconnected_devices.append(device_id)
            
            for device_id in disconnected_devices:
                del self.device_connections[device_id]
        
        if disconnected_devices:
            logger.info(f"Conexiones desconectadas limpiadas: {len(disconnected_devices)} dispositivos")

# Instancia global del gestor de WebSockets
websocket_manager = WebSocketManager()

# Tarea de heartbeat para mantener conexiones vivas
async def heartbeat_task():
    """Tarea que envía heartbeat periódicamente"""
    while True:
        try:
            await websocket_manager.send_heartbeat()
            await websocket_manager.cleanup_disconnected()
            await asyncio.sleep(30)  # Heartbeat cada 30 segundos
        except Exception as e:
            logger.error(f"Error en heartbeat task: {e}")
            await asyncio.sleep(5)