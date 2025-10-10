import sqlite3
import aiosqlite
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from models import EmergencyAlert, AlertStatus, AccidentType, DeviceInfo, NotificationLog, SensorEvent, SensorEventType
import logging

logger = logging.getLogger(__name__)

class Database:
    """Gestor de base de datos para AlertaRaven API"""
    
    def __init__(self, db_path: str = "alertas.db"):
        self.db_path = db_path
    
    async def init_db(self):
        """Inicializa las tablas de la base de datos"""
        async with aiosqlite.connect(self.db_path) as db:
            # Tabla de alertas de emergencia
            await db.execute("""
                CREATE TABLE IF NOT EXISTS emergency_alerts (
                    alert_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    user_id TEXT,
                    accident_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    confidence REAL NOT NULL,
                    acceleration_magnitude REAL NOT NULL,
                    gyroscope_magnitude REAL NOT NULL,
                    location_data TEXT,
                    medical_info TEXT,
                    emergency_contacts TEXT,
                    status TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    additional_data TEXT
                )
            """)
            
            # Tabla de dispositivos
            await db.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    device_model TEXT,
                    os_version TEXT,
                    app_version TEXT,
                    last_seen DATETIME NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Tabla de logs de notificaciones
            await db.execute("""
                CREATE TABLE IF NOT EXISTS notification_logs (
                    log_id TEXT PRIMARY KEY,
                    alert_id TEXT NOT NULL,
                    notification_type TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    error_message TEXT,
                    FOREIGN KEY (alert_id) REFERENCES emergency_alerts (alert_id)
                )
            """)

            # Tabla de contactos por dispositivo
            await db.execute("""
                CREATE TABLE IF NOT EXISTS device_contacts (
                    device_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    relationship TEXT,
                    is_primary BOOLEAN NOT NULL DEFAULT 0,
                    updated_at DATETIME NOT NULL,
                    PRIMARY KEY (device_id, phone)
                )
            """)
            
            # Índices para mejorar rendimiento
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_device_id ON emergency_alerts(device_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON emergency_alerts(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON emergency_alerts(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_notifications_alert_id ON notification_logs(alert_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_contacts_device_id ON device_contacts(device_id)")

            # Tabla de eventos de sensores para entrenamiento
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sensor_events (
                    event_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    acceleration_magnitude REAL NOT NULL,
                    gyroscope_magnitude REAL NOT NULL,
                    accel_variance REAL,
                    gyro_variance REAL,
                    accel_jerk REAL,
                    predicted_label TEXT,
                    prediction_confidence REAL,
                    raw_data TEXT,
                    created_at DATETIME NOT NULL
                )
            """)

            await db.execute("CREATE INDEX IF NOT EXISTS idx_sensor_events_device ON sensor_events(device_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_sensor_events_label ON sensor_events(label)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_sensor_events_time ON sensor_events(timestamp)")
            
            # Intentar agregar columnas si la tabla existe desde antes
            try:
                await db.execute("ALTER TABLE sensor_events ADD COLUMN predicted_label TEXT")
            except Exception:
                pass
            try:
                await db.execute("ALTER TABLE sensor_events ADD COLUMN prediction_confidence REAL")
            except Exception:
                pass

            # Tabla de snapshots de métricas de modelo
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS model_metrics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at DATETIME NOT NULL,
                    model_version TEXT,
                    precision REAL,
                    recall REAL,
                    f1 REAL,
                    classes_json TEXT,
                    confusion_matrix_json TEXT
                )
                """
            )

            await db.execute("CREATE INDEX IF NOT EXISTS idx_model_metrics_created ON model_metrics_snapshots(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_model_metrics_version ON model_metrics_snapshots(model_version)")

            await db.commit()
            logger.info("Base de datos inicializada correctamente")
    
    async def save_alert(self, alert: EmergencyAlert) -> str:
        """Guarda una alerta de emergencia en la base de datos"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO emergency_alerts (
                    alert_id, device_id, user_id, accident_type, timestamp,
                    confidence, acceleration_magnitude, gyroscope_magnitude,
                    location_data, medical_info, emergency_contacts,
                    status, created_at, updated_at, additional_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.device_id,
                alert.user_id,
                alert.accident_type.value,
                alert.timestamp.isoformat(),
                alert.confidence,
                alert.acceleration_magnitude,
                alert.gyroscope_magnitude,
                json.dumps(alert.location_data) if alert.location_data else None,
                json.dumps(alert.medical_info) if alert.medical_info else None,
                json.dumps(alert.emergency_contacts),
                alert.status.value,
                alert.created_at.isoformat(),
                alert.updated_at.isoformat(),
                json.dumps(alert.additional_data) if alert.additional_data else None
            ))
            await db.commit()
            logger.info(f"Alerta guardada: {alert.alert_id}")
            return alert.alert_id
    
    async def get_alert(self, alert_id: str) -> Optional[EmergencyAlert]:
        """Obtiene una alerta por su ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM emergency_alerts WHERE alert_id = ?", 
                (alert_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_alert(row)
                return None
    
    async def get_alerts(
        self, 
        limit: int = 50, 
        offset: int = 0,
        device_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[EmergencyAlert]:
        """Obtiene lista de alertas con filtros opcionales"""
        query = "SELECT * FROM emergency_alerts WHERE 1=1"
        params = []
        
        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_alert(row) for row in rows]
    
    async def update_alert_status(self, alert_id: str, status: AlertStatus):
        """Actualiza el estado de una alerta"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE emergency_alerts 
                SET status = ?, updated_at = ? 
                WHERE alert_id = ?
            """, (status.value, datetime.now().isoformat(), alert_id))
            await db.commit()
            logger.info(f"Estado de alerta {alert_id} actualizado a {status.value}")
    
    async def save_device_info(self, device_info: DeviceInfo):
        """Guarda o actualiza información del dispositivo"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO devices (
                    device_id, user_id, device_model, os_version, 
                    app_version, last_seen, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                device_info.device_id,
                device_info.user_id,
                device_info.device_model,
                device_info.os_version,
                device_info.app_version,
                device_info.last_seen.isoformat(),
                device_info.is_active
            ))
            await db.commit()
    
    async def log_notification(self, notification_log: NotificationLog):
        """Registra un log de notificación"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO notification_logs (
                    log_id, alert_id, notification_type, recipient,
                    status, timestamp, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                notification_log.log_id,
                notification_log.alert_id,
                notification_log.notification_type,
                notification_log.recipient,
                notification_log.status,
                notification_log.timestamp.isoformat(),
                notification_log.error_message
            ))
            await db.commit()

    async def get_contacts(self, device_id: str) -> List[Dict[str, Any]]:
        """Obtiene los contactos de emergencia asociados a un dispositivo"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT name, phone, relationship, is_primary, updated_at FROM device_contacts WHERE device_id = ? ORDER BY is_primary DESC, name ASC",
                (device_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                contacts = []
                for row in rows:
                    contacts.append({
                        "name": row["name"],
                        "phone": row["phone"],
                        "relationship": row["relationship"],
                        "is_primary": bool(row["is_primary"]),
                        "updated_at": row["updated_at"],
                    })
                return contacts

    async def replace_contacts(self, device_id: str, contacts: List[Dict[str, Any]]) -> int:
        """Reemplaza los contactos de emergencia para un dispositivo"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM device_contacts WHERE device_id = ?", (device_id,))
            count = 0
            now = datetime.now().isoformat()
            for c in contacts:
                await db.execute(
                    """
                    INSERT INTO device_contacts (device_id, name, phone, relationship, is_primary, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        device_id,
                        c.get("name"),
                        c.get("phone"),
                        c.get("relationship"),
                        1 if c.get("is_primary") else 0,
                        now,
                    )
                )
                count += 1
            await db.commit()
            logger.info(f"Contactos reemplazados para {device_id}: {count}")
            return count

    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de alertas"""
        async with aiosqlite.connect(self.db_path) as db:
            # Total de alertas
            async with db.execute("SELECT COUNT(*) FROM emergency_alerts") as cursor:
                total_alerts = (await cursor.fetchone())[0]
            
            # Alertas por tipo
            alerts_by_type = {}
            async with db.execute("""
                SELECT accident_type, COUNT(*) 
                FROM emergency_alerts 
                GROUP BY accident_type
            """) as cursor:
                async for row in cursor:
                    alerts_by_type[row[0]] = row[1]
            
            # Alertas por estado
            alerts_by_status = {}
            async with db.execute("""
                SELECT status, COUNT(*) 
                FROM emergency_alerts 
                GROUP BY status
            """) as cursor:
                async for row in cursor:
                    alerts_by_status[row[0]] = row[1]
            
            # Alertas de hoy
            async with db.execute("""
                SELECT COUNT(*) FROM emergency_alerts 
                WHERE DATE(timestamp) = DATE('now')
            """) as cursor:
                alerts_today = (await cursor.fetchone())[0]
            
            # Confianza promedio
            async with db.execute("""
                SELECT AVG(confidence) FROM emergency_alerts
            """) as cursor:
                avg_confidence = (await cursor.fetchone())[0] or 0.0
            
            return {
                "total_alerts": total_alerts,
                "alerts_by_type": alerts_by_type,
                "alerts_by_status": alerts_by_status,
                "alerts_today": alerts_today,
                "average_confidence": round(avg_confidence, 2)
            }
    
    async def health_check(self) -> bool:
        """Verifica la salud de la base de datos"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def close(self):
        """Cierra la conexión a la base de datos"""
        # aiosqlite maneja las conexiones automáticamente
        logger.info("Conexiones de base de datos cerradas")

    # ==========================
    # CRUD de sensor_events
    # ==========================

    async def save_sensor_event(self, event: SensorEvent) -> str:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO sensor_events (
                    event_id, device_id, label, timestamp,
                    acceleration_magnitude, gyroscope_magnitude,
                    accel_variance, gyro_variance, accel_jerk,
                    predicted_label, prediction_confidence,
                    raw_data, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.device_id,
                    event.label.value,
                    event.timestamp.isoformat(),
                    event.acceleration_magnitude,
                    event.gyroscope_magnitude,
                    event.accel_variance,
                    event.gyro_variance,
                    event.accel_jerk,
                    event.predicted_label.value if event.predicted_label else None,
                    event.prediction_confidence,
                    json.dumps(event.raw_data) if event.raw_data else None,
                    event.created_at.isoformat(),
                ),
            )
            await db.commit()
            logger.info(f"SensorEvent guardado: {event.event_id}")
            return event.event_id

    async def get_sensor_events(
        self,
        label: Optional[str] = None,
        device_id: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM sensor_events WHERE 1=1"
        params: List[Any] = []

        if label:
            query += " AND label = ?"
            params.append(label.upper())
        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                results: List[Dict[str, Any]] = []
                for row in rows:
                    results.append({
                        "event_id": row["event_id"],
                        "device_id": row["device_id"],
                        "label": row["label"],
                        "timestamp": row["timestamp"],
                        "acceleration_magnitude": row["acceleration_magnitude"],
                        "gyroscope_magnitude": row["gyroscope_magnitude"],
                        "accel_variance": row["accel_variance"],
                        "gyro_variance": row["gyro_variance"],
                        "accel_jerk": row["accel_jerk"],
                        "predicted_label": row["predicted_label"],
                        "prediction_confidence": row["prediction_confidence"],
                        "raw_data": json.loads(row["raw_data"]) if row["raw_data"] else None,
                        "created_at": row["created_at"],
                    })
                return results

    async def get_sensor_event_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de eventos para métricas del dashboard"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Totales
            async with db.execute("SELECT COUNT(*) as total FROM sensor_events") as c:
                total = (await c.fetchone())["total"]

            # Por etiqueta
            async with db.execute("SELECT label, COUNT(*) as cnt FROM sensor_events GROUP BY label") as c:
                by_label_rows = await c.fetchall()
                by_label = {row["label"]: row["cnt"] for row in by_label_rows}

            # Últimas 24h y series por hora
            async with db.execute("SELECT COUNT(*) as cnt FROM sensor_events WHERE timestamp >= datetime('now','-1 day')") as c:
                last_24h = (await c.fetchone())["cnt"]

            async with db.execute(
                """
                SELECT strftime('%Y-%m-%d %H:00:00', timestamp) as hour, COUNT(*) as cnt
                FROM sensor_events
                WHERE timestamp >= datetime('now','-1 day')
                GROUP BY hour
                ORDER BY hour
                """
            ) as c:
                per_hour_rows = await c.fetchall()
                per_hour = [{"hour": row["hour"], "count": row["cnt"]} for row in per_hour_rows]

            # Dispositivos únicos
            async with db.execute("SELECT COUNT(DISTINCT device_id) as devices FROM sensor_events") as c:
                devices = (await c.fetchone())["devices"]

            return {
                "total_events": total,
                "by_label": by_label,
                "last_24h": last_24h,
                "per_hour": per_hour,
                "devices": devices,
            }

    async def get_sensor_event_confusion(self) -> List[Dict[str, Any]]:
        """Devuelve matriz de confusión agregada label vs predicted_label"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT label, predicted_label, COUNT(*) as cnt
                FROM sensor_events
                WHERE predicted_label IS NOT NULL
                GROUP BY label, predicted_label
                """
            ) as c:
                rows = await c.fetchall()
                return [{
                    "label": row["label"],
                    "predicted_label": row["predicted_label"],
                    "count": row["cnt"],
                } for row in rows]

    # ==========================
    # Snapshots de métricas de modelo
    # ==========================
    async def save_model_metrics_snapshot(self, metrics: Dict[str, Any], model_version: Optional[str] = None) -> int:
        """Guarda un snapshot de métricas del modelo y devuelve su ID"""
        created_at = datetime.now().isoformat()
        overall = metrics.get("overall", {})
        classes_json = json.dumps(metrics.get("classes", []))
        confusion_json = json.dumps(metrics.get("confusion_matrix", {}))

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO model_metrics_snapshots (
                    created_at, model_version, precision, recall, f1,
                    classes_json, confusion_matrix_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    model_version,
                    overall.get("precision", 0.0),
                    overall.get("recall", 0.0),
                    overall.get("f1", 0.0),
                    classes_json,
                    confusion_json,
                )
            )
            await db.commit()
            async with db.execute("SELECT last_insert_rowid()") as c:
                row = await c.fetchone()
                snapshot_id = row[0]
                logger.info(f"Snapshot de métricas guardado: {snapshot_id}")
                return snapshot_id

    async def get_model_metrics_history(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 50,
        model_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Obtiene historial de snapshots de métricas del modelo"""
        query = "SELECT id, created_at, model_version, precision, recall, f1, classes_json, confusion_matrix_json FROM model_metrics_snapshots WHERE 1=1"
        params: List[Any] = []
        if start:
            query += " AND created_at >= ?"
            params.append(start)
        if end:
            query += " AND created_at <= ?"
            params.append(end)
        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as c:
                rows = await c.fetchall()
                history: List[Dict[str, Any]] = []
                for row in rows:
                    history.append({
                        "id": row["id"],
                        "created_at": row["created_at"],
                        "model_version": row["model_version"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                        "classes": json.loads(row["classes_json"]) if row["classes_json"] else [],
                        "confusion_matrix": json.loads(row["confusion_matrix_json"]) if row["confusion_matrix_json"] else {},
                    })
                return history
    
    async def get_alerts(self, status: str = None, accident_type: str = None, limit: int = 50, offset: int = 0):
        """Obtiene lista de alertas con filtros opcionales"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = "SELECT * FROM emergency_alerts WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status.upper())
            
            if accident_type:
                query += " AND accident_type = ?"
                params.append(accident_type.upper())
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
    
    async def get_alert_by_id(self, alert_id: str):
        """Obtiene una alerta específica por ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM emergency_alerts WHERE alert_id = ?", 
                (alert_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_dict(row)
                return None
    
    async def get_statistics(self):
        """Obtiene estadísticas del sistema"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Total de alertas
            async with db.execute("SELECT COUNT(*) as total FROM emergency_alerts") as cursor:
                total_alerts = (await cursor.fetchone())['total']
            
            # Alertas por estado
            async with db.execute("""
                SELECT status, COUNT(*) as count 
                FROM emergency_alerts 
                GROUP BY status
            """) as cursor:
                status_counts = {row['status']: row['count'] for row in await cursor.fetchall()}
            
            # Alertas por tipo de accidente
            async with db.execute("""
                SELECT accident_type, COUNT(*) as count 
                FROM emergency_alerts 
                GROUP BY accident_type
            """) as cursor:
                accident_type_counts = {row['accident_type']: row['count'] for row in await cursor.fetchall()}
            
            # Alertas de las últimas 24 horas
            async with db.execute("""
                SELECT COUNT(*) as count 
                FROM emergency_alerts 
                WHERE created_at >= datetime('now', '-1 day')
            """) as cursor:
                last_24h = (await cursor.fetchone())['count']
            
            # Alertas activas (no completadas/canceladas)
            async with db.execute("""
                SELECT COUNT(*) as count 
                FROM emergency_alerts 
                WHERE status NOT IN ('COMPLETED', 'CANCELLED', 'FAILED')
            """) as cursor:
                active_alerts = (await cursor.fetchone())['count']
            
            return {
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "last_24h": last_24h,
                "status_distribution": status_counts,
                "accident_type_distribution": accident_type_counts
            }
    
    async def update_alert_status(self, alert_id: str, new_status):
        """Actualiza el estado de una alerta"""
        # Si new_status es un enum AlertStatus, usar su valor de forma explícita
        if isinstance(new_status, AlertStatus):
            status_value = new_status.value
        else:
            # Si es un string u otro tipo, convertir a string y mayúsculas
            status_value = str(new_status).upper()
            
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE emergency_alerts SET status = ?, updated_at = ? WHERE alert_id = ?",
                (status_value, datetime.now().isoformat(), alert_id)
            )
            await db.commit()
            return await self.get_alert_by_id(alert_id)
    
    def _row_to_dict(self, row):
        """Convierte una fila de la base de datos a un diccionario"""
        return {
            "alert_id": row['alert_id'],
            "device_id": row['device_id'],
            "user_id": row['user_id'],
            "accident_type": row['accident_type'],
            "timestamp": row['timestamp'],
            "confidence": row['confidence'],
            "acceleration_magnitude": row['acceleration_magnitude'],
            "gyroscope_magnitude": row['gyroscope_magnitude'],
            "location_data": json.loads(row['location_data']) if row['location_data'] else None,
            "medical_info": json.loads(row['medical_info']) if row['medical_info'] else None,
            "emergency_contacts": json.loads(row['emergency_contacts']) if row['emergency_contacts'] else [],
            "status": row['status'],
            "created_at": row['created_at'],
            "updated_at": row['updated_at'],
            "additional_data": json.loads(row['additional_data']) if row['additional_data'] else None
        }
    
    def _row_to_alert(self, row) -> EmergencyAlert:
        """Convierte una fila de la base de datos a un objeto EmergencyAlert"""
        return EmergencyAlert(
            alert_id=row['alert_id'],
            device_id=row['device_id'],
            user_id=row['user_id'],
            accident_type=AccidentType(row['accident_type']),
            timestamp=datetime.fromisoformat(row['timestamp']),
            confidence=row['confidence'],
            acceleration_magnitude=row['acceleration_magnitude'],
            gyroscope_magnitude=row['gyroscope_magnitude'],
            location_data=json.loads(row['location_data']) if row['location_data'] else None,
            medical_info=json.loads(row['medical_info']) if row['medical_info'] else None,
            emergency_contacts=json.loads(row['emergency_contacts']) if row['emergency_contacts'] else [],
            status=AlertStatus(row['status']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            additional_data=json.loads(row['additional_data']) if row['additional_data'] else None
        )