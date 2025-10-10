import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4
import os

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi import Request
from jose import jwt, JWTError
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager, suppress

from models import *
from database import Database
from websocket_manager import WebSocketManager, heartbeat_task
from ml import RFAccidentClassifier

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de tareas periódicas
METRICS_SNAPSHOT_INTERVAL = int(os.getenv("METRICS_SNAPSHOT_INTERVAL", "3600"))  # 1 hora por defecto
DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", None)
ML_MODEL_DIR = os.getenv("ML_MODEL_DIR", "models")
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", os.path.join(ML_MODEL_DIR, "random_forest.pkl"))
ML_META_PATH = os.getenv("ML_META_PATH", os.path.join(ML_MODEL_DIR, "random_forest_meta.json"))
ACCIDENT_CONFIDENCE_THRESHOLD = float(os.getenv("ACCIDENT_CONFIDENCE_THRESHOLD", "0.7"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializar la base de datos y tareas de background
    await db.init_db()
    hb_task = asyncio.create_task(heartbeat_task())
    # Programar snapshots automáticos de métricas
    metrics_task = asyncio.create_task(model_metrics_snapshot_task(METRICS_SNAPSHOT_INTERVAL, DEFAULT_MODEL_VERSION))
    # Intentar cargar modelo ML desde disco
    try:
        if os.path.exists(ML_MODEL_PATH):
            rf_classifier.load(ML_MODEL_PATH, ML_META_PATH)
            logger.info("Modelo RandomForest cargado correctamente")
        else:
            logger.info("No se encontró modelo RandomForest en disco, se entrenará cuando se solicite")
        # Aplicar threshold desde entorno si procede
        rf_classifier.configure(confidence_threshold=ACCIDENT_CONFIDENCE_THRESHOLD)
    except Exception as e:
        logger.error(f"Error cargando modelo ML: {e}")
    try:
        yield
    finally:
        await db.close()
        hb_task.cancel()
        metrics_task.cancel()
        # Suprimir CancelledError al cerrar la tarea
        with suppress(asyncio.CancelledError):
            await hb_task
        with suppress(asyncio.CancelledError):
            await metrics_task

app = FastAPI(
    title="AlertaRaven API",
    description="API para recibir y gestionar alertas de emergencia de la aplicación AlertaRaven",
    version="1.0.0",
    lifespan=lifespan,
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes
db = Database()
websocket_manager = WebSocketManager()
rf_classifier = RFAccidentClassifier()

# Configuración de autenticación
security = HTTPBearer()
VALID_API_KEYS = {"alertaraven_mobile_key_2024"}  # En producción usar variables de entorno

def get_api_key(api_key: str):
    return api_key in VALID_API_KEYS

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="API key inválida")
    return credentials.credentials


# ================================
# Tarea periódica: snapshots de métricas
# ================================
async def model_metrics_snapshot_task(interval_seconds: int = 3600, model_version: Optional[str] = None):
    """Toma snapshots automáticos de métricas del modelo cada `interval_seconds`"""
    while True:
        try:
            # Obtener datos de confusión actuales
            confusion_rows = await db.get_sensor_event_confusion()
            if not confusion_rows:
                logger.info("No hay datos de predicción para snapshot de métricas; esperando próximo ciclo")
                await asyncio.sleep(interval_seconds)
                continue

            labels = set(r["label"] for r in confusion_rows)
            predicted = set(r["predicted_label"] for r in confusion_rows)
            classes = sorted(list(labels.union(predicted)))

            matrix = {c: {p: 0 for p in classes} for c in classes}
            for r in confusion_rows:
                matrix[r["label"]][r["predicted_label"]] = r["count"]

            total_tp = 0
            total_fp = 0
            total_fn = 0
            for c in classes:
                tp = matrix.get(c, {}).get(c, 0)
                fp = sum(matrix[l][c] for l in classes if l != c)
                fn = sum(matrix[c][p] for p in classes if p != c)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

            metrics_payload = {
                "available": True,
                "classes": classes,
                "confusion_matrix": matrix,
                "overall": {
                    "precision": round(micro_precision, 4),
                    "recall": round(micro_recall, 4),
                    "f1": round(micro_f1, 4),
                }
            }

            snapshot_id = await db.save_model_metrics_snapshot(metrics_payload, model_version)
            logger.info(f"Snapshot automático de métricas creado (ID={snapshot_id})")
        except Exception as e:
            logger.error(f"Error en tarea de snapshot de métricas: {e}")
        finally:
            await asyncio.sleep(interval_seconds)

# =============================
# Autenticación para la web (dashboard)
# =============================
WEB_JWT_SECRET = os.getenv("ALERTARAVEN_WEB_SECRET", "dev_secret")
WEB_JWT_ALGORITHM = "HS256"
ADMIN_USER = os.getenv("ALERTARAVEN_ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ALERTARAVEN_ADMIN_PASS", "admin")

def create_web_token(username: str) -> str:
    payload = {
        "sub": username,
        "role": "admin",
        "iat": int(datetime.now().timestamp()),
    }
    return jwt.encode(payload, WEB_JWT_SECRET, algorithm=WEB_JWT_ALGORITHM)

def verify_web_auth(request: Request) -> str:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="No autenticado")
    try:
        payload = jwt.decode(token, WEB_JWT_SECRET, algorithms=[WEB_JWT_ALGORITHM])
        return payload.get("sub", "")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")

# Modelos de datos para la API
class LocationData(BaseModel):
    latitude: float = Field(..., description="Latitud de la ubicación")
    longitude: float = Field(..., description="Longitud de la ubicación")
    accuracy: Optional[float] = Field(None, description="Precisión en metros")
    altitude: Optional[float] = Field(None, description="Altitud")
    speed: Optional[float] = Field(None, description="Velocidad")
    timestamp: Optional[str] = Field(None, description="Timestamp de la ubicación")

class MedicalInfo(BaseModel):
    blood_type: Optional[str] = Field(None, description="Tipo de sangre")
    allergies: Optional[List[str]] = Field(None, description="Lista de alergias")
    medications: Optional[List[str]] = Field(None, description="Lista de medicamentos")
    medical_conditions: Optional[List[str]] = Field(None, description="Condiciones médicas")
    emergency_medical_info: Optional[str] = Field(None, description="Información médica adicional")

class EmergencyContact(BaseModel):
    name: str = Field(..., description="Nombre del contacto")
    phone: str = Field(..., description="Número de teléfono")
    relationship: Optional[str] = Field(None, description="Relación con el usuario")
    is_primary: bool = Field(False, description="Si es contacto primario")

class DeviceContactsResponse(BaseModel):
    device_id: str
    contacts: List[EmergencyContact]

class AccidentEventData(BaseModel):
    accident_type: str = Field(..., description="Tipo de accidente")
    timestamp: str = Field(..., description="Timestamp del evento como string")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Nivel de confianza (0.0-1.0)")
    acceleration_magnitude: float = Field(..., description="Magnitud de aceleración")
    gyroscope_magnitude: float = Field(..., description="Magnitud del giroscopio")
    location_data: Optional[LocationData] = Field(None, description="Datos de ubicación")
    additional_sensor_data: Optional[Dict[str, Any]] = Field(None, description="Datos adicionales de sensores")

class EmergencyAlertRequest(BaseModel):
    device_id: str = Field(..., description="ID único del dispositivo")
    user_id: Optional[str] = Field(None, description="ID del usuario")
    accident_event: AccidentEventData = Field(..., description="Datos del evento de accidente")
    medical_info: Optional[MedicalInfo] = Field(None, description="Información médica del usuario")
    emergency_contacts: List[EmergencyContact] = Field(default_factory=list, description="Contactos de emergencia")
    api_key: str = Field(..., description="Clave de API para autenticación")

class AlertResponse(BaseModel):
    alert_id: str = Field(..., description="ID único de la alerta")
    status: str = Field(..., description="Estado de la alerta")
    message: str = Field(..., description="Mensaje de respuesta")
    timestamp: datetime = Field(..., description="Timestamp de procesamiento")

# ================================
# Modelos para eventos de sensores
# ================================

class SensorEventIn(BaseModel):
    device_id: str
    # Si no se envía label, se clasifica con heurística
    label: Optional[SensorEventType] = None
    predicted_label: Optional[SensorEventType] = None
    prediction_confidence: Optional[float] = None
    acceleration_magnitude: float
    gyroscope_magnitude: float
    accel_variance: Optional[float] = None
    gyro_variance: Optional[float] = None
    accel_jerk: Optional[float] = None
    timestamp: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

# Eventos de ciclo de vida migrados a `lifespan` arriba

@app.get("/")
async def root():
    """Endpoint raíz de la API"""
    return {
        "message": "AlertaRaven API - Sistema de Alertas de Emergencia",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    """Verificación de salud de la API"""
    try:
        # Verificar conexión a la base de datos
        db_healthy = await db.health_check()
        ws_stats = await websocket_manager.get_connection_stats()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "database": "connected" if db_healthy else "disconnected",
            "timestamp": datetime.now(),
            "websocket_connections": ws_stats["total_connections"],
            "services": {
                "database": db_healthy,
                "websockets": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/api/v1/emergency-alert-debug", response_model=AlertResponse)
async def receive_emergency_alert_debug(alert_request: EmergencyAlertRequest):
    """
    Endpoint de depuración: acepta el mismo payload que la app móvil
    y devuelve un AlertResponse completo sin persistir datos.
    """
    logger.info(f"=== DEBUG ENDPOINT ===")
    logger.info(f"Datos recibidos (debug): {alert_request.dict()}")
    return AlertResponse(
        alert_id=str(uuid4()),
        status="received",
        message="Datos registrados para debugging",
        timestamp=datetime.now()
    )

@app.post("/api/v1/emergency-alert", response_model=AlertResponse)
async def receive_emergency_alert(
    alert_request: EmergencyAlertRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Recibe una alerta de emergencia desde la aplicación móvil
    """
    try:
        logger.info(f"Recibida alerta de emergencia del dispositivo: {alert_request.device_id}")
        logger.info(f"Datos completos recibidos: {alert_request.dict()}")
        
        # Crear objeto de alerta
        try:
            # Convertir el tipo de accidente a mayúsculas para que coincida con el enum
            accident_type_str = alert_request.accident_event.accident_type.upper()
            logger.info(f"Convirtiendo tipo de accidente: '{alert_request.accident_event.accident_type}' -> '{accident_type_str}'")
            accident_type_enum = AccidentType(accident_type_str)
            logger.info(f"Enum creado exitosamente: {accident_type_enum}")
        except ValueError as e:
            logger.warning(f"Tipo de accidente desconocido: {alert_request.accident_event.accident_type}, error: {e}")
            accident_type_enum = AccidentType.UNKNOWN
        
        # Parsear el timestamp (viene como string ISO)
        try:
            # Intentar parsear como ISO format
            timestamp = datetime.fromisoformat(alert_request.accident_event.timestamp.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Fallback: intentar como timestamp en milisegundos
                timestamp = datetime.fromtimestamp(float(alert_request.accident_event.timestamp) / 1000)
            except (ValueError, TypeError):
                logger.warning(f"No se pudo parsear timestamp: {alert_request.accident_event.timestamp}")
                timestamp = datetime.now()
            
        alert = EmergencyAlert(
            alert_id=str(uuid4()),
            device_id=alert_request.device_id,
            user_id=alert_request.user_id,
            accident_type=accident_type_enum,
            timestamp=timestamp,
            location_data=alert_request.accident_event.location_data.dict() if alert_request.accident_event.location_data else None,
            medical_info=alert_request.medical_info.dict() if alert_request.medical_info else None,
            emergency_contacts=[contact.dict() for contact in alert_request.emergency_contacts],
            confidence=alert_request.accident_event.confidence,
            acceleration_magnitude=alert_request.accident_event.acceleration_magnitude,
            gyroscope_magnitude=alert_request.accident_event.gyroscope_magnitude,
            status=AlertStatus.RECEIVED,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Guardar en base de datos
        alert_id = await db.save_alert(alert)
        
        # Procesar alerta en background
        background_tasks.add_task(process_emergency_alert, alert_id)
        
        # Notificar via WebSocket a clientes conectados
        await websocket_manager.notify_new_alert({
            "alert_id": alert_id,
            "device_id": alert_request.device_id,
            "accident_type": alert_request.accident_event.accident_type,
            "confidence": alert_request.accident_event.confidence,
            "timestamp": alert.timestamp.isoformat(),
            "location": alert_request.accident_event.location_data.dict() if alert_request.accident_event.location_data else None
        })
        
        logger.info(f"Alerta procesada exitosamente. ID: {alert_id}")
        
        return AlertResponse(
            alert_id=alert_id,
            status="received",
            message="Alerta de emergencia recibida y procesada correctamente",
            timestamp=datetime.now()
        )
        
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=f"Datos inválidos: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Error procesando alerta de emergencia: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/api/v1/alerts/{alert_id}")
async def get_alert_status(alert_id: str):
    """
    Obtiene el estado de una alerta específica
    """
    try:
        alert = await db.get_alert(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alerta no encontrada")
        
        return {
            "alert_id": alert.alert_id,
            "status": alert.status.value if hasattr(alert.status, "value") else str(alert.status),
            "device_id": alert.device_id,
            "created_at": alert.created_at,
            "updated_at": alert.updated_at,
            "accident_type": alert.accident_type.value if hasattr(alert.accident_type, "value") else str(alert.accident_type),
            "confidence": alert.confidence,
            "location_data": alert.location_data,
            "emergency_contacts_count": len(alert.emergency_contacts or [])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo estado de alerta {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/api/v1/alerts")
async def get_alerts(
    limit: int = 50,
    offset: int = 0,
    device_id: Optional[str] = None,
    status: Optional[str] = None
):
    """
    Obtiene lista de alertas con filtros opcionales
    """
    try:
        alerts = await db.get_alerts(
            limit=limit,
            offset=offset,
            device_id=device_id,
            status=status
        )
        stats = await db.get_alert_statistics()
        
        return {
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "status": a.status.value if hasattr(a.status, "value") else str(a.status),
                    "device_id": a.device_id,
                    "created_at": a.created_at,
                    "updated_at": a.updated_at,
                    "accident_type": a.accident_type.value if hasattr(a.accident_type, "value") else str(a.accident_type),
                    "confidence": a.confidence,
                    "location_data": a.location_data,
                    "emergency_contacts_count": len(a.emergency_contacts or [])
                }
                for a in alerts
            ],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": len(alerts)
            },
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error obteniendo alertas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

async def process_emergency_alert(alert_id: str):
    """
    Procesa una alerta de emergencia en background
    """
    try:
        alert = await db.get_alert(alert_id)
        if not alert:
            logger.error(f"Alerta {alert_id} no encontrada para procesamiento")
            return
        
        logger.info(f"Procesando alerta {alert_id}")
        
        # Simular procesamiento inicial
        await asyncio.sleep(1)
        
        # Actualizar estado a procesando
        await db.update_alert_status(alert_id, AlertStatus.PROCESSING)
        await websocket_manager.notify_alert_status_change(
            alert_id, "received", "processing", alert.device_id
        )
        
        # Simular validación y procesamiento de datos
        await asyncio.sleep(2)
        
        # Determinar si la alerta es válida basado en la confianza
        if alert.confidence >= 0.7:
            # Alta confianza - marcar como confirmada
            await db.update_alert_status(alert_id, AlertStatus.CONFIRMED)
            await websocket_manager.notify_alert_status_change(
                alert_id, "processing", "confirmed", alert.device_id
            )
            
            # Simular envío de notificaciones a contactos de emergencia
            await asyncio.sleep(1)
            
            await db.update_alert_status(alert_id, AlertStatus.COMPLETED)
            await websocket_manager.notify_alert_status_change(
                alert_id, "confirmed", "completed", alert.device_id
            )
        else:
            # Baja confianza - marcar como pendiente de revisión
            await db.update_alert_status(alert_id, AlertStatus.PENDING_REVIEW)
            await websocket_manager.notify_alert_status_change(
                alert_id, "processing", "pending_review", alert.device_id
            )
        
        logger.info(f"Alerta {alert_id} procesada exitosamente")
        
    except Exception as e:
        logger.error(f"Error procesando alerta {alert_id}: {e}")
        await db.update_alert_status(alert_id, AlertStatus.FAILED)
        await websocket_manager.notify_alert_status_change(
            alert_id, "processing", "failed", None
        )

# ================================
# Endpoints de eventos de sensores (entrenamiento)
# ================================

def classify_event_heuristic(acc_mag: float, gyro_mag: float, accel_var: Optional[float] = None, gyro_var: Optional[float] = None) -> SensorEventType:
    """Clasificador simple por umbrales para etiquetar si no se proporciona label"""
    # Umbrales orientativos (ajustables)
    if acc_mag > 25 and gyro_mag < 5:
        return SensorEventType.PHONE_DROP
    if acc_mag > 18 and gyro_mag > 12:
        return SensorEventType.VEHICLE_ACCIDENT
    return SensorEventType.NORMAL

@app.post("/api/v1/sensor-events", dependencies=[Depends(verify_api_key)])
async def ingest_sensor_event(event_in: SensorEventIn):
    """Recibe un evento agregado de sensores (o ventana) para entrenamiento"""
    try:
        label = event_in.label or classify_event_heuristic(
            event_in.acceleration_magnitude,
            event_in.gyroscope_magnitude,
            event_in.accel_variance,
            event_in.gyro_variance,
        )
        # Preparar datos para posible predicción del modelo
        row_data = {
            "acceleration_magnitude": event_in.acceleration_magnitude,
            "gyroscope_magnitude": event_in.gyroscope_magnitude,
            "accel_variance": event_in.accel_variance,
            "gyro_variance": event_in.gyro_variance,
            "accel_jerk": event_in.accel_jerk,
        }
        pred_label = event_in.predicted_label
        pred_conf = event_in.prediction_confidence
        # Si hay modelo entrenado y no se proporcionó predicted_label, inferirlo
        if pred_label is None and rf_classifier.is_ready():
            try:
                plabel_str, conf = rf_classifier.predict(row_data)
                # Mapear a enum si posible
                try:
                    pred_label = SensorEventType[plabel_str]
                except Exception:
                    pred_label = SensorEventType.OTHER
                pred_conf = conf
            except Exception as e:
                logger.error(f"Error en predicción ML: {e}")

        event = SensorEvent(
            device_id=event_in.device_id,
            label=label,
            timestamp=datetime.fromisoformat(event_in.timestamp) if event_in.timestamp else datetime.now(),
            acceleration_magnitude=event_in.acceleration_magnitude,
            gyroscope_magnitude=event_in.gyroscope_magnitude,
            accel_variance=event_in.accel_variance,
            gyro_variance=event_in.gyro_variance,
            accel_jerk=event_in.accel_jerk,
            predicted_label=pred_label,
            prediction_confidence=pred_conf,
            raw_data=event_in.raw_data,
        )
        event_id = await db.save_sensor_event(event)
        return {"ok": True, "event_id": event_id, "label": event.label.value}
    except Exception as e:
        logger.error(f"Error guardando sensor event: {e}")
        raise HTTPException(status_code=500, detail="Error guardando evento de sensores")

@app.get("/api/v1/sensor-events", dependencies=[Depends(verify_api_key)])
async def list_sensor_events(
    label: Optional[str] = Query(None),
    device_id: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    limit: int = Query(500),
    offset: int = Query(0)
):
    try:
        rows = await db.get_sensor_events(label=label, device_id=device_id, start=start, end=end, limit=limit, offset=offset)
        items = [
            {
                "event_id": r.get("event_id"),
                "ok": True,
                "message": r.get("label")
            }
            for r in rows
        ]
        return {"events": items, "pagination": {"limit": limit, "offset": offset, "total": len(rows)}}
    except Exception as e:
        logger.error(f"Error listando sensor events: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo eventos de sensores")

@app.get("/api/v1/sensor-events/export", dependencies=[Depends(verify_api_key)])
async def export_sensor_events_csv(
    label: Optional[str] = Query(None), device_id: Optional[str] = Query(None)
):
    """Exporta CSV con columnas útiles para entrenamiento"""
    try:
        rows = await db.get_sensor_events(label=label, device_id=device_id, limit=100000, offset=0)
        # Construir CSV en memoria
        headers = [
            "event_id","device_id","label","timestamp",
            "acceleration_magnitude","gyroscope_magnitude",
            "accel_variance","gyro_variance","accel_jerk"
        ]
        import io, csv
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in headers})
        content = buf.getvalue()
        return HTMLResponse(content, media_type="text/csv")
    except Exception as e:
        logger.error(f"Error exportando sensor events: {e}")
        raise HTTPException(status_code=500, detail="Error exportando datos")

# ================================
# ENDPOINTS PARA APLICACIÓN WEB
# ================================

# Configurar archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# PWA: servir manifest y service worker desde la raíz para mayor alcance
@app.get("/manifest.webmanifest")
async def pwa_manifest():
    return FileResponse("static/manifest.webmanifest", media_type="application/manifest+json")

@app.get("/service-worker.js")
async def pwa_service_worker():
    return FileResponse("static/service-worker.js", media_type="application/javascript")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket para notificaciones en tiempo real"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Mantener la conexión activa
            await websocket.receive_text()
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)

# Rutas web del dashboard

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Página principal del dashboard (protegida)"""
    try:
        verify_web_auth(request)
        return FileResponse("static/index.html")
    except HTTPException:
        return RedirectResponse(url="/login")

@app.get("/components/{component_name}")
async def get_component(component_name: str):
    """Servir componentes individuales"""
    component_path = f"static/components/{component_name}.html"
    if os.path.exists(component_path):
        return FileResponse(component_path)
    else:
        raise HTTPException(status_code=404, detail="Componente no encontrado")

# Endpoints específicos para cada página (opcional)
@app.get("/alerts-page", response_class=HTMLResponse)
async def alerts_page():
    """Página de alertas"""
    return FileResponse("static/index.html")

# Páginas públicas
@app.get("/landing", response_class=HTMLResponse)
async def landing_page():
    return FileResponse("static/landing.html")

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return FileResponse("static/login.html")

# API de autenticación web
@app.post("/api/auth/login")
async def web_login(data: Dict[str, str]):
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if username == ADMIN_USER and password == ADMIN_PASS:
        token = create_web_token(username)
        response = JSONResponse({"ok": True, "message": "Login exitoso"})
        # Cookie HttpOnly
        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=60*60*8
        )
        return response
    return JSONResponse({"ok": False, "message": "Credenciales inválidas"}, status_code=401)

@app.get("/logout")
async def web_logout():
    response = RedirectResponse(url="/landing")
    response.delete_cookie("access_token")
    return response

@app.get("/map-page", response_class=HTMLResponse)
async def map_page():
    """Página del mapa"""
    return FileResponse("static/index.html")

@app.get("/statistics-page", response_class=HTMLResponse)
async def statistics_page():
    """Página de estadísticas"""
    return FileResponse("static/index.html")

@app.get("/system-page", response_class=HTMLResponse)
async def system_page():
    """Página del sistema"""
    return FileResponse("static/index.html")

    

@app.get("/api/alerts", response_model=List[Dict[str, Any]])
async def get_alerts(
    status: Optional[str] = Query(None, description="Filtrar por estado"),
    accident_type: Optional[str] = Query(None, description="Filtrar por tipo de accidente"),
    limit: int = Query(50, description="Número máximo de alertas"),
    offset: int = Query(0, description="Offset para paginación")
):
    """Obtener lista de alertas con filtros opcionales"""
    try:
        alerts = await db.get_alerts(status=status, accident_type=accident_type, limit=limit, offset=offset)
        return alerts
    except Exception as e:
        logger.error(f"Error obteniendo alertas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/api/alerts/{alert_id}")
async def get_alert_details(alert_id: str):
    """Obtener detalles completos de una alerta específica"""
    try:
        alert = await db.get_alert_by_id(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alerta no encontrada")
        return alert
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo alerta {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/api/statistics")
async def get_statistics():
    """Obtener estadísticas del sistema para el dashboard"""
    try:
        stats = await db.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# ================================
# Endpoints de métricas de modelo
# ================================

@app.get("/api/v1/metrics/sensor-events-summary")
async def sensor_events_summary():
    """Resumen de eventos de sensores para dashboard"""
    try:
        summary = await db.get_sensor_event_summary()
        return summary
    except Exception as e:
        logger.error(f"Error obteniendo resumen de sensor events: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo resumen")

@app.get("/api/v1/metrics/model")
async def model_metrics():
    """Métricas de modelo (precision, recall, F1 y matriz de confusión)"""
    try:
        confusion_rows = await db.get_sensor_event_confusion()
        # Si no hay predicciones, devolver bandera
        if not confusion_rows:
            summary = await db.get_sensor_event_summary()
            return {
                "available": False,
                "reason": "No hay datos de predicción (predicted_label) registrados",
                "summary": summary,
            }

        # Construir matriz y métricas por clase
        labels = set(r["label"] for r in confusion_rows)
        predicted = set(r["predicted_label"] for r in confusion_rows)
        classes = sorted(list(labels.union(predicted)))

        # Mapa (label -> predicted -> count)
        matrix = {c: {p: 0 for p in classes} for c in classes}
        for r in confusion_rows:
            matrix[r["label"]][r["predicted_label"]] = r["count"]

        per_class = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for c in classes:
            tp = matrix.get(c, {}).get(c, 0)
            fp = sum(matrix[l][c] for l in classes if l != c)
            fn = sum(matrix[c][p] for p in classes if p != c)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            per_class[c] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn,
            }

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

        return {
            "available": True,
            "classes": classes,
            "confusion_matrix": matrix,
            "per_class": per_class,
            "overall": {
                "precision": round(micro_precision, 4),
                "recall": round(micro_recall, 4),
                "f1": round(micro_f1, 4),
            }
        }
    except Exception as e:
        logger.error(f"Error obteniendo métricas de modelo: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo métricas de modelo")

@app.post("/api/v1/metrics/model/snapshot")
async def create_model_metrics_snapshot(model_version: Optional[str] = Query(None, description="Versión del modelo")):
    """Calcula métricas actuales y guarda un snapshot en la base de datos"""
    try:
        # Reutilizar la lógica de model_metrics para calcular métricas
        confusion_rows = await db.get_sensor_event_confusion()
        if not confusion_rows:
            raise HTTPException(status_code=400, detail="No hay datos de predicción (predicted_label) para generar snapshot")

        labels = set(r["label"] for r in confusion_rows)
        predicted = set(r["predicted_label"] for r in confusion_rows)
        classes = sorted(list(labels.union(predicted)))

        matrix = {c: {p: 0 for p in classes} for c in classes}
        for r in confusion_rows:
            matrix[r["label"]][r["predicted_label"]] = r["count"]

        total_tp = 0
        total_fp = 0
        total_fn = 0
        for c in classes:
            tp = matrix.get(c, {}).get(c, 0)
            fp = sum(matrix[l][c] for l in classes if l != c)
            fn = sum(matrix[c][p] for p in classes if p != c)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

        metrics_payload = {
            "available": True,
            "classes": classes,
            "confusion_matrix": matrix,
            "overall": {
                "precision": round(micro_precision, 4),
                "recall": round(micro_recall, 4),
                "f1": round(micro_f1, 4),
            }
        }

        snapshot_id = await db.save_model_metrics_snapshot(metrics_payload, model_version)
        return {"ok": True, "snapshot_id": snapshot_id, "metrics": metrics_payload}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creando snapshot de métricas: {e}")
        raise HTTPException(status_code=500, detail="Error creando snapshot de métricas")

# ================================
# Entrenamiento y predicción ML (RandomForest)
# ================================

class TrainParams(BaseModel):
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    test_size: float = 0.2
    random_state: int = 42
    use_cv: bool = False
    param_grid: Optional[Dict[str, List[Any]]] = None
    # Filtros de dataset
    start: Optional[str] = None
    end: Optional[str] = None
    device_id: Optional[str] = None
    labels_include: Optional[List[str]] = None
    min_per_class: int = 20

@app.post("/api/v1/model/train/randomforest", dependencies=[Depends(verify_api_key)])
async def train_random_forest(params: TrainParams):
    """Entrena un modelo RandomForest con datos de sensor_events y lo persiste."""
    try:
        # Obtener datos etiquetados de la BD
        rows = await db.get_sensor_events(
            label=None,
            device_id=params.device_id,
            start=params.start,
            end=params.end,
            limit=100000,
            offset=0
        )
        if not rows:
            raise HTTPException(status_code=400, detail="No hay eventos de sensores para entrenar")

        # Filtrar filas que tengan 'label'
        labeled_rows = [r for r in rows if r.get("label")]
        # Aplicar filtro por etiquetas si se especifica
        if params.labels_include:
            allow = set([str(l).upper() for l in params.labels_include])
            labeled_rows = [r for r in labeled_rows if str(r.get("label")).upper() in allow]
        if len(labeled_rows) < 5:
            raise HTTPException(status_code=400, detail="Dataset insuficiente para entrenar (menos de 5 ejemplos)")

        # Reporte de balance por clase
        from collections import Counter
        counts = Counter([str(r.get("label")) for r in labeled_rows])
        total = sum(counts.values())
        distribution = {k: {"count": v, "percent": round((v / total) * 100, 2)} for k, v in counts.items()}
        warnings = [f"Clase {k} con {v} muestras (< {params.min_per_class})" for k, v in counts.items() if v < params.min_per_class]

        if params.use_cv:
            metrics = rf_classifier.train_cv(
                labeled_rows,
                param_grid=params.param_grid,
                cv_splits=3,
                random_state=params.random_state,
            )
        else:
            metrics = rf_classifier.train(
                labeled_rows,
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                min_samples_leaf=params.min_samples_leaf,
                test_size=params.test_size,
                random_state=params.random_state,
            )

        # Guardar modelo en disco
        os.makedirs(ML_MODEL_DIR, exist_ok=True)
        rf_classifier.save(ML_MODEL_PATH, ML_META_PATH)

        return {
            "ok": True,
            "message": "Modelo entrenado y guardado",
            "model_path": ML_MODEL_PATH,
            "metrics": metrics,
            "classes": rf_classifier.classes_,
            "params": rf_classifier.params,
            "dataset": {
                "total": total,
                "by_label": distribution,
                "warnings": warnings,
                "filters": {
                    "start": params.start,
                    "end": params.end,
                    "device_id": params.device_id,
                    "labels_include": params.labels_include,
                    "min_per_class": params.min_per_class,
                },
                "labeled_rows": len(labeled_rows)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error entrenando modelo: {e}")
        raise HTTPException(status_code=500, detail="Error entrenando modelo")

class PredictIn(BaseModel):
    acceleration_magnitude: float
    gyroscope_magnitude: float
    accel_variance: Optional[float] = None
    gyro_variance: Optional[float] = None
    accel_jerk: Optional[float] = None

@app.post("/api/v1/model/predict")
async def model_predict(payload: PredictIn):
    """Predice etiqueta y confianza para características agregadas usando el modelo actual."""
    try:
        if not rf_classifier.is_ready():
            raise HTTPException(status_code=400, detail="Modelo no entrenado/cargado")
        row = {
            "acceleration_magnitude": payload.acceleration_magnitude,
            "gyroscope_magnitude": payload.gyroscope_magnitude,
            "accel_variance": payload.accel_variance,
            "gyro_variance": payload.gyro_variance,
            "accel_jerk": payload.accel_jerk,
        }
        label_str, conf = rf_classifier.predict(row)
        # Mapear a enum si existe
        display_label = label_str
        try:
            display_label = SensorEventType[label_str].value
        except Exception:
            pass
        return {
            "ok": True,
            "predicted_label": display_label,
            "confidence": round(conf, 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail="Error en predicción")

@app.get("/api/v1/model/status")
async def model_status():
    """Estado del modelo ML cargado en la API."""
    try:
        return {
            "ready": rf_classifier.is_ready(),
            "classes": rf_classifier.classes_,
            "params": rf_classifier.params,
            "model_path": ML_MODEL_PATH if os.path.exists(ML_MODEL_PATH) else None,
            "accident_labels": rf_classifier.accident_labels,
            "confidence_threshold": rf_classifier.confidence_threshold,
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado del modelo: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo estado del modelo")

@app.get("/api/v1/metrics/model/history")
async def get_model_metrics_history(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    limit: int = Query(50),
    model_version: Optional[str] = Query(None)
):
    """Obtiene historial de snapshots de métricas"""
    try:
        history = await db.get_model_metrics_history(start=start, end=end, limit=limit, model_version=model_version)
        return {"items": history}
    except Exception as e:
        logger.error(f"Error obteniendo historial de métricas: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo historial de métricas")

@app.put("/api/alerts/{alert_id}/status")
async def update_alert_status_endpoint(alert_id: str, request: dict):
    """Actualizar el estado de una alerta (para paramédicos/aseguradoras)"""
    try:
        new_status = request.get("status")
        if not new_status:
            raise HTTPException(status_code=400, detail="Status requerido")
            
        valid_statuses = ["PENDING", "IN_PROGRESS", "COMPLETED", "CANCELLED", "FAILED"]
        if new_status.upper() not in valid_statuses:
            raise HTTPException(status_code=400, detail="Estado inválido")
        
        updated_alert = await db.update_alert_status(alert_id, new_status.upper())
        if not updated_alert:
            raise HTTPException(status_code=404, detail="Alerta no encontrada")
        
        # Notificar cambio de estado via WebSocket
        await websocket_manager.notify_alert_status_change(alert_id, "manual", new_status.lower(), None)
        
        return {
            "alert_id": updated_alert.get("alert_id"),
            "status": updated_alert.get("status"),
            "device_id": updated_alert.get("device_id"),
            "created_at": updated_alert.get("created_at"),
            "updated_at": updated_alert.get("updated_at"),
            "accident_type": updated_alert.get("accident_type"),
            "confidence": updated_alert.get("confidence"),
            "location_data": updated_alert.get("location_data"),
            "emergency_contacts_count": len(updated_alert.get("emergency_contacts") or [])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error actualizando estado de alerta {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# ==========================
# Endpoints de contactos (sincronización móvil)
# ==========================

@app.get("/api/v1/contacts/{device_id}", response_model=DeviceContactsResponse)
async def get_device_contacts(
    device_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Obtiene los contactos de emergencia para un dispositivo"""
    try:
        contacts = await db.get_contacts(device_id)
        api_contacts = [EmergencyContact(**{
            "name": c.get("name"),
            "phone": c.get("phone"),
            "relationship": c.get("relationship"),
            "is_primary": c.get("is_primary", False)
        }) for c in contacts]
        return {"device_id": device_id, "contacts": api_contacts}
    except Exception as e:
        logger.error(f"Error obteniendo contactos para {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo contactos")

@app.put("/api/v1/contacts/{device_id}", response_model=DeviceContactsResponse)
async def replace_device_contacts(
    device_id: str,
    contacts: List[EmergencyContact],
    api_key: str = Depends(verify_api_key)
):
    """Reemplaza los contactos de emergencia para un dispositivo y devuelve los guardados"""
    try:
        unique = {}
        for c in contacts:
            unique[c.phone] = c
        normalized = [{
            "name": c.name,
            "phone": c.phone,
            "relationship": c.relationship,
            "is_primary": c.is_primary
        } for c in unique.values()]
        await db.replace_contacts(device_id, normalized)
        stored = await db.get_contacts(device_id)
        api_contacts = [EmergencyContact(**{
            "name": c.get("name"),
            "phone": c.get("phone"),
            "relationship": c.get("relationship"),
            "is_primary": c.get("is_primary", False)
        }) for c in stored]
        return {"device_id": device_id, "contacts": api_contacts}
    except Exception as e:
        logger.error(f"Error reemplazando contactos para {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Error guardando contactos")

 
class ModelConfigIn(BaseModel):
    accident_labels: Optional[List[str]] = None
    confidence_threshold: Optional[float] = None

@app.get("/api/v1/model/config")
async def get_model_config():
    return {
        "accident_labels": rf_classifier.accident_labels,
        "confidence_threshold": rf_classifier.confidence_threshold,
    }

@app.post("/api/v1/model/config")
async def set_model_config(cfg: ModelConfigIn):
    try:
        rf_classifier.configure(
            accident_labels=cfg.accident_labels,
            confidence_threshold=cfg.confidence_threshold,
        )
        # Persistir si hay modelo en disco
        if rf_classifier.is_ready():
            os.makedirs(ML_MODEL_DIR, exist_ok=True)
            rf_classifier.save(ML_MODEL_PATH, ML_META_PATH)
        return {"ok": True, "config": {
            "accident_labels": rf_classifier.accident_labels,
            "confidence_threshold": rf_classifier.confidence_threshold,
        }}
    except Exception as e:
        logger.error(f"Error configurando modelo: {e}")
        raise HTTPException(status_code=500, detail="Error configurando modelo")
@app.get("/api/v1/model/dataset/summary")
async def dataset_summary(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    device_id: Optional[str] = Query(None),
    labels_include: Optional[List[str]] = Query(None),
):
    """Devuelve distribución por clase del dataset de entrenamiento con filtros opcionales."""
    try:
        rows = await db.get_sensor_events(
            label=None,
            device_id=device_id,
            start=start,
            end=end,
            limit=100000,
            offset=0
        )
        labeled_rows = [r for r in rows if r.get("label")]
        if labels_include:
            allow = set([str(l).upper() for l in labels_include])
            labeled_rows = [r for r in labeled_rows if str(r.get("label")).upper() in allow]
        from collections import Counter
        counts = Counter([str(r.get("label")) for r in labeled_rows])
        total = sum(counts.values())
        distribution = {k: {"count": v, "percent": round((v / total) * 100, 2)} for k, v in counts.items()}
        return {
            "total": total,
            "by_label": distribution,
            "filters": {
                "start": start,
                "end": end,
                "device_id": device_id,
                "labels_include": labels_include,
            }
        }
    except Exception as e:
        logger.error(f"Error obteniendo resumen de dataset: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo resumen de dataset")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)