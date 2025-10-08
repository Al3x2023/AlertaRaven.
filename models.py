from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid

class SensorEventType(Enum):
    """Tipos de evento para datos de sensores"""
    NORMAL = "NORMAL"
    PHONE_DROP = "PHONE_DROP"
    VEHICLE_ACCIDENT = "VEHICLE_ACCIDENT"
    OTHER = "OTHER"
    # Categorías adicionales para compatibilidad con la app móvil
    COLLISION = "COLLISION"
    SUDDEN_STOP = "SUDDEN_STOP"
    ROLLOVER = "ROLLOVER"
    FALL = "FALL"

class SensorEvent(BaseModel):
    """Evento agregado de sensores para entrenamiento"""
    event_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str = Field(..., description="ID del dispositivo")
    label: SensorEventType = Field(..., description="Etiqueta del evento")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Magnitudes y características agregadas
    acceleration_magnitude: float = Field(..., description="Magnitud de aceleración")
    gyroscope_magnitude: float = Field(..., description="Magnitud de giroscopio")
    accel_variance: Optional[float] = Field(None, description="Varianza de aceleración en ventana")
    gyro_variance: Optional[float] = Field(None, description="Varianza de giroscopio en ventana")
    accel_jerk: Optional[float] = Field(None, description="Jerk promedio")

    # Predicción del modelo (opcional) para métricas
    predicted_label: Optional[SensorEventType] = Field(None, description="Etiqueta predicha por el modelo")
    prediction_confidence: Optional[float] = Field(None, description="Confianza de la predicción (0.0-1.0)")

    # Datos crudos opcionales (para exportación)
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Datos crudos (arrays)")

    created_at: datetime = Field(default_factory=datetime.now)

class AccidentType(Enum):
    """Tipos de accidentes detectables"""
    COLLISION = "COLLISION"
    SUDDEN_STOP = "SUDDEN_STOP"
    ROLLOVER = "ROLLOVER"
    FALL = "FALL"
    UNKNOWN = "UNKNOWN"

class AlertStatus(Enum):
    """Estados de una alerta de emergencia"""
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    NOTIFIED = "NOTIFIED"
    CONFIRMED = "CONFIRMED"
    COMPLETED = "COMPLETED"
    PENDING_REVIEW = "PENDING_REVIEW"
    FAILED = "FAILED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"

class EmergencyAlert(BaseModel):
    """Modelo principal para alertas de emergencia"""
    alert_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str = Field(..., description="ID único del dispositivo")
    user_id: Optional[str] = Field(None, description="ID del usuario")
    
    # Datos del accidente
    accident_type: AccidentType = Field(..., description="Tipo de accidente detectado")
    timestamp: datetime = Field(..., description="Momento del accidente")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Nivel de confianza")
    
    # Datos de sensores
    acceleration_magnitude: float = Field(..., description="Magnitud de aceleración")
    gyroscope_magnitude: float = Field(..., description="Magnitud del giroscopio")
    
    # Ubicación
    location_data: Optional[Dict[str, Any]] = Field(None, description="Datos de ubicación")
    
    # Información médica y contactos
    medical_info: Optional[Dict[str, Any]] = Field(None, description="Información médica")
    emergency_contacts: List[Dict[str, Any]] = Field(default_factory=list, description="Contactos de emergencia")
    
    # Estado y metadatos
    status: AlertStatus = Field(default=AlertStatus.RECEIVED, description="Estado actual")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Datos adicionales
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Datos adicionales")
    
    # Removido Config basado en clase para compatibilidad con Pydantic v2

class AlertStatistics(BaseModel):
    """Estadísticas de alertas"""
    total_alerts: int = 0
    alerts_by_type: Dict[str, int] = Field(default_factory=dict)
    alerts_by_status: Dict[str, int] = Field(default_factory=dict)
    alerts_today: int = 0
    alerts_this_week: int = 0
    alerts_this_month: int = 0
    average_confidence: float = 0.0
    
class DeviceInfo(BaseModel):
    """Información del dispositivo"""
    device_id: str
    user_id: Optional[str] = None
    device_model: Optional[str] = None
    os_version: Optional[str] = None
    app_version: Optional[str] = None
    last_seen: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    
class NotificationLog(BaseModel):
    """Log de notificaciones enviadas"""
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_id: str
    notification_type: str  # SMS, CALL, EMAIL, PUSH
    recipient: str
    status: str  # SENT, FAILED, DELIVERED
    timestamp: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None

class PushSubscriptionPayload(BaseModel):
    """Payload de suscripción push del navegador (PushManager.subscribe)"""
    endpoint: str
    keys: Dict[str, str]
    device_id: Optional[str] = None

class PushNotificationRequest(BaseModel):
    """Solicitud para enviar una notificación push"""
    title: Optional[str] = 'AlertaRaven'
    body: Optional[str] = 'Tienes una nueva alerta'
    url: Optional[str] = '/dashboard'
    alert_id: Optional[str] = None