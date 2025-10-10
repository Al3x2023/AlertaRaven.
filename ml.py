import os
import json
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


class RFAccidentClassifier:
    """Wrapper simple para RandomForest en clasificación de eventos de sensores.

    - Entrena con características agregadas de `sensor_events`
    - Persiste el modelo a disco (joblib)
    - Expone predicción con confianza (probabilidad máxima)
    """

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.classes_: List[str] = []
        self.params: Dict[str, Any] = {}
        # Configuración de dominio
        self.accident_labels: List[str] = [
            "VEHICLE_ACCIDENT", "COLLISION", "SUDDEN_STOP", "ROLLOVER", "FALL"
        ]
        self.confidence_threshold: float = 0.7

    def is_ready(self) -> bool:
        return self.model is not None and len(self.classes_) > 0

    @staticmethod
    def _features_from_row(row: Dict[str, Any]) -> List[float]:
        # Tomar características disponibles, imputando 0 para None
        am = row.get("acceleration_magnitude") or 0.0
        gm = row.get("gyroscope_magnitude") or 0.0
        av = row.get("accel_variance") if row.get("accel_variance") is not None else 0.0
        gv = row.get("gyro_variance") if row.get("gyro_variance") is not None else 0.0
        aj = row.get("accel_jerk") if row.get("accel_jerk") is not None else 0.0
        return [float(am), float(gm), float(av), float(gv), float(aj)]

    def train(self, rows: List[Dict[str, Any]],
              n_estimators: int = 200,
              max_depth: Optional[int] = None,
              min_samples_leaf: int = 1,
              test_size: float = 0.2,
              random_state: int = 42) -> Dict[str, Any]:
        """Entrena el modelo con filas de la BD y devuelve métricas de evaluación."""
        if not rows:
            raise ValueError("No hay datos para entrenar")

        X: List[List[float]] = []
        y: List[str] = []
        for r in rows:
            label = r.get("label")
            if not label:
                # Saltar registros sin etiqueta
                continue
            X.append(self._features_from_row(r))
            y.append(str(label))

        if len(X) < 5:
            raise ValueError("Dataset insuficiente para entrenar (menos de 5 ejemplos)")

        X_train, X_test, y_train, y_test = train_test_split(
            np.array(X), np.array(y), test_size=test_size, random_state=random_state, stratify=np.array(y)
        )

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # Evaluación simple
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        labels_sorted = sorted(list(set(list(y_test) + list(y_pred))))
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

        self.model = rf
        self.classes_ = list(rf.classes_)
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "test_size": test_size,
            "random_state": random_state,
        }

        return {
            "accuracy": acc,
            "report": report,
            "labels": labels_sorted,
            "confusion_matrix": cm.tolist(),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        }

    def predict(self, row: Dict[str, Any]) -> Tuple[str, float]:
        """Predice etiqueta y confianza para una fila de evento."""
        if not self.is_ready():
            raise RuntimeError("Modelo no entrenado/cargado")
        x = np.array([self._features_from_row(row)])
        proba = self.model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = self.model.classes_[idx]
        confidence = float(proba[idx])
        # Aplicar umbral de confianza para accidentes
        if label in self.accident_labels and confidence < self.confidence_threshold:
            # Degradar a NORMAL para priorizar precisión
            return "NORMAL", confidence
        return str(label), confidence

    def save(self, model_path: str, meta_path: Optional[str] = None):
        if not self.is_ready():
            raise RuntimeError("No hay modelo para guardar")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        if meta_path:
            meta = {
                "classes": self.classes_,
                "params": self.params,
                "accident_labels": self.accident_labels,
                "confidence_threshold": self.confidence_threshold,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)

    def load(self, model_path: str, meta_path: Optional[str] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        self.model = joblib.load(model_path)
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.classes_ = meta.get("classes", [])
                self.params = meta.get("params", {})
                self.accident_labels = meta.get("accident_labels", self.accident_labels)
                self.confidence_threshold = float(meta.get("confidence_threshold", self.confidence_threshold))
        else:
            # Recuperar clases del modelo si no hay meta
            self.classes_ = list(getattr(self.model, "classes_", []))
            self.params = {}

    def configure(self, accident_labels: Optional[List[str]] = None, confidence_threshold: Optional[float] = None):
        """Configura etiquetas consideradas accidentes y el umbral de confianza."""
        if accident_labels is not None and len(accident_labels) > 0:
            self.accident_labels = [str(l).upper() for l in accident_labels]
        if confidence_threshold is not None:
            self.confidence_threshold = float(confidence_threshold)

    def train_cv(self, rows: List[Dict[str, Any]], param_grid: Optional[Dict[str, List[Any]]] = None, cv_splits: int = 3,
                 random_state: int = 42) -> Dict[str, Any]:
        """Entrena usando GridSearchCV con validación cruzada estratificada y devuelve mejores parámetros y métricas."""
        if not rows:
            raise ValueError("No hay datos para entrenar")

        X: List[List[float]] = []
        y: List[str] = []
        for r in rows:
            label = r.get("label")
            if not label:
                continue
            X.append(self._features_from_row(r))
            y.append(str(label))

        if len(X) < 10:
            raise ValueError("Dataset insuficiente para CV (menos de 10 ejemplos)")

        X_np = np.array(X)
        y_np = np.array(y)

        if param_grid is None:
            param_grid = {
                "n_estimators": [200, 300, 400],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 4],
            }

        base = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        grid = GridSearchCV(base, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        grid.fit(X_np, y_np)

        best: RandomForestClassifier = grid.best_estimator_
        self.model = best
        self.classes_ = list(best.classes_)
        self.params = grid.best_params_

        # Informe aproximado sobre todo el set (validación cruzada ya evaluó)
        y_pred = best.predict(X_np)
        acc = accuracy_score(y_np, y_pred)
        report = classification_report(y_np, y_pred, output_dict=True)
        labels_sorted = sorted(list(set(list(y_np) + list(y_pred))))
        cm = confusion_matrix(y_np, y_pred, labels=labels_sorted)

        return {
            "accuracy": acc,
            "report": report,
            "labels": labels_sorted,
            "confusion_matrix": cm.tolist(),
            "best_params": self.params,
            "cv_splits": cv_splits,
        }