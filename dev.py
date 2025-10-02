#!/usr/bin/env python3
import uvicorn
import os

if __name__ == "__main__":
    # Configuración para desarrollo
    uvicorn.run(
        "main:app",  # Reemplaza "main" con el nombre de tu archivo
        host="0.0.0.0",
        port=8000,
        reload=True,  # Recarga automática en desarrollo
        reload_dirs=["static"],  # Observar cambios en la carpeta static
        log_level="debug"
    )