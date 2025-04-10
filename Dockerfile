# Usar una imagen base de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos del proyecto
COPY . .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puertos necesarios
EXPOSE 6379  # Puerto principal del caché
EXPOSE 8000  # Puerto para API de administración
EXPOSE 9090  # Puerto para métricas Prometheus

# Variables de entorno por defecto
ENV MAX_MEMORY_MB=1000 \
    ENABLE_SECURITY=true \
    ENABLE_MONITORING=true \
    PROMETHEUS_PORT=9090 \
    ADMIN_API_PORT=8000

# Comando para iniciar el servicio
CMD ["python", "-m", "cache_system"]