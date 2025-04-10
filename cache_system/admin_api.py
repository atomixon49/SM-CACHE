"""
API REST de administración para SM-CACHE.
"""
from fastapi import FastAPI, Security, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import time
from datetime import datetime, timedelta
import ssl
import json
import logging
from .security import CacheSecurity
from .intelligent_cache import IntelligentCache

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS = 100  # requests per window
client_request_counts: Dict[str, list] = {}

app = FastAPI(title="SM-CACHE Admin API", version="1.0.0")
api_key_header = APIKeyHeader(name="X-API-Key")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancias globales
cache_instance: Optional[IntelligentCache] = None
security_instance: Optional[CacheSecurity] = None
ssl_context: Optional[ssl.SSLContext] = None

def load_ssl_config():
    """Carga la configuración SSL."""
    try:
        with open("config/ssl/ssl_config.json") as f:
            config = json.load(f)
        
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(
            config["ssl_settings"]["cert_file"],
            config["ssl_settings"]["key_file"]
        )
        ctx.load_verify_locations(config["client_auth"]["trusted_ca"])
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.set_ciphers(config["ssl_settings"]["ciphers"])
        
        return ctx
    except Exception as e:
        logging.error(f"Error loading SSL config: {e}")
        return None

def init_admin_api(cache: IntelligentCache, security: CacheSecurity):
    """Inicializa la API con las instancias necesarias."""
    global cache_instance, security_instance, ssl_context
    cache_instance = cache
    security_instance = security
    ssl_context = load_ssl_config()

def check_rate_limit(client_id: str) -> bool:
    """Verifica el rate limit para un cliente."""
    now = time.time()
    if client_id not in client_request_counts:
        client_request_counts[client_id] = []
    
    # Limpiar requests antiguos
    client_request_counts[client_id] = [
        timestamp for timestamp in client_request_counts[client_id]
        if now - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Verificar límite
    if len(client_request_counts[client_id]) >= MAX_REQUESTS:
        return False
    
    # Registrar nuevo request
    client_request_counts[client_id].append(now)
    return True

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verifica la API key y los permisos."""
    if not security_instance or not security_instance.validate_api_key(api_key):
        raise HTTPException(status_code=403, detail="API key inválida")
    return api_key

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Middleware para seguridad y rate limiting."""
    # Verificar SSL
    if not request.url.scheme == "https":
        raise HTTPException(status_code=400, detail="HTTPS required")
    
    # Obtener API key del header
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Verificar rate limit
    if not check_rate_limit(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Registrar acceso
    logging.info(f"Access from {request.client.host} to {request.url.path}")
    
    response = await call_next(request)
    return response

@app.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Obtiene estadísticas del caché."""
    if not cache_instance:
        raise HTTPException(status_code=503, detail="Caché no inicializado")
    return cache_instance.get_stats()

@app.post("/flush")
async def flush_cache(api_key: str = Depends(verify_api_key)):
    """Limpia todo el caché."""
    if not cache_instance:
        raise HTTPException(status_code=503, detail="Caché no inicializado")
    if not security_instance.check_permission(api_key, "admin"):
        raise HTTPException(status_code=403, detail="Permiso denegado")
    cache_instance.clear()
    return {"status": "success", "message": "Caché limpiado"}

@app.get("/metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """Obtiene métricas detalladas del caché."""
    if not cache_instance or not cache_instance.metrics_collector:
        raise HTTPException(status_code=503, detail="Métricas no disponibles")
    return cache_instance.metrics_collector.get_metrics_summary()

@app.post("/export-metrics")
async def export_metrics(api_key: str = Depends(verify_api_key)):
    """Exporta las métricas actuales."""
    if not cache_instance or not cache_instance.metrics_monitor:
        raise HTTPException(status_code=503, detail="Monitor de métricas no disponible")
    if not security_instance.check_permission(api_key, "admin"):
        raise HTTPException(status_code=403, detail="Permiso denegado")
    success = cache_instance.metrics_monitor.export_metrics()
    return {"status": "success" if success else "error"}

@app.delete("/items/{key}")
async def delete_item(key: str, api_key: str = Depends(verify_api_key)):
    """Elimina un elemento específico del caché."""
    if not cache_instance:
        raise HTTPException(status_code=503, detail="Caché no inicializado")
    if not security_instance.check_permission(api_key, "write"):
        raise HTTPException(status_code=403, detail="Permiso denegado")
    cache_instance.remove(key)
    return {"status": "success", "message": f"Elemento {key} eliminado"}

@app.get("/health")
async def health_check():
    """Endpoint para health check."""
    if not cache_instance:
        raise HTTPException(status_code=503, detail="Caché no inicializado")
    
    stats = cache_instance.get_stats()
    memory_usage = stats["memory_usage"]["memory_usage_percent"]
    
    status = "healthy"
    if memory_usage > 90:
        status = "warning"
    
    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "memory_usage": memory_usage,
        "cache_size": stats["size"]
    }

@app.get("/cluster/nodes")
async def get_cluster_nodes(api_key: str = Depends(verify_api_key)):
    """Obtiene información sobre los nodos del cluster."""
    if not cache_instance or not cache_instance.distributed_enabled:
        raise HTTPException(status_code=503, detail="Caché distribuido no habilitado")
    if not security_instance.check_permission(api_key, "admin"):
        raise HTTPException(status_code=403, detail="Permiso denegado")
    
    return cache_instance.distributed.get_cluster_info()