"""
Optimizaciones de rendimiento para el sistema de caché.
"""
from functools import lru_cache
import zlib
from typing import Any, Optional, Dict

class OptimizedStore:
    """Implementación optimizada del almacenamiento en caché."""
    
    def __init__(self, max_size: int = 1000, compression_level: int = 6):
        self._cache: Dict[Any, bytes] = {}
        self.compression_level = compression_level
        self.max_size = max_size
    
    @lru_cache(maxsize=1000)
    def get(self, key: Any) -> Optional[Any]:
        """Obtiene un valor del caché con LRU integrado."""
        if key not in self._cache:
            return None
        return self._decompress(self._cache[key])
    
    def put(self, key: Any, value: Any) -> None:
        """Almacena un valor en el caché con compresión."""
        compressed = self._compress(value)
        self._cache[key] = compressed
    
    def _compress(self, data: Any) -> bytes:
        """Comprime los datos para optimizar el almacenamiento."""
        try:
            return zlib.compress(str(data).encode(), self.compression_level)
        except Exception:
            # Si falla la compresión, almacenar sin comprimir
            return str(data).encode()
    
    def _decompress(self, data: bytes) -> Any:
        """Descomprime los datos almacenados."""
        try:
            return zlib.decompress(data).decode()
        except Exception:
            # Si falla la descompresión, devolver los datos sin procesar
            return data.decode()
    
    def clear(self) -> None:
        """Limpia el caché y el LRU cache."""
        self._cache.clear()
        self.get.cache_clear()  # Limpiar LRU cache