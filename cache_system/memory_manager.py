"""
Gestor de memoria para el sistema de caché.
"""
from typing import Dict, Any, List, Optional
import sys
import time
import threading
import logging


class MemoryManager:
    """Gestiona el uso de memoria del caché."""
    
    def __init__(self, max_size_bytes: int):
        """
        Inicializa el gestor de memoria.
        
        Args:
            max_size_bytes: Tamaño máximo en bytes
        """
        self.max_size_bytes = max_size_bytes
        self.current_size = 0
        self.item_sizes: Dict[str, int] = {}
        self.access_times: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MemoryManager")

    def add_item(self, key: str, value: Any) -> bool:
        """
        Añade un elemento y registra su tamaño.
        
        Args:
            key: Clave del elemento
            value: Valor a almacenar
            
        Returns:
            True si se añadió correctamente, False si no hay espacio
        """
        size = self._estimate_size(value)
        
        with self._lock:
            if self.current_size + size > self.max_size_bytes:
                return False
                
            self.item_sizes[key] = size
            self.current_size += size
            self.access_times[key] = time.time()
            
            return True

    def remove_item(self, key: str) -> None:
        """
        Elimina un elemento y actualiza el uso de memoria.
        
        Args:
            key: Clave del elemento a eliminar
        """
        with self._lock:
            if key in self.item_sizes:
                self.current_size -= self.item_sizes[key]
                del self.item_sizes[key]
                del self.access_times[key]

    def update_item(self, key: str, value: Any) -> bool:
        """
        Actualiza un elemento existente.
        
        Args:
            key: Clave del elemento
            value: Nuevo valor
            
        Returns:
            True si se actualizó correctamente, False si no hay espacio
        """
        with self._lock:
            # Eliminar tamaño anterior
            if key in self.item_sizes:
                self.current_size -= self.item_sizes[key]
                
            # Calcular nuevo tamaño
            new_size = self._estimate_size(value)
            
            # Verificar espacio
            if self.current_size + new_size > self.max_size_bytes:
                # Restaurar tamaño anterior si existía
                if key in self.item_sizes:
                    self.current_size += self.item_sizes[key]
                return False
                
            # Actualizar
            self.item_sizes[key] = new_size
            self.current_size += new_size
            self.access_times[key] = time.time()
            
            return True

    def get_eviction_candidates(self, count: int) -> List[str]:
        """
        Obtiene candidatos para evicción basado en acceso y tamaño.
        
        Args:
            count: Número de candidatos a retornar
            
        Returns:
            Lista de claves candidatas para evicción
        """
        with self._lock:
            # Ordenar por último acceso (más antiguo primero)
            sorted_items = sorted(
                self.access_times.items(),
                key=lambda x: (x[1], -self.item_sizes.get(x[0], 0))
            )
            
            return [key for key, _ in sorted_items[:count]]

    def is_memory_full(self) -> bool:
        """
        Verifica si la memoria está llena.
        
        Returns:
            True si la memoria está llena, False en caso contrario
        """
        return self.current_size >= self.max_size_bytes

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de uso de memoria.
        
        Returns:
            Diccionario con estadísticas de memoria
        """
        with self._lock:
            return {
                'current_size': self.current_size,
                'max_size': self.max_size_bytes,
                'item_count': len(self.item_sizes),
                'usage_percent': (self.current_size / self.max_size_bytes) * 100
            }

    def _estimate_size(self, value: Any) -> int:
        """
        Estima el tamaño en bytes de un valor.
        
        Args:
            value: Valor a estimar
            
        Returns:
            Tamaño estimado en bytes
        """
        size = sys.getsizeof(value)
        
        # Estimación recursiva para contenedores
        if isinstance(value, (list, tuple, set)):
            size += sum(self._estimate_size(item) for item in value)
        elif isinstance(value, dict):
            size += sum(self._estimate_size(k) + self._estimate_size(v) 
                       for k, v in value.items())
        elif isinstance(value, str):
            size = len(value.encode('utf-8'))
            
        return size
