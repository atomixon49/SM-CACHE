"""
Módulo para la gestión automática de memoria en el sistema de caché inteligente.
"""
from typing import Dict, Any, List, Tuple, Callable, Optional
import time
import sys
from .usage_tracker import UsageTracker


class MemoryManager:
    """
    Clase que gestiona automáticamente la memoria del caché,
    decidiendo qué elementos mantener y cuáles descartar.
    """

    def __init__(self,
                 usage_tracker: UsageTracker,
                 max_size: int = 1000,
                 max_memory_mb: float = 100.0,
                 size_estimator: Optional[Callable[[Any], int]] = None):
        """
        Inicializa el gestor de memoria.

        Args:
            usage_tracker: El rastreador de uso que proporciona datos históricos
            max_size: Número máximo de elementos en el caché
            max_memory_mb: Tamaño máximo de memoria en MB
            size_estimator: Función para estimar el tamaño de un valor en bytes
        """
        self.usage_tracker = usage_tracker
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_size = 0
        self.current_memory_bytes = 0
        self.size_estimator = size_estimator or self._default_size_estimator
        self.item_sizes: Dict[Any, int] = {}

    def _default_size_estimator(self, value: Any) -> int:
        """
        Estima el tamaño en bytes de un valor.

        Args:
            value: El valor a estimar

        Returns:
            Tamaño estimado en bytes
        """
        return sys.getsizeof(value)

    def add_item(self, key: Any, value: Any) -> None:
        """
        Registra un nuevo elemento en el gestor de memoria.

        Args:
            key: La clave del elemento
            value: El valor del elemento
        """
        # Estimar el tamaño del valor
        size = self.size_estimator(value)

        # Actualizar el tamaño total
        if key in self.item_sizes:
            self.current_memory_bytes -= self.item_sizes[key]

        self.item_sizes[key] = size
        self.current_memory_bytes += size
        self.current_size = len(self.item_sizes)

    def remove_item(self, key: Any) -> None:
        """
        Elimina un elemento del gestor de memoria.

        Args:
            key: La clave del elemento a eliminar
        """
        if key in self.item_sizes:
            self.current_memory_bytes -= self.item_sizes[key]
            del self.item_sizes[key]
            self.current_size = len(self.item_sizes)

    def is_memory_full(self) -> bool:
        """
        Verifica si la memoria está llena según los límites establecidos.

        Returns:
            True si la memoria está llena, False en caso contrario
        """
        return (self.current_size >= self.max_size or
                self.current_memory_bytes >= self.max_memory_bytes)

    def get_eviction_candidates(self, count: int = 1) -> List[Any]:
        """
        Obtiene las claves candidatas para ser eliminadas del caché.

        Args:
            count: Número de candidatos a devolver

        Returns:
            Lista de claves candidatas para evicción
        """
        # Calcular puntuación para cada elemento
        scores = []
        current_time = time.time()

        for key in self.item_sizes:
            # Factores para la puntuación:
            # 1. Frecuencia de acceso (menor es mejor para evicción)
            frequency = self.usage_tracker.get_access_frequency(key)

            # 2. Tiempo desde el último acceso (mayor es mejor para evicción)
            last_access = self.usage_tracker.get_last_access_time(key)
            recency = current_time - last_access if last_access > 0 else current_time

            # 3. Tamaño del elemento (mayor es mejor para evicción)
            size = self.item_sizes[key]

            # Calcular puntuación (mayor puntuación = mejor candidato para evicción)
            # Normalizar cada factor para que tenga un peso adecuado
            if frequency == 0:
                frequency = 0.1  # Evitar división por cero

            score = (recency * size) / frequency
            scores.append((key, score))

        # Ordenar por puntuación (mayor primero)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Devolver las claves con mayor puntuación
        return [key for key, _ in scores[:count]]

    def get_memory_usage(self) -> int:
        """
        Obtiene el uso actual de memoria en bytes.

        Returns:
            Uso de memoria en bytes
        """
        return self.current_memory_bytes

    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas sobre el uso de memoria.

        Returns:
            Diccionario con estadísticas de uso de memoria
        """
        return {
            'current_size': self.current_size,
            'max_size': self.max_size,
            'current_memory_mb': self.current_memory_bytes / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'memory_usage_percent': (self.current_memory_bytes / self.max_memory_bytes) * 100
                                    if self.max_memory_bytes > 0 else 0,
            'size_usage_percent': (self.current_size / self.max_size) * 100
                                  if self.max_size > 0 else 0
        }
