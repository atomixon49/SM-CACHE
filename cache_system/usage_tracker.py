"""
Módulo para el seguimiento de patrones de uso en el sistema de caché inteligente.
"""
import time
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple


class UsageTracker:
    """
    Clase que rastrea los patrones de uso de datos en el caché.
    Registra cuándo y con qué frecuencia se accede a cada clave.
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Inicializa el rastreador de uso.
        
        Args:
            max_history_size: Tamaño máximo del historial de accesos a mantener
        """
        self.access_times: Dict[Any, List[float]] = defaultdict(list)
        self.access_count: Counter = Counter()
        self.access_sequence: List[Any] = []
        self.max_history_size = max_history_size
        
    def record_access(self, key: Any) -> None:
        """
        Registra un acceso a una clave específica.
        
        Args:
            key: La clave a la que se accedió
        """
        current_time = time.time()
        self.access_times[key].append(current_time)
        self.access_count[key] += 1
        
        # Mantener el historial de secuencia de acceso
        self.access_sequence.append(key)
        if len(self.access_sequence) > self.max_history_size:
            # Eliminar el elemento más antiguo
            oldest_key = self.access_sequence.pop(0)
            # También podríamos limpiar access_times para claves antiguas si es necesario
            
    def get_access_frequency(self, key: Any) -> int:
        """
        Obtiene la frecuencia de acceso para una clave.
        
        Args:
            key: La clave para consultar
            
        Returns:
            El número de veces que se ha accedido a la clave
        """
        return self.access_count[key]
    
    def get_last_access_time(self, key: Any) -> float:
        """
        Obtiene el último tiempo de acceso para una clave.
        
        Args:
            key: La clave para consultar
            
        Returns:
            El timestamp del último acceso o 0 si nunca se ha accedido
        """
        times = self.access_times.get(key, [])
        return times[-1] if times else 0
    
    def get_access_pattern(self, key: Any, window: int = 5) -> List[Any]:
        """
        Obtiene el patrón de acceso que precede a una clave específica.
        
        Args:
            key: La clave para analizar
            window: El tamaño de la ventana de contexto anterior
            
        Returns:
            Lista de claves que suelen preceder a la clave especificada
        """
        patterns = []
        for i, k in enumerate(self.access_sequence):
            if k == key and i >= window:
                patterns.append(self.access_sequence[i-window:i])
        return patterns
    
    def get_most_accessed_keys(self, n: int = 10) -> List[Tuple[Any, int]]:
        """
        Obtiene las n claves más accedidas.
        
        Args:
            n: Número de claves a devolver
            
        Returns:
            Lista de tuplas (clave, frecuencia) ordenadas por frecuencia
        """
        return self.access_count.most_common(n)
    
    def clear_history(self) -> None:
        """Limpia todo el historial de acceso."""
        self.access_times.clear()
        self.access_count.clear()
        self.access_sequence.clear()
