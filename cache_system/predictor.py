"""
Módulo para la predicción de datos que serán necesarios en el sistema de caché inteligente.
"""
from typing import Dict, Any, List, Set, Optional
from collections import defaultdict
import math
from .usage_tracker import UsageTracker


class CachePredictor:
    """
    Clase que predice qué datos serán necesarios en el futuro basándose
    en patrones de uso históricos.
    """
    
    def __init__(self, usage_tracker: UsageTracker, confidence_threshold: float = 0.6):
        """
        Inicializa el predictor de caché.
        
        Args:
            usage_tracker: El rastreador de uso que proporciona datos históricos
            confidence_threshold: Umbral de confianza para hacer predicciones
        """
        self.usage_tracker = usage_tracker
        self.confidence_threshold = confidence_threshold
        self.sequence_patterns: Dict[tuple, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.last_n_keys: List[Any] = []
        self.max_pattern_length = 5
        
    def update_patterns(self, key: Any) -> None:
        """
        Actualiza los patrones de secuencia con una nueva clave accedida.
        
        Args:
            key: La clave a la que se acaba de acceder
        """
        # Actualizar la lista de las últimas n claves
        self.last_n_keys.append(key)
        if len(self.last_n_keys) > self.max_pattern_length:
            self.last_n_keys.pop(0)
        
        # Actualizar patrones para diferentes longitudes de secuencia
        for i in range(1, min(len(self.last_n_keys), self.max_pattern_length)):
            pattern = tuple(self.last_n_keys[-i-1:-1])  # Patrón anterior a la clave actual
            self.sequence_patterns[pattern][key] += 1
    
    def predict_next_keys(self, n: int = 3) -> List[Any]:
        """
        Predice las próximas n claves que probablemente se accederán.
        
        Args:
            n: Número de claves a predecir
            
        Returns:
            Lista de claves predichas ordenadas por probabilidad
        """
        if len(self.last_n_keys) < 2:
            # No hay suficientes datos para hacer predicciones
            return []
        
        candidates = defaultdict(float)
        
        # Considerar patrones de diferentes longitudes
        for i in range(1, min(len(self.last_n_keys), self.max_pattern_length)):
            pattern = tuple(self.last_n_keys[-i:])
            
            if pattern in self.sequence_patterns:
                total = sum(self.sequence_patterns[pattern].values())
                for key, count in self.sequence_patterns[pattern].items():
                    # Calcular probabilidad y ajustar por longitud del patrón
                    probability = count / total
                    # Los patrones más largos tienen más peso
                    weight = i / self.max_pattern_length
                    candidates[key] += probability * weight
        
        # Filtrar por umbral de confianza y ordenar
        predictions = [(k, p) for k, p in candidates.items() 
                      if p >= self.confidence_threshold]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return [k for k, _ in predictions[:n]]
    
    def predict_related_keys(self, key: Any, n: int = 3) -> List[Any]:
        """
        Predice claves relacionadas con una clave específica.
        
        Args:
            key: La clave para la que buscar relaciones
            n: Número de claves relacionadas a devolver
            
        Returns:
            Lista de claves relacionadas
        """
        related = defaultdict(int)
        
        # Buscar patrones donde aparece la clave
        for pattern, next_keys in self.sequence_patterns.items():
            if key in pattern:
                for next_key, count in next_keys.items():
                    if next_key != key:
                        related[next_key] += count
        
        # Ordenar por frecuencia
        sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_related[:n]]
    
    def get_prefetch_candidates(self) -> Set[Any]:
        """
        Obtiene un conjunto de claves candidatas para prefetch.
        
        Returns:
            Conjunto de claves que deberían ser precargadas
        """
        # Combinar predicciones de próximas claves y claves relacionadas
        candidates = set(self.predict_next_keys(5))
        
        # Añadir claves relacionadas con las últimas accedidas
        if self.last_n_keys:
            for recent_key in self.last_n_keys[-3:]:  # Últimas 3 claves
                related = self.predict_related_keys(recent_key, 2)
                candidates.update(related)
        
        return candidates
