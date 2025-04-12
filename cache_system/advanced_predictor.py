"""
Predictor avanzado para el sistema de caché inteligente.
Implementa algoritmos de aprendizaje automático para mejorar la precisión de las predicciones.
"""
from typing import Dict, Any, List, Set, Tuple, Optional
import time
import math
import numpy as np
from collections import Counter, defaultdict, deque
import logging
from .usage_tracker import UsageTracker


class AdvancedPredictor:
    """
    Predictor avanzado que combina múltiples algoritmos para mejorar la precisión
    de las predicciones de acceso a caché.
    """
    
    def __init__(self, usage_tracker: UsageTracker, 
                 learning_rate: float = 0.1,
                 decay_factor: float = 0.95,
                 max_history: int = 10000):
        """
        Inicializa el predictor avanzado.
        
        Args:
            usage_tracker: Rastreador de uso para obtener datos históricos
            learning_rate: Tasa de aprendizaje para actualizar pesos
            decay_factor: Factor de decaimiento para dar más peso a eventos recientes
            max_history: Tamaño máximo del historial a mantener
        """
        self.usage_tracker = usage_tracker
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        
        # Historial de acceso reciente
        self.access_history = deque(maxlen=max_history)
        self.pattern_history: Dict[Tuple[Any, ...], Counter] = defaultdict(Counter)
        
        # Modelos de predicción
        self.markov_weights = {}  # Pesos para cadenas de Markov de diferentes órdenes
        self.frequency_weights = {}  # Pesos para diferentes ventanas de tiempo
        
        # Inicializar pesos
        for order in range(1, 6):  # Órdenes 1-5
            self.markov_weights[order] = 0.5 / order  # Mayor peso a órdenes menores
            
        for window in [10, 100, 1000]:
            self.frequency_weights[window] = 0.5 / math.log2(window)  # Mayor peso a ventanas más pequeñas
        
        # Métricas de rendimiento
        self.hit_count = 0
        self.miss_count = 0
        self.predictions: Dict[Any, float] = {}  # Última predicción para cada clave
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AdvancedPredictor")
    
    def update_patterns(self, key: Any) -> None:
        """
        Actualiza los patrones de acceso con una nueva clave.
        
        Args:
            key: Clave a la que se accedió
        """
        # Registrar en historial
        self.access_history.append(key)
        
        # Actualizar patrones de Markov para diferentes órdenes
        for order in range(1, 6):
            if len(self.access_history) >= order:
                # Obtener contexto (últimas 'order' claves)
                context = tuple(list(self.access_history)[-order:])
                
                # Predecir la siguiente clave basada en este contexto
                if len(context) == order:
                    self.pattern_history[context[:-1]][context[-1]] += 1
        
        # Verificar si la clave fue predicha correctamente
        if key in self.predictions:
            self.hit_count += 1
            
            # Ajustar pesos basados en el éxito
            prediction_score = self.predictions[key]
            if prediction_score > 0:
                # Aumentar peso de los modelos que contribuyeron a esta predicción
                for order, weight in self.markov_weights.items():
                    if self._get_markov_prediction(order, key) > 0:
                        self.markov_weights[order] += self.learning_rate * (1 - weight)
                    else:
                        self.markov_weights[order] -= self.learning_rate * weight
                
                for window, weight in self.frequency_weights.items():
                    if self._get_frequency_prediction(window, key) > 0:
                        self.frequency_weights[window] += self.learning_rate * (1 - weight)
                    else:
                        self.frequency_weights[window] -= self.learning_rate * weight
                
                # Normalizar pesos
                self._normalize_weights()
        else:
            self.miss_count += 1
        
        # Limpiar predicciones anteriores
        self.predictions = {}
    
    def predict_next_keys(self, count: int = 5) -> List[Any]:
        """
        Predice las próximas claves que serán accedidas.
        
        Args:
            count: Número de claves a predecir
            
        Returns:
            Lista de claves predichas ordenadas por probabilidad
        """
        # Combinar predicciones de diferentes modelos
        combined_scores = Counter()
        
        # Predicciones basadas en Markov
        for order, weight in self.markov_weights.items():
            predictions = self._predict_markov(order, count * 2)
            for key, score in predictions:
                combined_scores[key] += score * weight
        
        # Predicciones basadas en frecuencia
        for window, weight in self.frequency_weights.items():
            predictions = self._predict_frequency(window, count * 2)
            for key, score in predictions:
                combined_scores[key] += score * weight
        
        # Guardar predicciones para evaluación
        self.predictions = {k: s for k, s in combined_scores.items()}
        
        # Devolver las mejores predicciones
        return [key for key, _ in combined_scores.most_common(count)]
    
    def _predict_markov(self, order: int, count: int) -> List[Tuple[Any, float]]:
        """
        Realiza predicciones usando modelo de Markov de orden específico.
        
        Args:
            order: Orden del modelo de Markov
            count: Número de predicciones a devolver
            
        Returns:
            Lista de tuplas (clave, puntuación)
        """
        if len(self.access_history) < order:
            return []
        
        # Obtener contexto actual
        context = tuple(list(self.access_history)[-order:])
        
        # Buscar el contexto más largo que coincida
        for i in range(order, 0, -1):
            if i <= len(context):
                sub_context = context[-i:]
                if sub_context in self.pattern_history:
                    # Calcular probabilidades
                    counter = self.pattern_history[sub_context]
                    total = sum(counter.values())
                    
                    if total > 0:
                        # Normalizar puntuaciones
                        return [(key, count/total * (i/order)) 
                                for key, count in counter.most_common(count)]
        
        return []
    
    def _predict_frequency(self, window: int, count: int) -> List[Tuple[Any, float]]:
        """
        Realiza predicciones basadas en frecuencia de acceso reciente.
        
        Args:
            window: Tamaño de la ventana de tiempo
            count: Número de predicciones a devolver
            
        Returns:
            Lista de tuplas (clave, puntuación)
        """
        if not self.access_history:
            return []
        
        # Obtener ventana de accesos recientes
        if len(self.access_history) <= window:
            recent = list(self.access_history)
        else:
            recent = list(self.access_history)[-window:]
        
        # Contar frecuencias
        counter = Counter(recent)
        total = len(recent)
        
        # Normalizar y devolver
        return [(key, count/total) for key, count in counter.most_common(count)]
    
    def _get_markov_prediction(self, order: int, key: Any) -> float:
        """
        Obtiene la puntuación de predicción de Markov para una clave específica.
        
        Args:
            order: Orden del modelo de Markov
            key: Clave a evaluar
            
        Returns:
            Puntuación de predicción
        """
        if len(self.access_history) < order:
            return 0.0
        
        # Obtener contexto actual
        context = tuple(list(self.access_history)[-order:])
        
        # Buscar el contexto más largo que coincida
        for i in range(order, 0, -1):
            if i <= len(context):
                sub_context = context[-i:]
                if sub_context in self.pattern_history:
                    # Calcular probabilidad
                    counter = self.pattern_history[sub_context]
                    total = sum(counter.values())
                    
                    if total > 0 and key in counter:
                        return counter[key] / total * (i/order)
        
        return 0.0
    
    def _get_frequency_prediction(self, window: int, key: Any) -> float:
        """
        Obtiene la puntuación de predicción de frecuencia para una clave específica.
        
        Args:
            window: Tamaño de la ventana de tiempo
            key: Clave a evaluar
            
        Returns:
            Puntuación de predicción
        """
        if not self.access_history:
            return 0.0
        
        # Obtener ventana de accesos recientes
        if len(self.access_history) <= window:
            recent = list(self.access_history)
        else:
            recent = list(self.access_history)[-window:]
        
        # Contar frecuencia de la clave
        count = recent.count(key)
        total = len(recent)
        
        return count / total if total > 0 else 0.0
    
    def _normalize_weights(self) -> None:
        """Normaliza los pesos para que sumen 1.0"""
        # Normalizar pesos de Markov
        markov_sum = sum(self.markov_weights.values())
        if markov_sum > 0:
            for order in self.markov_weights:
                self.markov_weights[order] /= markov_sum
        
        # Normalizar pesos de frecuencia
        freq_sum = sum(self.frequency_weights.values())
        if freq_sum > 0:
            for window in self.frequency_weights:
                self.frequency_weights[window] /= freq_sum
    
    def get_prefetch_candidates(self) -> Set[Any]:
        """
        Obtiene un conjunto de claves candidatas para prefetch.
        
        Returns:
            Conjunto de claves que deberían ser precargadas
        """
        # Predecir próximas claves
        candidates = set(self.predict_next_keys(10))
        
        # Añadir claves con alta frecuencia de acceso
        if len(self.access_history) > 0:
            # Contar frecuencias en los últimos 100 accesos
            recent = list(self.access_history)[-100:] if len(self.access_history) > 100 else list(self.access_history)
            counter = Counter(recent)
            
            # Añadir claves con alta frecuencia
            threshold = 0.1 * len(recent)  # 10% de los accesos recientes
            for key, count in counter.items():
                if count >= threshold and key not in candidates:
                    candidates.add(key)
        
        return candidates
    
    def get_prediction_accuracy(self) -> float:
        """
        Obtiene la precisión de las predicciones.
        
        Returns:
            Porcentaje de aciertos (0-100)
        """
        total = self.hit_count + self.miss_count
        return (self.hit_count / total) * 100 if total > 0 else 0.0
    
    def get_model_weights(self) -> Dict[str, Dict[Any, float]]:
        """
        Obtiene los pesos actuales de los modelos.
        
        Returns:
            Diccionario con los pesos de cada modelo
        """
        return {
            'markov': self.markov_weights,
            'frequency': self.frequency_weights
        }
