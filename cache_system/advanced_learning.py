"""
Sistema de aprendizaje avanzado y predicción de patrones.
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
import logging
import time
from collections import deque, Counter, defaultdict
import math
from .usage_tracker import UsageTracker


class PatternAnalyzer:
    """Analizador de patrones para detección de anomalías y predicciones."""
    
    def __init__(self, window_size: int = 1000,
                 anomaly_threshold: float = 0.95,
                 trend_window: int = 60):
        """
        Inicializa el analizador de patrones.
        
        Args:
            window_size: Tamaño de la ventana de datos históricos
            anomaly_threshold: Umbral para detección de anomalías
            trend_window: Ventana para análisis de tendencias (segundos)
        """
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.trend_window = trend_window
        
        # Históricos
        self.memory_history: deque = deque(maxlen=window_size)
        self.latency_history: deque = deque(maxlen=window_size)
        self.hit_rate_history: deque = deque(maxlen=window_size)
        
        # Modelos
        self.isolation_forest = IsolationForest(contamination=0.1)
        self.model_trained = False
        
        # Estado
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
        # Logging
        self.logger = logging.getLogger("PatternAnalyzer")
    
    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Añade nuevas métricas al análisis.
        
        Args:
            metrics: Diccionario con métricas actuales
        """
        timestamp = time.time()
        
        # Extraer métricas relevantes
        memory_usage = metrics.get('memory_usage', {}).get('memory_usage_percent', 0)
        latency = metrics.get('performance', {}).get('avg_get_time', 0)
        hit_rate = metrics.get('performance', {}).get('hit_rate', 0)
        
        # Almacenar datos
        self.memory_history.append((timestamp, memory_usage))
        self.latency_history.append((timestamp, latency))
        self.hit_rate_history.append((timestamp, hit_rate))
        
        # Entrenar modelo si hay suficientes datos
        if len(self.memory_history) >= 100 and not self.model_trained:
            self._train_models()
    
    def _train_models(self) -> None:
        """Entrena los modelos de detección de anomalías."""
        try:
            # Preparar datos
            X = np.array([
                [m[1] for m in self.memory_history],
                [l[1] for l in self.latency_history],
                [h[1] for h in self.hit_rate_history]
            ]).T
            
            # Entrenar modelo
            self.isolation_forest.fit(X)
            self.model_trained = True
            
            # Calcular líneas base
            self.baseline_stats = {
                'memory': {
                    'mean': np.mean([m[1] for m in self.memory_history]),
                    'std': np.std([m[1] for m in self.memory_history])
                },
                'latency': {
                    'mean': np.mean([l[1] for l in self.latency_history]),
                    'std': np.std([l[1] for l in self.latency_history])
                },
                'hit_rate': {
                    'mean': np.mean([h[1] for h in self.hit_rate_history]),
                    'std': np.std([h[1] for h in self.hit_rate_history])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelos: {e}")
    
    def detect_anomalies(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detecta anomalías en las métricas actuales.
        
        Args:
            current_metrics: Métricas actuales
            
        Returns:
            Lista de anomalías detectadas
        """
        anomalies = []
        
        if not self.model_trained:
            return anomalies
            
        try:
            # Preparar datos actuales
            X = np.array([[
                current_metrics.get('memory_usage', {}).get('memory_usage_percent', 0),
                current_metrics.get('performance', {}).get('avg_get_time', 0),
                current_metrics.get('performance', {}).get('hit_rate', 0)
            ]])
            
            # Detectar anomalías
            prediction = self.isolation_forest.predict(X)
            
            if prediction[0] == -1:  # Anomalía detectada
                # Analizar qué métricas son anómalas
                memory = X[0][0]
                latency = X[0][1]
                hit_rate = X[0][2]
                
                for metric, value, baseline in [
                    ('memory', memory, self.baseline_stats['memory']),
                    ('latency', latency, self.baseline_stats['latency']),
                    ('hit_rate', hit_rate, self.baseline_stats['hit_rate'])
                ]:
                    z_score = (value - baseline['mean']) / baseline['std']
                    
                    if abs(z_score) > stats.norm.ppf(self.anomaly_threshold):
                        anomalies.append({
                            'metric': metric,
                            'value': value,
                            'z_score': z_score,
                            'baseline_mean': baseline['mean'],
                            'baseline_std': baseline['std'],
                            'severity': 'high' if abs(z_score) > 3 else 'medium'
                        })
                        
        except Exception as e:
            self.logger.error(f"Error detectando anomalías: {e}")
            
        return anomalies
    
    def analyze_trends(self) -> Dict[str, Dict[str, Any]]:
        """
        Analiza tendencias en las métricas.
        
        Returns:
            Diccionario con análisis de tendencias
        """
        current_time = time.time()
        window_start = current_time - self.trend_window
        
        trends = {}
        
        for metric_name, history in [
            ('memory', self.memory_history),
            ('latency', self.latency_history),
            ('hit_rate', self.hit_rate_history)
        ]:
            # Filtrar datos en la ventana de tiempo
            recent_data = [
                (t, v) for t, v in history
                if t >= window_start
            ]
            
            if len(recent_data) < 2:
                continue
                
            # Calcular tendencia
            times, values = zip(*recent_data)
            slope, _, r_value, _, _ = stats.linregress(times, values)
            
            # Determinar dirección y fuerza de la tendencia
            direction = 'stable'
            if abs(slope) > 0.01:
                direction = 'increasing' if slope > 0 else 'decreasing'
                
            strength = abs(r_value)
            
            trends[metric_name] = {
                'direction': direction,
                'strength': strength,
                'slope': slope,
                'r_squared': r_value ** 2
            }
            
        return trends
    
    def predict_threshold_breach(self, metric: str,
                               threshold: float,
                               horizon: int = 300) -> Optional[float]:
        """
        Predice tiempo hasta alcanzar un umbral.
        
        Args:
            metric: Nombre de la métrica
            threshold: Valor umbral
            horizon: Horizonte de predicción en segundos
            
        Returns:
            Tiempo estimado hasta alcanzar el umbral (None si no se alcanza)
        """
        history = {
            'memory': self.memory_history,
            'latency': self.latency_history,
            'hit_rate': self.hit_rate_history
        }.get(metric)
        
        if not history or len(history) < 2:
            return None
            
        try:
            times, values = zip(*history)
            slope, intercept, _, _, _ = stats.linregress(times, values)
            
            if abs(slope) < 1e-6:  # Pendiente casi plana
                return None
                
            current_time = time.time()
            breach_time = (threshold - intercept) / slope
            
            if current_time <= breach_time <= current_time + horizon:
                return breach_time - current_time
                
        except Exception as e:
            self.logger.error(f"Error prediciendo umbral: {e}")
            
        return None
    
    def get_dynamic_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula umbrales dinámicos basados en el comportamiento histórico.
        
        Returns:
            Diccionario con umbrales por métrica
        """
        thresholds = {}
        
        for metric, history in [
            ('memory', self.memory_history),
            ('latency', self.latency_history),
            ('hit_rate', self.hit_rate_history)
        ]:
            if not history:
                continue
                
            values = [v for _, v in history]
            mean = np.mean(values)
            std = np.std(values)
            
            thresholds[metric] = {
                'warning': mean + 2 * std,
                'critical': mean + 3 * std,
                'baseline': mean,
                'lower_warning': mean - 2 * std,
                'lower_critical': mean - 3 * std
            }
            
        return thresholds


class MarkovChainPredictor:
    """
    Predictor basado en cadenas de Markov para modelar transiciones entre claves.
    """

    def __init__(self, order: int = 2, decay_factor: float = 0.95):
        """
        Inicializa el predictor de cadenas de Markov.

        Args:
            order: Orden de la cadena de Markov (longitud del contexto)
            decay_factor: Factor de decaimiento para dar más peso a eventos recientes
        """
        self.order = order
        self.decay_factor = decay_factor
        self.transitions: Dict[tuple, Counter] = defaultdict(Counter)
        self.last_update_time: Dict[tuple, float] = {}
        self.sequence: List[Any] = []

    def update(self, key: Any) -> None:
        """
        Actualiza el modelo con una nueva clave.

        Args:
            key: La clave a la que se accedió
        """
        self.sequence.append(key)

        # Mantener la secuencia con longitud limitada
        if len(self.sequence) > self.order * 10:
            self.sequence = self.sequence[-self.order * 10:]

        # Actualizar transiciones para diferentes órdenes
        for o in range(1, min(self.order + 1, len(self.sequence))):
            if len(self.sequence) <= o:
                continue

            context = tuple(self.sequence[-(o+1):-1])
            target = self.sequence[-1]

            # Aplicar decaimiento a las transiciones anteriores
            current_time = time.time()
            if context in self.last_update_time:
                time_diff = current_time - self.last_update_time[context]
                if time_diff > 0.1:  # Evitar decaimiento en actualizaciones muy cercanas
                    decay = self.decay_factor ** (time_diff * 10)  # Normalizar a unidades de ~0.1 segundos
                    for k in self.transitions[context]:
                        self.transitions[context][k] *= decay

            # Actualizar la transición actual
            self.transitions[context][target] += 1
            self.last_update_time[context] = current_time

    def predict_next(self, k: int = 3) -> List[Tuple[Any, float]]:
        """
        Predice las próximas k claves más probables.

        Args:
            k: Número de predicciones a devolver

        Returns:
            Lista de tuplas (clave, probabilidad) ordenadas por probabilidad
        """
        if len(self.sequence) < 1:
            return []

        candidates = Counter()

        # Considerar diferentes órdenes para la predicción
        for o in range(1, min(self.order + 1, len(self.sequence) + 1)):
            if len(self.sequence) < o:
                continue

            context = tuple(self.sequence[-o:])

            # Buscar el contexto más largo que coincida
            while len(context) > 0 and context not in self.transitions:
                context = context[1:]

            if not context:
                continue

            # Calcular probabilidades
            total = sum(self.transitions[context].values())
            if total > 0:
                weight = o / self.order  # Dar más peso a contextos más largos
                for target, count in self.transitions[context].items():
                    candidates[target] += (count / total) * weight

        # Devolver las k predicciones más probables
        return candidates.most_common(k)


class FrequencyBasedPredictor:
    """
    Predictor basado en frecuencias de acceso con ventanas temporales.
    """

    def __init__(self, window_sizes: List[int] = [10, 100, 1000], weights: List[float] = [0.6, 0.3, 0.1]):
        """
        Inicializa el predictor basado en frecuencias.

        Args:
            window_sizes: Tamaños de ventanas para diferentes períodos de tiempo
            weights: Pesos para cada ventana (deben sumar 1)
        """
        if len(window_sizes) != len(weights):
            raise ValueError("window_sizes y weights deben tener la misma longitud")

        if abs(sum(weights) - 1.0) > 0.001:
            raise ValueError("La suma de weights debe ser 1")

        self.window_sizes = window_sizes
        self.weights = weights
        self.access_history: List[Any] = []
        self.last_access_time: Dict[Any, float] = {}

    def update(self, key: Any) -> None:
        """
        Actualiza el modelo con una nueva clave.

        Args:
            key: La clave a la que se accedió
        """
        self.access_history.append(key)
        self.last_access_time[key] = time.time()

        # Limitar el tamaño del historial
        max_window = max(self.window_sizes)
        if len(self.access_history) > max_window * 2:
            self.access_history = self.access_history[-max_window:]

    def predict_next(self, k: int = 3) -> List[Tuple[Any, float]]:
        """
        Predice las próximas k claves más probables.

        Args:
            k: Número de predicciones a devolver

        Returns:
            Lista de tuplas (clave, probabilidad) ordenadas por probabilidad
        """
        if not self.access_history:
            return []

        scores = Counter()

        # Calcular frecuencias en diferentes ventanas
        for i, window_size in enumerate(self.window_sizes):
            if len(self.access_history) < window_size:
                window = self.access_history
            else:
                window = self.access_history[-window_size:]

            # Contar frecuencias en esta ventana
            window_counts = Counter(window)
            total = len(window)

            # Añadir puntuaciones ponderadas
            weight = self.weights[i]
            for key, count in window_counts.items():
                scores[key] += (count / total) * weight

        # Ajustar por recencia (dar bonus a elementos accedidos recientemente)
        current_time = time.time()
        for key in scores:
            if key in self.last_access_time:
                time_diff = current_time - self.last_access_time[key]
                recency_factor = math.exp(-time_diff / 60)  # Decae exponencialmente con el tiempo (minutos)
                scores[key] *= (1 + recency_factor)

        # Devolver las k predicciones más probables
        return scores.most_common(k)


class AssociationRulePredictor:
    """
    Predictor basado en reglas de asociación para identificar patrones complejos.
    """

    def __init__(self, window_size: int = 5, min_support: float = 0.1, min_confidence: float = 0.5):
        """
        Inicializa el predictor basado en reglas de asociación.

        Args:
            window_size: Tamaño de la ventana para buscar asociaciones
            min_support: Soporte mínimo para considerar un conjunto frecuente
            min_confidence: Confianza mínima para considerar una regla válida
        """
        self.window_size = window_size
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions: List[Set[Any]] = []
        self.current_transaction: Set[Any] = set()
        self.last_key: Optional[Any] = None
        self.rules: Dict[frozenset, Dict[Any, float]] = {}
        self.update_rules_counter = 0

    def update(self, key: Any) -> None:
        """
        Actualiza el modelo con una nueva clave.

        Args:
            key: La clave a la que se accedió
        """
        # Si es la misma clave que la última, no hacer nada
        if key == self.last_key:
            return

        self.last_key = key

        # Añadir a la transacción actual
        self.current_transaction.add(key)

        # Si la transacción alcanza el tamaño de ventana, guardarla y crear una nueva
        if len(self.current_transaction) >= self.window_size:
            self.transactions.append(self.current_transaction)
            self.current_transaction = set([key])  # Comenzar nueva transacción con la clave actual

            # Limitar el número de transacciones
            if len(self.transactions) > 1000:
                self.transactions = self.transactions[-1000:]

            # Actualizar reglas periódicamente
            self.update_rules_counter += 1
            if self.update_rules_counter >= 10:
                self._update_association_rules()
                self.update_rules_counter = 0

    def _update_association_rules(self) -> None:
        """Actualiza las reglas de asociación basadas en las transacciones actuales."""
        if len(self.transactions) < 5:
            return

        # Paso 1: Encontrar conjuntos de elementos frecuentes
        item_counts = Counter()
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1

        # Filtrar por soporte mínimo
        n_transactions = len(self.transactions)
        frequent_items = {item for item, count in item_counts.items()
                         if count / n_transactions >= self.min_support}

        # Paso 2: Generar reglas de asociación
        self.rules = {}

        for transaction in self.transactions:
            # Considerar solo elementos frecuentes en esta transacción
            items = transaction.intersection(frequent_items)

            # Generar todas las posibles reglas X -> Y
            for item in items:
                antecedent = frozenset(items - {item})
                if not antecedent:
                    continue

                if antecedent not in self.rules:
                    self.rules[antecedent] = Counter()

                self.rules[antecedent][item] += 1

        # Calcular confianza y filtrar reglas
        for antecedent, consequents in list(self.rules.items()):
            antecedent_count = sum(1 for t in self.transactions if antecedent.issubset(t))

            if antecedent_count == 0:
                del self.rules[antecedent]
                continue

            # Filtrar consecuentes por confianza mínima
            for consequent, count in list(consequents.items()):
                confidence = count / antecedent_count
                if confidence < self.min_confidence:
                    del consequents[consequent]
                else:
                    # Guardar la confianza en lugar del conteo
                    consequents[consequent] = confidence

            # Eliminar reglas sin consecuentes
            if not consequents:
                del self.rules[antecedent]

    def predict_next(self, current_items: Set[Any], k: int = 3) -> List[Tuple[Any, float]]:
        """
        Predice las próximas k claves más probables basadas en los elementos actuales.

        Args:
            current_items: Conjunto de elementos actuales
            k: Número de predicciones a devolver

        Returns:
            Lista de tuplas (clave, confianza) ordenadas por confianza
        """
        if not self.rules:
            return []

        candidates = Counter()

        # Buscar reglas que coincidan con los elementos actuales
        for antecedent, consequents in self.rules.items():
            if antecedent.issubset(current_items):
                for consequent, confidence in consequents.items():
                    if consequent not in current_items:  # Solo predecir elementos que no están en el conjunto actual
                        candidates[consequent] += confidence

        # Devolver las k predicciones más probables
        return candidates.most_common(k)


class EnsemblePredictor:
    """
    Predictor de conjunto que combina múltiples algoritmos de predicción.
    """

    def __init__(self, usage_tracker: UsageTracker):
        """
        Inicializa el predictor de conjunto.

        Args:
            usage_tracker: El rastreador de uso que proporciona datos históricos
        """
        self.usage_tracker = usage_tracker

        # Inicializar predictores individuales
        self.markov_predictor = MarkovChainPredictor(order=3)
        self.frequency_predictor = FrequencyBasedPredictor()
        self.association_predictor = AssociationRulePredictor()

        # Pesos para cada predictor
        self.weights = {
            'markov': 0.5,
            'frequency': 0.3,
            'association': 0.2
        }

        # Historial reciente para la predicción de asociación
        self.recent_keys: Set[Any] = set()
        self.max_recent_keys = 10

    def update(self, key: Any) -> None:
        """
        Actualiza todos los predictores con una nueva clave.

        Args:
            key: La clave a la que se accedió
        """
        self.markov_predictor.update(key)
        self.frequency_predictor.update(key)
        self.association_predictor.update(key)

        # Actualizar conjunto de claves recientes
        self.recent_keys.add(key)
        if len(self.recent_keys) > self.max_recent_keys:
            # Eliminar la clave menos reciente
            oldest_key = None
            oldest_time = float('inf')

            for k in self.recent_keys:
                last_time = self.usage_tracker.get_last_access_time(k)
                if last_time < oldest_time:
                    oldest_time = last_time
                    oldest_key = k

            if oldest_key:
                self.recent_keys.remove(oldest_key)

    def predict_next_keys(self, n: int = 3) -> List[Any]:
        """
        Predice las próximas n claves que probablemente se accederán.

        Args:
            n: Número de claves a predecir

        Returns:
            Lista de claves predichas ordenadas por probabilidad
        """
        # Obtener predicciones de cada predictor
        markov_predictions = self.markov_predictor.predict_next(n * 2)
        frequency_predictions = self.frequency_predictor.predict_next(n * 2)
        association_predictions = self.association_predictor.predict_next(self.recent_keys, n * 2)

        # Combinar predicciones con pesos
        combined_scores = Counter()

        for key, score in markov_predictions:
            combined_scores[key] += score * self.weights['markov']

        for key, score in frequency_predictions:
            combined_scores[key] += score * self.weights['frequency']

        for key, score in association_predictions:
            combined_scores[key] += score * self.weights['association']

        # Devolver las n predicciones con mayor puntuación
        return [key for key, _ in combined_scores.most_common(n)]

    def adapt_weights(self, hit_rates: Dict[str, float]) -> None:
        """
        Adapta los pesos de los predictores basándose en sus tasas de acierto.

        Args:
            hit_rates: Diccionario con las tasas de acierto de cada predictor
        """
        total = sum(hit_rates.values())
        if total > 0:
            # Normalizar para que sumen 1
            self.weights = {k: v / total for k, v in hit_rates.items()}

    def get_prefetch_candidates(self) -> Set[Any]:
        """
        Obtiene un conjunto de claves candidatas para prefetch.

        Returns:
            Conjunto de claves que deberían ser precargadas
        """
        # Obtener predicciones combinadas
        predictions = self.predict_next_keys(5)

        # Añadir algunas predicciones específicas de cada modelo para diversificar
        markov_only = [k for k, _ in self.markov_predictor.predict_next(2)
                      if k not in predictions]
        freq_only = [k for k, _ in self.frequency_predictor.predict_next(2)
                    if k not in predictions and k not in markov_only]

        # Combinar todas las predicciones
        candidates = set(predictions)
        candidates.update(markov_only)
        candidates.update(freq_only)

        return candidates
