"""
Módulo para algoritmos de aprendizaje avanzados en el sistema de caché inteligente.
"""
from typing import Dict, Any, List, Tuple, Set, Optional, Counter as CounterType
from collections import Counter, defaultdict
import time
import math
from .usage_tracker import UsageTracker


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
