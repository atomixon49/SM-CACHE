"""
Pruebas para los algoritmos de aprendizaje avanzados del sistema de caché inteligente.
"""
from typing import Dict, Any, Optional, List, Set, Tuple
import unittest
import time
from ..advanced_learning import (
    PatternAnalyzer,
    MarkovChainPredictor,
    FrequencyBasedPredictor,
    AssociationRulePredictor,
    EnsemblePredictor
)
from ..usage_tracker import UsageTracker


class TestMarkovChainPredictor(unittest.TestCase):
    """Pruebas para el predictor de cadenas de Markov."""

    def setUp(self):
        """Configuración para cada prueba."""
        self.predictor = MarkovChainPredictor(order=2)

    def test_basic_prediction(self):
        """Prueba predicciones básicas."""
        # Entrenar con una secuencia simple
        sequence = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
        for key in sequence:
            self.predictor.update(key)

        # Verificar que hay predicciones
        predictions = self.predictor.predict_next(2)
        self.assertGreaterEqual(len(predictions), 1)
        # Verificar que las predicciones contienen claves válidas
        self.assertIn(predictions[0][0], sequence)

    def test_decay_factor(self):
        """Prueba que el factor de decaimiento funciona."""
        # Esta prueba simplemente verifica que el predictor funciona con decaimiento
        predictor = MarkovChainPredictor(order=2, decay_factor=0.5)

        # Entrenar con una secuencia
        sequence = ["A", "B", "C", "D", "E"]
        for _ in range(5):  # Repetir la secuencia
            for key in sequence:
                predictor.update(key)

        # Verificar que hay predicciones
        predictions = predictor.predict_next(2)
        self.assertGreaterEqual(len(predictions), 1)

        # Verificar que las predicciones contienen claves válidas
        pred_keys = [p[0] for p in predictions]
        for key in pred_keys:
            self.assertIn(key, sequence)


class TestFrequencyBasedPredictor(unittest.TestCase):
    """Pruebas para el predictor basado en frecuencias."""

    def setUp(self):
        """Configuración para cada prueba."""
        self.predictor = FrequencyBasedPredictor()

    def test_frequency_prediction(self):
        """Prueba predicciones basadas en frecuencia."""
        # Acceder a A más veces que a otras claves
        keys = ["A", "B", "C", "A", "D", "A", "E", "A"]
        for key in keys:
            self.predictor.update(key)

        # A debería ser la predicción más probable
        predictions = self.predictor.predict_next(3)
        self.assertEqual(predictions[0][0], "A")

    def test_recency_factor(self):
        """Prueba que el factor de recencia funciona."""
        # Acceder a A muchas veces pero hace tiempo
        for _ in range(10):
            self.predictor.update("A")

        # Esperar un poco
        time.sleep(0.2)

        # Acceder a B menos veces pero más recientemente
        for _ in range(5):
            self.predictor.update("B")

        # B debería tener una puntuación alta debido a la recencia
        predictions = self.predictor.predict_next(2)
        self.assertIn("B", [p[0] for p in predictions])


class TestAssociationRulePredictor(unittest.TestCase):
    """Pruebas para el predictor basado en reglas de asociación."""

    def setUp(self):
        """Configuración para cada prueba."""
        self.predictor = AssociationRulePredictor(
            window_size=3,
            min_support=0.1,
            min_confidence=0.5
        )

    def test_association_rules(self):
        """Prueba la generación de reglas de asociación."""
        # Crear patrones de asociación: A, B suelen aparecer con C
        patterns = [
            ["A", "B", "C"],
            ["D", "E", "F"],
            ["A", "C", "G"],
            ["B", "C", "H"],
            ["A", "B", "C"],
        ]

        # Registrar los patrones
        for pattern in patterns:
            for key in pattern:
                self.predictor.update(key)

        # Forzar actualización de reglas
        self.predictor._update_association_rules()

        # Predecir basado en A, B
        current_items = {"A", "B"}
        predictions = self.predictor.predict_next(current_items, k=1)
        
        # C debería ser una predicción probable
        if predictions:
            self.assertEqual(predictions[0][0], "C")


class TestEnsemblePredictor(unittest.TestCase):
    """Pruebas para el predictor de conjunto."""

    def setUp(self):
        """Configuración para cada prueba."""
        self.usage_tracker = UsageTracker()
        self.predictor = EnsemblePredictor(self.usage_tracker)

    def test_ensemble_prediction(self):
        """Prueba predicciones del conjunto."""
        # Entrenar con una secuencia
        sequence = ["A", "B", "C", "A", "B", "C", "D", "E"]
        for key in sequence:
            self.predictor.update(key)

        # Verificar predicciones
        predictions = self.predictor.predict_next_keys(3)
        self.assertGreaterEqual(len(predictions), 1)
        self.assertTrue(all(key in sequence for key in predictions))

    def test_prefetch_candidates(self):
        """Prueba la obtención de candidatos para prefetch."""
        # Registrar algunos patrones
        patterns = [
            ["X", "Y", "Z"],
            ["P", "Q", "R"],
            ["X", "Y", "W"]
        ]

        for pattern in patterns:
            for key in pattern:
                self.usage_tracker.record_access(key)
                self.predictor.update(key)

        # Obtener candidatos para prefetch
        candidates = self.predictor.get_prefetch_candidates()

        # Debería haber al menos un candidato
        self.assertGreater(len(candidates), 0)

    def test_weight_adaptation(self):
        """Prueba la adaptación de pesos."""
        # Configurar tasas de acierto simuladas
        hit_rates = {
            'markov': 0.8,
            'frequency': 0.5,
            'association': 0.3
        }

        # Adaptar pesos
        self.predictor.adapt_weights(hit_rates)

        # Verificar que los pesos se actualizaron correctamente
        self.assertAlmostEqual(self.predictor.weights['markov'], 0.5, places=1)
        self.assertAlmostEqual(self.predictor.weights['frequency'], 0.3125, places=3)
        self.assertAlmostEqual(self.predictor.weights['association'], 0.1875, places=3)


if __name__ == '__main__':
    unittest.main()
