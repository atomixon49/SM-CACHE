"""
Pruebas para el componente de predicción del sistema de caché inteligente.
"""
import unittest
from cache_system.usage_tracker import UsageTracker
from cache_system.predictor import CachePredictor


class TestCachePredictor(unittest.TestCase):
    """Pruebas para la clase CachePredictor."""

    def setUp(self):
        """Configuración para cada prueba."""
        self.usage_tracker = UsageTracker()
        self.predictor = CachePredictor(self.usage_tracker, confidence_threshold=0.1)

    def test_pattern_learning(self):
        """Prueba el aprendizaje de patrones de secuencia."""
        # Simular un patrón de acceso A -> B -> C
        keys = ["A", "B", "C"]

        # Repetir el patrón varias veces para que el predictor lo aprenda
        for _ in range(5):
            for key in keys:
                self.usage_tracker.record_access(key)
                self.predictor.update_patterns(key)

        # Verificar que después de A y B, se predice C
        self.predictor.last_n_keys = ["A", "B"]
        predictions = self.predictor.predict_next_keys(1)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0], "C")

    def test_related_keys(self):
        """Prueba la predicción de claves relacionadas."""
        # Simular patrones donde X aparece con diferentes claves
        patterns = [
            ["X", "A", "B"],
            ["C", "X", "D"],
            ["E", "F", "X"],
            ["X", "A", "G"]
        ]

        # Registrar los patrones
        for pattern in patterns:
            for key in pattern:
                self.usage_tracker.record_access(key)
                self.predictor.update_patterns(key)

        # Obtener claves relacionadas con X
        related = self.predictor.predict_related_keys("X", 3)

        # Verificar que A está entre las relacionadas (aparece 2 veces con X)
        self.assertIn("A", related)

    def test_prefetch_candidates(self):
        """Prueba la obtención de candidatos para prefetch."""
        # Simular varios patrones de acceso
        patterns = [
            ["A", "B", "C"],
            ["D", "E", "F"],
            ["A", "B", "G"]
        ]

        # Registrar los patrones varias veces
        for _ in range(3):
            for pattern in patterns:
                for key in pattern:
                    self.usage_tracker.record_access(key)
                    self.predictor.update_patterns(key)

        # Establecer últimas claves accedidas
        self.predictor.last_n_keys = ["A", "B"]

        # Obtener candidatos para prefetch
        candidates = self.predictor.get_prefetch_candidates()

        # Verificar que C y G están entre los candidatos
        self.assertTrue("C" in candidates or "G" in candidates)

    def test_confidence_threshold(self):
        """Prueba el umbral de confianza para predicciones."""
        # Crear predictor con umbral alto
        high_threshold_predictor = CachePredictor(self.usage_tracker, confidence_threshold=0.9)

        # Simular un patrón débil (pocas repeticiones)
        keys = ["P", "Q", "R"]
        for key in keys:
            self.usage_tracker.record_access(key)
            high_threshold_predictor.update_patterns(key)

        # Establecer últimas claves accedidas
        high_threshold_predictor.last_n_keys = ["P", "Q"]

        # No debería predecir nada con un umbral alto y pocas repeticiones
        predictions = high_threshold_predictor.predict_next_keys(1)
        self.assertEqual(len(predictions), 0)

        # Repetir el patrón muchas veces
        for _ in range(10):
            for key in keys:
                self.usage_tracker.record_access(key)
                high_threshold_predictor.update_patterns(key)

        # Ahora debería predecir algo
        predictions = high_threshold_predictor.predict_next_keys(1)
        self.assertEqual(len(predictions), 1)
        # Verificar que la predicción es una de las claves válidas
        self.assertIn(predictions[0], keys)

    def test_multiple_patterns(self):
        """Prueba el manejo de múltiples patrones que se solapan."""
        # Simular dos patrones que comparten un prefijo
        pattern1 = ["A", "B", "C", "D"]
        pattern2 = ["A", "B", "E", "F"]

        # Registrar el primer patrón más veces
        for _ in range(7):
            for key in pattern1:
                self.usage_tracker.record_access(key)
                self.predictor.update_patterns(key)

        # Registrar el segundo patrón menos veces
        for _ in range(3):
            for key in pattern2:
                self.usage_tracker.record_access(key)
                self.predictor.update_patterns(key)

        # Establecer últimas claves accedidas como A, B
        self.predictor.last_n_keys = ["A", "B"]

        # Debería predecir al menos una clave
        predictions = self.predictor.predict_next_keys(2)
        self.assertGreaterEqual(len(predictions), 1)
        # Verificar que las predicciones son claves válidas
        for pred in predictions:
            self.assertIn(pred, ["C", "E"])


if __name__ == '__main__':
    unittest.main()
