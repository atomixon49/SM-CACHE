"""
Pruebas para el componente de gestión de memoria del sistema de caché inteligente.
"""
import unittest
import time
from cache_system.usage_tracker import UsageTracker
from cache_system.memory_manager import MemoryManager


class TestMemoryManager(unittest.TestCase):
    """Pruebas para la clase MemoryManager."""

    def setUp(self):
        """Configuración para cada prueba."""
        self.usage_tracker = UsageTracker()
        self.memory_manager = MemoryManager(
            self.usage_tracker,
            max_size=5,
            max_memory_mb=0.1  # 100 KB
        )

    def test_add_remove_item(self):
        """Prueba añadir y eliminar elementos."""
        # Añadir elementos
        self.memory_manager.add_item("key1", "value1")
        self.memory_manager.add_item("key2", "value2")

        # Verificar tamaño
        self.assertEqual(self.memory_manager.current_size, 2)

        # Eliminar un elemento
        self.memory_manager.remove_item("key1")

        # Verificar tamaño actualizado
        self.assertEqual(self.memory_manager.current_size, 1)

    def test_memory_full_detection(self):
        """Prueba la detección de memoria llena."""
        # Inicialmente no está lleno
        self.assertFalse(self.memory_manager.is_memory_full())

        # Añadir elementos grandes hasta llenar
        for i in range(10):
            # Cada elemento es aproximadamente 10KB
            large_value = "X" * 10000
            self.memory_manager.add_item(f"key{i}", large_value)

            # Si hemos superado el límite, debería detectarlo
            if i >= 5 or self.memory_manager.current_memory_bytes >= 0.1 * 1024 * 1024:
                self.assertTrue(self.memory_manager.is_memory_full())
                break

    def test_eviction_candidates(self):
        """Prueba la selección de candidatos para evicción."""
        # Añadir elementos
        for i in range(5):
            self.memory_manager.add_item(f"key{i}", f"value{i}")
            self.usage_tracker.record_access(f"key{i}")

        # Acceder más a algunos elementos
        for _ in range(5):
            self.usage_tracker.record_access("key0")
            self.usage_tracker.record_access("key1")

        # Esperar un poco para que key4 sea el más antiguo
        time.sleep(0.1)

        # Obtener candidatos para evicción
        candidates = self.memory_manager.get_eviction_candidates(2)

        # Los candidatos deberían ser los menos accedidos y más antiguos
        self.assertEqual(len(candidates), 2)
        # Verificar que los candidatos son claves válidas (no key0 o key1 que fueron accedidos más veces)
        for candidate in candidates:
            self.assertNotIn(candidate, ["key0", "key1"])

    def test_memory_usage_stats(self):
        """Prueba las estadísticas de uso de memoria."""
        # Añadir algunos elementos
        for i in range(3):
            self.memory_manager.add_item(f"key{i}", "X" * 1000)

        # Obtener estadísticas
        stats = self.memory_manager.get_memory_usage_stats()

        # Verificar campos de estadísticas
        self.assertEqual(stats['current_size'], 3)
        self.assertEqual(stats['max_size'], 5)
        self.assertGreater(stats['current_memory_mb'], 0)
        self.assertEqual(stats['max_memory_mb'], 0.1)
        self.assertGreater(stats['memory_usage_percent'], 0)
        self.assertEqual(stats['size_usage_percent'], 60.0)  # 3/5 * 100

    def test_size_estimation(self):
        """Prueba la estimación de tamaño de elementos."""
        # Definir un estimador personalizado
        def custom_estimator(value):
            if isinstance(value, str):
                return len(value) * 2  # Cada carácter cuenta como 2 bytes
            return 100  # Valor por defecto

        # Crear gestor con estimador personalizado
        memory_manager = MemoryManager(
            self.usage_tracker,
            max_size=10,
            max_memory_mb=1.0,
            size_estimator=custom_estimator
        )

        # Añadir un elemento
        memory_manager.add_item("key", "abcde")  # 5 caracteres

        # Verificar tamaño estimado (5 * 2 = 10 bytes)
        self.assertEqual(memory_manager.item_sizes["key"], 10)

    def test_eviction_with_different_sizes(self):
        """Prueba la evicción considerando diferentes tamaños de elementos."""
        # Añadir elementos de diferentes tamaños
        self.memory_manager.add_item("small", "x")
        self.memory_manager.add_item("medium", "x" * 1000)
        self.memory_manager.add_item("large", "x" * 10000)

        # Registrar accesos (todos con la misma frecuencia)
        for key in ["small", "medium", "large"]:
            self.usage_tracker.record_access(key)

        # Obtener candidato para evicción
        candidates = self.memory_manager.get_eviction_candidates(1)

        # El elemento más grande debería ser el candidato preferido
        self.assertEqual(candidates[0], "large")


if __name__ == '__main__':
    unittest.main()
