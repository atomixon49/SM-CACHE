"""
Pruebas unitarias para el gestor de memoria.
"""
import unittest
from unittest.mock import MagicMock
from cache_system.memory_manager import MemoryManager


class TestMemoryManager(unittest.TestCase):
    """Pruebas para el gestor de memoria."""
    
    def setUp(self):
        """Inicializa el entorno de prueba."""
        self.memory_manager = MemoryManager(max_size_bytes=1024)  # 1KB for testing

    def test_add_remove_item(self):
        """Prueba añadir y eliminar elementos."""
        # Añadir un elemento
        success = self.memory_manager.add_item("key1", "value1")
        self.assertTrue(success)
        self.assertIn("key1", self.memory_manager.item_sizes)
        
        # Eliminar el elemento
        self.memory_manager.remove_item("key1")
        self.assertNotIn("key1", self.memory_manager.item_sizes)

    def test_memory_full_detection(self):
        """Prueba la detección de memoria llena."""
        # Añadir elementos hasta llenar la memoria
        big_data = "x" * 1000  # ~1000 bytes
        success1 = self.memory_manager.add_item("key1", big_data)
        self.assertTrue(success1)
        
        # Intentar añadir más allá del límite
        success2 = self.memory_manager.add_item("key2", big_data)
        self.assertFalse(success2)
        
        self.assertTrue(self.memory_manager.is_memory_full())

    def test_eviction_candidates(self):
        """Prueba la selección de candidatos para evicción."""
        # Añadir varios elementos
        self.memory_manager.add_item("key1", "value1")
        self.memory_manager.access_times["key1"] = 100  # Simular acceso antiguo
        
        self.memory_manager.add_item("key2", "value2")
        self.memory_manager.access_times["key2"] = 200  # Acceso más reciente
        
        candidates = self.memory_manager.get_eviction_candidates(1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0], "key1")  # Debería elegir el menos usado

    def test_memory_usage_stats(self):
        """Prueba las estadísticas de uso de memoria."""
        self.memory_manager.add_item("key1", "value1")
        self.memory_manager.add_item("key2", "value2")
        
        stats = self.memory_manager.get_memory_usage()
        self.assertIn('current_size', stats)
        self.assertIn('max_size', stats)
        self.assertIn('item_count', stats)
        self.assertIn('usage_percent', stats)

    def test_size_estimation(self):
        """Prueba la estimación de tamaño de elementos."""
        # Probar con diferentes tipos de datos
        test_data = [
            ("string", "test"),
            ("number", 42),
            ("list", [1, 2, 3]),
            ("dict", {"a": 1, "b": 2})
        ]
        
        for name, value in test_data:
            size = self.memory_manager._estimate_size(value)
            self.assertGreater(size, 0, f"Size estimation failed for {name}")

    def test_eviction_with_different_sizes(self):
        """Prueba la evicción considerando diferentes tamaños de elementos."""
        # Añadir elementos de diferentes tamaños
        self.memory_manager.add_item("small", "x")
        self.memory_manager.add_item("medium", "x" * 100)
        self.memory_manager.add_item("large", "x" * 500)
        
        # Verificar que los elementos más grandes son candidatos prioritarios
        candidates = self.memory_manager.get_eviction_candidates(2)
        self.assertEqual(len(candidates), 2)
        self.assertIn("large", candidates)  # El elemento más grande debería estar en la lista


if __name__ == '__main__':
    unittest.main()
