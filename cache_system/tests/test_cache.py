"""
Pruebas unitarias para el sistema de caché inteligente.
"""
import unittest
import threading
import time
from ..intelligent_cache import IntelligentCache

class TestIntelligentCache(unittest.TestCase):
    """Pruebas para el caché inteligente."""
    
    def setUp(self):
        """Inicializa el entorno de prueba."""
        self.cache = IntelligentCache(
            max_size=1024,  # 1KB para pruebas
            monitoring_enabled=True,
            persistence_enabled=False,
            distributed_enabled=False,
            prefetch_enabled=False  # Desactivar prefetch para pruebas
        )

    def tearDown(self):
        """Limpia después de cada prueba."""
        self.cache = None

    def test_basic_operations(self):
        """Prueba operaciones básicas de caché."""
        # Almacenar
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Obtener valor inexistente
        self.assertIsNone(self.cache.get("nonexistent"))
        
        # Actualizar valor
        self.cache.put("key1", "new_value")
        self.assertEqual(self.cache.get("key1"), "new_value")

    def test_ttl_expiration(self):
        """Prueba la expiración por tiempo de vida."""
        self.cache = IntelligentCache(
            max_size=1024,
            ttl_enabled=True,
            default_ttl=1  # 1 segundo
        )
        
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Esperar a que expire
        time.sleep(1.1)
        self.assertIsNone(self.cache.get("key1"))

    def test_memory_management(self):
        """Prueba la gestión automática de memoria."""
        # Llenar el caché
        big_value = "x" * 900  # ~900 bytes
        self.cache.put("key1", big_value)
        
        # Intentar añadir más allá del límite
        self.cache.put("key2", big_value)
        
        # Verificar que key1 fue eviccionada
        self.assertIsNone(self.cache.get("key1"))
        self.assertEqual(self.cache.get("key2"), big_value)

    def test_prefetch(self):
        """Prueba la funcionalidad de prefetch."""
        self.cache = IntelligentCache(
            max_size=1024,
            prefetch_enabled=True,
            prediction_threshold=0.5
        )
        
        # Entrenar el predictor
        sequence = ["A", "B", "C"]
        for _ in range(3):
            for key in sequence:
                self.cache.get(key)
                
        # El predictor debería precargar C después de B
        self.cache.put("A", "value_a")
        self.cache.put("B", "value_b")
        self.cache.put("C", "value_c")
        
        self.cache.get("A")
        self.cache.get("B")
        time.sleep(0.1)  # Dar tiempo para prefetch
        
        # Verificar que C está en caché
        self.assertIsNotNone(self.cache.get("C"))

    def test_concurrent_access(self):
        """Prueba acceso concurrente al caché."""
        def worker(tid: int):
            for i in range(100):
                key = f"key{tid}_{i}"
                self.cache.put(key, f"value{tid}_{i}")
                self.cache.get(key)
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()

    def test_clear(self):
        """Prueba la limpieza del caché."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        self.cache.clear()
        
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))

    def test_data_loader(self):
        """Prueba el cargador de datos automático."""
        def loader(key):
            return f"loaded_{key}"
            
        self.cache.set_data_loader(loader)
        
        # Primera carga desde loader
        value = self.cache.get("key1")
        self.assertEqual(value, "loaded_key1")
        
        # Segunda obtención desde caché
        value = self.cache.get("key1")
        self.assertEqual(value, "loaded_key1")

if __name__ == '__main__':
    unittest.main()
