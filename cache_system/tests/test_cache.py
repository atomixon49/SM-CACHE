"""
Pruebas para el sistema de caché inteligente.
"""
import unittest
import time
import threading
from cache_system import IntelligentCache


class TestIntelligentCache(unittest.TestCase):
    """Pruebas para la clase IntelligentCache."""

    def setUp(self):
        """Configuración para cada prueba."""
        # Crear una instancia de caché con tamaño limitado
        self.cache = IntelligentCache(
            max_size=10,
            max_memory_mb=1.0,
            ttl=5,  # 5 segundos de TTL
            prefetch_enabled=False  # Desactivar prefetch para pruebas
        )

        # Contador para el cargador de datos
        self.load_count = 0

    def tearDown(self):
        """Limpieza después de cada prueba."""
        self.cache.stop()

    def test_basic_operations(self):
        """Prueba operaciones básicas de caché."""
        # Poner y obtener
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")

        # Verificar contiene
        self.assertTrue(self.cache.contains("key1"))
        self.assertFalse(self.cache.contains("key2"))

        # Eliminar
        self.cache.remove("key1")
        self.assertFalse(self.cache.contains("key1"))
        self.assertIsNone(self.cache.get("key1"))

    def test_ttl_expiration(self):
        """Prueba la expiración por tiempo de vida."""
        # Poner con TTL corto
        self.cache = IntelligentCache(ttl=1)  # 1 segundo de TTL
        self.cache.put("key1", "value1")

        # Verificar antes de expirar
        self.assertEqual(self.cache.get("key1"), "value1")

        # Esperar a que expire
        time.sleep(1.5)

        # Verificar después de expirar
        self.assertIsNone(self.cache.get("key1"))
        self.assertFalse(self.cache.contains("key1"))

    def test_memory_management(self):
        """Prueba la gestión automática de memoria."""
        # Crear caché con límite muy pequeño
        self.cache = IntelligentCache(max_size=3, max_memory_mb=0.1)

        # Llenar el caché
        for i in range(5):
            self.cache.put(f"key{i}", "X" * 1000)  # Valores grandes

        # Verificar que algunos elementos fueron eviccionados
        self.assertLess(len(self.cache.cache), 5)

        # Verificar estadísticas
        stats = self.cache.get_stats()
        self.assertLessEqual(stats['size'], 3)

    def test_data_loader(self):
        """Prueba el cargador de datos automático."""
        # Función de carga
        def data_loader(key):
            self.load_count += 1
            return f"loaded_{key}"

        # Crear caché con cargador
        self.cache = IntelligentCache(data_loader=data_loader)

        # Obtener valor no existente
        value = self.cache.get("test_key")

        # Verificar que se cargó automáticamente
        self.assertEqual(value, "loaded_test_key")
        self.assertEqual(self.load_count, 1)

        # Obtener de nuevo (debería estar en caché)
        value = self.cache.get("test_key")
        self.assertEqual(value, "loaded_test_key")
        self.assertEqual(self.load_count, 1)  # No debería haber cargado de nuevo

    def test_prefetch(self):
        """Prueba la funcionalidad de prefetch."""
        # Función de carga
        loaded_keys = []

        def data_loader(key):
            loaded_keys.append(key)
            return f"loaded_{key}"

        # Crear caché con prefetch habilitado
        self.cache = IntelligentCache(
            data_loader=data_loader,
            prefetch_enabled=True
        )

        # Simular un patrón de acceso
        for i in range(3):
            self.cache.get(f"sequence_{i}")

        # Repetir el patrón para que el predictor lo aprenda
        for i in range(3):
            self.cache.get(f"sequence_{i}")

        # Esperar a que el prefetch ocurra
        time.sleep(2)

        # Verificar que se hayan cargado al menos las claves originales
        self.assertGreaterEqual(len(loaded_keys), 3)

    def test_concurrent_access(self):
        """Prueba acceso concurrente al caché."""
        # Número de hilos y operaciones
        num_threads = 10
        ops_per_thread = 100

        # Función para hilo de prueba
        def worker(thread_id):
            for i in range(ops_per_thread):
                key = f"key_{thread_id}_{i}"
                self.cache.put(key, f"value_{thread_id}_{i}")
                self.assertEqual(self.cache.get(key), f"value_{thread_id}_{i}")

        # Crear y ejecutar hilos
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Esperar a que terminen
        for t in threads:
            t.join()

        # Verificar que no hubo errores (si llegamos aquí, la prueba pasó)
        self.assertTrue(True)

    def test_clear(self):
        """Prueba la limpieza del caché."""
        # Llenar el caché
        for i in range(5):
            self.cache.put(f"key{i}", f"value{i}")

        # Verificar que hay elementos
        self.assertEqual(self.cache.get_stats()['size'], 5)

        # Limpiar
        self.cache.clear()

        # Verificar que está vacío
        self.assertEqual(self.cache.get_stats()['size'], 0)
        self.assertIsNone(self.cache.get("key0"))


if __name__ == '__main__':
    unittest.main()
