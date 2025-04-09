"""
Pruebas para la persistencia del sistema de caché inteligente.
"""
import unittest
import os
import time
import shutil
import tempfile
from cache_system.persistence import CachePersistence
from cache_system import IntelligentCache


class TestCachePersistence(unittest.TestCase):
    """Pruebas para la clase CachePersistence."""

    def setUp(self):
        """Configuración para cada prueba."""
        # Crear directorio temporal para las pruebas
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = CachePersistence(storage_dir=self.temp_dir)
        self.persistence.set_cache_name("test_cache")

    def tearDown(self):
        """Limpieza después de cada prueba."""
        # Eliminar directorio temporal
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_load_cache(self):
        """Prueba guardar y cargar el caché."""
        # Datos de prueba
        cache = {"key1": "value1", "key2": "value2"}
        expiry_times = {"key1": time.time() + 100}
        metadata = {"test_meta": "test_value"}

        # Guardar el caché
        success = self.persistence.save_cache(cache, expiry_times, metadata)
        self.assertTrue(success)

        # Verificar que se crearon los archivos
        cache_file = os.path.join(self.temp_dir, "test_cache.json")
        meta_file = os.path.join(self.temp_dir, "test_cache.meta.json")
        self.assertTrue(os.path.exists(cache_file))
        self.assertTrue(os.path.exists(meta_file))

        # Cargar el caché
        loaded_cache, loaded_expiry, loaded_meta = self.persistence.load_cache()

        # Verificar que los datos se cargaron correctamente
        self.assertEqual(len(loaded_cache), 2)
        self.assertEqual(loaded_cache["key1"], "value1")
        self.assertEqual(loaded_cache["key2"], "value2")
        self.assertEqual(len(loaded_expiry), 1)
        self.assertIn("key1", loaded_expiry)
        self.assertEqual(loaded_meta["test_meta"], "test_value")

    def test_clear_cache_files(self):
        """Prueba eliminar archivos de caché."""
        # Guardar un caché primero
        cache = {"key1": "value1"}
        self.persistence.save_cache(cache, {}, {})

        # Verificar que se crearon los archivos
        cache_file = os.path.join(self.temp_dir, "test_cache.json")
        meta_file = os.path.join(self.temp_dir, "test_cache.meta.json")
        self.assertTrue(os.path.exists(cache_file))
        self.assertTrue(os.path.exists(meta_file))

        # Eliminar archivos
        success = self.persistence.clear_cache_files()
        self.assertTrue(success)

        # Verificar que se eliminaron los archivos
        self.assertFalse(os.path.exists(cache_file))
        self.assertFalse(os.path.exists(meta_file))

    def test_get_cache_info(self):
        """Prueba obtener información del caché."""
        # Guardar un caché primero
        cache = {"key1": "value1", "key2": "value2"}
        metadata = {"test_meta": "test_value"}
        self.persistence.save_cache(cache, {}, metadata)

        # Obtener información
        info = self.persistence.get_cache_info()

        # Verificar información
        self.assertEqual(info["item_count"], 2)
        self.assertEqual(info["test_meta"], "test_value")
        self.assertIn("file_size_bytes", info)
        self.assertIn("file_size_mb", info)


class TestIntelligentCachePersistence(unittest.TestCase):
    """Pruebas para la persistencia en IntelligentCache."""

    def setUp(self):
        """Configuración para cada prueba."""
        # Crear directorio temporal para las pruebas
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Limpieza después de cada prueba."""
        # Eliminar directorio temporal
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_persistence(self):
        """Prueba la persistencia en el caché inteligente."""
        # Crear caché con persistencia
        cache = IntelligentCache(
            max_size=10,
            persistence_enabled=True,
            persistence_dir=self.temp_dir,
            cache_name="test_intelligent_cache"
        )

        # Añadir algunos elementos
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Guardar explícitamente
        success = cache.save()
        self.assertTrue(success)

        # Verificar que se crearon los archivos
        cache_file = os.path.join(self.temp_dir, "test_intelligent_cache.json")
        meta_file = os.path.join(self.temp_dir, "test_intelligent_cache.meta.json")
        self.assertTrue(os.path.exists(cache_file))
        self.assertTrue(os.path.exists(meta_file))

        # Detener el caché
        cache.stop()

        # Crear un nuevo caché que cargue desde disco
        new_cache = IntelligentCache(
            max_size=10,
            persistence_enabled=True,
            persistence_dir=self.temp_dir,
            cache_name="test_intelligent_cache"
        )

        # Verificar que los elementos se cargaron
        self.assertEqual(new_cache.get("key1"), "value1")
        self.assertEqual(new_cache.get("key2"), "value2")
        self.assertEqual(new_cache.get("key3"), "value3")

        # Verificar estadísticas
        stats = new_cache.get_stats()
        self.assertEqual(stats["size"], 3)
        self.assertTrue(stats["persistence_enabled"])
        self.assertIn("persistence", stats)

        # Detener el nuevo caché
        new_cache.stop()

    def test_auto_save(self):
        """Prueba el guardado automático."""
        # Crear caché con guardado automático
        cache = IntelligentCache(
            max_size=10,
            persistence_enabled=True,
            persistence_dir=self.temp_dir,
            cache_name="test_auto_save",
            auto_save_interval=1  # 1 segundo
        )

        # Añadir algunos elementos
        cache.put("key1", "value1")

        # Guardar explícitamente para asegurar que se crea el archivo
        success = cache.save()
        self.assertTrue(success)

        # Verificar que se crearon los archivos
        cache_file = os.path.join(self.temp_dir, "test_auto_save.json")
        self.assertTrue(os.path.exists(cache_file))

        # Añadir más elementos
        cache.put("key2", "value2")

        # Guardar explícitamente de nuevo
        cache.save()

        # Detener el caché
        cache.stop()

        # Crear un nuevo caché que cargue desde disco
        new_cache = IntelligentCache(
            max_size=10,
            persistence_enabled=True,
            persistence_dir=self.temp_dir,
            cache_name="test_auto_save"
        )

        # Verificar que todos los elementos se cargaron
        self.assertEqual(new_cache.get("key1"), "value1")
        self.assertEqual(new_cache.get("key2"), "value2")

        # Detener el nuevo caché
        new_cache.stop()


if __name__ == '__main__':
    unittest.main()
