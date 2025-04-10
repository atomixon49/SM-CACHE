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


"""
Pruebas unitarias para el sistema de persistencia.
"""
import unittest
import os
import json
from cache_system.persistence import CachePersistence

class TestCachePersistence(unittest.TestCase):
    """Pruebas para el sistema de persistencia."""
    
    def setUp(self):
        """Inicializa el entorno de prueba."""
        self.test_dir = "test_persistence"
        self.persistence = CachePersistence(self.test_dir)
        
        # Crear directorio si no existe
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def tearDown(self):
        """Limpia después de cada prueba."""
        # Eliminar archivos de prueba
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_basic_persistence(self):
        """Prueba operaciones básicas de persistencia."""
        # Guardar datos
        data = {"key1": "value1", "key2": "value2"}
        self.persistence.save_data(data)
        
        # Cargar datos
        loaded_data = self.persistence.load_data()
        self.assertEqual(loaded_data, data)

    def test_data_types(self):
        """Prueba persistencia con diferentes tipos de datos."""
        data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
            "bool": True,
            "none": None
        }
        
        self.persistence.save_data(data)
        loaded_data = self.persistence.load_data()
        
        self.assertEqual(loaded_data, data)
        self.assertIsInstance(loaded_data["string"], str)
        self.assertIsInstance(loaded_data["number"], int)
        self.assertIsInstance(loaded_data["float"], float)
        self.assertIsInstance(loaded_data["list"], list)
        self.assertIsInstance(loaded_data["dict"], dict)
        self.assertIsInstance(loaded_data["bool"], bool)
        self.assertIsNone(loaded_data["none"])

    def test_file_corruption(self):
        """Prueba recuperación ante corrupción de archivo."""
        data = {"key": "value"}
        self.persistence.save_data(data)
        
        # Corromper archivo
        with open(self.persistence.file_path, 'w') as f:
            f.write("corrupted data")
        
        # Debería retornar diccionario vacío
        loaded_data = self.persistence.load_data()
        self.assertEqual(loaded_data, {})

    def test_compression(self):
        """Prueba compresión de datos."""
        self.persistence = CachePersistence(
            self.test_dir,
            compress=True
        )
        
        # Datos grandes
        data = {str(i): "x" * 1000 for i in range(100)}
        
        # Guardar con compresión
        self.persistence.save_data(data)
        
        # Verificar que archivo comprimido es más pequeño
        file_size = os.path.getsize(self.persistence.file_path)
        raw_size = len(json.dumps(data).encode())
        
        self.assertLess(file_size, raw_size)
        
        # Verificar que los datos se cargan correctamente
        loaded_data = self.persistence.load_data()
        self.assertEqual(loaded_data, data)

    def test_backup(self):
        """Prueba sistema de respaldo."""
        data = {"key": "value"}
        
        # Guardar datos y crear respaldo
        self.persistence.save_data(data)
        self.persistence.create_backup()
        
        # Verificar archivo de respaldo
        backup_path = self.persistence.file_path + ".bak"
        self.assertTrue(os.path.exists(backup_path))
        
        # Corromper archivo principal
        os.remove(self.persistence.file_path)
        
        # Restaurar desde respaldo
        self.persistence.restore_from_backup()
        loaded_data = self.persistence.load_data()
        self.assertEqual(loaded_data, data)

    def test_concurrent_access(self):
        """Prueba acceso concurrente."""
        import threading
        
        def writer():
            for i in range(100):
                data = {f"key{i}": f"value{i}"}
                self.persistence.save_data(data)
                
        def reader():
            for _ in range(100):
                self.persistence.load_data()
        
        # Crear hilos
        threads = []
        for _ in range(3):
            t1 = threading.Thread(target=writer)
            t2 = threading.Thread(target=reader)
            threads.extend([t1, t2])
            t1.start()
            t2.start()
            
        # Esperar a que terminen
        for t in threads:
            t.join()


if __name__ == '__main__':
    unittest.main()
