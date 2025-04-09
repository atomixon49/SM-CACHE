"""
Pruebas para el caché distribuido del sistema de caché inteligente.
"""
import unittest
import time
import threading
from cache_system.distributed import Node, Message, DistributedCache
from cache_system import IntelligentCache


class TestNode(unittest.TestCase):
    """Pruebas para la clase Node."""
    
    def test_node_creation(self):
        """Prueba la creación de nodos."""
        # Crear nodo con ID automático
        node1 = Node("localhost", 5000)
        self.assertEqual(node1.host, "localhost")
        self.assertEqual(node1.port, 5000)
        self.assertIsNotNone(node1.node_id)
        
        # Crear nodo con ID específico
        node2 = Node("localhost", 5001, "test_node")
        self.assertEqual(node2.node_id, "test_node")
        
        # Verificar igualdad
        node3 = Node("localhost", 5000, node1.node_id)
        self.assertEqual(node1, node3)
        self.assertNotEqual(node1, node2)


class TestMessage(unittest.TestCase):
    """Pruebas para la clase Message."""
    
    def test_message_serialization(self):
        """Prueba la serialización y deserialización de mensajes."""
        # Crear nodo y mensaje
        node = Node("localhost", 5000, "test_node")
        data = {"key": "value", "number": 123}
        message = Message(Message.TYPE_SYNC, node, data)
        
        # Serializar a JSON
        json_str = message.to_json()
        self.assertIsInstance(json_str, str)
        
        # Deserializar desde JSON
        deserialized = Message.from_json(json_str)
        self.assertEqual(deserialized.type, Message.TYPE_SYNC)
        self.assertEqual(deserialized.sender.node_id, "test_node")
        self.assertEqual(deserialized.data["key"], "value")
        self.assertEqual(deserialized.data["number"], 123)


class TestDistributedCache(unittest.TestCase):
    """Pruebas para la clase DistributedCache."""
    
    def setUp(self):
        """Configuración para cada prueba."""
        # Crear caché distribuido
        self.cache = DistributedCache(
            host="localhost",
            port=5000,
            node_id="test_node",
            sync_interval=1,
            heartbeat_interval=1
        )
        
        # Configurar callbacks
        self.invalidated_keys = []
        self.requested_keys = []
        self.sync_requested_count = 0
        self.sync_received_data = None
        
        self.cache.on_key_invalidated = lambda key: self.invalidated_keys.append(key)
        self.cache.on_key_requested = lambda key: (self.requested_keys.append(key), ("value_for_" + key, True))
        self.cache.on_sync_requested = lambda: {"key1": "value1", "key2": "value2"}
        self.cache.on_sync_received = lambda data: setattr(self, "sync_received_data", data)
        
    def tearDown(self):
        """Limpieza después de cada prueba."""
        if self.cache.running:
            self.cache.stop()
    
    def test_start_stop(self):
        """Prueba iniciar y detener el caché distribuido."""
        # Iniciar
        success = self.cache.start()
        self.assertTrue(success)
        self.assertTrue(self.cache.running)
        
        # Verificar que los hilos están ejecutándose
        self.assertIsNotNone(self.cache.server_thread)
        self.assertTrue(self.cache.server_thread.is_alive())
        
        # Detener
        self.cache.stop()
        self.assertFalse(self.cache.running)
        
        # Verificar que los hilos se detuvieron
        time.sleep(0.1)  # Dar tiempo a que los hilos terminen
        self.assertFalse(self.cache.server_thread.is_alive())
    
    def test_cluster_info(self):
        """Prueba obtener información del cluster."""
        # Iniciar caché
        self.cache.start()
        
        # Obtener información
        info = self.cache.get_cluster_info()
        
        # Verificar información
        self.assertEqual(info["local_node"]["node_id"], "test_node")
        self.assertEqual(info["local_node"]["host"], "localhost")
        self.assertEqual(info["local_node"]["port"], 5000)
        self.assertEqual(info["node_count"], 1)  # Solo el nodo local
        self.assertTrue(info["running"])
        
        # Detener caché
        self.cache.stop()


class TestIntelligentCacheDistributed(unittest.TestCase):
    """Pruebas para la integración de caché distribuido en IntelligentCache."""
    
    def setUp(self):
        """Configuración para cada prueba."""
        # Crear caché con distribución habilitada
        self.cache = IntelligentCache(
            max_size=10,
            distributed_enabled=True,
            distributed_host="localhost",
            distributed_port=5100
        )
        
    def tearDown(self):
        """Limpieza después de cada prueba."""
        self.cache.stop()
    
    def test_distributed_enabled(self):
        """Prueba que el caché distribuido está habilitado."""
        self.assertTrue(self.cache.distributed_enabled)
        self.assertIsNotNone(self.cache.distributed)
        
        # Verificar que los callbacks están configurados
        self.assertIsNotNone(self.cache.distributed.on_key_invalidated)
        self.assertIsNotNone(self.cache.distributed.on_key_requested)
        self.assertIsNotNone(self.cache.distributed.on_sync_requested)
        self.assertIsNotNone(self.cache.distributed.on_sync_received)
    
    def test_distributed_callbacks(self):
        """Prueba los callbacks del caché distribuido."""
        # Probar callback de invalidación
        self.cache.put("test_key", "test_value")
        self.assertTrue(self.cache.contains("test_key"))
        
        # Simular invalidación desde otro nodo
        self.cache._on_distributed_key_invalidated("test_key")
        self.assertFalse(self.cache.contains("test_key"))
        
        # Probar callback de solicitud de clave
        self.cache.put("requested_key", "requested_value")
        value, found = self.cache._on_distributed_key_requested("requested_key")
        self.assertEqual(value, "requested_value")
        self.assertTrue(found)
        
        # Probar callback de solicitud de sincronización
        self.cache.put("sync_key1", "sync_value1")
        self.cache.put("sync_key2", "sync_value2")
        sync_data = self.cache._on_distributed_sync_requested()
        self.assertIn("sync_key1", sync_data)
        self.assertIn("sync_key2", sync_data)
        
        # Probar callback de recepción de sincronización
        self.cache._on_distributed_sync_received({"new_key": "new_value"})
        self.assertTrue(self.cache.contains("new_key"))
        self.assertEqual(self.cache.get("new_key"), "new_value")


if __name__ == '__main__':
    unittest.main()
