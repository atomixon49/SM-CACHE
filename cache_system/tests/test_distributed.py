"""
Pruebas para el módulo de caché distribuido.
"""
import unittest
import asyncio
import time
from unittest.mock import Mock, patch
from cache_system.distributed import Node, DistributedCache
from cache_system.security import CacheSecurity

class TestNode(unittest.TestCase):
    """Pruebas para la clase Node."""

    def test_node_creation(self):
        """Prueba la creación básica de un nodo."""
        node = Node("localhost", 5000)
        self.assertEqual(node.host, "localhost")
        self.assertEqual(node.port, 5000)
        self.assertEqual(node.status, "unknown")
        self.assertEqual(node.reconnect_attempts, 0)

    def test_node_id_generation(self):
        """Prueba la generación de IDs únicos."""
        node1 = Node("localhost", 5000)
        node2 = Node("localhost", 5001)
        self.assertNotEqual(node1.id, node2.id)

    def test_node_string_representation(self):
        """Prueba la representación en string del nodo."""
        node = Node("localhost", 5000)
        self.assertIn("localhost:5000", str(node))
        self.assertIn("id=", str(node))

class TestDistributedCache(unittest.TestCase):
    """Pruebas para la clase DistributedCache."""

    def setUp(self):
        """Configuración para cada prueba."""
        self.security = CacheSecurity()
        self.cache = DistributedCache(
            "localhost",
            5000,
            self.security
        )

    def tearDown(self):
        """Limpieza después de cada prueba."""
        if self.cache.running:
            asyncio.run(self.cache.stop())

    def test_cache_initialization(self):
        """Prueba la inicialización del caché distribuido."""
        self.assertEqual(self.cache.node.host, "localhost")
        self.assertEqual(self.cache.node.port, 5000)
        self.assertFalse(self.cache.running)
        self.assertEqual(len(self.cache.nodes), 1)  # Solo el nodo actual

    @patch('aiohttp.ClientSession.post')
    async def test_send_message(self, mock_post):
        """Prueba el envío de mensajes."""
        # Configurar el mock
        mock_response = Mock()
        mock_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_response

        # Iniciar el caché
        await self.cache.start()

        # Crear un nodo destino
        target_node = Node("localhost", 5001)
        self.cache.nodes[target_node.id] = target_node

        # Enviar mensaje
        message = {
            'type': 'test',
            'data': 'test_data'
        }
        success = await self.cache._send_message(target_node, message)

        self.assertTrue(success)
        self.assertEqual(target_node.status, "healthy")
        self.assertEqual(target_node.reconnect_attempts, 0)

    async def test_node_failure_handling(self):
        """Prueba el manejo de fallos de nodos."""
        # Crear un nodo que fallará
        failing_node = Node("localhost", 5001)
        failing_node.reconnect_attempts = failing_node.max_reconnect_attempts
        self.cache.nodes[failing_node.id] = failing_node
        self.cache.leader_id = failing_node.id

        # Simular fallo
        await self.cache._handle_node_failure(failing_node)

        # Verificar que el nodo se marcó como fallido
        self.assertEqual(failing_node.status, "failed")
        self.assertIsNone(self.cache.leader_id)

    @patch('aiohttp.ClientSession.post')
    async def test_broadcast_message(self, mock_post):
        """Prueba el broadcast de mensajes."""
        # Configurar el mock
        mock_response = Mock()
        mock_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_response

        # Iniciar el caché
        await self.cache.start()

        # Añadir algunos nodos
        for port in range(5001, 5003):
            node = Node("localhost", port)
            self.cache.nodes[node.id] = node

        # Enviar mensaje broadcast
        message = {
            'type': 'test_broadcast',
            'data': 'broadcast_data'
        }
        await self.cache._broadcast_message(message)

        # Verificar que se llamó al método post para cada nodo
        self.assertEqual(mock_post.call_count, 2)

    async def test_cluster_info(self):
        """Prueba la obtención de información del cluster."""
        # Añadir algunos nodos
        for port in range(5001, 5003):
            node = Node("localhost", port)
            node.status = "healthy"
            self.cache.nodes[node.id] = node

        # Obtener info del cluster
        info = self.cache.get_cluster_info()

        # Verificar la información
        self.assertEqual(len(info['nodes']), 3)  # 2 nodos + el nodo actual
        self.assertIn('leader_id', info)
        self.assertIn('is_leader', info)
        self.assertIn('network_status', info)

    async def test_message_handling(self):
        """Prueba el manejo de diferentes tipos de mensajes."""
        # Iniciar el caché
        await self.cache.start()

        # Probar diferentes tipos de mensajes
        messages = [
            {
                'type': 'node_leaving',
                'node_id': 'test_node',
                'timestamp': '2025-04-09T12:00:00'
            },
            {
                'type': 'leadership_announcement',
                'leader_id': 'leader_node',
                'timestamp': '2025-04-09T12:00:00'
            },
            {
                'type': 'cluster_update',
                'nodes': [],
                'timestamp': '2025-04-09T12:00:00'
            }
        ]

        for message in messages:
            # Procesar mensaje
            await self.cache._process_message(message, 'sender_id')

    def test_network_status_monitoring(self):
        """Prueba el monitoreo del estado de la red."""
        # Verificar valores iniciales
        self.assertEqual(self.cache.network_status['packet_loss'], 0.0)
        self.assertEqual(self.cache.network_status['latency'], 0.0)
        self.assertEqual(self.cache.network_status['bandwidth'], 0.0)

if __name__ == '__main__':
    unittest.main()
