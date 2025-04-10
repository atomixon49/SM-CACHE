"""
Sistema de caché distribuido con tolerancia a fallos mejorada.
"""
import socket
import threading
import time
import json
import logging
import asyncio
import aiohttp
from aiohttp import web
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from .security import CacheSecurity

class Node:
    def __init__(self, host: str, port: int, node_id: str = None):
        self.host = host
        self.port = port
        self.id = node_id or self._generate_id()
        self.last_seen = time.time()
        self.status = "unknown"
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0  # segundos, se incrementará exponencialmente

    def _generate_id(self) -> str:
        return f"{self.host}:{self.port}-{hex(int(time.time() * 1000))[2:]}"

    def __str__(self) -> str:
        return f"Node({self.host}:{self.port}, id={self.id})"

class DistributedCache:
    """Implementación del caché distribuido con tolerancia a fallos."""
    
    def __init__(self, host: str, port: int, security: CacheSecurity,
                 retry_timeout: float = 5.0,
                 max_retries: int = 3):
        self.node = Node(host, port)
        self.security = security
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        
        # Estado del cluster
        self.nodes: Dict[str, Node] = {self.node.id: self.node}
        self.leader_id: Optional[str] = None
        self.is_leader = False
        
        # Control y sincronización
        self.running = False
        self._lock = threading.RLock()
        self.sync_interval = 10.0  # segundos
        
        # Cola de mensajes pendientes para nodos desconectados
        self.pending_messages: Dict[str, List[Dict]] = {}
        
        # Hilos de trabajo
        self.server_thread: Optional[threading.Thread] = None
        self.message_processor: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.sync_thread: Optional[threading.Thread] = None
        
        # Cliente HTTP asíncrono
        self.http_client: Optional[aiohttp.ClientSession] = None
        
        # Estado de la red
        self.network_status = {
            'packet_loss': 0.0,
            'latency': 0.0,
            'bandwidth': 0.0
        }
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DistributedCache")
    
    async def start(self) -> bool:
        """Inicia el caché distribuido."""
        if self.running:
            return True
            
        try:
            self.running = True
            
            # Iniciar cliente HTTP
            self.http_client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.retry_timeout)
            )
            
            # Iniciar hilos
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.message_processor = threading.Thread(
                target=self._process_messages,
                daemon=True
            )
            self.heartbeat_thread = threading.Thread(
                target=self._send_heartbeats,
                daemon=True
            )
            self.sync_thread = threading.Thread(
                target=self._sync_data,
                daemon=True
            )
            
            self.server_thread.start()
            self.message_processor.start()
            self.heartbeat_thread.start()
            self.sync_thread.start()
            
            self.logger.info(f"Nodo iniciado en {self.node.host}:{self.node.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al iniciar nodo: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> None:
        """Detiene el caché distribuido."""
        if not self.running:
            return
            
        self.running = False
        
        # Notificar salida a otros nodos
        if self.nodes:
            await self._broadcast_message({
                'type': 'node_leaving',
                'node_id': self.node.id,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Cerrar cliente HTTP
        if self.http_client:
            await self.http_client.close()
            
        # Esperar a que terminen los hilos
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        if self.message_processor:
            self.message_processor.join(timeout=2.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2.0)
        if self.sync_thread:
            self.sync_thread.join(timeout=2.0)
            
        self.logger.info("Nodo detenido")
    
    async def _send_message(self, node: Node, message: Dict) -> bool:
        """
        Envía un mensaje a un nodo con reintentos y manejo de fallos.
        
        Args:
            node: Nodo destino
            message: Mensaje a enviar
            
        Returns:
            True si el mensaje se envió correctamente
        """
        for attempt in range(self.max_retries):
            try:
                url = f"http://{node.host}:{node.port}/message"
                headers = {
                    "X-Node-ID": self.node.id,
                    "Content-Type": "application/json"
                }
                
                async with self.http_client.post(
                    url, json=message, headers=headers,
                    timeout=self.retry_timeout
                ) as response:
                    if response.status == 200:
                        node.reconnect_attempts = 0
                        node.status = "healthy"
                        return True
                        
                    self.logger.warning(
                        f"Error enviando mensaje a {node}: {response.status}"
                    )
                    
            except aiohttp.ClientError as e:
                self.logger.error(f"Error al enviar mensaje a {node}: {e}")
                node.reconnect_attempts += 1
                
                if node.reconnect_attempts >= node.max_reconnect_attempts:
                    await self._handle_node_failure(node)
                    return False
                    
                # Retraso exponencial
                delay = min(30, node.reconnect_delay * (2 ** attempt))
                await asyncio.sleep(delay)
                
        return False
    
    async def _handle_node_failure(self, node: Node) -> None:
        """
        Maneja el fallo de un nodo.
        
        Args:
            node: Nodo que ha fallado
        """
        self.logger.warning(f"Nodo {node} ha fallado")
        
        with self._lock:
            # Guardar mensajes pendientes
            if node.id not in self.pending_messages:
                self.pending_messages[node.id] = []
                
            # Actualizar estado del nodo
            node.status = "failed"
            
            # Si era el líder, iniciar nueva elección
            if node.id == self.leader_id:
                self.leader_id = None
                await self._start_election()
    
    async def _broadcast_message(self, message: Dict) -> None:
        """
        Envía un mensaje a todos los nodos del cluster.
        
        Args:
            message: Mensaje a enviar
        """
        tasks = []
        for node_id, node in self.nodes.items():
            if node_id != self.node.id and node.status != "failed":
                tasks.append(self._send_message(node, message))
                
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed = sum(1 for r in results if isinstance(r, Exception) or not r)
            if failed:
                self.logger.warning(f"{failed} nodos no recibieron el mensaje")
    
    def _run_server(self) -> None:
        """Ejecuta el servidor HTTP para recibir mensajes."""
        try:
            app = self._create_server_app()
            self.logger.info("Hilo de servidor iniciado")
            web.run_app(app, host=self.node.host, port=self.node.port)
        except Exception as e:
            self.logger.error(f"Error en servidor: {e}")
    
    def _create_server_app(self) -> web.Application:
        """Crea la aplicación del servidor."""
        app = web.Application()
        app.router.add_post('/message', self._handle_message)
        app.router.add_post('/join', self._handle_join)
        app.router.add_post('/heartbeat', self._handle_heartbeat)
        return app
    
    async def _handle_message(self, request: web.Request) -> web.Response:
        """Maneja mensajes entrantes."""
        try:
            node_id = request.headers.get('X-Node-ID')
            if not node_id or node_id not in self.nodes:
                return web.Response(status=403)
                
            message = await request.json()
            await self._process_message(message, node_id)
            
            return web.Response(status=200)
            
        except Exception as e:
            self.logger.error(f"Error procesando mensaje: {e}")
            return web.Response(status=500)
    
    async def _process_message(self, message: Dict, sender_id: str) -> None:
        """
        Procesa un mensaje recibido.
        
        Args:
            message: Mensaje recibido
            sender_id: ID del nodo emisor
        """
        message_type = message.get('type')
        
        if message_type == 'node_leaving':
            await self._handle_node_leaving(message)
        elif message_type == 'leadership_announcement':
            await self._handle_leadership_announcement(message)
        elif message_type == 'cluster_update':
            await self._handle_cluster_update(message)
        elif message_type == 'sync_request':
            await self._handle_sync_request(message, sender_id)
        elif message_type == 'sync_response':
            await self._handle_sync_response(message)
    
    async def _send_heartbeats(self) -> None:
        """Envía heartbeats periódicos."""
        while self.running:
            try:
                message = {
                    'type': 'heartbeat',
                    'node_id': self.node.id,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self._broadcast_message(message)
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                self.logger.error(f"Error enviando heartbeats: {e}")
                await asyncio.sleep(1)
    
    async def _sync_data(self) -> None:
        """Sincroniza datos entre nodos."""
        while self.running:
            try:
                if self.is_leader:
                    # El líder inicia la sincronización
                    await self._initiate_sync()
                    
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                self.logger.error(f"Error en sincronización: {e}")
                await asyncio.sleep(1)
    
    async def _initiate_sync(self) -> None:
        """Inicia el proceso de sincronización."""
        message = {
            'type': 'sync_request',
            'node_id': self.node.id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._broadcast_message(message)
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Obtiene información del cluster.
        
        Returns:
            Diccionario con información del cluster
        """
        with self._lock:
            return {
                'nodes': {
                    node_id: {
                        'host': node.host,
                        'port': node.port,
                        'status': node.status,
                        'last_seen': node.last_seen
                    }
                    for node_id, node in self.nodes.items()
                },
                'leader_id': self.leader_id,
                'is_leader': self.is_leader,
                'network_status': self.network_status
            }
