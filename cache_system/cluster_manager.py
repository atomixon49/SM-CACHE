"""
Gestor de cluster y failover automático para SM-CACHE.
"""
import asyncio
import logging
import time
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
import json
import aiohttp
from .security import CacheSecurity

class ClusterManager:
    """Gestiona el cluster de nodos de caché y el failover automático."""
    
    def __init__(self, node_id: str, host: str, port: int,
                 security: CacheSecurity, ssl_context: Optional[dict] = None):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.security = security
        self.ssl_context = ssl_context
        
        # Estado del cluster
        self.nodes: Dict[str, dict] = {}
        self.leader_id: Optional[str] = None
        self.is_leader = False
        
        # Estado de salud
        self.last_heartbeat: Dict[str, float] = {}
        self.node_health: Dict[str, str] = {}
        
        # Configuración
        self.heartbeat_interval = 5.0  # segundos
        self.failover_timeout = 15.0   # segundos
        
        # Control
        self.running = False
        self._tasks: Set[asyncio.Task] = set()
        
        # Logging
        self.logger = logging.getLogger("ClusterManager")
    
    async def start(self, initial_nodes: List[Tuple[str, int]] = None) -> None:
        """Inicia el gestor de cluster."""
        self.running = True
        
        # Registrar nodo actual
        self.nodes[self.node_id] = {
            'host': self.host,
            'port': self.port,
            'joined_at': datetime.utcnow().isoformat(),
            'role': 'unknown'
        }
        
        # Iniciar tareas principales
        self._tasks.add(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.add(asyncio.create_task(self._health_check_loop()))
        self._tasks.add(asyncio.create_task(self._leader_election_loop()))
        
        # Conectar con nodos iniciales
        if initial_nodes:
            for host, port in initial_nodes:
                await self._try_join_node(host, port)
    
    async def stop(self) -> None:
        """Detiene el gestor de cluster."""
        self.running = False
        
        # Cancelar todas las tareas
        for task in self._tasks:
            task.cancel()
        
        # Esperar a que terminen
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        # Notificar salida a otros nodos
        if self.nodes:
            await self._notify_leaving()
    
    async def _heartbeat_loop(self) -> None:
        """Envía y monitorea heartbeats periódicamente."""
        while self.running:
            try:
                timestamp = time.time()
                self.last_heartbeat[self.node_id] = timestamp
                
                # Enviar heartbeat a todos los nodos
                for node_id, node in self.nodes.items():
                    if node_id != self.node_id:
                        try:
                            async with aiohttp.ClientSession() as session:
                                url = f"https://{node['host']}:{node['port']}/cluster/heartbeat"
                                headers = {"X-Node-ID": self.node_id}
                                async with session.post(url, headers=headers, ssl=self.ssl_context) as resp:
                                    if resp.status == 200:
                                        self.last_heartbeat[node_id] = timestamp
                        except Exception as e:
                            self.logger.warning(f"Error sending heartbeat to {node_id}: {e}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(1)
    
    async def _health_check_loop(self) -> None:
        """Monitorea la salud de los nodos."""
        while self.running:
            try:
                current_time = time.time()
                
                # Verificar último heartbeat de cada nodo
                for node_id in list(self.nodes.keys()):
                    if node_id == self.node_id:
                        continue
                        
                    last_seen = self.last_heartbeat.get(node_id, 0)
                    if current_time - last_seen > self.failover_timeout:
                        self.node_health[node_id] = "unreachable"
                        if self.is_leader:
                            await self._handle_node_failure(node_id)
                    else:
                        self.node_health[node_id] = "healthy"
                
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(1)
    
    async def _leader_election_loop(self) -> None:
        """Gestiona la elección de líder."""
        while self.running:
            try:
                # Si no hay líder, iniciar elección
                if not self.leader_id:
                    await self._start_election()
                
                # Si el líder no responde, iniciar nueva elección
                elif self.leader_id in self.node_health:
                    if self.node_health[self.leader_id] == "unreachable":
                        self.logger.info("Leader unreachable, starting new election")
                        self.leader_id = None
                        await self._start_election()
                
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in leader election loop: {e}")
                await asyncio.sleep(1)
    
    async def _start_election(self) -> None:
        """Inicia una elección de líder."""
        # Implementar algoritmo de elección (por ejemplo, Bully Algorithm)
        highest_id = self.node_id
        for node_id in self.nodes:
            if node_id > highest_id and self.node_health.get(node_id) == "healthy":
                highest_id = node_id
        
        # Si este nodo tiene el ID más alto, se convierte en líder
        if highest_id == self.node_id:
            self.is_leader = True
            self.leader_id = self.node_id
            self.nodes[self.node_id]['role'] = 'leader'
            await self._announce_leadership()
        else:
            self.is_leader = False
            self.leader_id = highest_id
            self.nodes[self.node_id]['role'] = 'follower'
    
    async def _handle_node_failure(self, node_id: str) -> None:
        """Maneja el fallo de un nodo."""
        if not self.is_leader:
            return
            
        self.logger.warning(f"Node {node_id} failed, initiating failover")
        
        # Remover nodo fallido
        if node_id in self.nodes:
            del self.nodes[node_id]
            del self.last_heartbeat[node_id]
            del self.node_health[node_id]
        
        # Notificar a otros nodos
        await self._broadcast_cluster_update()
        
        # Si era el líder, iniciar nueva elección
        if node_id == self.leader_id:
            self.leader_id = None
            await self._start_election()
    
    async def _announce_leadership(self) -> None:
        """Anuncia el liderazgo a otros nodos."""
        message = {
            'type': 'leadership_announcement',
            'leader_id': self.node_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._broadcast_message(message)
    
    async def _notify_leaving(self) -> None:
        """Notifica a otros nodos que este nodo está saliendo del cluster."""
        message = {
            'type': 'node_leaving',
            'node_id': self.node_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._broadcast_message(message)
    
    async def _broadcast_message(self, message: dict) -> None:
        """Envía un mensaje a todos los nodos del cluster."""
        for node_id, node in self.nodes.items():
            if node_id != self.node_id:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = f"https://{node['host']}:{node['port']}/cluster/message"
                        headers = {
                            "X-Node-ID": self.node_id,
                            "Content-Type": "application/json"
                        }
                        async with session.post(url, json=message, headers=headers, ssl=self.ssl_context) as resp:
                            if resp.status != 200:
                                self.logger.warning(f"Failed to send message to {node_id}")
                except Exception as e:
                    self.logger.error(f"Error broadcasting message to {node_id}: {e}")
    
    async def _broadcast_cluster_update(self) -> None:
        """Envía actualización del estado del cluster a todos los nodos."""
        message = {
            'type': 'cluster_update',
            'nodes': self.nodes,
            'leader_id': self.leader_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self._broadcast_message(message)
    
    async def _try_join_node(self, host: str, port: int) -> bool:
        """Intenta unirse a un nodo existente."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://{host}:{port}/cluster/join"
                headers = {
                    "X-Node-ID": self.node_id,
                    "Content-Type": "application/json"
                }
                data = {
                    'host': self.host,
                    'port': self.port,
                }
                async with session.post(url, json=data, headers=headers, ssl=self.ssl_context) as resp:
                    if resp.status == 200:
                        cluster_state = await resp.json()
                        self.nodes.update(cluster_state['nodes'])
                        self.leader_id = cluster_state['leader_id']
                        return True
            return False
        except Exception as e:
            self.logger.error(f"Error joining node {host}:{port}: {e}")
            return False
    
    def get_cluster_state(self) -> dict:
        """Obtiene el estado actual del cluster."""
        return {
            'nodes': self.nodes,
            'leader_id': self.leader_id,
            'node_health': self.node_health,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def is_healthy(self) -> bool:
        """Verifica si el cluster está saludable."""
        # Debe haber al menos un líder
        if not self.leader_id:
            return False
        
        # El líder debe estar saludable
        if self.leader_id in self.node_health:
            if self.node_health[self.leader_id] != "healthy":
                return False
        
        # Al menos 50% de los nodos deben estar saludables
        healthy_nodes = sum(1 for status in self.node_health.values() if status == "healthy")
        return healthy_nodes >= len(self.nodes) / 2