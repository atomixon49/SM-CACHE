"""
Módulo para soporte de caché distribuido en el sistema de caché inteligente.
"""
import socket
import threading
import time
import json
import logging
import queue
import hashlib
from typing import Dict, Any, List, Set, Optional, Callable, Tuple, Union
import random
from .utils import CacheSerializer


class Node:
    """
    Representa un nodo en el sistema de caché distribuido.
    """
    
    def __init__(self, host: str, port: int, node_id: Optional[str] = None):
        """
        Inicializa un nodo.
        
        Args:
            host: Dirección IP o nombre de host del nodo
            port: Puerto del nodo
            node_id: Identificador único del nodo (generado automáticamente si es None)
        """
        self.host = host
        self.port = port
        self.node_id = node_id or self._generate_node_id(host, port)
        
    def _generate_node_id(self, host: str, port: int) -> str:
        """
        Genera un ID único para el nodo basado en host y puerto.
        
        Args:
            host: Dirección IP o nombre de host
            port: Puerto
            
        Returns:
            ID único del nodo
        """
        # Usar hash para generar un ID único
        node_str = f"{host}:{port}:{random.randint(0, 10000)}"
        return hashlib.md5(node_str.encode()).hexdigest()[:8]
        
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id
        
    def __hash__(self):
        return hash(self.node_id)
        
    def __str__(self):
        return f"Node({self.host}:{self.port}, id={self.node_id})"


class Message:
    """
    Representa un mensaje en el sistema de caché distribuido.
    """
    
    # Tipos de mensajes
    TYPE_JOIN = "JOIN"           # Unirse al cluster
    TYPE_LEAVE = "LEAVE"         # Salir del cluster
    TYPE_HEARTBEAT = "HEARTBEAT" # Mensaje de latido para detectar nodos caídos
    TYPE_SYNC = "SYNC"           # Sincronizar datos de caché
    TYPE_INVALIDATE = "INVALIDATE" # Invalidar una clave en todos los nodos
    TYPE_GET = "GET"             # Solicitar un valor
    TYPE_RESPONSE = "RESPONSE"   # Respuesta a una solicitud
    
    def __init__(self, msg_type: str, sender: Node, data: Dict[str, Any]):
        """
        Inicializa un mensaje.
        
        Args:
            msg_type: Tipo de mensaje
            sender: Nodo que envía el mensaje
            data: Datos del mensaje
        """
        self.type = msg_type
        self.sender = sender
        self.data = data
        self.timestamp = time.time()
        
    def to_json(self) -> str:
        """
        Convierte el mensaje a formato JSON.
        
        Returns:
            Cadena JSON que representa el mensaje
        """
        msg_dict = {
            "type": self.type,
            "sender": {
                "host": self.sender.host,
                "port": self.sender.port,
                "node_id": self.sender.node_id
            },
            "data": self.data,
            "timestamp": self.timestamp
        }
        return json.dumps(msg_dict)
        
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """
        Crea un mensaje a partir de una cadena JSON.
        
        Args:
            json_str: Cadena JSON que representa el mensaje
            
        Returns:
            Instancia de Message
        """
        msg_dict = json.loads(json_str)
        sender_dict = msg_dict["sender"]
        sender = Node(
            host=sender_dict["host"],
            port=sender_dict["port"],
            node_id=sender_dict["node_id"]
        )
        
        message = cls(
            msg_type=msg_dict["type"],
            sender=sender,
            data=msg_dict["data"]
        )
        message.timestamp = msg_dict["timestamp"]
        return message


class DistributedCache:
    """
    Implementación de caché distribuido que sincroniza múltiples nodos.
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 5000,
                 node_id: Optional[str] = None,
                 cluster_nodes: Optional[List[Tuple[str, int]]] = None,
                 sync_interval: int = 30,
                 heartbeat_interval: int = 10,
                 node_timeout: int = 30):
        """
        Inicializa el caché distribuido.
        
        Args:
            host: Dirección IP o nombre de host de este nodo
            port: Puerto de este nodo
            node_id: Identificador único de este nodo (generado automáticamente si es None)
            cluster_nodes: Lista de nodos (host, port) en el cluster
            sync_interval: Intervalo en segundos para sincronizar datos
            heartbeat_interval: Intervalo en segundos para enviar latidos
            node_timeout: Tiempo en segundos después del cual un nodo se considera caído
        """
        # Configuración del nodo
        self.local_node = Node(host, port, node_id)
        self.cluster_nodes: Set[Node] = set()
        
        # Añadir nodos iniciales si se proporcionan
        if cluster_nodes:
            for node_host, node_port in cluster_nodes:
                if node_host != host or node_port != port:  # No añadir este nodo
                    self.cluster_nodes.add(Node(node_host, node_port))
        
        # Intervalos y timeouts
        self.sync_interval = sync_interval
        self.heartbeat_interval = heartbeat_interval
        self.node_timeout = node_timeout
        
        # Estado interno
        self.running = False
        self.last_heartbeat: Dict[str, float] = {}  # node_id -> timestamp
        self.message_queue = queue.Queue()
        
        # Callbacks
        self.on_key_invalidated: Optional[Callable[[Any], None]] = None
        self.on_key_requested: Optional[Callable[[Any], Tuple[Any, bool]]] = None
        self.on_sync_requested: Optional[Callable[[], Dict[Any, Any]]] = None
        self.on_sync_received: Optional[Callable[[Dict[Any, Any]], None]] = None
        
        # Hilos
        self.server_thread: Optional[threading.Thread] = None
        self.client_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.sync_thread: Optional[threading.Thread] = None
        self.processor_thread: Optional[threading.Thread] = None
        
        # Sockets
        self.server_socket: Optional[socket.socket] = None
        
        # Eventos para detener hilos
        self.stop_event = threading.Event()
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DistributedCache")
    
    def start(self) -> bool:
        """
        Inicia el caché distribuido.
        
        Returns:
            True si se inició correctamente, False en caso contrario
        """
        if self.running:
            return True
            
        try:
            # Iniciar servidor
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.local_node.host, self.local_node.port))
            self.server_socket.listen(10)
            self.server_socket.settimeout(1.0)  # Timeout para poder detener el hilo
            
            # Iniciar hilos
            self.stop_event.clear()
            
            self.server_thread = threading.Thread(target=self._server_worker)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.processor_thread = threading.Thread(target=self._message_processor)
            self.processor_thread.daemon = True
            self.processor_thread.start()
            
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()
            
            self.sync_thread = threading.Thread(target=self._sync_worker)
            self.sync_thread.daemon = True
            self.sync_thread.start()
            
            # Unirse al cluster
            self._join_cluster()
            
            self.running = True
            self.logger.info(f"Nodo iniciado en {self.local_node.host}:{self.local_node.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al iniciar nodo: {e}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Detiene el caché distribuido."""
        if not self.running:
            return
            
        # Notificar salida del cluster
        self._leave_cluster()
        
        # Detener hilos
        self.stop_event.set()
        
        # Cerrar socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            
        # Esperar a que terminen los hilos
        for thread in [self.server_thread, self.processor_thread, 
                      self.heartbeat_thread, self.sync_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
                
        self.running = False
        self.logger.info("Nodo detenido")
    
    def invalidate_key(self, key: Any) -> None:
        """
        Invalida una clave en todos los nodos del cluster.
        
        Args:
            key: Clave a invalidar
        """
        if not self.running:
            return
            
        # Crear mensaje de invalidación
        message = Message(
            msg_type=Message.TYPE_INVALIDATE,
            sender=self.local_node,
            data={"key": str(key)}
        )
        
        # Enviar a todos los nodos
        self._broadcast_message(message)
    
    def request_key(self, key: Any) -> Tuple[Any, bool]:
        """
        Solicita una clave a otros nodos del cluster.
        
        Args:
            key: Clave a solicitar
            
        Returns:
            Tupla (valor, encontrado)
        """
        if not self.running or not self.cluster_nodes:
            return None, False
            
        # Crear mensaje de solicitud
        message = Message(
            msg_type=Message.TYPE_GET,
            sender=self.local_node,
            data={"key": str(key)}
        )
        
        # Enviar a un nodo aleatorio
        random_node = random.choice(list(self.cluster_nodes))
        response = self._send_message_and_wait_response(random_node, message)
        
        if response and "value" in response.data:
            return response.data["value"], response.data.get("found", False)
            
        return None, False
    
    def sync_cache(self, cache_data: Dict[Any, Any]) -> None:
        """
        Sincroniza datos de caché con otros nodos.
        
        Args:
            cache_data: Datos de caché a sincronizar
        """
        if not self.running:
            return
            
        # Serializar datos de caché
        serialized_data = CacheSerializer.serialize_cache(
            cache_data, {}, key_serializer=str
        )
        
        # Crear mensaje de sincronización
        message = Message(
            msg_type=Message.TYPE_SYNC,
            sender=self.local_node,
            data={"cache": serialized_data}
        )
        
        # Enviar a todos los nodos
        self._broadcast_message(message)
    
    def _server_worker(self) -> None:
        """Hilo de servidor que acepta conexiones entrantes."""
        self.logger.info("Hilo de servidor iniciado")
        
        while not self.stop_event.is_set():
            try:
                # Aceptar conexión
                client_socket, address = self.server_socket.accept()
                client_socket.settimeout(5.0)
                
                # Recibir mensaje
                data = b""
                while True:
                    try:
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    except socket.timeout:
                        break
                
                if data:
                    # Procesar mensaje
                    try:
                        message = Message.from_json(data.decode())
                        self.message_queue.put(message)
                    except Exception as e:
                        self.logger.error(f"Error al procesar mensaje: {e}")
                
                # Cerrar conexión
                client_socket.close()
                
            except socket.timeout:
                # Timeout para comprobar stop_event
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    self.logger.error(f"Error en hilo de servidor: {e}")
                    time.sleep(1)
    
    def _message_processor(self) -> None:
        """Hilo que procesa mensajes recibidos."""
        self.logger.info("Hilo de procesador de mensajes iniciado")
        
        while not self.stop_event.is_set():
            try:
                # Obtener mensaje de la cola
                try:
                    message = self.message_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Actualizar último latido
                self.last_heartbeat[message.sender.node_id] = time.time()
                
                # Añadir nodo si no está en la lista
                if message.sender.node_id != self.local_node.node_id:
                    self.cluster_nodes.add(message.sender)
                
                # Procesar según tipo de mensaje
                if message.type == Message.TYPE_JOIN:
                    self._handle_join(message)
                elif message.type == Message.TYPE_LEAVE:
                    self._handle_leave(message)
                elif message.type == Message.TYPE_HEARTBEAT:
                    self._handle_heartbeat(message)
                elif message.type == Message.TYPE_SYNC:
                    self._handle_sync(message)
                elif message.type == Message.TYPE_INVALIDATE:
                    self._handle_invalidate(message)
                elif message.type == Message.TYPE_GET:
                    self._handle_get(message)
                
                # Marcar como procesado
                self.message_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error en procesador de mensajes: {e}")
    
    def _heartbeat_worker(self) -> None:
        """Hilo que envía latidos periódicos y detecta nodos caídos."""
        self.logger.info("Hilo de latidos iniciado")
        
        while not self.stop_event.is_set():
            try:
                # Enviar latido
                message = Message(
                    msg_type=Message.TYPE_HEARTBEAT,
                    sender=self.local_node,
                    data={}
                )
                self._broadcast_message(message)
                
                # Detectar nodos caídos
                current_time = time.time()
                dead_nodes = []
                
                for node in self.cluster_nodes:
                    last_time = self.last_heartbeat.get(node.node_id, 0)
                    if current_time - last_time > self.node_timeout:
                        dead_nodes.append(node)
                
                # Eliminar nodos caídos
                for node in dead_nodes:
                    self.cluster_nodes.remove(node)
                    self.logger.info(f"Nodo {node} eliminado por timeout")
                
                # Esperar hasta el próximo latido
                self.stop_event.wait(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error en hilo de latidos: {e}")
                time.sleep(1)
    
    def _sync_worker(self) -> None:
        """Hilo que sincroniza periódicamente el caché."""
        self.logger.info("Hilo de sincronización iniciado")
        
        while not self.stop_event.is_set():
            try:
                # Solicitar datos de caché
                if self.on_sync_requested:
                    cache_data = self.on_sync_requested()
                    if cache_data:
                        self.sync_cache(cache_data)
                
                # Esperar hasta la próxima sincronización
                self.stop_event.wait(self.sync_interval)
                
            except Exception as e:
                self.logger.error(f"Error en hilo de sincronización: {e}")
                time.sleep(1)
    
    def _join_cluster(self) -> None:
        """Envía mensaje de unión al cluster."""
        message = Message(
            msg_type=Message.TYPE_JOIN,
            sender=self.local_node,
            data={}
        )
        self._broadcast_message(message)
        self.logger.info("Mensaje de unión enviado")
    
    def _leave_cluster(self) -> None:
        """Envía mensaje de salida del cluster."""
        if not self.running:
            return
            
        message = Message(
            msg_type=Message.TYPE_LEAVE,
            sender=self.local_node,
            data={}
        )
        self._broadcast_message(message)
        self.logger.info("Mensaje de salida enviado")
    
    def _broadcast_message(self, message: Message) -> None:
        """
        Envía un mensaje a todos los nodos del cluster.
        
        Args:
            message: Mensaje a enviar
        """
        for node in self.cluster_nodes:
            self._send_message(node, message)
    
    def _send_message(self, node: Node, message: Message) -> bool:
        """
        Envía un mensaje a un nodo específico.
        
        Args:
            node: Nodo destinatario
            message: Mensaje a enviar
            
        Returns:
            True si se envió correctamente, False en caso contrario
        """
        try:
            # Crear socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5.0)
            
            # Conectar al nodo
            client_socket.connect((node.host, node.port))
            
            # Enviar mensaje
            client_socket.sendall(message.to_json().encode())
            
            # Cerrar conexión
            client_socket.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error al enviar mensaje a {node}: {e}")
            return False
    
    def _send_message_and_wait_response(self, node: Node, message: Message, 
                                       timeout: float = 5.0) -> Optional[Message]:
        """
        Envía un mensaje a un nodo y espera respuesta.
        
        Args:
            node: Nodo destinatario
            message: Mensaje a enviar
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            Mensaje de respuesta o None si hay error o timeout
        """
        try:
            # Crear socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(timeout)
            
            # Conectar al nodo
            client_socket.connect((node.host, node.port))
            
            # Enviar mensaje
            client_socket.sendall(message.to_json().encode())
            
            # Esperar respuesta
            data = b""
            while True:
                try:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                except socket.timeout:
                    break
            
            # Cerrar conexión
            client_socket.close()
            
            if data:
                return Message.from_json(data.decode())
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error al enviar mensaje y esperar respuesta de {node}: {e}")
            return None
    
    def _handle_join(self, message: Message) -> None:
        """
        Maneja un mensaje de unión al cluster.
        
        Args:
            message: Mensaje recibido
        """
        if message.sender.node_id != self.local_node.node_id:
            self.cluster_nodes.add(message.sender)
            self.logger.info(f"Nodo {message.sender} se ha unido al cluster")
            
            # Enviar datos de caché actuales
            if self.on_sync_requested:
                cache_data = self.on_sync_requested()
                if cache_data:
                    sync_message = Message(
                        msg_type=Message.TYPE_SYNC,
                        sender=self.local_node,
                        data={"cache": CacheSerializer.serialize_cache(cache_data, {}, key_serializer=str)}
                    )
                    self._send_message(message.sender, sync_message)
    
    def _handle_leave(self, message: Message) -> None:
        """
        Maneja un mensaje de salida del cluster.
        
        Args:
            message: Mensaje recibido
        """
        if message.sender.node_id != self.local_node.node_id:
            if message.sender in self.cluster_nodes:
                self.cluster_nodes.remove(message.sender)
                self.logger.info(f"Nodo {message.sender} ha salido del cluster")
    
    def _handle_heartbeat(self, message: Message) -> None:
        """
        Maneja un mensaje de latido.
        
        Args:
            message: Mensaje recibido
        """
        # Actualizar último latido (ya se hace en el procesador de mensajes)
        pass
    
    def _handle_sync(self, message: Message) -> None:
        """
        Maneja un mensaje de sincronización.
        
        Args:
            message: Mensaje recibido
        """
        if "cache" in message.data and self.on_sync_received:
            try:
                serialized_data = message.data["cache"]
                cache_data, _ = CacheSerializer.deserialize_cache(serialized_data)
                self.on_sync_received(cache_data)
                self.logger.info(f"Datos de caché sincronizados desde {message.sender}")
            except Exception as e:
                self.logger.error(f"Error al procesar sincronización: {e}")
    
    def _handle_invalidate(self, message: Message) -> None:
        """
        Maneja un mensaje de invalidación de clave.
        
        Args:
            message: Mensaje recibido
        """
        if "key" in message.data and self.on_key_invalidated:
            key = message.data["key"]
            self.on_key_invalidated(key)
            self.logger.info(f"Clave {key} invalidada por {message.sender}")
    
    def _handle_get(self, message: Message) -> None:
        """
        Maneja un mensaje de solicitud de clave.
        
        Args:
            message: Mensaje recibido
        """
        if "key" in message.data and self.on_key_requested:
            key = message.data["key"]
            value, found = self.on_key_requested(key)
            
            # Enviar respuesta
            response = Message(
                msg_type=Message.TYPE_RESPONSE,
                sender=self.local_node,
                data={
                    "key": key,
                    "value": value,
                    "found": found
                }
            )
            self._send_message(message.sender, response)
            self.logger.info(f"Solicitud de clave {key} atendida para {message.sender}")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el cluster.
        
        Returns:
            Diccionario con información del cluster
        """
        return {
            "local_node": {
                "host": self.local_node.host,
                "port": self.local_node.port,
                "node_id": self.local_node.node_id
            },
            "cluster_nodes": [
                {
                    "host": node.host,
                    "port": node.port,
                    "node_id": node.node_id,
                    "last_heartbeat": self.last_heartbeat.get(node.node_id, 0)
                }
                for node in self.cluster_nodes
            ],
            "node_count": len(self.cluster_nodes) + 1,
            "running": self.running
        }
