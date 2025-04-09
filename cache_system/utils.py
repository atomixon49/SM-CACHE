"""
Utilidades para el sistema de caché inteligente.
"""
import sys
import time
import json
from typing import Dict, Any, List, Callable, Optional
import threading


def estimate_size(obj: Any) -> int:
    """
    Estima el tamaño en bytes de un objeto de forma más precisa.
    
    Args:
        obj: El objeto a medir
        
    Returns:
        Tamaño estimado en bytes
    """
    if isinstance(obj, (str, bytes, bytearray)):
        return sys.getsizeof(obj)
    elif isinstance(obj, (int, float, bool, type(None))):
        return sys.getsizeof(obj)
    elif isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(estimate_size(k) + estimate_size(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        return sys.getsizeof(obj) + sum(estimate_size(i) for i in obj)
    else:
        # Para objetos complejos, usar el tamaño base
        return sys.getsizeof(obj)


class AsyncDataLoader:
    """
    Cargador asíncrono de datos que puede utilizarse con el caché inteligente.
    """
    
    def __init__(self, 
                 load_function: Callable[[Any], Any],
                 max_workers: int = 5,
                 timeout: float = 30.0):
        """
        Inicializa el cargador asíncrono.
        
        Args:
            load_function: Función que carga los datos para una clave
            max_workers: Número máximo de trabajadores concurrentes
            timeout: Tiempo máximo de espera para una carga en segundos
        """
        self.load_function = load_function
        self.max_workers = max_workers
        self.timeout = timeout
        self.semaphore = threading.Semaphore(max_workers)
        self.results: Dict[Any, Any] = {}
        self.errors: Dict[Any, Exception] = {}
        self.locks: Dict[Any, threading.Lock] = {}
        self.threads: Dict[Any, threading.Thread] = {}
    
    def _worker(self, key: Any) -> None:
        """
        Función de trabajo para cargar datos de forma asíncrona.
        
        Args:
            key: La clave para la que cargar datos
        """
        try:
            result = self.load_function(key)
            self.results[key] = result
        except Exception as e:
            self.errors[key] = e
        finally:
            self.semaphore.release()
            if key in self.locks:
                lock = self.locks.pop(key)
                lock.release()
            if key in self.threads:
                del self.threads[key]
    
    def load(self, key: Any, block: bool = True) -> Any:
        """
        Carga datos para una clave, posiblemente de forma asíncrona.
        
        Args:
            key: La clave para la que cargar datos
            block: Si se debe bloquear hasta que los datos estén disponibles
            
        Returns:
            Los datos cargados si block=True, o None si block=False
            
        Raises:
            Exception: Si ocurre un error durante la carga y block=True
        """
        # Verificar si ya tenemos resultados o errores
        if key in self.results:
            result = self.results.pop(key)
            return result
        elif key in self.errors:
            error = self.errors.pop(key)
            raise error
        
        # Verificar si ya hay una carga en progreso
        if key in self.threads:
            if block:
                # Esperar a que termine la carga
                lock = self.locks.get(key)
                if lock:
                    lock.acquire(timeout=self.timeout)
                    lock.release()
                
                # Verificar resultados o errores
                if key in self.results:
                    result = self.results.pop(key)
                    return result
                elif key in self.errors:
                    error = self.errors.pop(key)
                    raise error
            return None
        
        # Iniciar una nueva carga
        self.semaphore.acquire(blocking=block)
        if not block and not self.semaphore._value:  # type: ignore
            self.semaphore.release()
            return None
        
        # Crear un lock para esta carga
        lock = threading.Lock()
        lock.acquire()
        self.locks[key] = lock
        
        # Iniciar hilo de trabajo
        thread = threading.Thread(target=self._worker, args=(key,))
        thread.daemon = True
        self.threads[key] = thread
        thread.start()
        
        if block:
            # Esperar a que termine la carga
            lock.acquire(timeout=self.timeout)
            lock.release()
            
            # Verificar resultados o errores
            if key in self.results:
                result = self.results.pop(key)
                return result
            elif key in self.errors:
                error = self.errors.pop(key)
                raise error
            else:
                raise TimeoutError(f"Tiempo de espera agotado al cargar datos para clave {key}")
        
        return None


class CacheSerializer:
    """
    Clase para serializar y deserializar el estado del caché.
    """
    
    @staticmethod
    def serialize_cache(cache: Dict[Any, Any], 
                        expiry_times: Dict[Any, float],
                        key_serializer: Optional[Callable[[Any], str]] = None,
                        value_serializer: Optional[Callable[[Any], Any]] = None) -> str:
        """
        Serializa el estado del caché a JSON.
        
        Args:
            cache: Diccionario de caché
            expiry_times: Diccionario de tiempos de expiración
            key_serializer: Función para serializar claves no JSON-serializables
            value_serializer: Función para serializar valores no JSON-serializables
            
        Returns:
            Cadena JSON con el estado serializado
        """
        key_serializer = key_serializer or str
        
        serialized_data = {
            'timestamp': time.time(),
            'items': []
        }
        
        for key, value in cache.items():
            serialized_key = key_serializer(key)
            serialized_value = value_serializer(value) if value_serializer else value
            
            expiry = expiry_times.get(key, None)
            
            item = {
                'key': serialized_key,
                'value': serialized_value,
                'expiry': expiry
            }
            
            serialized_data['items'].append(item)
        
        return json.dumps(serialized_data)
    
    @staticmethod
    def deserialize_cache(serialized_data: str,
                          key_deserializer: Optional[Callable[[str], Any]] = None,
                          value_deserializer: Optional[Callable[[Any], Any]] = None) -> tuple:
        """
        Deserializa el estado del caché desde JSON.
        
        Args:
            serialized_data: Cadena JSON con el estado serializado
            key_deserializer: Función para deserializar claves
            value_deserializer: Función para deserializar valores
            
        Returns:
            Tupla (cache, expiry_times)
        """
        data = json.loads(serialized_data)
        
        cache = {}
        expiry_times = {}
        
        for item in data['items']:
            key = key_deserializer(item['key']) if key_deserializer else item['key']
            value = value_deserializer(item['value']) if value_deserializer else item['value']
            expiry = item.get('expiry')
            
            cache[key] = value
            if expiry is not None:
                expiry_times[key] = expiry
        
        return cache, expiry_times
