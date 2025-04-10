"""
Implementación del sistema de caché inteligente.
"""
from typing import Any, Dict, Optional, List, Set, Tuple, Union, Callable
import time
import threading
import logging
from .memory_manager import MemoryManager
from .predictor import CachePredictor
from .monitoring import MetricsCollector, MetricsMonitor
from .persistence import CachePersistence
from .security import CacheSecurity
from .usage_tracker import UsageTracker
from .distributed import DistributedCache

class IntelligentCache:
    """Implementación de caché inteligente con múltiples características."""

    def __init__(self,
                 max_size: int = 1000,
                 ttl_enabled: bool = False,
                 default_ttl: Optional[int] = None,
                 prefetch_enabled: bool = True,
                 prediction_threshold: float = 0.7,
                 monitoring_enabled: bool = True,
                 persistence_enabled: bool = False,
                 persistence_dir: str = ".cache",
                 distributed_enabled: bool = False,
                 host: str = "localhost",
                 port: int = 5000,
                 security_enabled: bool = False,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Inicializa el caché inteligente.
        
        Args:
            max_size: Tamaño máximo en bytes
            ttl_enabled: Habilitar tiempo de vida
            default_ttl: Tiempo de vida por defecto en segundos
            prefetch_enabled: Habilitar precarga
            prediction_threshold: Umbral de confianza para predicciones
            monitoring_enabled: Habilitar monitoreo
            persistence_enabled: Habilitar persistencia
            persistence_dir: Directorio para persistencia
            distributed_enabled: Habilitar modo distribuido
            host: Host para modo distribuido
            port: Puerto para modo distribuido
            security_enabled: Habilitar seguridad
            alert_thresholds: Umbrales para alertas
        """
        # Configuración básica
        self.max_size = max_size
        self.ttl_enabled = ttl_enabled
        self.default_ttl = default_ttl
        self.prefetch_enabled = prefetch_enabled
        
        # Estado interno
        self._cache: Dict[Any, Any] = {}
        self._ttl: Dict[Any, float] = {}
        self._lock = threading.Lock()
        self.running = True
        
        # Componentes
        self.memory_manager = MemoryManager(max_size)
        self.predictor = CachePredictor(prediction_threshold)
        self.usage_tracker = UsageTracker()
        
        # Métricas y monitoreo
        if monitoring_enabled:
            self.metrics = MetricsCollector()
            self.monitor = MetricsMonitor(
                self.metrics,
                export_interval=60,
                alert_thresholds=alert_thresholds
            )
            self.monitor.start()
        else:
            self.metrics = None
            self.monitor = None
        
        # Persistencia
        self.persistence_enabled = persistence_enabled
        if persistence_enabled:
            self.persistence = CachePersistence(
                cache_dir=persistence_dir,
                base_interval=60
            )
            self._load_from_disk()
        else:
            self.persistence = None
        
        # Distribución
        self.distributed_enabled = distributed_enabled
        if distributed_enabled:
            self.security = CacheSecurity() if security_enabled else None
            self.distributed = DistributedCache(host, port, self.security)
        else:
            self.distributed = None
            
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("IntelligentCache")

    # ... resto del código existente ...

    def _start_prefetch_thread(self) -> None:
        """Inicia el hilo de precarga de datos."""
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            return

        self.stop_prefetch.clear()
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True
        )
        self.prefetch_thread.start()
        self.logger.info("Hilo de prefetch iniciado")

    def _prefetch_worker(self) -> None:
        """Función de trabajo para el hilo de precarga."""
        while not self.stop_prefetch.is_set():
            try:
                # Obtener candidatos para prefetch
                candidates = self.predictor.get_prefetch_candidates()

                # Filtrar los que ya están en caché
                with self.lock:
                    candidates = [k for k in candidates if k not in self.cache]

                # Precargar datos (máximo 3 a la vez para no sobrecargar)
                for key in list(candidates)[:3]:
                    if self.stop_prefetch.is_set():
                        break

                    if self.data_loader and key not in self.cache:
                        try:
                            value = self.data_loader(key)
                            self.put(key, value, is_prefetch=True)
                            self.logger.debug(f"Prefetch exitoso para clave: {key}")
                        except Exception as e:
                            self.logger.error(f"Error en prefetch para clave {key}: {e}")

                # Esperar antes de la siguiente ronda de prefetch
                self.stop_prefetch.wait(timeout=1.0)

            except Exception as e:
                self.logger.error(f"Error en hilo de prefetch: {e}")
                # Esperar un poco antes de reintentar
                time.sleep(0.5)

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Obtiene un valor del caché.

        Args:
            key: La clave a buscar
            default: Valor por defecto si la clave no existe

        Returns:
            El valor asociado a la clave o el valor por defecto
        """
        start_time = time.time()
        hit = False
        from_prefetch = False
        distributed_hit = False

        try:
            with self.lock:
                # Verificar si la clave existe y no ha expirado
                if key in self.cache:
                    # Verificar expiración
                    if self.ttl is not None and key in self.expiry_times:
                        if time.time() > self.expiry_times[key]:
                            # El elemento ha expirado
                            self._remove_item(key)
                            return self._load_missing_data(key, default)

                    # Registrar acceso
                    self.usage_tracker.record_access(key)
                    if hasattr(self.predictor, 'update_patterns'):
                        self.predictor.update_patterns(key)
                    else:
                        self.predictor.update(key)

                    hit = True
                    return self.cache[key]

                # Intentar obtener de otros nodos si está habilitado el caché distribuido
                if self.distributed_enabled and self.distributed:
                    value, found = self.distributed.request_key(key)
                    if found:
                        # Guardar en caché local
                        self.put(key, value, is_prefetch=True)
                        hit = True
                        distributed_hit = True
                        return value

                return self._load_missing_data(key, default)
        finally:
            # Registrar métricas si está habilitado el monitoreo
            if self.monitoring_enabled and self.metrics_collector:
                elapsed_time = time.time() - start_time
                self.metrics_collector.record_get(key, hit, elapsed_time, from_prefetch)

                if distributed_hit:
                    self.metrics_collector.record_distributed_operation('get', True)

    def _load_missing_data(self, key: Any, default: Any) -> Any:
        """
        Intenta cargar datos que no están en el caché.

        Args:
            key: La clave a cargar
            default: Valor por defecto si no se puede cargar

        Returns:
            El valor cargado o el valor por defecto
        """
        # Si hay un cargador de datos, intentar cargar
        if self.data_loader:
            try:
                value = self.data_loader(key)
                self.put(key, value)
                return value
            except Exception as e:
                self.logger.error(f"Error al cargar datos para clave {key}: {e}")

        return default

    def put(self, key: Any, value: Any, is_prefetch: bool = False, notify_distributed: bool = True) -> None:
        """
        Almacena un valor en el caché.

        Args:
            key: La clave para almacenar
            value: El valor a almacenar
            is_prefetch: Indica si es una operación de precarga
            notify_distributed: Si se debe notificar a otros nodos
        """
        start_time = time.time()

        try:
            with self.lock:
                # Verificar si necesitamos hacer espacio
                if key not in self.cache and self.memory_manager.is_memory_full():
                    self._evict_items()

                # Almacenar el valor
                self.cache[key] = value

                # Actualizar tiempo de expiración si es necesario
                if self.ttl is not None:
                    self.expiry_times[key] = time.time() + self.ttl

                # Registrar en el gestor de memoria
                self.memory_manager.add_item(key, value)

                # Registrar acceso (solo si no es prefetch)
                if not is_prefetch:
                    self.usage_tracker.record_access(key)
                    if hasattr(self.predictor, 'update_patterns'):
                        self.predictor.update_patterns(key)
                    else:
                        self.predictor.update(key)

                    # Sincronizar con otros nodos si está habilitado el caché distribuido
                    if notify_distributed and self.distributed_enabled and self.distributed:
                        # Crear un diccionario con solo esta clave
                        sync_data = {key: value}
                        self.distributed.sync_cache(sync_data)

                        # Registrar operación distribuida
                        if self.monitoring_enabled and self.metrics_collector:
                            self.metrics_collector.record_distributed_operation('put', True)
        finally:
            # Registrar métricas si está habilitado el monitoreo
            if self.monitoring_enabled and self.metrics_collector:
                elapsed_time = time.time() - start_time
                self.metrics_collector.record_put(key, elapsed_time)

                # Registrar uso de memoria
                memory_usage = self.memory_manager.get_memory_usage()
                self.metrics_collector.record_memory_usage(memory_usage, len(self.cache))

    def _evict_items(self, count: int = 1) -> None:
        """
        Elimina elementos del caché según la política de evicción.

        Args:
            count: Número de elementos a eliminar
        """
        # Obtener candidatos para evicción
        candidates = self.memory_manager.get_eviction_candidates(count)

        # Eliminar los candidatos
        for key in candidates:
            self._remove_item(key)
            self.logger.debug(f"Elemento eviccionado: {key}")

            # Registrar evicción en métricas
            if self.monitoring_enabled and self.metrics_collector:
                self.metrics_collector.record_eviction(key)

    def _remove_item(self, key: Any) -> None:
        """
        Elimina un elemento del caché y estructuras relacionadas.

        Args:
            key: La clave a eliminar
        """
        is_expired = False
        if self.ttl is not None and key in self.expiry_times:
            is_expired = time.time() > self.expiry_times[key]

        if key in self.cache:
            del self.cache[key]

        if key in self.expiry_times:
            del self.expiry_times[key]

        self.memory_manager.remove_item(key)

        # Registrar expiración en métricas
        if is_expired and self.monitoring_enabled and self.metrics_collector:
            self.metrics_collector.record_expiration(key)

    def remove(self, key: Any, notify_distributed: bool = True) -> None:
        """
        Elimina explícitamente un elemento del caché.

        Args:
            key: La clave a eliminar
            notify_distributed: Si se debe notificar a otros nodos
        """
        with self.lock:
            self._remove_item(key)

            # Invalidar en otros nodos si está habilitado el caché distribuido
            if notify_distributed and self.distributed_enabled and self.distributed:
                self.distributed.invalidate_key(key)

    def clear(self, notify_distributed: bool = True) -> None:
        """Limpia todo el caché."""
        with self.lock:
            # Guardar claves para invalidar en otros nodos
            keys_to_invalidate = list(self.cache.keys()) if notify_distributed else []

            # Limpiar caché local
            self.cache.clear()
            self.expiry_times.clear()
            self.usage_tracker.clear_history()

            # Reiniciar el gestor de memoria
            max_size = self.memory_manager.max_size
            max_memory_bytes = self.memory_manager.max_memory_bytes
            size_estimator = self.memory_manager.size_estimator
            self.memory_manager = MemoryManager(
                self.usage_tracker,
                max_size=max_size,
                max_memory_mb=max_memory_bytes/(1024*1024),
                size_estimator=size_estimator
            )

            # Invalidar claves en otros nodos
            if notify_distributed and self.distributed_enabled and self.distributed:
                for key in keys_to_invalidate:
                    self.distributed.invalidate_key(key)

    def contains(self, key: Any) -> bool:
        """
        Verifica si una clave existe en el caché y no ha expirado.

        Args:
            key: La clave a verificar

        Returns:
            True si la clave existe y no ha expirado, False en caso contrario
        """
        with self.lock:
            if key not in self.cache:
                return False

            # Verificar expiración
            if self.ttl is not None and key in self.expiry_times:
                if time.time() > self.expiry_times[key]:
                    self._remove_item(key)
                    return False

            return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del caché.

        Returns:
            Diccionario con estadísticas del caché
        """
        with self.lock:
            memory_stats = self.memory_manager.get_memory_usage_stats()

            # Calcular tasa de aciertos si hay suficientes datos
            hit_rate = 0.0
            if hasattr(self, 'hits') and hasattr(self, 'misses'):
                total = self.hits + self.misses
                hit_rate = (self.hits / total) * 100 if total > 0 else 0

            stats = {
                'size': len(self.cache),
                'memory_usage': memory_stats,
                'hit_rate': hit_rate,
                'prefetch_enabled': self.prefetch_enabled,
                'ttl_enabled': self.ttl is not None,
                'advanced_learning': self.using_advanced_learning,
                'predictor_type': type(self.predictor).__name__,
                'persistence_enabled': self.persistence_enabled,
                'distributed_enabled': self.distributed_enabled,
                'monitoring_enabled': self.monitoring_enabled
            }

            # Añadir información de persistencia si está habilitada
            if self.persistence_enabled and self.persistence:
                cache_info = self.persistence.get_cache_info()
                if cache_info:
                    stats['persistence'] = cache_info

            # Añadir información de distribución si está habilitada
            if self.distributed_enabled and self.distributed:
                stats['distributed'] = self.distributed.get_cluster_info()

            # Añadir información de monitoreo si está habilitado
            if self.monitoring_enabled and self.metrics_collector:
                stats['metrics'] = self.metrics_collector.get_metrics_summary()

            return stats

    def stop(self) -> None:
        """Detiene todos los hilos y limpia recursos."""
        # Detener hilo de prefetch
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.stop_prefetch.set()
            self.prefetch_thread.join(timeout=2.0)
            self.logger.info("Hilo de prefetch detenido")

        # Detener guardado automático y guardar estado final
        if self.persistence_enabled and self.persistence:
            # Guardar estado final
            self._save_to_disk()

            # Detener guardado automático
            if hasattr(self.persistence, 'stop_auto_save'):
                self.persistence.stop_auto_save()

        # Detener caché distribuido
        if self.distributed_enabled and self.distributed:
            self.distributed.stop()
            self.logger.info("Caché distribuido detenido")

        # Detener monitor de métricas
        if self.monitoring_enabled and self.metrics_monitor:
            # Exportar métricas finales
            if self.metrics_collector:
                self.metrics_monitor.export_metrics()
                self.metrics_monitor.export_all_historical_metrics()

            self.metrics_monitor.stop()
            self.logger.info("Monitor de métricas detenido")

    def _load_from_disk(self) -> bool:
        """
        Carga el estado del caché desde disco.

        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        if not self.persistence_enabled or not self.persistence:
            return False

        try:
            # Cargar el caché desde disco
            loaded_cache, loaded_expiry_times, metadata = self.persistence.load_cache()

            if loaded_cache is None:
                self.logger.info("No se encontró caché guardado para cargar")
                return False

            # Actualizar el estado interno
            with self.lock:
                self.cache = loaded_cache
                self.expiry_times = loaded_expiry_times

                # Actualizar el gestor de memoria
                for key, value in loaded_cache.items():
                    self.memory_manager.add_item(key, value)

            self.logger.info(f"Caché cargado desde disco ({len(loaded_cache)} elementos)")

            # Iniciar guardado automático si está configurado
            if self.persistence.auto_save_interval:
                self._start_auto_save()

            return True

        except Exception as e:
            self.logger.error(f"Error al cargar caché desde disco: {e}")
            return False

    def _save_to_disk(self) -> bool:
        """
        Guarda el estado del caché en disco.

        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        if not self.persistence_enabled or not self.persistence:
            return False

        try:
            with self.lock:
                # Crear metadatos adicionales
                metadata = {
                    'max_size': self.memory_manager.max_size,
                    'max_memory_mb': self.memory_manager.max_memory_bytes / (1024 * 1024),
                    'ttl_enabled': self.ttl is not None,
                    'prefetch_enabled': self.prefetch_enabled,
                    'advanced_learning': self.using_advanced_learning,
                    'predictor_type': type(self.predictor).__name__
                }

                # Guardar el caché en disco
                success = self.persistence.save_cache(self.cache, self.expiry_times, metadata)

            if success:
                self.logger.info(f"Caché guardado en disco ({len(self.cache)} elementos)")

            return success

        except Exception as e:
            self.logger.error(f"Error al guardar caché en disco: {e}")
            return False

    def _start_auto_save(self) -> None:
        """
        Inicia el guardado automático del caché.
        """
        if not self.persistence_enabled or not self.persistence:
            return

        # Función para proporcionar el estado actual del caché
        def cache_provider():
            with self.lock:
                metadata = {
                    'max_size': self.memory_manager.max_size,
                    'max_memory_mb': self.memory_manager.max_memory_bytes / (1024 * 1024),
                    'ttl_enabled': self.ttl is not None,
                    'prefetch_enabled': self.prefetch_enabled,
                    'advanced_learning': self.using_advanced_learning,
                    'predictor_type': type(self.predictor).__name__
                }
                return self.cache.copy(), self.expiry_times.copy(), metadata

        # Iniciar guardado automático
        self.persistence.start_auto_save(cache_provider)

    def save(self) -> bool:
        """
        Guarda explícitamente el estado del caché en disco.

        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        return self._save_to_disk()

    def _on_distributed_key_invalidated(self, key: Any) -> None:
        """
        Callback cuando una clave es invalidada por otro nodo.

        Args:
            key: Clave invalidada
        """
        with self.lock:
            if key in self.cache:
                self._remove_item(key)
                self.logger.info(f"Clave {key} invalidada por otro nodo")

    def _on_distributed_key_requested(self, key: Any) -> Tuple[Any, bool]:
        """
        Callback cuando otro nodo solicita una clave.

        Args:
            key: Clave solicitada

        Returns:
            Tupla (valor, encontrado)
        """
        with self.lock:
            if key in self.cache:
                # Verificar expiración
                if self.ttl is not None and key in self.expiry_times:
                    if time.time() > self.expiry_times[key]:
                        self._remove_item(key)
                        return None, False

                return self.cache[key], True

            return None, False

    def _on_distributed_sync_requested(self) -> Dict[Any, Any]:
        """
        Callback cuando se solicita una sincronización de caché.

        Returns:
            Diccionario con los datos de caché
        """
        with self.lock:
            # Filtrar elementos expirados
            if self.ttl is not None:
                current_time = time.time()
                expired_keys = [k for k, t in self.expiry_times.items()
                              if current_time > t]

                for key in expired_keys:
                    self._remove_item(key)

            return self.cache.copy()

    def _on_distributed_sync_received(self, cache_data: Dict[Any, Any]) -> None:
        """
        Callback cuando se reciben datos de sincronización.

        Args:
            cache_data: Datos de caché recibidos
        """
        with self.lock:
            # Actualizar caché con los datos recibidos
            for key, value in cache_data.items():
                if key not in self.cache:
                    self.put(key, value, is_prefetch=True)

    def __del__(self) -> None:
        """Destructor que asegura la limpieza de recursos."""
        # Guardar el caché antes de destruir el objeto
        if self.persistence_enabled and self.persistence:
            self._save_to_disk()

        self.stop()
