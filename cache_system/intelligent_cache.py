"""
Módulo principal para el sistema de caché inteligente.
"""
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import threading
import time
import logging
import os
from .usage_tracker import UsageTracker
from .predictor import CachePredictor
from .memory_manager import MemoryManager
from .advanced_learning import EnsemblePredictor
from .persistence import CachePersistence
from .distributed import DistributedCache
from .monitoring import MetricsCollector, MetricsMonitor


class IntelligentCache:
    """
    Sistema de caché inteligente que aprende patrones de uso,
    predice qué datos serán necesarios y gestiona automáticamente la memoria.
    """

    def __init__(self,
                 max_size: int = 1000,
                 max_memory_mb: float = 100.0,
                 ttl: Optional[int] = None,
                 prefetch_enabled: bool = True,
                 data_loader: Optional[Callable[[Any], Any]] = None,
                 size_estimator: Optional[Callable[[Any], int]] = None,
                 use_advanced_learning: bool = False,
                 persistence_enabled: bool = False,
                 persistence_dir: str = ".cache",
                 cache_name: str = "default",
                 auto_save_interval: Optional[int] = None,
                 distributed_enabled: bool = False,
                 distributed_host: str = "localhost",
                 distributed_port: int = 5000,
                 cluster_nodes: Optional[List[Tuple[str, int]]] = None,
                 monitoring_enabled: bool = False,
                 metrics_export_interval: Optional[int] = None,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Inicializa el caché inteligente.

        Args:
            max_size: Número máximo de elementos en el caché
            max_memory_mb: Tamaño máximo de memoria en MB
            ttl: Tiempo de vida de los elementos en segundos (None = sin expiración)
            prefetch_enabled: Si se debe habilitar la precarga de datos
            data_loader: Función para cargar datos cuando no están en caché
            size_estimator: Función para estimar el tamaño de un valor en bytes
            use_advanced_learning: Si se deben usar algoritmos de aprendizaje avanzados
            persistence_enabled: Si se debe habilitar la persistencia en disco
            persistence_dir: Directorio donde se guardarán los archivos de caché
            cache_name: Nombre del caché para identificar los archivos guardados
            auto_save_interval: Intervalo en segundos para guardado automático (None = desactivado)
            distributed_enabled: Si se debe habilitar el caché distribuido
            distributed_host: Dirección IP o nombre de host de este nodo
            distributed_port: Puerto de este nodo
            cluster_nodes: Lista de nodos (host, port) en el cluster
            monitoring_enabled: Si se debe habilitar el monitoreo de métricas
            metrics_export_interval: Intervalo en segundos para exportar métricas (None = desactivado)
            alert_thresholds: Umbrales para alertas (ej: {'hit_rate': 50.0})
        """
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("IntelligentCache")

        # Componentes principales
        self.usage_tracker = UsageTracker(max_history_size=max_size * 2)

        # Seleccionar el predictor según la configuración
        if use_advanced_learning:
            self.predictor = EnsemblePredictor(self.usage_tracker)
        else:
            self.predictor = CachePredictor(self.usage_tracker)

        self.memory_manager = MemoryManager(
            self.usage_tracker,
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            size_estimator=size_estimator
        )

        # Guardar el tipo de predictor para estadísticas
        self.using_advanced_learning = use_advanced_learning

        # Configuración
        self.ttl = ttl
        self.prefetch_enabled = prefetch_enabled
        self.data_loader = data_loader

        # Estado interno
        self.cache: Dict[Any, Any] = {}
        self.expiry_times: Dict[Any, float] = {}
        self.lock = threading.RLock()
        self.prefetch_thread: Optional[threading.Thread] = None
        self.stop_prefetch = threading.Event()

        # Configuración de persistencia
        self.persistence_enabled = persistence_enabled
        self.persistence = None

        if persistence_enabled:
            self.persistence = CachePersistence(
                storage_dir=persistence_dir,
                auto_save_interval=auto_save_interval
            )
            self.persistence.set_cache_name(cache_name)

            # Intentar cargar el caché desde disco
            self._load_from_disk()

        # Configuración de caché distribuido
        self.distributed_enabled = distributed_enabled
        self.distributed = None

        if distributed_enabled:
            self.distributed = DistributedCache(
                host=distributed_host,
                port=distributed_port,
                cluster_nodes=cluster_nodes
            )

            # Configurar callbacks
            self.distributed.on_key_invalidated = self._on_distributed_key_invalidated
            self.distributed.on_key_requested = self._on_distributed_key_requested
            self.distributed.on_sync_requested = self._on_distributed_sync_requested
            self.distributed.on_sync_received = self._on_distributed_sync_received

            # Iniciar el caché distribuido
            self.distributed.start()

        # Configuración de monitoreo
        self.monitoring_enabled = monitoring_enabled
        self.metrics_collector = None
        self.metrics_monitor = None

        if monitoring_enabled:
            self.metrics_collector = MetricsCollector()
            self.metrics_monitor = MetricsMonitor(
                metrics_collector=self.metrics_collector,
                export_interval=metrics_export_interval,
                alert_thresholds=alert_thresholds
            )

            # Iniciar el monitor de métricas
            self.metrics_monitor.start()
            self.logger.info("Monitor de métricas iniciado")

        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("IntelligentCache")

        # Iniciar el hilo de prefetch si está habilitado
        if self.prefetch_enabled and self.data_loader:
            self._start_prefetch_thread()

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
                    self.predictor.update_patterns(key)

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
                    self.predictor.update_patterns(key)

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
