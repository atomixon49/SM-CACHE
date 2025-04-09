"""
Módulo para la persistencia del caché en el sistema de caché inteligente.
"""
import os
import json
import pickle
import time
import logging
from typing import Dict, Any, Optional, Callable, Tuple, List, Set, Union
import threading
from .utils import CacheSerializer


class CachePersistence:
    """
    Clase para gestionar la persistencia del caché en disco.
    """

    def __init__(self,
                 storage_dir: str = ".cache",
                 serialization_format: str = "json",
                 auto_save_interval: Optional[int] = None,
                 key_serializer: Optional[Callable[[Any], str]] = None,
                 key_deserializer: Optional[Callable[[str], Any]] = None,
                 value_serializer: Optional[Callable[[Any], Any]] = None,
                 value_deserializer: Optional[Callable[[Any], Any]] = None):
        """
        Inicializa el gestor de persistencia.

        Args:
            storage_dir: Directorio donde se almacenarán los archivos de caché
            serialization_format: Formato de serialización ('json' o 'pickle')
            auto_save_interval: Intervalo en segundos para guardado automático (None = desactivado)
            key_serializer: Función para serializar claves no serializables
            key_deserializer: Función para deserializar claves
            value_serializer: Función para serializar valores no serializables
            value_deserializer: Función para deserializar valores
        """
        self.storage_dir = storage_dir
        self.serialization_format = serialization_format
        self.auto_save_interval = auto_save_interval

        # Serializadores personalizados
        self.key_serializer = key_serializer or str
        self.key_deserializer = key_deserializer
        self.value_serializer = value_serializer
        self.value_deserializer = value_deserializer

        # Estado interno
        self.cache_name = "default"
        self.auto_save_thread = None
        self.auto_save_stop_event = threading.Event()
        self.last_save_time = 0

        # Configurar logging
        self.logger = logging.getLogger("CachePersistence")

        # Crear directorio de almacenamiento si no existe
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def set_cache_name(self, name: str) -> None:
        """
        Establece el nombre del caché para identificar los archivos guardados.

        Args:
            name: Nombre del caché
        """
        self.cache_name = name

    def _get_cache_filename(self) -> str:
        """
        Obtiene el nombre de archivo para el caché actual.

        Returns:
            Ruta completa al archivo de caché
        """
        if self.serialization_format == "json":
            extension = "json"
        else:
            extension = "pkl"

        return os.path.join(self.storage_dir, f"{self.cache_name}.{extension}")

    def _get_metadata_filename(self) -> str:
        """
        Obtiene el nombre de archivo para los metadatos del caché.

        Returns:
            Ruta completa al archivo de metadatos
        """
        return os.path.join(self.storage_dir, f"{self.cache_name}.meta.json")

    def save_cache(self,
                  cache: Dict[Any, Any],
                  expiry_times: Dict[Any, float],
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Guarda el estado del caché en disco.

        Args:
            cache: Diccionario de caché
            expiry_times: Diccionario de tiempos de expiración
            metadata: Metadatos adicionales a guardar

        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            cache_filename = self._get_cache_filename()
            metadata_filename = self._get_metadata_filename()

            # Guardar el caché
            if self.serialization_format == "json":
                # Usar el serializador JSON
                serialized_data = CacheSerializer.serialize_cache(
                    cache,
                    expiry_times,
                    key_serializer=self.key_serializer,
                    value_serializer=self.value_serializer
                )

                with open(cache_filename, 'w', encoding='utf-8') as f:
                    f.write(serialized_data)
            else:
                # Usar pickle para serialización binaria
                with open(cache_filename, 'wb') as f:
                    pickle.dump((cache, expiry_times), f)

            # Guardar metadatos
            if metadata is None:
                metadata = {}

            metadata.update({
                'timestamp': time.time(),
                'format': self.serialization_format,
                'cache_name': self.cache_name,
                'item_count': len(cache)
            })

            with open(metadata_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            self.last_save_time = time.time()
            self.logger.info(f"Caché guardado en {cache_filename} ({len(cache)} elementos)")
            return True

        except Exception as e:
            self.logger.error(f"Error al guardar caché: {e}")
            return False

    def load_cache(self) -> Tuple[Optional[Dict[Any, Any]], Optional[Dict[Any, float]], Optional[Dict[str, Any]]]:
        """
        Carga el estado del caché desde disco.

        Returns:
            Tupla (cache, expiry_times, metadata) o (None, None, None) si hay error
        """
        try:
            cache_filename = self._get_cache_filename()
            metadata_filename = self._get_metadata_filename()

            # Verificar si existen los archivos
            if not os.path.exists(cache_filename) or not os.path.exists(metadata_filename):
                self.logger.warning(f"No se encontraron archivos de caché para {self.cache_name}")
                return None, None, None

            # Cargar metadatos
            with open(metadata_filename, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Cargar el caché según el formato
            if metadata.get('format', 'json') == "json":
                with open(cache_filename, 'r', encoding='utf-8') as f:
                    serialized_data = f.read()

                cache, expiry_times = CacheSerializer.deserialize_cache(
                    serialized_data,
                    key_deserializer=self.key_deserializer,
                    value_deserializer=self.value_deserializer
                )
            else:
                with open(cache_filename, 'rb') as f:
                    cache, expiry_times = pickle.load(f)

            self.logger.info(f"Caché cargado desde {cache_filename} ({len(cache)} elementos)")
            return cache, expiry_times, metadata

        except Exception as e:
            self.logger.error(f"Error al cargar caché: {e}")
            return None, None, None

    def start_auto_save(self,
                       cache_provider: Callable[[], Tuple[Dict[Any, Any], Dict[Any, float], Dict[str, Any]]]) -> None:
        """
        Inicia el guardado automático periódico.

        Args:
            cache_provider: Función que devuelve (cache, expiry_times, metadata)
        """
        if self.auto_save_interval is None or self.auto_save_interval <= 0:
            return

        if self.auto_save_thread is not None and self.auto_save_thread.is_alive():
            return

        self.auto_save_stop_event.clear()
        self.auto_save_thread = threading.Thread(
            target=self._auto_save_worker,
            args=(cache_provider,),
            daemon=True
        )
        self.auto_save_thread.start()
        self.logger.info(f"Guardado automático iniciado (intervalo: {self.auto_save_interval}s)")

    def _auto_save_worker(self,
                         cache_provider: Callable[[], Tuple[Dict[Any, Any], Dict[Any, float], Dict[str, Any]]]) -> None:
        """
        Función de trabajo para el hilo de guardado automático.

        Args:
            cache_provider: Función que devuelve (cache, expiry_times, metadata)
        """
        while not self.auto_save_stop_event.is_set():
            try:
                # Esperar el intervalo configurado
                self.auto_save_stop_event.wait(self.auto_save_interval)
                if self.auto_save_stop_event.is_set():
                    break

                # Obtener el estado actual del caché
                cache, expiry_times, metadata = cache_provider()

                # Guardar el caché
                self.save_cache(cache, expiry_times, metadata)

            except Exception as e:
                self.logger.error(f"Error en guardado automático: {e}")
                # Esperar un poco antes de reintentar
                time.sleep(1)

    def stop_auto_save(self) -> None:
        """Detiene el guardado automático."""
        if self.auto_save_thread and self.auto_save_thread.is_alive():
            self.auto_save_stop_event.set()
            self.auto_save_thread.join(timeout=2.0)
            self.logger.info("Guardado automático detenido")

    def clear_cache_files(self) -> bool:
        """
        Elimina los archivos de caché guardados.

        Returns:
            True si se eliminaron correctamente, False en caso contrario
        """
        try:
            cache_filename = self._get_cache_filename()
            metadata_filename = self._get_metadata_filename()

            # Eliminar archivos si existen
            if os.path.exists(cache_filename):
                os.remove(cache_filename)

            if os.path.exists(metadata_filename):
                os.remove(metadata_filename)

            self.logger.info(f"Archivos de caché eliminados para {self.cache_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error al eliminar archivos de caché: {e}")
            return False

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el caché guardado.

        Returns:
            Diccionario con información del caché o diccionario vacío si no existe
        """
        try:
            metadata_filename = self._get_metadata_filename()

            if not os.path.exists(metadata_filename):
                return {}

            with open(metadata_filename, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Añadir información sobre el archivo
            cache_filename = self._get_cache_filename()
            if os.path.exists(cache_filename):
                file_size = os.path.getsize(cache_filename)
                metadata['file_size_bytes'] = file_size
                metadata['file_size_mb'] = file_size / (1024 * 1024)

            return metadata

        except Exception as e:
            self.logger.error(f"Error al obtener información del caché: {e}")
            return {}
