"""
Sistema de persistencia para el caché con guardado adaptativo.
"""
import json
import os
import time
import threading
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

class CachePersistence:
    """Gestiona la persistencia del caché con guardado adaptativo."""
    
    def __init__(self, cache_dir: str = ".cache",
                 base_interval: int = 300,
                 min_interval: int = 60,
                 max_interval: int = 900,
                 change_threshold: float = 0.1):
        """
        Inicializa el sistema de persistencia.
        
        Args:
            cache_dir: Directorio para archivos de caché
            base_interval: Intervalo base de guardado en segundos
            min_interval: Intervalo mínimo de guardado
            max_interval: Intervalo máximo de guardado
            change_threshold: Umbral de cambios para ajustar intervalo
        """
        self.cache_dir = cache_dir
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.change_threshold = change_threshold
        
        # Estado interno
        self.running = False
        self.last_save_time = 0
        self.last_save_size = 0
        self.current_interval = base_interval
        self._lock = threading.RLock()
        
        # Control de guardado automático
        self.auto_save_thread: Optional[threading.Thread] = None
        self.stop_auto_save = threading.Event()
        
        # Métricas de cambios
        self.changes_since_last_save = 0
        self.total_operations = 0
        
        # Hooks
        self.pre_save_hook: Optional[Callable[[], None]] = None
        self.post_save_hook: Optional[Callable[[], None]] = None
        
        # Crear directorio si no existe
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CachePersistence")
    
    def start_auto_save(self) -> None:
        """Inicia el guardado automático adaptativo."""
        if self.running:
            return
            
        self.running = True
        self.stop_auto_save.clear()
        
        self.auto_save_thread = threading.Thread(
            target=self._auto_save_loop,
            daemon=True
        )
        self.auto_save_thread.start()
        
        self.logger.info(f"Guardado automático iniciado (intervalo: {self.current_interval}s)")
    
    def stop_auto_save(self) -> None:
        """Detiene el guardado automático."""
        if not self.running:
            return
            
        self.running = False
        self.stop_auto_save.set()
        
        if self.auto_save_thread:
            self.auto_save_thread.join(timeout=5.0)
            
        self.logger.info("Guardado automático detenido")
    
    def save_cache(self, cache_data: Dict[str, Any], filename: str) -> bool:
        """
        Guarda los datos del caché en disco.
        
        Args:
            cache_data: Datos a guardar
            filename: Nombre del archivo
            
        Returns:
            True si se guardó correctamente
        """
        try:
            if self.pre_save_hook:
                self.pre_save_hook()
                
            filepath = os.path.join(self.cache_dir, filename)
            
            # Crear backup antes de guardar
            if os.path.exists(filepath):
                backup_path = f"{filepath}.bak"
                try:
                    os.replace(filepath, backup_path)
                except Exception as e:
                    self.logger.warning(f"Error creando backup: {e}")
            
            # Guardar datos
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.utcnow().isoformat(),
                    'data': cache_data
                }, f, indent=2)
                
            self.last_save_time = time.time()
            self.last_save_size = len(cache_data)
            
            # Actualizar métricas
            with self._lock:
                self.changes_since_last_save = 0
                self._adjust_save_interval()
            
            if self.post_save_hook:
                self.post_save_hook()
                
            self.logger.info(f"Caché guardado en {filepath} ({len(cache_data)} elementos)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando caché: {e}")
            return False
    
    def load_cache(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Carga datos del caché desde disco.
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Datos cargados o None si hay error
        """
        try:
            filepath = os.path.join(self.cache_dir, filename)
            
            # Si existe backup y el archivo principal está corrupto, restaurar
            backup_path = f"{filepath}.bak"
            if os.path.exists(backup_path):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        json.load(f)  # Verificar si es JSON válido
                except Exception:
                    self.logger.warning("Archivo principal corrupto, restaurando backup")
                    os.replace(backup_path, filepath)
            
            # Cargar datos
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            cache_data = data.get('data', {})
            self.last_save_size = len(cache_data)
            
            self.logger.info(f"Caché cargado desde {filepath} ({len(cache_data)} elementos)")
            return cache_data
            
        except FileNotFoundError:
            self.logger.info(f"No se encontró archivo de caché: {filename}")
            return None
        except Exception as e:
            self.logger.error(f"Error cargando caché: {e}")
            return None
    
    def record_operation(self, is_write: bool = False) -> None:
        """
        Registra una operación para ajustar el intervalo de guardado.
        
        Args:
            is_write: Si la operación es de escritura
        """
        with self._lock:
            self.total_operations += 1
            if is_write:
                self.changes_since_last_save += 1
    
    def _adjust_save_interval(self) -> None:
        """Ajusta el intervalo de guardado basado en la tasa de cambios."""
        if self.total_operations == 0:
            return
            
        change_rate = self.changes_since_last_save / max(1, self.total_operations)
        
        # Ajustar intervalo basado en la tasa de cambios
        if change_rate > self.change_threshold:
            # Más cambios -> guardar más frecuentemente
            self.current_interval = max(
                self.min_interval,
                self.current_interval * 0.8
            )
        else:
            # Menos cambios -> guardar menos frecuentemente
            self.current_interval = min(
                self.max_interval,
                self.current_interval * 1.2
            )
        
        # Reiniciar contadores
        self.total_operations = 0
        self.changes_since_last_save = 0
    
    def _auto_save_loop(self) -> None:
        """Loop principal para guardado automático."""
        while self.running:
            try:
                next_save = self.last_save_time + self.current_interval
                wait_time = max(0, next_save - time.time())
                
                if self.stop_auto_save.wait(timeout=wait_time):
                    break
                    
                # Si hay hooks configurados, ejecutar
                if self.pre_save_hook:
                    self.pre_save_hook()
                    
                # El guardado real se maneja en save_cache()
                # Este loop solo maneja la temporización
                
                if self.post_save_hook:
                    self.post_save_hook()
                    
            except Exception as e:
                self.logger.error(f"Error en auto_save_loop: {e}")
                time.sleep(1)
