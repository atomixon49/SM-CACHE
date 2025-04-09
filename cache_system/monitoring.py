"""
Módulo para monitoreo y métricas del sistema de caché inteligente.
"""
import time
import threading
import logging
import json
import os
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from collections import deque


class MetricsCollector:
    """
    Recolector de métricas para el sistema de caché inteligente.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Inicializa el recolector de métricas.
        
        Args:
            max_history: Número máximo de puntos de datos históricos a mantener
        """
        # Métricas básicas
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.evictions = 0
        self.expirations = 0
        
        # Métricas de rendimiento
        self.get_times: deque = deque(maxlen=max_history)
        self.put_times: deque = deque(maxlen=max_history)
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        
        # Métricas de memoria
        self.memory_usage: deque = deque(maxlen=max_history)
        self.item_count: deque = deque(maxlen=max_history)
        
        # Métricas de predicción
        self.prediction_hits = 0
        self.prediction_misses = 0
        
        # Métricas de distribución
        self.distributed_gets = 0
        self.distributed_puts = 0
        self.distributed_hits = 0
        self.distributed_misses = 0
        
        # Historial de métricas con timestamps
        self.history: Dict[str, deque] = {
            'hit_rate': deque(maxlen=max_history),
            'memory_usage': deque(maxlen=max_history),
            'item_count': deque(maxlen=max_history),
            'get_latency': deque(maxlen=max_history),
            'put_latency': deque(maxlen=max_history),
            'prediction_accuracy': deque(maxlen=max_history)
        }
        
        # Configuración
        self.max_history = max_history
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MetricsCollector")
    
    def record_get(self, key: Any, hit: bool, elapsed_time: float, from_prefetch: bool = False) -> None:
        """
        Registra una operación de obtención.
        
        Args:
            key: Clave accedida
            hit: Si fue un acierto o no
            elapsed_time: Tiempo de ejecución en segundos
            from_prefetch: Si el valor se obtuvo de una precarga
        """
        if hit:
            self.hits += 1
            if from_prefetch:
                self.prefetch_hits += 1
        else:
            self.misses += 1
            if from_prefetch:
                self.prefetch_misses += 1
                
        self.get_times.append(elapsed_time)
        
        # Actualizar historial
        timestamp = time.time()
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        self.history['hit_rate'].append((timestamp, hit_rate))
        self.history['get_latency'].append((timestamp, elapsed_time))
    
    def record_put(self, key: Any, elapsed_time: float) -> None:
        """
        Registra una operación de almacenamiento.
        
        Args:
            key: Clave almacenada
            elapsed_time: Tiempo de ejecución en segundos
        """
        self.puts += 1
        self.put_times.append(elapsed_time)
        
        # Actualizar historial
        timestamp = time.time()
        self.history['put_latency'].append((timestamp, elapsed_time))
    
    def record_eviction(self, key: Any) -> None:
        """
        Registra una evicción de caché.
        
        Args:
            key: Clave eviccionada
        """
        self.evictions += 1
    
    def record_expiration(self, key: Any) -> None:
        """
        Registra una expiración de caché.
        
        Args:
            key: Clave expirada
        """
        self.expirations += 1
    
    def record_memory_usage(self, bytes_used: int, item_count: int) -> None:
        """
        Registra el uso de memoria.
        
        Args:
            bytes_used: Bytes utilizados
            item_count: Número de elementos
        """
        timestamp = time.time()
        self.memory_usage.append((timestamp, bytes_used))
        self.item_count.append((timestamp, item_count))
        
        # Actualizar historial
        self.history['memory_usage'].append((timestamp, bytes_used))
        self.history['item_count'].append((timestamp, item_count))
    
    def record_prediction(self, success: bool) -> None:
        """
        Registra el resultado de una predicción.
        
        Args:
            success: Si la predicción fue exitosa
        """
        if success:
            self.prediction_hits += 1
        else:
            self.prediction_misses += 1
            
        # Actualizar historial
        timestamp = time.time()
        total = self.prediction_hits + self.prediction_misses
        accuracy = (self.prediction_hits / total) * 100 if total > 0 else 0
        self.history['prediction_accuracy'].append((timestamp, accuracy))
    
    def record_distributed_operation(self, operation: str, success: bool) -> None:
        """
        Registra una operación distribuida.
        
        Args:
            operation: Tipo de operación ('get' o 'put')
            success: Si la operación fue exitosa
        """
        if operation == 'get':
            self.distributed_gets += 1
            if success:
                self.distributed_hits += 1
            else:
                self.distributed_misses += 1
        elif operation == 'put':
            self.distributed_puts += 1
    
    def get_hit_rate(self) -> float:
        """
        Obtiene la tasa de aciertos.
        
        Returns:
            Tasa de aciertos como porcentaje
        """
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0
    
    def get_prefetch_hit_rate(self) -> float:
        """
        Obtiene la tasa de aciertos de precarga.
        
        Returns:
            Tasa de aciertos de precarga como porcentaje
        """
        total = self.prefetch_hits + self.prefetch_misses
        return (self.prefetch_hits / total) * 100 if total > 0 else 0
    
    def get_prediction_accuracy(self) -> float:
        """
        Obtiene la precisión de las predicciones.
        
        Returns:
            Precisión de predicciones como porcentaje
        """
        total = self.prediction_hits + self.prediction_misses
        return (self.prediction_hits / total) * 100 if total > 0 else 0
    
    def get_average_get_time(self) -> float:
        """
        Obtiene el tiempo promedio de obtención.
        
        Returns:
            Tiempo promedio en segundos
        """
        return sum(self.get_times) / len(self.get_times) if self.get_times else 0
    
    def get_average_put_time(self) -> float:
        """
        Obtiene el tiempo promedio de almacenamiento.
        
        Returns:
            Tiempo promedio en segundos
        """
        return sum(self.put_times) / len(self.put_times) if self.put_times else 0
    
    def get_distributed_hit_rate(self) -> float:
        """
        Obtiene la tasa de aciertos distribuidos.
        
        Returns:
            Tasa de aciertos distribuidos como porcentaje
        """
        total = self.distributed_hits + self.distributed_misses
        return (self.distributed_hits / total) * 100 if total > 0 else 0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de las métricas.
        
        Returns:
            Diccionario con métricas resumidas
        """
        return {
            'operations': {
                'gets': self.hits + self.misses,
                'hits': self.hits,
                'misses': self.misses,
                'puts': self.puts,
                'evictions': self.evictions,
                'expirations': self.expirations
            },
            'performance': {
                'hit_rate': self.get_hit_rate(),
                'avg_get_time': self.get_average_get_time(),
                'avg_put_time': self.get_average_put_time()
            },
            'prefetch': {
                'hits': self.prefetch_hits,
                'misses': self.prefetch_misses,
                'hit_rate': self.get_prefetch_hit_rate()
            },
            'prediction': {
                'hits': self.prediction_hits,
                'misses': self.prediction_misses,
                'accuracy': self.get_prediction_accuracy()
            },
            'distributed': {
                'gets': self.distributed_gets,
                'puts': self.distributed_puts,
                'hits': self.distributed_hits,
                'misses': self.distributed_misses,
                'hit_rate': self.get_distributed_hit_rate()
            }
        }
    
    def get_historical_metrics(self, metric_name: str, 
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None) -> List[Tuple[float, float]]:
        """
        Obtiene métricas históricas para un período específico.
        
        Args:
            metric_name: Nombre de la métrica
            start_time: Tiempo de inicio (None = desde el principio)
            end_time: Tiempo de fin (None = hasta el final)
            
        Returns:
            Lista de tuplas (timestamp, valor)
        """
        if metric_name not in self.history:
            return []
            
        if start_time is None and end_time is None:
            return list(self.history[metric_name])
            
        start_time = start_time or 0
        end_time = end_time or float('inf')
        
        return [(t, v) for t, v in self.history[metric_name] 
               if start_time <= t <= end_time]
    
    def reset(self) -> None:
        """Reinicia todas las métricas."""
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.evictions = 0
        self.expirations = 0
        self.get_times.clear()
        self.put_times.clear()
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.memory_usage.clear()
        self.item_count.clear()
        self.prediction_hits = 0
        self.prediction_misses = 0
        self.distributed_gets = 0
        self.distributed_puts = 0
        self.distributed_hits = 0
        self.distributed_misses = 0
        
        for queue in self.history.values():
            queue.clear()


class MetricsExporter:
    """
    Exportador de métricas para el sistema de caché inteligente.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, export_dir: str = ".metrics"):
        """
        Inicializa el exportador de métricas.
        
        Args:
            metrics_collector: Recolector de métricas
            export_dir: Directorio donde se exportarán las métricas
        """
        self.metrics_collector = metrics_collector
        self.export_dir = export_dir
        
        # Crear directorio si no existe
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MetricsExporter")
    
    def export_to_json(self, filename: str = "metrics.json") -> bool:
        """
        Exporta las métricas a un archivo JSON.
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            True si se exportó correctamente, False en caso contrario
        """
        try:
            # Obtener métricas
            metrics = self.metrics_collector.get_metrics_summary()
            
            # Añadir timestamp
            metrics['timestamp'] = time.time()
            
            # Guardar a archivo
            filepath = os.path.join(self.export_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
                
            self.logger.info(f"Métricas exportadas a {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al exportar métricas: {e}")
            return False
    
    def export_historical_metrics(self, metric_name: str, 
                                 filename: Optional[str] = None) -> bool:
        """
        Exporta métricas históricas a un archivo JSON.
        
        Args:
            metric_name: Nombre de la métrica
            filename: Nombre del archivo (None = usar nombre de métrica)
            
        Returns:
            True si se exportó correctamente, False en caso contrario
        """
        try:
            # Obtener métricas históricas
            metrics = self.metrics_collector.get_historical_metrics(metric_name)
            
            if not metrics:
                self.logger.warning(f"No hay métricas históricas para {metric_name}")
                return False
                
            # Crear estructura de datos
            data = {
                'metric': metric_name,
                'timestamp': time.time(),
                'data': [{'timestamp': t, 'value': v} for t, v in metrics]
            }
            
            # Guardar a archivo
            if filename is None:
                filename = f"{metric_name}_history.json"
                
            filepath = os.path.join(self.export_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Métricas históricas exportadas a {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al exportar métricas históricas: {e}")
            return False
    
    def export_all_historical_metrics(self) -> bool:
        """
        Exporta todas las métricas históricas.
        
        Returns:
            True si todas se exportaron correctamente, False si alguna falló
        """
        success = True
        for metric_name in self.metrics_collector.history.keys():
            if not self.export_historical_metrics(metric_name):
                success = False
                
        return success


class MetricsMonitor:
    """
    Monitor de métricas para el sistema de caché inteligente.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, 
                export_interval: Optional[int] = None,
                alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Inicializa el monitor de métricas.
        
        Args:
            metrics_collector: Recolector de métricas
            export_interval: Intervalo en segundos para exportar métricas (None = desactivado)
            alert_thresholds: Umbrales para alertas (ej: {'hit_rate': 50.0})
        """
        self.metrics_collector = metrics_collector
        self.export_interval = export_interval
        self.alert_thresholds = alert_thresholds or {}
        
        # Estado interno
        self.running = False
        self.exporter = MetricsExporter(metrics_collector)
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitor = threading.Event()
        
        # Callbacks
        self.on_alert: Optional[Callable[[str, float, float], None]] = None
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MetricsMonitor")
    
    def start(self) -> bool:
        """
        Inicia el monitor de métricas.
        
        Returns:
            True si se inició correctamente, False en caso contrario
        """
        if self.running:
            return True
            
        try:
            # Iniciar hilo de monitoreo
            self.stop_monitor.clear()
            self.monitor_thread = threading.Thread(
                target=self._monitor_worker,
                daemon=True
            )
            self.monitor_thread.start()
            
            self.running = True
            self.logger.info("Monitor de métricas iniciado")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al iniciar monitor de métricas: {e}")
            return False
    
    def stop(self) -> None:
        """Detiene el monitor de métricas."""
        if not self.running:
            return
            
        # Detener hilo de monitoreo
        self.stop_monitor.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
        self.running = False
        self.logger.info("Monitor de métricas detenido")
    
    def _monitor_worker(self) -> None:
        """Función de trabajo para el hilo de monitoreo."""
        last_export_time = 0
        
        while not self.stop_monitor.is_set():
            try:
                # Verificar alertas
                self._check_alerts()
                
                # Exportar métricas si es necesario
                if self.export_interval is not None:
                    current_time = time.time()
                    if current_time - last_export_time >= self.export_interval:
                        self.exporter.export_to_json()
                        last_export_time = current_time
                
                # Esperar antes de la siguiente verificación
                self.stop_monitor.wait(timeout=1.0)
                
            except Exception as e:
                self.logger.error(f"Error en hilo de monitoreo: {e}")
                time.sleep(1.0)
    
    def _check_alerts(self) -> None:
        """Verifica si se deben generar alertas."""
        # Obtener métricas actuales
        metrics = self.metrics_collector.get_metrics_summary()
        
        # Verificar umbrales
        for metric_path, threshold in self.alert_thresholds.items():
            # Obtener valor actual
            parts = metric_path.split('.')
            value = metrics
            for part in parts:
                if part in value:
                    value = value[part]
                else:
                    value = None
                    break
            
            if value is None:
                continue
                
            # Verificar umbral
            if isinstance(value, (int, float)) and value < threshold:
                self._trigger_alert(metric_path, value, threshold)
    
    def _trigger_alert(self, metric_path: str, value: float, threshold: float) -> None:
        """
        Genera una alerta.
        
        Args:
            metric_path: Ruta de la métrica
            value: Valor actual
            threshold: Umbral configurado
        """
        message = f"Alerta: {metric_path} = {value:.2f} (umbral: {threshold:.2f})"
        self.logger.warning(message)
        
        # Llamar callback si está configurado
        if self.on_alert:
            self.on_alert(metric_path, value, threshold)
    
    def export_metrics(self) -> bool:
        """
        Exporta las métricas actuales.
        
        Returns:
            True si se exportó correctamente, False en caso contrario
        """
        return self.exporter.export_to_json()
    
    def export_all_historical_metrics(self) -> bool:
        """
        Exporta todas las métricas históricas.
        
        Returns:
            True si todas se exportaron correctamente, False si alguna falló
        """
        return self.exporter.export_all_historical_metrics()
