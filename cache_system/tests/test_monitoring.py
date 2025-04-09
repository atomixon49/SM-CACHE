"""
Pruebas para el monitoreo y métricas del sistema de caché inteligente.
"""
import unittest
import time
import os
import tempfile
import json
from cache_system.monitoring import MetricsCollector, MetricsExporter, MetricsMonitor
from cache_system import IntelligentCache


class TestMetricsCollector(unittest.TestCase):
    """Pruebas para la clase MetricsCollector."""
    
    def setUp(self):
        """Configuración para cada prueba."""
        self.metrics = MetricsCollector()
    
    def test_record_get(self):
        """Prueba el registro de operaciones de obtención."""
        # Registrar algunos accesos
        self.metrics.record_get("key1", True, 0.01)  # Hit
        self.metrics.record_get("key2", False, 0.02)  # Miss
        self.metrics.record_get("key3", True, 0.03, from_prefetch=True)  # Prefetch hit
        
        # Verificar contadores
        self.assertEqual(self.metrics.hits, 2)
        self.assertEqual(self.metrics.misses, 1)
        self.assertEqual(self.metrics.prefetch_hits, 1)
        
        # Verificar tasa de aciertos
        self.assertAlmostEqual(self.metrics.get_hit_rate(), 66.67, delta=0.01)
    
    def test_record_put(self):
        """Prueba el registro de operaciones de almacenamiento."""
        # Registrar algunas operaciones
        self.metrics.record_put("key1", 0.01)
        self.metrics.record_put("key2", 0.02)
        
        # Verificar contador
        self.assertEqual(self.metrics.puts, 2)
        
        # Verificar tiempo promedio
        self.assertAlmostEqual(self.metrics.get_average_put_time(), 0.015, delta=0.001)
    
    def test_record_eviction_expiration(self):
        """Prueba el registro de evicciones y expiraciones."""
        # Registrar eventos
        self.metrics.record_eviction("key1")
        self.metrics.record_eviction("key2")
        self.metrics.record_expiration("key3")
        
        # Verificar contadores
        self.assertEqual(self.metrics.evictions, 2)
        self.assertEqual(self.metrics.expirations, 1)
    
    def test_record_memory_usage(self):
        """Prueba el registro de uso de memoria."""
        # Registrar uso de memoria
        self.metrics.record_memory_usage(1024, 10)
        self.metrics.record_memory_usage(2048, 20)
        
        # Verificar que se registró correctamente
        self.assertEqual(len(self.metrics.memory_usage), 2)
        self.assertEqual(len(self.metrics.item_count), 2)
        
        # Verificar que el historial se actualizó
        self.assertEqual(len(self.metrics.history['memory_usage']), 2)
        self.assertEqual(len(self.metrics.history['item_count']), 2)
    
    def test_get_metrics_summary(self):
        """Prueba la obtención del resumen de métricas."""
        # Registrar algunas métricas
        self.metrics.record_get("key1", True, 0.01)
        self.metrics.record_put("key2", 0.02)
        self.metrics.record_eviction("key3")
        self.metrics.record_memory_usage(1024, 10)
        
        # Obtener resumen
        summary = self.metrics.get_metrics_summary()
        
        # Verificar estructura
        self.assertIn('operations', summary)
        self.assertIn('performance', summary)
        self.assertIn('prefetch', summary)
        self.assertIn('prediction', summary)
        self.assertIn('distributed', summary)
        
        # Verificar valores
        self.assertEqual(summary['operations']['hits'], 1)
        self.assertEqual(summary['operations']['puts'], 1)
        self.assertEqual(summary['operations']['evictions'], 1)
    
    def test_reset(self):
        """Prueba el reinicio de métricas."""
        # Registrar algunas métricas
        self.metrics.record_get("key1", True, 0.01)
        self.metrics.record_put("key2", 0.02)
        
        # Reiniciar
        self.metrics.reset()
        
        # Verificar que se reiniciaron
        self.assertEqual(self.metrics.hits, 0)
        self.assertEqual(self.metrics.puts, 0)
        self.assertEqual(len(self.metrics.get_times), 0)


class TestMetricsExporter(unittest.TestCase):
    """Pruebas para la clase MetricsExporter."""
    
    def setUp(self):
        """Configuración para cada prueba."""
        self.metrics = MetricsCollector()
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = MetricsExporter(self.metrics, export_dir=self.temp_dir)
    
    def tearDown(self):
        """Limpieza después de cada prueba."""
        # Eliminar archivos temporales
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_export_to_json(self):
        """Prueba la exportación a JSON."""
        # Registrar algunas métricas
        self.metrics.record_get("key1", True, 0.01)
        self.metrics.record_put("key2", 0.02)
        
        # Exportar
        filename = "test_metrics.json"
        success = self.exporter.export_to_json(filename)
        
        # Verificar
        self.assertTrue(success)
        filepath = os.path.join(self.temp_dir, filename)
        self.assertTrue(os.path.exists(filepath))
        
        # Verificar contenido
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertIn('operations', data)
            self.assertEqual(data['operations']['hits'], 1)
    
    def test_export_historical_metrics(self):
        """Prueba la exportación de métricas históricas."""
        # Registrar algunas métricas
        self.metrics.record_get("key1", True, 0.01)
        self.metrics.record_get("key2", False, 0.02)
        
        # Exportar
        success = self.exporter.export_historical_metrics('hit_rate')
        
        # Verificar
        self.assertTrue(success)
        filepath = os.path.join(self.temp_dir, "hit_rate_history.json")
        self.assertTrue(os.path.exists(filepath))
        
        # Verificar contenido
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertEqual(data['metric'], 'hit_rate')
            self.assertEqual(len(data['data']), 2)


class TestMetricsMonitor(unittest.TestCase):
    """Pruebas para la clase MetricsMonitor."""
    
    def setUp(self):
        """Configuración para cada prueba."""
        self.metrics = MetricsCollector()
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = MetricsMonitor(
            metrics_collector=self.metrics,
            export_interval=1,
            alert_thresholds={'performance.hit_rate': 50.0}
        )
        self.monitor.exporter = MetricsExporter(self.metrics, export_dir=self.temp_dir)
        
        # Variables para alertas
        self.alert_triggered = False
        self.alert_metric = None
        
        # Configurar callback de alerta
        def on_alert(metric, value, threshold):
            self.alert_triggered = True
            self.alert_metric = metric
            
        self.monitor.on_alert = on_alert
    
    def tearDown(self):
        """Limpieza después de cada prueba."""
        if self.monitor.running:
            self.monitor.stop()
            
        # Eliminar archivos temporales
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_start_stop(self):
        """Prueba iniciar y detener el monitor."""
        # Iniciar
        success = self.monitor.start()
        self.assertTrue(success)
        self.assertTrue(self.monitor.running)
        
        # Verificar que el hilo está ejecutándose
        self.assertIsNotNone(self.monitor.monitor_thread)
        self.assertTrue(self.monitor.monitor_thread.is_alive())
        
        # Detener
        self.monitor.stop()
        self.assertFalse(self.monitor.running)
        
        # Verificar que el hilo se detuvo
        time.sleep(0.1)  # Dar tiempo a que el hilo termine
        self.assertFalse(self.monitor.monitor_thread.is_alive())
    
    def test_export_metrics(self):
        """Prueba la exportación manual de métricas."""
        # Registrar algunas métricas
        self.metrics.record_get("key1", True, 0.01)
        
        # Exportar
        success = self.monitor.export_metrics()
        
        # Verificar
        self.assertTrue(success)
        filepath = os.path.join(self.temp_dir, "metrics.json")
        self.assertTrue(os.path.exists(filepath))
    
    def test_alert_system(self):
        """Prueba el sistema de alertas."""
        # Configurar métricas para generar alerta
        self.metrics.hits = 4
        self.metrics.misses = 6  # Hit rate = 40%
        
        # Verificar alertas
        self.monitor._check_alerts()
        
        # Verificar que se generó la alerta
        self.assertTrue(self.alert_triggered)
        self.assertEqual(self.alert_metric, 'performance.hit_rate')


class TestIntelligentCacheMonitoring(unittest.TestCase):
    """Pruebas para la integración de monitoreo en IntelligentCache."""
    
    def setUp(self):
        """Configuración para cada prueba."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Crear caché con monitoreo habilitado
        self.cache = IntelligentCache(
            max_size=10,
            monitoring_enabled=True,
            metrics_export_interval=1,
            alert_thresholds={'performance.hit_rate': 50.0}
        )
        
        # Configurar directorio de exportación
        if self.cache.metrics_monitor and self.cache.metrics_monitor.exporter:
            self.cache.metrics_monitor.exporter.export_dir = self.temp_dir
    
    def tearDown(self):
        """Limpieza después de cada prueba."""
        self.cache.stop()
        
        # Eliminar archivos temporales
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
    
    def test_monitoring_enabled(self):
        """Prueba que el monitoreo está habilitado."""
        self.assertTrue(self.cache.monitoring_enabled)
        self.assertIsNotNone(self.cache.metrics_collector)
        self.assertIsNotNone(self.cache.metrics_monitor)
    
    def test_metrics_collection(self):
        """Prueba la recolección de métricas durante operaciones de caché."""
        # Realizar operaciones
        self.cache.put("key1", "value1")
        self.cache.get("key1")
        self.cache.get("key2")  # Miss
        
        # Verificar métricas
        metrics = self.cache.metrics_collector.get_metrics_summary()
        self.assertEqual(metrics['operations']['puts'], 1)
        self.assertEqual(metrics['operations']['hits'], 1)
        self.assertEqual(metrics['operations']['misses'], 1)
        self.assertAlmostEqual(metrics['performance']['hit_rate'], 50.0)
    
    def test_metrics_in_stats(self):
        """Prueba que las métricas se incluyen en las estadísticas."""
        # Realizar operaciones
        self.cache.put("key1", "value1")
        self.cache.get("key1")
        
        # Obtener estadísticas
        stats = self.cache.get_stats()
        
        # Verificar que incluye métricas
        self.assertIn('metrics', stats)
        self.assertIn('operations', stats['metrics'])
        self.assertEqual(stats['metrics']['operations']['puts'], 1)
        self.assertEqual(stats['metrics']['operations']['hits'], 1)
    
    def test_export_on_stop(self):
        """Prueba que las métricas se exportan al detener el caché."""
        # Realizar operaciones
        self.cache.put("key1", "value1")
        self.cache.get("key1")
        
        # Detener caché (debería exportar métricas)
        self.cache.stop()
        
        # Verificar que se exportaron
        metrics_file = os.path.join(self.temp_dir, "metrics.json")
        self.assertTrue(os.path.exists(metrics_file))


if __name__ == '__main__':
    unittest.main()
