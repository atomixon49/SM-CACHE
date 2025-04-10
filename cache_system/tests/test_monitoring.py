"""
Pruebas unitarias para el sistema de monitoreo.
"""
import unittest
from prometheus_client import CollectorRegistry
from cache_system.monitoring import MetricsCollector

class TestMetricsCollector(unittest.TestCase):
    """Pruebas para el recolector de métricas."""
    
    def setUp(self):
        """Inicializa el entorno de prueba."""
        self.metrics = MetricsCollector()

    def test_hit_miss_tracking(self):
        """Prueba el seguimiento de hits y misses."""
        # Registrar hits
        self.metrics.record_hit()
        self.metrics.record_hit()
        
        # Registrar misses
        self.metrics.record_miss()
        
        # Verificar contadores
        self.assertEqual(self.metrics.hits, 2)
        self.assertEqual(self.metrics.misses, 1)
        
        # Verificar métricas Prometheus
        self.assertEqual(2, self.metrics.prom_hits._value.get())
        self.assertEqual(1, self.metrics.prom_misses._value.get())

    def test_memory_tracking(self):
        """Prueba el seguimiento de uso de memoria."""
        # Registrar uso de memoria
        self.metrics.record_memory_usage(1024)  # 1KB
        
        # Verificar métrica
        self.assertEqual(1024, self.metrics.prom_memory._value.get())

    def test_operation_tracking(self):
        """Prueba el seguimiento de operaciones."""
        # Registrar puts
        self.metrics.record_put()
        self.metrics.record_put()
        
        # Registrar evictions
        self.metrics.record_eviction()
        
        # Verificar contadores
        self.assertEqual(self.metrics.puts, 2)
        self.assertEqual(self.metrics.evictions, 1)
        
        # Verificar métricas Prometheus
        self.assertEqual(2, self.metrics.prom_puts._value.get())
        self.assertEqual(1, self.metrics.prom_evictions._value.get())

    def test_expiration_tracking(self):
        """Prueba el seguimiento de expiraciones."""
        # Registrar expiraciones
        self.metrics.record_expiration()
        self.metrics.record_expiration()
        
        # Verificar contador
        self.assertEqual(self.metrics.expirations, 2)
        
        # Verificar métrica Prometheus
        self.assertEqual(2, self.metrics.prom_expirations._value.get())

    def test_item_count_tracking(self):
        """Prueba el seguimiento del número de elementos."""
        # Registrar elementos
        self.metrics.record_item_count(5)
        
        # Verificar métrica Prometheus
        self.assertEqual(5, self.metrics.prom_items._value.get())

    def test_latency_tracking(self):
        """Prueba el seguimiento de latencia."""
        # Registrar latencias de get
        with self.metrics.track_get_latency():
            pass  # Simular operación get
            
        # Registrar latencias de put
        with self.metrics.track_put_latency():
            pass  # Simular operación put
            
        # Verificar que se registraron las latencias
        self.assertGreater(
            self.metrics.prom_get_latency._summary.sum.get(), 
            0
        )
        self.assertGreater(
            self.metrics.prom_put_latency._summary.sum.get(), 
            0
        )

    def test_registry_isolation(self):
        """Prueba que cada instancia tiene su propio registro."""
        metrics1 = MetricsCollector()
        metrics2 = MetricsCollector()
        
        # Registrar hits en cada instancia
        metrics1.record_hit()
        metrics2.record_hit()
        metrics2.record_hit()
        
        # Verificar que los contadores son independientes
        self.assertEqual(1, metrics1.prom_hits._value.get())
        self.assertEqual(2, metrics2.prom_hits._value.get())

    def test_clear_metrics(self):
        """Prueba la limpieza de métricas."""
        # Registrar algunas métricas
        self.metrics.record_hit()
        self.metrics.record_miss()
        self.metrics.record_put()
        
        # Limpiar métricas
        self.metrics.clear()
        
        # Verificar que los contadores se reiniciaron
        self.assertEqual(self.metrics.hits, 0)
        self.assertEqual(self.metrics.misses, 0)
        self.assertEqual(self.metrics.puts, 0)
        
        # Verificar que las métricas Prometheus se reiniciaron
        self.assertEqual(0, self.metrics.prom_hits._value.get())
        self.assertEqual(0, self.metrics.prom_misses._value.get())
        self.assertEqual(0, self.metrics.prom_puts._value.get())

    def test_get_metrics(self):
        """Prueba la obtención de métricas."""
        # Registrar algunas métricas
        self.metrics.record_hit()
        self.metrics.record_miss()
        self.metrics.record_put()
        self.metrics.record_memory_usage(1024)
        
        # Obtener métricas
        metrics = self.metrics.get_metrics()
        
        # Verificar que el diccionario contiene las métricas correctas
        self.assertEqual(metrics['hits'], 1)
        self.assertEqual(metrics['misses'], 1)
        self.assertEqual(metrics['puts'], 1)
        self.assertEqual(metrics['memory_usage'], 1024)


if __name__ == '__main__':
    unittest.main()
