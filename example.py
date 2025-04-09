"""
Ejemplo de uso del Sistema de Caché Inteligente.
"""
import time
import random
import threading
from cache_system import IntelligentCache


def simulate_expensive_data_load(key):
    """
    Simula una carga de datos costosa.

    Args:
        key: La clave para la que cargar datos

    Returns:
        Los datos cargados
    """
    print(f"Cargando datos para clave: {key}")
    # Simular latencia de red o acceso a base de datos
    time.sleep(0.2)
    return f"Datos para {key} (cargados en {time.strftime('%H:%M:%S')})"


def print_stats(cache):
    """
    Imprime estadísticas del caché.

    Args:
        cache: Instancia de IntelligentCache
    """
    stats = cache.get_stats()
    print("\n--- Estadísticas del Caché ---")
    print(f"Tamaño actual: {stats['size']} elementos")
    print(f"Uso de memoria: {stats['memory_usage']['current_memory_mb']:.2f} MB " +
          f"de {stats['memory_usage']['max_memory_mb']:.2f} MB " +
          f"({stats['memory_usage']['memory_usage_percent']:.1f}%)")
    print("-----------------------------\n")


def simulate_access_pattern(cache, pattern, repetitions=3, add_noise=False):
    """
    Simula un patrón de acceso al caché.

    Args:
        cache: Instancia de IntelligentCache
        pattern: Lista de claves que forman el patrón
        repetitions: Número de veces a repetir el patrón
        add_noise: Si se debe añadir ruido aleatorio al patrón
    """
    print(f"\nSimulando patrón de acceso: {' -> '.join(pattern)}")

    for _ in range(repetitions):
        # Patrón normal
        for key in pattern:
            start_time = time.time()
            value = cache.get(key)
            elapsed = (time.time() - start_time) * 1000

            print(f"Acceso a '{key}': {elapsed:.1f}ms - {value[:30]}...")
            time.sleep(0.1)  # Pequeña pausa entre accesos

        # Añadir accesos aleatorios como ruido si se solicita
        if add_noise:
            noise_key = random.choice(["X", "Y", "Z"])
            start_time = time.time()
            value = cache.get(noise_key)
            elapsed = (time.time() - start_time) * 1000
            print(f"Ruido - Acceso a '{noise_key}': {elapsed:.1f}ms - {value[:30]}...")


def main():
    """Función principal del ejemplo."""
    print("=== Sistema de Caché Inteligente - Ejemplo ===\n")

    # Determinar si usar aprendizaje avanzado
    use_advanced = input("\u00bfDesea usar algoritmos de aprendizaje avanzados? (s/n): ").lower() == 's'

    # Determinar si usar persistencia
    use_persistence = input("\u00bfDesea habilitar la persistencia del caché? (s/n): ").lower() == 's'

    # Determinar si usar caché distribuido
    use_distributed = input("\u00bfDesea habilitar el caché distribuido? (s/n): ").lower() == 's'

    # Determinar si usar monitoreo
    use_monitoring = input("\u00bfDesea habilitar el monitoreo y métricas? (s/n): ").lower() == 's'

    # Crear instancia de caché
    cache = IntelligentCache(
        max_size=100,
        max_memory_mb=10.0,
        ttl=60,  # 60 segundos de TTL
        prefetch_enabled=True,
        data_loader=simulate_expensive_data_load,
        use_advanced_learning=use_advanced,
        persistence_enabled=use_persistence,
        persistence_dir=".cache_example",
        cache_name="example_cache",
        auto_save_interval=30 if use_persistence else None,  # Guardar cada 30 segundos
        distributed_enabled=use_distributed,
        distributed_host="localhost",
        distributed_port=5000,
        monitoring_enabled=use_monitoring,
        metrics_export_interval=30 if use_monitoring else None,  # Exportar cada 30 segundos
        alert_thresholds={'performance.hit_rate': 50.0} if use_monitoring else None  # Alertar si tasa de aciertos < 50%
    )

    print(f"\nUsando predictor: {type(cache.predictor).__name__}")

    try:
        # Demostrar operaciones básicas
        print("1. Operaciones básicas:")
        cache.put("clave1", "Este es un valor de ejemplo")
        print(f"Valor para 'clave1': {cache.get('clave1')}")

        # Demostrar carga automática
        print("\n2. Carga automática de datos:")
        value = cache.get("clave_nueva")
        print(f"Valor cargado: {value}")

        # Demostrar aprendizaje de patrones
        print("\n3. Aprendizaje de patrones de acceso:")

        # Patrón 1: secuencia A -> B -> C
        pattern1 = ["A", "B", "C"]
        simulate_access_pattern(cache, pattern1, repetitions=3, add_noise=use_advanced)

        if use_advanced:
            # Patrón 2: secuencia D -> E -> F (para algoritmos avanzados)
            print("\nEntrenando con patrón adicional para algoritmos avanzados:")
            pattern2 = ["D", "E", "F"]
            simulate_access_pattern(cache, pattern2, repetitions=2, add_noise=True)

        print("\nEsperando a que el prefetch ocurra...")
        time.sleep(1)

        # Verificar si C ya está en caché después de acceder a A y B
        print("\nAccediendo a A y B de nuevo:")
        for key in ["A", "B"]:
            start_time = time.time()
            value = cache.get(key)
            elapsed = (time.time() - start_time) * 1000
            print(f"Acceso a '{key}': {elapsed:.1f}ms")

        # Verificar si C ya está en caché (debería ser rápido si el prefetch funcionó)
        print("\nVerificando si C ya está en caché:")
        start_time = time.time()
        value = cache.get("C")
        elapsed = (time.time() - start_time) * 1000
        print(f"Acceso a 'C': {elapsed:.1f}ms - {'PREFETCH EXITOSO' if elapsed < 50 else 'PREFETCH FALLIDO'}")

        # Demostrar gestión de memoria
        print("\n4. Gestión automática de memoria:")
        print("Añadiendo 50 elementos grandes al caché...")

        for i in range(50):
            # Cada elemento es aproximadamente 100KB
            large_value = f"Elemento grande #{i}: " + "X" * 100000
            cache.put(f"grande_{i}", large_value)

            # Acceder más a algunos elementos para influir en la evicción
            if i % 10 == 0:
                for _ in range(5):
                    cache.get(f"grande_{i}")

        # Mostrar estadísticas
        print_stats(cache)

        # Mostrar información del predictor, persistencia y distribución
        stats = cache.get_stats()
        print("\nInformación del predictor:")
        print(f"Tipo de predictor: {stats['predictor_type']}")
        print(f"Aprendizaje avanzado: {'Activado' if stats['advanced_learning'] else 'Desactivado'}")

        if use_persistence:
            print("\nInformación de persistencia:")
            print(f"Persistencia: {'Activada' if stats['persistence_enabled'] else 'Desactivada'}")

            # Guardar explícitamente el caché
            print("\nGuardando caché en disco...")
            success = cache.save()
            print(f"Guardado {'exitoso' if success else 'fallido'}")

            if 'persistence' in stats:
                print(f"Archivo: {stats['persistence'].get('cache_name', 'default')}.json")
                print(f"Elementos guardados: {stats['persistence'].get('item_count', 0)}")
                if 'file_size_mb' in stats['persistence']:
                    print(f"Tamaño del archivo: {stats['persistence']['file_size_mb']:.2f} MB")

        if use_distributed:
            print("\nInformación de caché distribuido:")
            print(f"Distribución: {'Activada' if stats.get('distributed_enabled', False) else 'Desactivada'}")

            if cache.distributed:
                cluster_info = cache.distributed.get_cluster_info()
                print(f"Nodo local: {cluster_info['local_node']['host']}:{cluster_info['local_node']['port']}")
                print(f"ID del nodo: {cluster_info['local_node']['node_id']}")
                print(f"Nodos en el cluster: {cluster_info['node_count']}")

                # Demostrar operaciones distribuidas
                print("\nDemostrando operaciones distribuidas:")
                print("Invalidando clave 'clave1' en todos los nodos...")
                cache.remove("clave1", notify_distributed=True)
                print("Clave invalidada.")

        if use_monitoring:
            print("\nInformación de monitoreo y métricas:")
            print(f"Monitoreo: {'Activado' if stats.get('monitoring_enabled', False) else 'Desactivado'}")

            if 'metrics' in stats:
                metrics = stats['metrics']
                print("\nEstadísticas de operaciones:")
                print(f"Total de operaciones: {metrics['operations']['gets'] + metrics['operations']['puts']}")
                print(f"Aciertos: {metrics['operations']['hits']}")
                print(f"Fallos: {metrics['operations']['misses']}")
                print(f"Tasa de aciertos: {metrics['performance']['hit_rate']:.2f}%")
                print(f"Tiempo promedio de obtención: {metrics['performance']['avg_get_time']*1000:.2f} ms")
                print(f"Tiempo promedio de almacenamiento: {metrics['performance']['avg_put_time']*1000:.2f} ms")

                # Exportar métricas manualmente
                if cache.metrics_monitor:
                    print("\nExportando métricas...")
                    success = cache.metrics_monitor.export_metrics()
                    print(f"Exportación {'exitosa' if success else 'fallida'}")
                    print(f"Directorio de métricas: {cache.metrics_monitor.exporter.export_dir}")

        # Demostrar TTL
        print("\n5. Expiración por tiempo de vida (TTL):")
        cache_with_short_ttl = IntelligentCache(ttl=2)  # 2 segundos de TTL
        cache_with_short_ttl.put("temp", "Este valor expirará pronto")

        print(f"Valor inicial: {cache_with_short_ttl.get('temp')}")
        print("Esperando a que expire (3 segundos)...")
        time.sleep(3)
        print(f"Valor después de TTL: {cache_with_short_ttl.get('temp')}")

        # Demostrar acceso concurrente
        print("\n6. Acceso concurrente:")

        def worker(worker_id, num_operations):
            for i in range(num_operations):
                key = f"worker_{worker_id}_{i}"
                cache.get(key)  # Esto cargará el dato automáticamente
                time.sleep(0.05)

        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        print("Acceso concurrente completado.")
        print_stats(cache)

    finally:
        # Asegurar que se limpien los recursos
        cache.stop()
        print("\n=== Ejemplo finalizado ===")


if __name__ == "__main__":
    main()
