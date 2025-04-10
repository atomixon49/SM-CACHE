"""
Demostración completa del Sistema de Caché Inteligente SM-CACHE.

Este script muestra todas las características principales del sistema,
incluyendo operaciones básicas, aprendizaje avanzado, gestión de memoria,
persistencia, monitoreo y caché distribuido.
"""

import time
import random
import threading
from typing import Any, Dict
from cache_system import IntelligentCache


def simular_carga_datos(key: Any) -> str:
    """Simula la carga de datos desde una fuente externa."""
    time.sleep(0.1)  # Simular latencia de red/BD
    return f"Datos para {key}"


def imprimir_estadisticas(cache: IntelligentCache) -> None:
    """Imprime las estadísticas actuales del caché."""
    stats = cache.get_stats()
    print("\nEstadísticas del caché:")
    print(f"- Tamaño actual: {stats['size']} elementos")
    print(f"- Uso de memoria: {stats['memory_usage']['memory_usage_percent']:.1f}%")
    
    if 'metrics' in stats:
        metricas = stats['metrics']
        print(f"- Tasa de aciertos: {metricas['performance']['hit_rate']:.1f}%")
        if 'predictions' in metricas:
            print(f"- Predicciones exitosas: {metricas['predictions']['accuracy']:.1f}%")
        print(f"- Tiempo promedio de acceso: {metricas['performance']['avg_get_time']*1000:.2f} ms")


def demo_operaciones_basicas(cache: IntelligentCache) -> None:
    """Demuestra las operaciones básicas del caché."""
    print("\n=== Demostración de operaciones básicas ===")
    
    # Operaciones básicas de put/get
    print("\nRealizando operaciones básicas...")
    cache.put("clave1", "valor1")
    cache.put("clave2", "valor2")
    print(f"Valor obtenido: {cache.get('clave1')}")
    print(f"¿Existe clave2?: {cache.contains('clave2')}")
    
    # Demostrar TTL
    print("\nProbando expiración TTL...")
    cache.put("temp", "temporal")
    print(f"Valor antes de expirar: {cache.get('temp')}")
    time.sleep(2)  # Esperar a que expire
    print(f"Valor después de expirar: {cache.get('temp')}")
    
    imprimir_estadisticas(cache)


def demo_aprendizaje(cache: IntelligentCache) -> None:
    """Demuestra las capacidades de aprendizaje y predicción."""
    print("\n=== Demostración de aprendizaje y predicción ===")
    
    # Simular un patrón de acceso
    secuencia = ["A", "B", "C", "D", "E"]
    print("\nEntrenando el predictor con un patrón...")
    
    for _ in range(3):  # Repetir el patrón varias veces
        for key in secuencia:
            cache.get(key)  # Esto activará el data_loader y entrenará el predictor
            time.sleep(0.1)
    
    print("\nEsperando a que el prefetch actúe...")
    time.sleep(2)  # Dar tiempo para que ocurra el prefetch
    
    # Verificar si algunos elementos fueron precargados
    print("\nVerificando elementos precargados:")
    for key in ["A", "B", "C"]:
        resultado = "precargado" if cache.contains(key) else "no precargado"
        print(f"Elemento {key}: {resultado}")
    
    imprimir_estadisticas(cache)


def demo_gestion_memoria(cache: IntelligentCache) -> None:
    """Demuestra la gestión automática de memoria."""
    print("\n=== Demostración de gestión de memoria ===")
    
    print("\nLlenando el caché más allá de su capacidad...")
    # Crear elementos grandes para forzar la evicción
    for i in range(20):
        # Crear un valor grande (aproximadamente 1MB cada uno)
        valor_grande = "X" * (1024 * 1024)  # 1MB de datos
        cache.put(f"grande_{i}", valor_grande)
        print(f"Añadido elemento grande_{i}")
        imprimir_estadisticas(cache)
        time.sleep(0.5)


def demo_persistencia(cache: IntelligentCache) -> None:
    """Demuestra las características de persistencia."""
    print("\n=== Demostración de persistencia ===")
    
    # Añadir algunos datos
    print("\nGuardando datos en el caché...")
    for i in range(5):
        cache.put(f"persistente_{i}", f"valor_{i}")
    
    # Guardar explícitamente
    print("\nGuardando estado en disco...")
    cache.save()
    
    # Simular reinicio creando una nueva instancia
    print("\nSimulando reinicio del sistema...")
    nuevo_cache = IntelligentCache(
        max_size=1000,
        persistence_enabled=True,
        persistence_dir=".cache",
        cache_name="demo_cache"
    )
    
    # Verificar datos cargados
    print("\nVerificando datos persistidos:")
    for i in range(5):
        valor = nuevo_cache.get(f"persistente_{i}")
        print(f"persistente_{i}: {valor}")
    
    nuevo_cache.stop()


def demo_distribuido(puerto_base: int = 5000) -> None:
    """Demuestra el caché distribuido."""
    print("\n=== Demostración de caché distribuido ===")
    
    # Crear dos nodos de caché
    nodo1 = IntelligentCache(
        max_size=1000,
        distributed_enabled=True,
        distributed_host="localhost",
        distributed_port=puerto_base,
        cluster_nodes=[("localhost", puerto_base + 1)]
    )
    
    nodo2 = IntelligentCache(
        max_size=1000,
        distributed_enabled=True,
        distributed_host="localhost",
        distributed_port=puerto_base + 1,
        cluster_nodes=[("localhost", puerto_base)]
    )
    
    # Demostrar sincronización
    print("\nProbando sincronización entre nodos...")
    nodo1.put("compartido1", "valor1")
    time.sleep(1)  # Dar tiempo para la sincronización
    
    valor_nodo2 = nodo2.get("compartido1")
    print(f"Valor en nodo2: {valor_nodo2}")
    
    # Limpiar
    nodo1.stop()
    nodo2.stop()


def main():
    """Función principal que ejecuta todas las demostraciones."""
    # Crear instancia principal de caché con todas las características
    cache = IntelligentCache(
        max_size=1000,
        max_memory_mb=10.0,  # 10MB para demo
        ttl=2,  # 2 segundos para demo
        prefetch_enabled=True,
        data_loader=simular_carga_datos,
        use_advanced_learning=True,
        persistence_enabled=True,
        persistence_dir=".cache",
        cache_name="demo_cache",
        auto_save_interval=30,
        monitoring_enabled=True,
        metrics_export_interval=5
    )
    
    try:
        # Ejecutar demostraciones
        demo_operaciones_basicas(cache)
        demo_aprendizaje(cache)
        demo_gestion_memoria(cache)
        demo_persistencia(cache)
        demo_distribuido()
        
        print("\n¡Demostración completada!")
        
    finally:
        # Limpiar recursos
        cache.stop()


if __name__ == "__main__":
    main()