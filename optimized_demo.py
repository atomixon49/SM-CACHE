"""
Demostración del sistema de caché optimizado.
"""
import time
import random
import string
import os
import sys
from typing import Any, Dict, List, Optional

from cache_system.optimized_cache import FastCache


def generate_random_string(length: int) -> str:
    """Genera una cadena aleatoria de longitud especificada."""
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def generate_random_data(size_kb: float) -> str:
    """Genera datos aleatorios de tamaño aproximado en KB."""
    length = int(size_kb * 1024)
    return generate_random_string(length)


def simular_carga_datos(key: str) -> str:
    """Simula la carga de datos desde una fuente externa."""
    print(f"Cargando datos para {key}...")
    time.sleep(0.1)  # Simular latencia
    return f"Datos para {key}: {generate_random_string(100)}"


def imprimir_estadisticas(cache: FastCache) -> None:
    """Imprime estadísticas del caché."""
    stats = cache.get_stats()
    print("\n--- Estadísticas del Caché ---")
    print(f"Elementos: {stats['size']}")
    print(f"Memoria: {stats['memory']['current_size'] / (1024*1024):.2f}MB / "
          f"{stats['memory']['max_size'] / (1024*1024):.2f}MB "
          f"({stats['memory']['usage_percent']:.1f}%)")
    print(f"Elementos prefetch: {stats['prefetched']}")
    print(f"Elementos comprimidos: {stats['compressed']}")

    if 'metrics' in stats:
        metrics = stats['metrics']
        ops = metrics.get('operations', {})
        perf = metrics.get('performance', {})

        print("\n--- Métricas de Rendimiento ---")
        print(f"Operaciones: {ops.get('gets', 0)} GETs, {ops.get('puts', 0)} PUTs")
        print(f"Hit rate: {perf.get('hit_rate', 0):.1f}%")
        print(f"Tiempo promedio: GET={perf.get('avg_get_time', 0)*1000:.2f}ms, "
              f"PUT={perf.get('avg_put_time', 0)*1000:.2f}ms")

    print("-----------------------------\n")


def demo_operaciones_basicas(cache: FastCache) -> None:
    """Demuestra operaciones básicas del caché."""
    print("\n=== Demostración de Operaciones Básicas ===")

    # Almacenar valores
    print("\nAlmacenando valores...")
    for i in range(5):
        key = f"clave_{i}"
        value = f"valor_{i}"
        cache.put(key, value)
        print(f"Almacenado: {key} = {value}")

    # Recuperar valores
    print("\nRecuperando valores...")
    for i in range(5):
        key = f"clave_{i}"
        value = cache.get(key)
        print(f"Recuperado: {key} = {value}")

    # Verificar existencia
    print("\nVerificando existencia...")
    print(f"¿Existe clave_2? {cache.contains('clave_2')}")
    print(f"¿Existe clave_inexistente? {cache.contains('clave_inexistente')}")

    # Estadísticas
    imprimir_estadisticas(cache)


def demo_compresion(cache: FastCache) -> None:
    """Demuestra la compresión de datos."""
    print("\n=== Demostración de Compresión ===")

    print("\nAlmacenando datos grandes...")
    for i in range(5):
        # Crear datos con patrones repetitivos (alta compresibilidad)
        key = f"compresible_{i}"
        value = "A" * 1000 + "B" * 1000 + "C" * 1000
        cache.put(key, value)
        print(f"Almacenado: {key} (tamaño: {len(value)} bytes)")

    # Almacenar datos aleatorios (baja compresibilidad)
    for i in range(5):
        key = f"aleatorio_{i}"
        value = generate_random_string(3000)
        cache.put(key, value)
        print(f"Almacenado: {key} (tamaño: {len(value)} bytes)")

    # Estadísticas
    imprimir_estadisticas(cache)

    # Recuperar y verificar
    print("\nRecuperando datos comprimidos...")
    for i in range(5):
        key = f"compresible_{i}"
        value = cache.get(key)
        print(f"Recuperado: {key} (tamaño: {len(value)} bytes)")


def demo_eviccion(cache: FastCache) -> None:
    """Demuestra la política de evicción adaptativa mejorada."""
    print("\n=== Demostración de Evicción Adaptativa Mejorada ===")

    print("\nLlenando el caché más allá de su capacidad...")
    # Crear elementos grandes para forzar la evicción
    for i in range(20):
        # Crear un valor grande (aproximadamente 1MB cada uno)
        valor_grande = "X" * (1024 * 1024)  # 1MB de datos
        cache.put(f"grande_{i}", valor_grande)
        print(f"Añadido elemento grande_{i}")

        # Mostrar estadísticas cada 5 elementos
        if (i + 1) % 5 == 0:
            imprimir_estadisticas(cache)

    # Crear patrones de acceso para entrenar el algoritmo adaptativo
    print("\nCreando patrones de acceso para entrenar el algoritmo adaptativo...")

    # Patrón 1: Acceder frecuentemente a algunos elementos (LFU debería mantenerlos)
    print("\nAccediendo frecuentemente a algunos elementos...")
    for _ in range(10):
        for i in range(5):
            key = f"grande_{i}"
            if cache.contains(key):
                cache.get(key)
                print(f"Accedido a {key}")

    # Patrón 2: Acceder recientemente a otros elementos (LRU debería mantenerlos)
    print("\nAccediendo recientemente a otros elementos...")
    for i in range(10, 15):
        key = f"grande_{i}"
        if cache.contains(key):
            cache.get(key)
            print(f"Accedido a {key}")

    # Forzar más evicciones
    print("\nAñadiendo más elementos para forzar evicciones...")
    for i in range(20, 30):
        valor_grande = "Y" * (1024 * 1024)  # 1MB de datos
        cache.put(f"nuevo_{i}", valor_grande)
        print(f"Añadido elemento nuevo_{i}")

    # Verificar qué elementos permanecen
    print("\nVerificando elementos después de evicción adaptativa...")

    # Verificar elementos frecuentemente accedidos
    print("\nElementos frecuentemente accedidos (deberían mantenerse):")
    for i in range(5):
        key = f"grande_{i}"
        existe = cache.contains(key)
        print(f"{key}: {'Presente' if existe else 'Eviccionado'}")

    # Verificar elementos recientemente accedidos
    print("\nElementos recientemente accedidos (deberían mantenerse):")
    for i in range(10, 15):
        key = f"grande_{i}"
        existe = cache.contains(key)
        print(f"{key}: {'Presente' if existe else 'Eviccionado'}")

    # Verificar elementos no accedidos (deberían ser eviccionados)
    print("\nElementos no accedidos (deberían ser eviccionados):")
    for i in range(5, 10):
        key = f"grande_{i}"
        existe = cache.contains(key)
        print(f"{key}: {'Presente' if existe else 'Eviccionado'}")

    # Mostrar estadísticas finales
    imprimir_estadisticas(cache)


def demo_ttl(cache: FastCache) -> None:
    """Demuestra el manejo de tiempo de vida (TTL)."""
    print("\n=== Demostración de TTL ===")

    # Almacenar con diferentes TTL
    print("\nAlmacenando elementos con diferentes TTL...")
    cache.put("ttl_corto", "Expira en 2 segundos", ttl=2)
    cache.put("ttl_medio", "Expira en 5 segundos", ttl=5)
    cache.put("ttl_largo", "Expira en 10 segundos", ttl=10)
    cache.put("sin_ttl", "No expira")

    # Verificar inicialmente
    print("\nVerificando elementos recién almacenados...")
    for key in ["ttl_corto", "ttl_medio", "ttl_largo", "sin_ttl"]:
        print(f"{key}: {cache.get(key)}")

    # Esperar y verificar
    print("\nEsperando 3 segundos...")
    time.sleep(3)
    print("\nVerificando después de 3 segundos...")
    for key in ["ttl_corto", "ttl_medio", "ttl_largo", "sin_ttl"]:
        value = cache.get(key)
        print(f"{key}: {value if value else 'Expirado'}")

    # Esperar más y verificar
    print("\nEsperando 3 segundos más...")
    time.sleep(3)
    print("\nVerificando después de 6 segundos...")
    for key in ["ttl_corto", "ttl_medio", "ttl_largo", "sin_ttl"]:
        value = cache.get(key)
        print(f"{key}: {value if value else 'Expirado'}")


def demo_prefetch(cache: FastCache) -> None:
    """Demuestra la funcionalidad de prefetch avanzado."""
    print("\n=== Demostración de Prefetch Avanzado ===")

    # Crear un patrón de acceso predecible
    print("\nCreando patrón de acceso secuencial...")
    for i in range(10):
        key = f"seq_{i}"
        cache.put(key, f"Valor secuencial {i}")

    # Acceder en secuencia para entrenar el predictor
    print("\nAccediendo en secuencia para entrenar el predictor...")
    for _ in range(3):  # Repetir patrón
        for i in range(10):
            key = f"seq_{i}"
            cache.get(key)
            time.sleep(0.1)  # Pequeña pausa

    # Verificar prefetch
    print("\nEsperando a que el prefetch actúe...")
    time.sleep(2)

    # Estadísticas
    imprimir_estadisticas(cache)

    # Mostrar predicciones
    if hasattr(cache.predictor, 'predict_next_keys'):
        print("\nPredicciones actuales:")
        predictions = cache.predictor.predict_next_keys(5)
        for i, key in enumerate(predictions):
            print(f"  {i+1}. {key}")

    # Crear un patrón más complejo
    print("\nCreando patrón de acceso más complejo...")
    # Patrón A-B-C-D-E
    for _ in range(5):
        for key in ["A", "B", "C", "D", "E"]:
            cache.put(key, f"Valor {key}")
            cache.get(key)
            time.sleep(0.1)

    # Iniciar el patrón pero no completarlo
    print("\nIniciando patrón para verificar predicción...")
    for key in ["A", "B", "C"]:
        cache.get(key)
        time.sleep(0.1)

    # Esperar y verificar predicciones
    print("\nEsperando a que el predictor aprenda...")
    time.sleep(1)

    # Mostrar predicciones actualizadas
    if hasattr(cache.predictor, 'predict_next_keys'):
        print("\nPredicciones actualizadas:")
        predictions = cache.predictor.predict_next_keys(3)
        for i, key in enumerate(predictions):
            print(f"  {i+1}. {key}")

        # Verificar si predijo correctamente
        if "D" in predictions and "E" in predictions:
            print("\n¡Predicción exitosa! El predictor aprendió el patrón A-B-C-D-E")
        else:
            print("\nEl predictor aún está aprendiendo el patrón")

    # Estadísticas finales
    imprimir_estadisticas(cache)

    # Mostrar precisión del predictor si está disponible
    if hasattr(cache.predictor, 'get_prediction_accuracy'):
        accuracy = cache.predictor.get_prediction_accuracy()
        print(f"\nPrecisión del predictor: {accuracy:.2f}%")


def demo_persistencia(cache: FastCache) -> None:
    """Demuestra la persistencia a disco."""
    print("\n=== Demostración de Persistencia ===")

    # Almacenar datos
    print("\nAlmacenando datos para persistencia...")
    for i in range(10):
        key = f"persist_{i}"
        value = f"Valor persistente {i} - {generate_random_string(100)}"
        cache.put(key, value)
        print(f"Almacenado: {key}")

    # Forzar guardado
    print("\nGuardando caché a disco...")
    cache._save_to_disk()

    # Verificar archivo
    cache_file = f".cache/{cache.cache_name}.cache"
    if os.path.exists(cache_file):
        print(f"Archivo de caché creado: {cache_file}")
        print(f"Tamaño: {os.path.getsize(cache_file) / 1024:.2f} KB")
    else:
        print("Error: No se creó el archivo de caché")

    # Crear nueva instancia y cargar
    print("\nCreando nueva instancia y cargando desde disco...")
    new_cache = FastCache(
        max_size=1000,
        max_memory_mb=100.0,
        persistence_enabled=True,
        persistence_dir=".cache",
        cache_name=cache.cache_name
    )

    # Verificar datos cargados
    print("\nVerificando datos cargados...")
    for i in range(10):
        key = f"persist_{i}"
        value = new_cache.get(key)
        if value:
            print(f"Recuperado: {key} (primeros 20 caracteres: {value[:20]}...)")
        else:
            print(f"No se pudo recuperar: {key}")


def main():
    """Función principal que ejecuta todas las demostraciones."""
    # Crear instancia de caché optimizado
    cache = FastCache(
        max_size=1000,
        max_memory_mb=100.0,  # 100MB
        ttl=None,  # Sin TTL global
        prefetch_enabled=True,
        data_loader=simular_carga_datos,
        eviction_policy="adaptive",
        compression_enabled=True,
        compression_level=6,
        monitoring_enabled=True,
        persistence_enabled=True,
        persistence_dir=".cache",
        cache_name="optimized_demo_cache",
        auto_save_interval=60,
        metrics_export_interval=30
    )

    try:
        # Ejecutar demostraciones
        demo_operaciones_basicas(cache)
        demo_compresion(cache)
        demo_eviccion(cache)
        demo_ttl(cache)
        demo_prefetch(cache)
        demo_persistencia(cache)

        print("\n¡Demostración completada!")

    finally:
        # Limpiar recursos
        cache.stop()


if __name__ == "__main__":
    main()
