"""
Benchmark para comparar el rendimiento entre la caché original y la optimizada.
"""
import time
import random
import string
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from cache_system import IntelligentCache
from cache_system.optimized_cache import FastCache


def generate_random_string(length: int) -> str:
    """Genera una cadena aleatoria de longitud especificada."""
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def generate_random_data(size_kb: float) -> str:
    """Genera datos aleatorios de tamaño aproximado en KB."""
    # Aproximadamente 1 byte por carácter
    length = int(size_kb * 1024)
    return generate_random_string(length)


def run_sequential_benchmark(cache, num_operations: int, value_size_kb: float) -> Dict[str, Any]:
    """
    Ejecuta un benchmark de acceso secuencial.
    
    Args:
        cache: Instancia de caché a probar
        num_operations: Número de operaciones a realizar
        value_size_kb: Tamaño aproximado de los valores en KB
        
    Returns:
        Diccionario con resultados del benchmark
    """
    # Generar datos de prueba
    test_data = {}
    for i in range(num_operations):
        key = f"key_{i}"
        test_data[key] = generate_random_data(value_size_kb)
    
    # Medir tiempo de inserción
    start_time = time.time()
    for key, value in test_data.items():
        cache.put(key, value)
    put_time = time.time() - start_time
    
    # Medir tiempo de lectura
    start_time = time.time()
    for key in test_data.keys():
        cache.get(key)
    get_time = time.time() - start_time
    
    # Obtener estadísticas
    stats = cache.get_stats() if hasattr(cache, 'get_stats') else {}
    
    return {
        'put_time': put_time,
        'get_time': get_time,
        'operations': num_operations,
        'avg_put_time': put_time / num_operations,
        'avg_get_time': get_time / num_operations,
        'stats': stats
    }


def run_random_benchmark(cache, num_operations: int, value_size_kb: float, 
                         num_keys: int) -> Dict[str, Any]:
    """
    Ejecuta un benchmark de acceso aleatorio.
    
    Args:
        cache: Instancia de caché a probar
        num_operations: Número de operaciones a realizar
        value_size_kb: Tamaño aproximado de los valores en KB
        num_keys: Número de claves distintas a usar
        
    Returns:
        Diccionario con resultados del benchmark
    """
    # Generar datos de prueba
    test_data = {}
    for i in range(num_keys):
        key = f"key_{i}"
        test_data[key] = generate_random_data(value_size_kb)
    
    # Insertar datos iniciales
    for key, value in test_data.items():
        cache.put(key, value)
    
    # Medir tiempo de acceso aleatorio
    keys = list(test_data.keys())
    start_time = time.time()
    for _ in range(num_operations):
        key = random.choice(keys)
        cache.get(key)
    get_time = time.time() - start_time
    
    # Medir tiempo de actualización aleatoria
    start_time = time.time()
    for _ in range(num_operations):
        key = random.choice(keys)
        value = generate_random_data(value_size_kb)
        cache.put(key, value)
    put_time = time.time() - start_time
    
    # Obtener estadísticas
    stats = cache.get_stats() if hasattr(cache, 'get_stats') else {}
    
    return {
        'put_time': put_time,
        'get_time': get_time,
        'operations': num_operations,
        'avg_put_time': put_time / num_operations,
        'avg_get_time': get_time / num_operations,
        'stats': stats
    }


def run_hotspot_benchmark(cache, num_operations: int, value_size_kb: float,
                         num_keys: int, hotspot_ratio: float = 0.8) -> Dict[str, Any]:
    """
    Ejecuta un benchmark con patrón de acceso hotspot (80/20).
    
    Args:
        cache: Instancia de caché a probar
        num_operations: Número de operaciones a realizar
        value_size_kb: Tamaño aproximado de los valores en KB
        num_keys: Número de claves distintas a usar
        hotspot_ratio: Proporción de accesos a claves "calientes"
        
    Returns:
        Diccionario con resultados del benchmark
    """
    # Generar datos de prueba
    test_data = {}
    for i in range(num_keys):
        key = f"key_{i}"
        test_data[key] = generate_random_data(value_size_kb)
    
    # Insertar datos iniciales
    for key, value in test_data.items():
        cache.put(key, value)
    
    # Definir claves "calientes" (20% del total)
    hot_keys_count = max(1, int(num_keys * 0.2))
    hot_keys = random.sample(list(test_data.keys()), hot_keys_count)
    all_keys = list(test_data.keys())
    
    # Medir tiempo de acceso con patrón hotspot
    start_time = time.time()
    for _ in range(num_operations):
        # 80% de accesos a claves calientes, 20% a todas las claves
        if random.random() < hotspot_ratio:
            key = random.choice(hot_keys)
        else:
            key = random.choice(all_keys)
        cache.get(key)
    get_time = time.time() - start_time
    
    # Obtener estadísticas
    stats = cache.get_stats() if hasattr(cache, 'get_stats') else {}
    
    return {
        'get_time': get_time,
        'operations': num_operations,
        'avg_get_time': get_time / num_operations,
        'hot_keys_count': hot_keys_count,
        'stats': stats
    }


def run_eviction_benchmark(cache, max_memory_mb: float, value_size_kb: float) -> Dict[str, Any]:
    """
    Ejecuta un benchmark para probar la política de evicción.
    
    Args:
        cache: Instancia de caché a probar
        max_memory_mb: Memoria máxima del caché en MB
        value_size_kb: Tamaño aproximado de los valores en KB
        
    Returns:
        Diccionario con resultados del benchmark
    """
    # Calcular cuántos elementos necesitamos para llenar el caché
    items_to_fill = int((max_memory_mb * 1024) / value_size_kb) + 10
    
    # Llenar el caché
    start_time = time.time()
    for i in range(items_to_fill):
        key = f"fill_key_{i}"
        value = generate_random_data(value_size_kb)
        cache.put(key, value)
    fill_time = time.time() - start_time
    
    # Verificar cuántos elementos quedaron (después de evicción)
    stats = cache.get_stats() if hasattr(cache, 'get_stats') else {}
    items_count = stats.get('size', 0) if 'size' in stats else len(cache._cache) if hasattr(cache, '_cache') else 0
    
    # Medir tiempo de acceso después de evicción
    start_time = time.time()
    for i in range(items_to_fill):
        key = f"fill_key_{i}"
        cache.get(key)
    get_time = time.time() - start_time
    
    return {
        'fill_time': fill_time,
        'get_time': get_time,
        'items_to_fill': items_to_fill,
        'items_after_eviction': items_count,
        'stats': stats
    }


def run_ttl_benchmark(cache, num_operations: int, value_size_kb: float, ttl: int) -> Dict[str, Any]:
    """
    Ejecuta un benchmark para probar el comportamiento con TTL.
    
    Args:
        cache: Instancia de caché a probar
        num_operations: Número de operaciones a realizar
        value_size_kb: Tamaño aproximado de los valores en KB
        ttl: Tiempo de vida en segundos
        
    Returns:
        Diccionario con resultados del benchmark
    """
    # Insertar datos con TTL
    start_time = time.time()
    for i in range(num_operations):
        key = f"ttl_key_{i}"
        value = generate_random_data(value_size_kb)
        if hasattr(cache, 'put') and 'ttl' in cache.put.__code__.co_varnames:
            cache.put(key, value, ttl=ttl)
        else:
            # Fallback para caché que no soporta TTL por elemento
            cache.put(key, value)
    put_time = time.time() - start_time
    
    # Esperar a que expiren algunos elementos
    wait_time = ttl / 2
    time.sleep(wait_time)
    
    # Medir accesos después de expiración parcial
    hits = 0
    start_time = time.time()
    for i in range(num_operations):
        key = f"ttl_key_{i}"
        value = cache.get(key)
        if value is not None:
            hits += 1
    get_time = time.time() - start_time
    
    # Esperar a que expiren todos
    time.sleep(ttl - wait_time + 0.1)
    
    # Verificar expiración completa
    all_expired = True
    for i in range(num_operations):
        key = f"ttl_key_{i}"
        if cache.get(key) is not None:
            all_expired = False
            break
    
    return {
        'put_time': put_time,
        'get_time': get_time,
        'operations': num_operations,
        'hits_after_partial_expiry': hits,
        'all_expired': all_expired
    }


def compare_caches(original_results: Dict[str, Any], 
                  optimized_results: Dict[str, Any],
                  test_name: str) -> Dict[str, Any]:
    """
    Compara los resultados de ambas implementaciones.
    
    Args:
        original_results: Resultados de la caché original
        optimized_results: Resultados de la caché optimizada
        test_name: Nombre de la prueba
        
    Returns:
        Diccionario con comparación
    """
    comparison = {
        'test': test_name,
        'metrics': {}
    }
    
    # Comparar métricas comunes
    for metric in ['put_time', 'get_time', 'avg_put_time', 'avg_get_time']:
        if metric in original_results and metric in optimized_results:
            original_value = original_results[metric]
            optimized_value = optimized_results[metric]
            improvement = ((original_value - optimized_value) / original_value) * 100
            
            comparison['metrics'][metric] = {
                'original': original_value,
                'optimized': optimized_value,
                'improvement_percent': improvement
            }
    
    return comparison


def plot_comparison(comparisons: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Genera gráficos comparativos.
    
    Args:
        comparisons: Lista de comparaciones
        output_dir: Directorio para guardar gráficos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Gráfico de mejora por prueba
    tests = []
    get_improvements = []
    put_improvements = []
    
    for comp in comparisons:
        tests.append(comp['test'])
        metrics = comp['metrics']
        
        if 'avg_get_time' in metrics:
            get_improvements.append(metrics['avg_get_time']['improvement_percent'])
        else:
            get_improvements.append(0)
            
        if 'avg_put_time' in metrics:
            put_improvements.append(metrics['avg_put_time']['improvement_percent'])
        else:
            put_improvements.append(0)
    
    # Crear gráfico de barras
    x = np.arange(len(tests))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, get_improvements, width, label='GET mejora %')
    rects2 = ax.bar(x + width/2, put_improvements, width, label='PUT mejora %')
    
    ax.set_ylabel('Mejora (%)')
    ax.set_title('Mejora de rendimiento por operación y prueba')
    ax.set_xticks(x)
    ax.set_xticklabels(tests)
    ax.legend()
    
    # Añadir etiquetas
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_by_test.png'))
    
    # Gráfico de tiempos absolutos
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, comp in enumerate(comparisons):
        metrics = comp['metrics']
        if 'avg_get_time' in metrics:
            original = metrics['avg_get_time']['original'] * 1000  # ms
            optimized = metrics['avg_get_time']['optimized'] * 1000  # ms
            
            x_pos = i * 3
            ax.bar([x_pos, x_pos + 1], [original, optimized], 
                   color=['blue', 'green'], 
                   tick_label=['Original', 'Optimizado'])
            ax.text(x_pos + 0.5, max(original, optimized) + 0.1, 
                    comp['test'], ha='center')
    
    ax.set_ylabel('Tiempo promedio (ms)')
    ax.set_title('Comparación de tiempos de GET')
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'get_times_comparison.png'))


def run_all_benchmarks(output_dir: str = 'benchmark_results'):
    """
    Ejecuta todos los benchmarks y genera informes.
    
    Args:
        output_dir: Directorio para guardar resultados
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuración común
    max_memory_mb = 100.0
    ttl = 5  # segundos
    
    # Crear instancias de caché
    original_cache = IntelligentCache(
        max_size=1000,
        max_memory_mb=max_memory_mb,
        ttl=ttl,
        prefetch_enabled=True,
        monitoring_enabled=True
    )
    
    optimized_cache = FastCache(
        max_size=1000,
        max_memory_mb=max_memory_mb,
        ttl=ttl,
        prefetch_enabled=True,
        compression_enabled=True,
        monitoring_enabled=True
    )
    
    # Definir pruebas
    benchmarks = [
        {
            'name': 'sequential_small',
            'func': run_sequential_benchmark,
            'params': {'num_operations': 1000, 'value_size_kb': 0.1}
        },
        {
            'name': 'sequential_large',
            'func': run_sequential_benchmark,
            'params': {'num_operations': 100, 'value_size_kb': 10.0}
        },
        {
            'name': 'random_access',
            'func': run_random_benchmark,
            'params': {'num_operations': 5000, 'value_size_kb': 1.0, 'num_keys': 1000}
        },
        {
            'name': 'hotspot',
            'func': run_hotspot_benchmark,
            'params': {'num_operations': 10000, 'value_size_kb': 0.5, 'num_keys': 500}
        },
        {
            'name': 'eviction',
            'func': run_eviction_benchmark,
            'params': {'max_memory_mb': max_memory_mb, 'value_size_kb': 1.0}
        },
        {
            'name': 'ttl',
            'func': run_ttl_benchmark,
            'params': {'num_operations': 100, 'value_size_kb': 0.5, 'ttl': ttl}
        }
    ]
    
    # Ejecutar pruebas
    results = []
    comparisons = []
    
    for benchmark in benchmarks:
        print(f"Ejecutando benchmark: {benchmark['name']}...")
        
        # Limpiar cachés
        original_cache.clear()
        optimized_cache.clear()
        
        # Ejecutar con caché original
        original_result = benchmark['func'](original_cache, **benchmark['params'])
        original_result['cache_type'] = 'original'
        original_result['benchmark'] = benchmark['name']
        
        # Ejecutar con caché optimizada
        optimized_result = benchmark['func'](optimized_cache, **benchmark['params'])
        optimized_result['cache_type'] = 'optimized'
        optimized_result['benchmark'] = benchmark['name']
        
        # Guardar resultados
        results.append(original_result)
        results.append(optimized_result)
        
        # Comparar
        comparison = compare_caches(original_result, optimized_result, benchmark['name'])
        comparisons.append(comparison)
        
        print(f"  Mejora en GET: {comparison['metrics'].get('avg_get_time', {}).get('improvement_percent', 0):.2f}%")
        if 'avg_put_time' in comparison['metrics']:
            print(f"  Mejora en PUT: {comparison['metrics']['avg_put_time']['improvement_percent']:.2f}%")
        print()
    
    # Guardar resultados
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(output_dir, 'comparisons.json'), 'w', encoding='utf-8') as f:
        json.dump(comparisons, f, indent=2)
    
    # Generar gráficos
    plot_comparison(comparisons, output_dir)
    
    # Generar informe
    generate_report(results, comparisons, output_dir)
    
    print(f"Benchmarks completados. Resultados guardados en {output_dir}")


def generate_report(results: List[Dict[str, Any]], 
                   comparisons: List[Dict[str, Any]],
                   output_dir: str) -> None:
    """
    Genera un informe HTML con los resultados.
    
    Args:
        results: Resultados de los benchmarks
        comparisons: Comparaciones entre implementaciones
        output_dir: Directorio para guardar el informe
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Informe de Benchmark - SM-CACHE</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .improvement { color: green; font-weight: bold; }
            .degradation { color: red; font-weight: bold; }
            .chart { margin: 20px 0; max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>Informe de Benchmark - SM-CACHE</h1>
        <p>Comparación de rendimiento entre la implementación original y optimizada.</p>
        
        <h2>Resumen de Mejoras</h2>
        <table>
            <tr>
                <th>Prueba</th>
                <th>Mejora en GET (%)</th>
                <th>Mejora en PUT (%)</th>
            </tr>
    """
    
    # Añadir filas de comparación
    for comp in comparisons:
        get_improvement = comp['metrics'].get('avg_get_time', {}).get('improvement_percent', 0)
        put_improvement = comp['metrics'].get('avg_put_time', {}).get('improvement_percent', 0)
        
        get_class = 'improvement' if get_improvement > 0 else 'degradation'
        put_class = 'improvement' if put_improvement > 0 else 'degradation'
        
        html += f"""
            <tr>
                <td>{comp['test']}</td>
                <td class="{get_class}">{get_improvement:.2f}%</td>
                <td class="{put_class}">{put_improvement:.2f}%</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Gráficos Comparativos</h2>
        <div class="chart">
            <img src="improvement_by_test.png" alt="Mejoras por prueba" width="800">
        </div>
        <div class="chart">
            <img src="get_times_comparison.png" alt="Comparación de tiempos GET" width="800">
        </div>
        
        <h2>Detalles por Prueba</h2>
    """
    
    # Añadir detalles de cada prueba
    benchmark_names = set(r['benchmark'] for r in results)
    for benchmark in benchmark_names:
        html += f"<h3>Prueba: {benchmark}</h3>"
        
        # Filtrar resultados para esta prueba
        benchmark_results = [r for r in results if r['benchmark'] == benchmark]
        original = next((r for r in benchmark_results if r['cache_type'] == 'original'), {})
        optimized = next((r for r in benchmark_results if r['cache_type'] == 'optimized'), {})
        
        # Crear tabla de comparación
        html += """
        <table>
            <tr>
                <th>Métrica</th>
                <th>Original</th>
                <th>Optimizado</th>
                <th>Mejora</th>
            </tr>
        """
        
        # Añadir métricas comunes
        metrics = set()
        for r in benchmark_results:
            metrics.update(k for k in r.keys() if k not in ['cache_type', 'benchmark', 'stats'])
        
        for metric in sorted(metrics):
            if metric in original and metric in optimized:
                orig_val = original[metric]
                opt_val = optimized[metric]
                
                # Calcular mejora si son números
                improvement = ""
                if isinstance(orig_val, (int, float)) and isinstance(opt_val, (int, float)) and orig_val != 0:
                    imp_percent = ((orig_val - opt_val) / orig_val) * 100
                    imp_class = 'improvement' if imp_percent > 0 else 'degradation'
                    improvement = f'<span class="{imp_class}">{imp_percent:.2f}%</span>'
                
                html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{orig_val}</td>
                    <td>{opt_val}</td>
                    <td>{improvement}</td>
                </tr>
                """
        
        html += """
        </table>
        """
    
    html += """
    </body>
    </html>
    """
    
    # Guardar informe
    with open(os.path.join(output_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark para SM-CACHE")
    parser.add_argument("--output", "-o", default="benchmark_results", 
                        help="Directorio de salida para resultados")
    
    args = parser.parse_args()
    run_all_benchmarks(args.output)
