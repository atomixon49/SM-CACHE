"""
Sistema de Benchmark para SM-CACHE

Este script proporciona un sistema completo para probar y evaluar
el rendimiento del sistema de caché inteligente SM-CACHE bajo
diferentes configuraciones y escenarios de uso.
"""

import time
import random
import threading
import json
import os
import argparse
from typing import Dict, List, Any, Callable, Tuple, Optional
from cache_system import IntelligentCache
from scenarios import (
    BasicScenario,
    SequentialAccessScenario,
    RandomAccessScenario,
    HotspotScenario,
    RealWorldScenario,
    ALL_SCENARIOS
)
from visualization import plot_results, create_comparison_charts
from report_generator import generate_report


class CacheBenchmark:
    """
    Sistema de benchmark para evaluar el rendimiento del caché inteligente.
    """
    
    def __init__(self, 
                 output_dir: str = "benchmark_results",
                 runs_per_test: int = 3,
                 warmup_runs: int = 1):
        """
        Inicializa el sistema de benchmark.
        
        Args:
            output_dir: Directorio donde se guardarán los resultados
            runs_per_test: Número de ejecuciones por prueba para promediar resultados
            warmup_runs: Número de ejecuciones de calentamiento antes de medir
        """
        self.output_dir = output_dir
        self.runs_per_test = runs_per_test
        self.warmup_runs = warmup_runs
        self.results = {}
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
    def run_benchmark(self, 
                      cache_configs: Dict[str, Dict[str, Any]],
                      scenarios: List[Any] = None,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Ejecuta el benchmark con diferentes configuraciones de caché y escenarios.
        
        Args:
            cache_configs: Diccionario de configuraciones de caché para probar
            scenarios: Lista de escenarios a ejecutar (si es None, usa todos)
            verbose: Si es True, muestra información detallada durante la ejecución
            
        Returns:
            Diccionario con los resultados del benchmark
        """
        if scenarios is None:
            scenarios = ALL_SCENARIOS
            
        results = {
            "timestamp": time.time(),
            "configs": cache_configs,
            "scenario_results": {}
        }
        
        for scenario_class in scenarios:
            scenario_name = scenario_class.__name__
            if verbose:
                print(f"\n{'='*50}")
                print(f"Ejecutando escenario: {scenario_name}")
                print(f"{'='*50}")
                
            scenario_results = {}
            
            for config_name, config in cache_configs.items():
                if verbose:
                    print(f"\n{'-'*40}")
                    print(f"Configuración: {config_name}")
                    print(f"{'-'*40}")
                
                # Ejecutar pruebas múltiples veces para obtener un promedio
                config_results = []
                
                for run in range(self.warmup_runs + self.runs_per_test):
                    is_warmup = run < self.warmup_runs
                    
                    if verbose:
                        run_type = "CALENTAMIENTO" if is_warmup else f"PRUEBA {run - self.warmup_runs + 1}"
                        print(f"\nEjecutando {run_type}...")
                    
                    # Crear instancia de caché con la configuración actual
                    cache = IntelligentCache(**config)
                    
                    # Crear y ejecutar el escenario
                    scenario = scenario_class(cache)
                    result = scenario.run(verbose=verbose and not is_warmup)
                    
                    # Guardar resultados si no es una ejecución de calentamiento
                    if not is_warmup:
                        config_results.append(result)
                        
                        if verbose:
                            print(f"Resultados de la prueba {run - self.warmup_runs + 1}:")
                            print(f"  Tasa de aciertos: {result['hit_rate']:.2f}%")
                            print(f"  Tiempo promedio de acceso: {result['avg_access_time']*1000:.2f} ms")
                    
                    # Detener el caché
                    cache.stop()
                
                # Calcular promedios
                avg_results = self._average_results(config_results)
                scenario_results[config_name] = avg_results
                
                if verbose:
                    print(f"\nResultados promedio para {config_name}:")
                    print(f"  Tasa de aciertos: {avg_results['hit_rate']:.2f}%")
                    print(f"  Tiempo promedio de acceso: {avg_results['avg_access_time']*1000:.2f} ms")
            
            results["scenario_results"][scenario_name] = scenario_results
        
        self.results = results
        return results
    
    def _average_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula el promedio de múltiples resultados.
        
        Args:
            results: Lista de resultados a promediar
            
        Returns:
            Diccionario con los resultados promediados
        """
        if not results:
            return {}
            
        avg_result = {}
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float)):
                avg_result[key] = sum(r[key] for r in results) / len(results)
            else:
                # Para valores no numéricos, usar el primer resultado
                avg_result[key] = results[0][key]
                
        return avg_result
    
    def save_results(self, filename: str = None) -> str:
        """
        Guarda los resultados del benchmark en un archivo JSON.
        
        Args:
            filename: Nombre del archivo (si es None, genera uno automáticamente)
            
        Returns:
            Ruta al archivo guardado
        """
        if not self.results:
            raise ValueError("No hay resultados para guardar. Ejecute run_benchmark primero.")
            
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Resultados guardados en: {filepath}")
        return filepath
    
    def generate_visualizations(self, results_file: str = None) -> str:
        """
        Genera visualizaciones a partir de los resultados del benchmark.
        
        Args:
            results_file: Ruta al archivo de resultados (si es None, usa los resultados actuales)
            
        Returns:
            Directorio donde se guardaron las visualizaciones
        """
        results = self.results
        
        if results_file:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        if not results:
            raise ValueError("No hay resultados para visualizar.")
            
        # Crear directorio para visualizaciones
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generar gráficos
        plot_results(results, vis_dir)
        create_comparison_charts(results, vis_dir)
        
        print(f"Visualizaciones generadas en: {vis_dir}")
        return vis_dir
    
    def generate_report(self, results_file: str = None, vis_dir: str = None) -> str:
        """
        Genera un informe HTML con los resultados del benchmark.
        
        Args:
            results_file: Ruta al archivo de resultados
            vis_dir: Directorio con las visualizaciones
            
        Returns:
            Ruta al informe generado
        """
        results = self.results
        
        if results_file:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        if not results:
            raise ValueError("No hay resultados para generar el informe.")
            
        if vis_dir is None:
            vis_dir = os.path.join(self.output_dir, "visualizations")
            
        # Generar informe
        report_path = generate_report(results, vis_dir, self.output_dir)
        
        print(f"Informe generado en: {report_path}")
        return report_path


def main():
    """Función principal para ejecutar el benchmark desde línea de comandos."""
    parser = argparse.ArgumentParser(description="Benchmark para SM-CACHE")
    parser.add_argument("--output", "-o", default="benchmark_results", 
                        help="Directorio de salida para resultados")
    parser.add_argument("--runs", "-r", type=int, default=3, 
                        help="Número de ejecuciones por prueba")
    parser.add_argument("--warmup", "-w", type=int, default=1, 
                        help="Número de ejecuciones de calentamiento")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Mostrar información detallada")
    
    args = parser.parse_args()
    
    # Configuraciones de caché para probar
    cache_configs = {
        "Básico": {
            "max_size": 1000,
            "max_memory_mb": 100.0,
            "ttl": None,
            "prefetch_enabled": False,
            "use_advanced_learning": False
        },
        "Con Prefetch": {
            "max_size": 1000,
            "max_memory_mb": 100.0,
            "ttl": None,
            "prefetch_enabled": True,
            "use_advanced_learning": False
        },
        "Aprendizaje Avanzado": {
            "max_size": 1000,
            "max_memory_mb": 100.0,
            "ttl": None,
            "prefetch_enabled": True,
            "use_advanced_learning": True
        },
        "Completo": {
            "max_size": 1000,
            "max_memory_mb": 100.0,
            "ttl": 3600,
            "prefetch_enabled": True,
            "use_advanced_learning": True,
            "persistence_enabled": True,
            "persistence_dir": ".cache_benchmark",
            "monitoring_enabled": True
        }
    }
    
    # Crear y ejecutar benchmark
    benchmark = CacheBenchmark(
        output_dir=args.output,
        runs_per_test=args.runs,
        warmup_runs=args.warmup
    )
    
    benchmark.run_benchmark(cache_configs, verbose=args.verbose)
    results_file = benchmark.save_results()
    vis_dir = benchmark.generate_visualizations(results_file)
    benchmark.generate_report(results_file, vis_dir)


if __name__ == "__main__":
    main()
