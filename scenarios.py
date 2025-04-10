"""
Escenarios de prueba para el benchmark del sistema de caché inteligente.

Este módulo define diferentes escenarios de acceso a datos para evaluar
el rendimiento del sistema de caché bajo diferentes patrones de uso.
"""

import time
import random
import threading
from typing import Dict, List, Any, Callable, Tuple, Optional
from abc import ABC, abstractmethod
from cache_system import IntelligentCache


class BenchmarkScenario(ABC):
    """Clase base para escenarios de benchmark."""
    
    def __init__(self, cache: IntelligentCache):
        """
        Inicializa el escenario.
        
        Args:
            cache: Instancia del caché a probar
        """
        self.cache = cache
        self.results = {}
        
    @abstractmethod
    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Ejecuta el escenario de benchmark.
        
        Args:
            verbose: Si es True, muestra información detallada durante la ejecución
            
        Returns:
            Diccionario con los resultados del benchmark
        """
        pass
    
    def _simulate_data_load(self, key: Any) -> Any:
        """
        Simula una carga de datos costosa.
        
        Args:
            key: Clave para la que cargar datos
            
        Returns:
            Datos cargados
        """
        # Simular latencia de red o acceso a base de datos
        time.sleep(0.05)
        return f"Datos para {key} (generados en {time.time()})"


class BasicScenario(BenchmarkScenario):
    """
    Escenario básico que prueba operaciones simples de caché.
    """
    
    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """Ejecuta el escenario básico."""
        if verbose:
            print("Ejecutando escenario básico...")
            
        # Configurar caché con cargador de datos
        self.cache.data_loader = self._simulate_data_load
        
        # Métricas a recopilar
        total_gets = 0
        hits = 0
        total_time = 0
        
        # Realizar operaciones básicas
        num_operations = 1000
        keys = [f"key_{i}" for i in range(100)]  # 100 claves posibles
        
        for i in range(num_operations):
            # Seleccionar una clave (con cierta probabilidad de repetición)
            key = random.choice(keys)
            
            # Medir tiempo de acceso
            start_time = time.time()
            value = self.cache.get(key)
            elapsed = time.time() - start_time
            
            # Actualizar métricas
            total_gets += 1
            total_time += elapsed
            
            # Determinar si fue un hit o miss
            if hasattr(self.cache, 'metrics_collector'):
                # Usar métricas internas si están disponibles
                hits = self.cache.metrics_collector.hits
            else:
                # Estimación aproximada basada en tiempo de respuesta
                if elapsed < 0.01:  # Umbral arbitrario para distinguir hits de misses
                    hits += 1
            
            # Ocasionalmente modificar valores
            if random.random() < 0.1:  # 10% de probabilidad
                self.cache.put(key, f"Nuevo valor para {key}")
                
        # Calcular métricas finales
        hit_rate = (hits / total_gets) * 100 if total_gets > 0 else 0
        avg_access_time = total_time / total_gets if total_gets > 0 else 0
        
        # Obtener estadísticas del caché
        cache_stats = self.cache.get_stats()
        
        # Preparar resultados
        self.results = {
            "scenario": "BasicScenario",
            "operations": total_gets,
            "hits": hits,
            "hit_rate": hit_rate,
            "avg_access_time": avg_access_time,
            "cache_stats": cache_stats
        }
        
        if verbose:
            print(f"Escenario básico completado. Tasa de aciertos: {hit_rate:.2f}%")
            
        return self.results


class SequentialAccessScenario(BenchmarkScenario):
    """
    Escenario que simula acceso secuencial a datos.
    Útil para probar la capacidad de predicción del caché.
    """
    
    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """Ejecuta el escenario de acceso secuencial."""
        if verbose:
            print("Ejecutando escenario de acceso secuencial...")
            
        # Configurar caché con cargador de datos
        self.cache.data_loader = self._simulate_data_load
        
        # Métricas a recopilar
        total_gets = 0
        hits = 0
        total_time = 0
        
        # Definir patrones secuenciales
        patterns = [
            ["A", "B", "C", "D", "E"],
            ["X", "Y", "Z"],
            ["1", "2", "3", "4"]
        ]
        
        # Ejecutar patrones varias veces para entrenar el predictor
        for _ in range(3):  # Fase de entrenamiento
            for pattern in patterns:
                for key in pattern:
                    self.cache.get(key)
                    time.sleep(0.01)  # Pequeña pausa entre accesos
        
        # Fase de prueba: medir rendimiento en patrones conocidos
        num_iterations = 5
        for _ in range(num_iterations):
            for pattern in patterns:
                for key in pattern:
                    # Medir tiempo de acceso
                    start_time = time.time()
                    value = self.cache.get(key)
                    elapsed = time.time() - start_time
                    
                    # Actualizar métricas
                    total_gets += 1
                    total_time += elapsed
                    
                    # Determinar si fue un hit o miss
                    if hasattr(self.cache, 'metrics_collector'):
                        # Usar métricas internas si están disponibles
                        hits = self.cache.metrics_collector.hits
                    else:
                        # Estimación aproximada basada en tiempo de respuesta
                        if elapsed < 0.01:
                            hits += 1
                    
                    time.sleep(0.01)  # Pequeña pausa entre accesos
        
        # Calcular métricas finales
        hit_rate = (hits / total_gets) * 100 if total_gets > 0 else 0
        avg_access_time = total_time / total_gets if total_gets > 0 else 0
        
        # Obtener estadísticas del caché
        cache_stats = self.cache.get_stats()
        
        # Preparar resultados
        self.results = {
            "scenario": "SequentialAccessScenario",
            "operations": total_gets,
            "hits": hits,
            "hit_rate": hit_rate,
            "avg_access_time": avg_access_time,
            "cache_stats": cache_stats
        }
        
        if verbose:
            print(f"Escenario secuencial completado. Tasa de aciertos: {hit_rate:.2f}%")
            
        return self.results


class RandomAccessScenario(BenchmarkScenario):
    """
    Escenario que simula acceso aleatorio a datos.
    Útil para probar el rendimiento en casos sin patrones claros.
    """
    
    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """Ejecuta el escenario de acceso aleatorio."""
        if verbose:
            print("Ejecutando escenario de acceso aleatorio...")
            
        # Configurar caché con cargador de datos
        self.cache.data_loader = self._simulate_data_load
        
        # Métricas a recopilar
        total_gets = 0
        hits = 0
        total_time = 0
        
        # Generar claves aleatorias
        num_operations = 1000
        key_space = 5000  # Espacio de claves mucho mayor que el tamaño del caché
        
        for i in range(num_operations):
            # Generar clave aleatoria
            key = f"random_key_{random.randint(1, key_space)}"
            
            # Medir tiempo de acceso
            start_time = time.time()
            value = self.cache.get(key)
            elapsed = time.time() - start_time
            
            # Actualizar métricas
            total_gets += 1
            total_time += elapsed
            
            # Determinar si fue un hit o miss
            if hasattr(self.cache, 'metrics_collector'):
                # Usar métricas internas si están disponibles
                hits = self.cache.metrics_collector.hits
            else:
                # Estimación aproximada basada en tiempo de respuesta
                if elapsed < 0.01:
                    hits += 1
        
        # Calcular métricas finales
        hit_rate = (hits / total_gets) * 100 if total_gets > 0 else 0
        avg_access_time = total_time / total_gets if total_gets > 0 else 0
        
        # Obtener estadísticas del caché
        cache_stats = self.cache.get_stats()
        
        # Preparar resultados
        self.results = {
            "scenario": "RandomAccessScenario",
            "operations": total_gets,
            "hits": hits,
            "hit_rate": hit_rate,
            "avg_access_time": avg_access_time,
            "cache_stats": cache_stats
        }
        
        if verbose:
            print(f"Escenario aleatorio completado. Tasa de aciertos: {hit_rate:.2f}%")
            
        return self.results


class HotspotScenario(BenchmarkScenario):
    """
    Escenario que simula acceso con puntos calientes (hotspots).
    Un pequeño conjunto de claves recibe la mayoría de los accesos.
    """
    
    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """Ejecuta el escenario de puntos calientes."""
        if verbose:
            print("Ejecutando escenario de puntos calientes...")
            
        # Configurar caché con cargador de datos
        self.cache.data_loader = self._simulate_data_load
        
        # Métricas a recopilar
        total_gets = 0
        hits = 0
        total_time = 0
        
        # Definir claves "calientes" y "frías"
        hot_keys = [f"hot_key_{i}" for i in range(10)]  # 10 claves calientes
        cold_keys = [f"cold_key_{i}" for i in range(1000)]  # 1000 claves frías
        
        # Realizar operaciones
        num_operations = 1000
        
        for i in range(num_operations):
            # 80% de probabilidad de acceder a una clave caliente
            if random.random() < 0.8:
                key = random.choice(hot_keys)
            else:
                key = random.choice(cold_keys)
            
            # Medir tiempo de acceso
            start_time = time.time()
            value = self.cache.get(key)
            elapsed = time.time() - start_time
            
            # Actualizar métricas
            total_gets += 1
            total_time += elapsed
            
            # Determinar si fue un hit o miss
            if hasattr(self.cache, 'metrics_collector'):
                # Usar métricas internas si están disponibles
                hits = self.cache.metrics_collector.hits
            else:
                # Estimación aproximada basada en tiempo de respuesta
                if elapsed < 0.01:
                    hits += 1
        
        # Calcular métricas finales
        hit_rate = (hits / total_gets) * 100 if total_gets > 0 else 0
        avg_access_time = total_time / total_gets if total_gets > 0 else 0
        
        # Obtener estadísticas del caché
        cache_stats = self.cache.get_stats()
        
        # Preparar resultados
        self.results = {
            "scenario": "HotspotScenario",
            "operations": total_gets,
            "hits": hits,
            "hit_rate": hit_rate,
            "avg_access_time": avg_access_time,
            "cache_stats": cache_stats
        }
        
        if verbose:
            print(f"Escenario de puntos calientes completado. Tasa de aciertos: {hit_rate:.2f}%")
            
        return self.results


class RealWorldScenario(BenchmarkScenario):
    """
    Escenario que simula un caso de uso del mundo real con múltiples patrones.
    Combina acceso secuencial, aleatorio y con puntos calientes.
    """
    
    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """Ejecuta el escenario del mundo real."""
        if verbose:
            print("Ejecutando escenario del mundo real...")
            
        # Configurar caché con cargador de datos
        self.cache.data_loader = self._simulate_data_load
        
        # Métricas a recopilar
        total_gets = 0
        hits = 0
        total_time = 0
        
        # Definir diferentes tipos de claves
        sequence_patterns = [
            ["page_1", "page_2", "page_3"],
            ["product_a", "product_b", "product_c", "product_d"],
            ["user_profile", "user_settings", "user_history"]
        ]
        hot_keys = [f"popular_item_{i}" for i in range(20)]
        cold_keys = [f"rare_item_{i}" for i in range(500)]
        
        # Simular múltiples usuarios concurrentes
        num_users = 5
        operations_per_user = 200
        
        def user_session(user_id):
            nonlocal total_gets, hits, total_time
            
            # Cada usuario tiene un comportamiento ligeramente diferente
            user_type = user_id % 3
            
            for i in range(operations_per_user):
                # Determinar tipo de acceso según el tipo de usuario
                if user_type == 0:
                    # Usuario que sigue patrones secuenciales
                    if i % 10 == 0:  # Cada 10 operaciones, iniciar un nuevo patrón
                        pattern = random.choice(sequence_patterns)
                        for key in pattern:
                            self._access_key(key)
                    else:
                        # Acceso aleatorio o a punto caliente
                        if random.random() < 0.7:
                            key = random.choice(hot_keys)
                        else:
                            key = random.choice(cold_keys)
                        self._access_key(key)
                        
                elif user_type == 1:
                    # Usuario con preferencia por puntos calientes
                    if random.random() < 0.9:
                        key = random.choice(hot_keys)
                    else:
                        key = random.choice(cold_keys)
                    self._access_key(key)
                    
                else:
                    # Usuario con comportamiento más aleatorio
                    if random.random() < 0.4:
                        key = random.choice(hot_keys)
                    elif random.random() < 0.7:
                        # Seleccionar una clave de un patrón aleatorio
                        pattern = random.choice(sequence_patterns)
                        key = random.choice(pattern)
                    else:
                        key = random.choice(cold_keys)
                    self._access_key(key)
                
                # Pequeña pausa entre operaciones
                time.sleep(0.01)
        
        def _access_key(self, key):
            nonlocal total_gets, hits, total_time
            
            # Medir tiempo de acceso
            start_time = time.time()
            value = self.cache.get(key)
            elapsed = time.time() - start_time
            
            # Actualizar métricas
            total_gets += 1
            total_time += elapsed
            
            # Determinar si fue un hit o miss
            if hasattr(self.cache, 'metrics_collector'):
                # Usar métricas internas si están disponibles
                hits = self.cache.metrics_collector.hits
            else:
                # Estimación aproximada basada en tiempo de respuesta
                if elapsed < 0.01:
                    hits += 1
            
            # Ocasionalmente actualizar valores
            if random.random() < 0.05:  # 5% de probabilidad
                self.cache.put(key, f"Valor actualizado para {key}")
        
        # Crear y ejecutar hilos de usuario
        threads = []
        for i in range(num_users):
            t = threading.Thread(target=user_session, args=(i,))
            threads.append(t)
            t.start()
        
        # Esperar a que todos los hilos terminen
        for t in threads:
            t.join()
        
        # Calcular métricas finales
        hit_rate = (hits / total_gets) * 100 if total_gets > 0 else 0
        avg_access_time = total_time / total_gets if total_gets > 0 else 0
        
        # Obtener estadísticas del caché
        cache_stats = self.cache.get_stats()
        
        # Preparar resultados
        self.results = {
            "scenario": "RealWorldScenario",
            "operations": total_gets,
            "hits": hits,
            "hit_rate": hit_rate,
            "avg_access_time": avg_access_time,
            "cache_stats": cache_stats
        }
        
        if verbose:
            print(f"Escenario del mundo real completado. Tasa de aciertos: {hit_rate:.2f}%")
            
        return self.results


# Lista de todos los escenarios disponibles
ALL_SCENARIOS = [
    BasicScenario,
    SequentialAccessScenario,
    RandomAccessScenario,
    HotspotScenario,
    RealWorldScenario
]
