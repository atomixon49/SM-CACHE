"""
Módulo para visualizar los resultados del benchmark del sistema de caché inteligente.

Este módulo proporciona funciones para generar gráficos y visualizaciones
a partir de los resultados del benchmark.
"""

import os
import json
import time
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Genera gráficos para los resultados del benchmark.
    
    Args:
        results: Diccionario con los resultados del benchmark
        output_dir: Directorio donde guardar los gráficos
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Extraer datos para los gráficos
    scenario_results = results.get("scenario_results", {})
    configs = results.get("configs", {})
    
    if not scenario_results or not configs:
        print("No hay suficientes datos para generar gráficos.")
        return
    
    # Generar gráficos para cada escenario
    for scenario_name, scenario_data in scenario_results.items():
        _plot_hit_rates(scenario_name, scenario_data, configs, output_dir)
        _plot_access_times(scenario_name, scenario_data, configs, output_dir)
        
    # Generar gráficos comparativos entre escenarios
    _plot_scenario_comparison(scenario_results, configs, output_dir)


def _plot_hit_rates(scenario_name: str, 
                   scenario_data: Dict[str, Dict[str, Any]], 
                   configs: Dict[str, Dict[str, Any]], 
                   output_dir: str) -> None:
    """
    Genera un gráfico de barras para las tasas de aciertos.
    
    Args:
        scenario_name: Nombre del escenario
        scenario_data: Datos del escenario
        configs: Configuraciones de caché
        output_dir: Directorio de salida
    """
    plt.figure(figsize=(10, 6))
    
    # Preparar datos
    config_names = list(scenario_data.keys())
    hit_rates = [scenario_data[config]["hit_rate"] for config in config_names]
    
    # Crear gráfico de barras
    bars = plt.bar(config_names, hit_rates, color='skyblue')
    
    # Añadir etiquetas de valor
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.2f}%', ha='center', va='bottom')
    
    # Configurar gráfico
    plt.title(f'Tasa de Aciertos por Configuración - {scenario_name}')
    plt.xlabel('Configuración de Caché')
    plt.ylabel('Tasa de Aciertos (%)')
    plt.ylim(0, 100)  # Tasa de aciertos entre 0% y 100%
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Guardar gráfico
    filename = f"{scenario_name}_hit_rates.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def _plot_access_times(scenario_name: str, 
                      scenario_data: Dict[str, Dict[str, Any]], 
                      configs: Dict[str, Dict[str, Any]], 
                      output_dir: str) -> None:
    """
    Genera un gráfico de barras para los tiempos de acceso.
    
    Args:
        scenario_name: Nombre del escenario
        scenario_data: Datos del escenario
        configs: Configuraciones de caché
        output_dir: Directorio de salida
    """
    plt.figure(figsize=(10, 6))
    
    # Preparar datos
    config_names = list(scenario_data.keys())
    access_times = [scenario_data[config]["avg_access_time"] * 1000 for config in config_names]  # Convertir a ms
    
    # Crear gráfico de barras
    bars = plt.bar(config_names, access_times, color='lightgreen')
    
    # Añadir etiquetas de valor
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f} ms', ha='center', va='bottom')
    
    # Configurar gráfico
    plt.title(f'Tiempo Promedio de Acceso - {scenario_name}')
    plt.xlabel('Configuración de Caché')
    plt.ylabel('Tiempo de Acceso (ms)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Guardar gráfico
    filename = f"{scenario_name}_access_times.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def _plot_scenario_comparison(scenario_results: Dict[str, Dict[str, Dict[str, Any]]], 
                             configs: Dict[str, Dict[str, Any]], 
                             output_dir: str) -> None:
    """
    Genera gráficos comparativos entre escenarios.
    
    Args:
        scenario_results: Resultados de todos los escenarios
        configs: Configuraciones de caché
        output_dir: Directorio de salida
    """
    # Comparación de tasas de aciertos entre escenarios
    plt.figure(figsize=(12, 8))
    
    # Preparar datos
    scenario_names = list(scenario_results.keys())
    config_names = list(configs.keys())
    
    # Posiciones de las barras
    x = np.arange(len(scenario_names))
    width = 0.8 / len(config_names)  # Ancho de las barras
    
    # Crear barras para cada configuración
    for i, config_name in enumerate(config_names):
        hit_rates = [scenario_results[scenario][config_name]["hit_rate"] 
                    for scenario in scenario_names]
        
        plt.bar(x + i*width - 0.4 + width/2, hit_rates, width, 
                label=config_name, alpha=0.7)
    
    # Configurar gráfico
    plt.title('Comparación de Tasas de Aciertos entre Escenarios')
    plt.xlabel('Escenario')
    plt.ylabel('Tasa de Aciertos (%)')
    plt.xticks(x, scenario_names, rotation=45)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Guardar gráfico
    filename = "scenario_comparison_hit_rates.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    # Comparación de tiempos de acceso entre escenarios
    plt.figure(figsize=(12, 8))
    
    # Crear barras para cada configuración
    for i, config_name in enumerate(config_names):
        access_times = [scenario_results[scenario][config_name]["avg_access_time"] * 1000  # Convertir a ms
                       for scenario in scenario_names]
        
        plt.bar(x + i*width - 0.4 + width/2, access_times, width, 
                label=config_name, alpha=0.7)
    
    # Configurar gráfico
    plt.title('Comparación de Tiempos de Acceso entre Escenarios')
    plt.xlabel('Escenario')
    plt.ylabel('Tiempo de Acceso (ms)')
    plt.xticks(x, scenario_names, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Guardar gráfico
    filename = "scenario_comparison_access_times.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def create_comparison_charts(results: Dict[str, Any], output_dir: str) -> None:
    """
    Genera gráficos de comparación adicionales.
    
    Args:
        results: Diccionario con los resultados del benchmark
        output_dir: Directorio donde guardar los gráficos
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Extraer datos para los gráficos
    scenario_results = results.get("scenario_results", {})
    configs = results.get("configs", {})
    
    if not scenario_results or not configs:
        print("No hay suficientes datos para generar gráficos de comparación.")
        return
    
    # Gráfico de radar para comparar configuraciones
    _create_radar_chart(scenario_results, configs, output_dir)
    
    # Gráfico de dispersión para relación entre tasa de aciertos y tiempo de acceso
    _create_scatter_plot(scenario_results, configs, output_dir)


def _create_radar_chart(scenario_results: Dict[str, Dict[str, Dict[str, Any]]], 
                       configs: Dict[str, Dict[str, Any]], 
                       output_dir: str) -> None:
    """
    Genera un gráfico de radar para comparar configuraciones.
    
    Args:
        scenario_results: Resultados de todos los escenarios
        configs: Configuraciones de caché
        output_dir: Directorio de salida
    """
    # Preparar datos
    scenario_names = list(scenario_results.keys())
    config_names = list(configs.keys())
    
    # Crear figura
    plt.figure(figsize=(10, 10))
    
    # Calcular ángulos para el gráfico de radar
    angles = np.linspace(0, 2*np.pi, len(scenario_names), endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo
    
    # Crear subgráfico polar
    ax = plt.subplot(111, polar=True)
    
    # Añadir cada configuración al gráfico
    for config_name in config_names:
        # Obtener tasas de aciertos para esta configuración
        values = [scenario_results[scenario][config_name]["hit_rate"] 
                 for scenario in scenario_names]
        values += values[:1]  # Cerrar el círculo
        
        # Dibujar línea y puntos
        ax.plot(angles, values, linewidth=2, label=config_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Configurar gráfico
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(scenario_names)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(0, 100)
    
    plt.title('Comparación de Configuraciones por Escenario')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Guardar gráfico
    filename = "radar_comparison.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def _create_scatter_plot(scenario_results: Dict[str, Dict[str, Dict[str, Any]]], 
                        configs: Dict[str, Dict[str, Any]], 
                        output_dir: str) -> None:
    """
    Genera un gráfico de dispersión para la relación entre tasa de aciertos y tiempo de acceso.
    
    Args:
        scenario_results: Resultados de todos los escenarios
        configs: Configuraciones de caché
        output_dir: Directorio de salida
    """
    plt.figure(figsize=(12, 8))
    
    # Colores para cada configuración
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Marcadores para cada escenario
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # Añadir puntos para cada combinación de escenario y configuración
    for i, scenario_name in enumerate(scenario_results.keys()):
        for j, config_name in enumerate(configs.keys()):
            result = scenario_results[scenario_name][config_name]
            
            hit_rate = result["hit_rate"]
            access_time = result["avg_access_time"] * 1000  # Convertir a ms
            
            plt.scatter(hit_rate, access_time, 
                       color=colors[j % len(colors)], 
                       marker=markers[i % len(markers)],
                       s=100,
                       label=f"{scenario_name} - {config_name}" if i == 0 or j == 0 else "")
            
            # Añadir etiqueta
            plt.annotate(f"{config_name[:3]}", 
                        (hit_rate, access_time),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center')
    
    # Configurar gráfico
    plt.title('Relación entre Tasa de Aciertos y Tiempo de Acceso')
    plt.xlabel('Tasa de Aciertos (%)')
    plt.ylabel('Tiempo de Acceso (ms)')
    plt.grid(linestyle='--', alpha=0.7)
    
    # Crear leyenda personalizada
    config_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=colors[j % len(colors)], 
                               markersize=10, label=config_name)
                    for j, config_name in enumerate(configs.keys())]
    
    scenario_handles = [plt.Line2D([0], [0], marker=markers[i % len(markers)], 
                                 color='black', markersize=10, 
                                 label=scenario_name)
                      for i, scenario_name in enumerate(scenario_results.keys())]
    
    plt.legend(handles=config_handles + scenario_handles, 
              loc='upper right', 
              title="Configuraciones y Escenarios")
    
    plt.tight_layout()
    
    # Guardar gráfico
    filename = "scatter_hit_rate_vs_time.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
