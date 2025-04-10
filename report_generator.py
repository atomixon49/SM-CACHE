"""
Módulo para generar informes de los resultados del benchmark del sistema de caché inteligente.

Este módulo proporciona funciones para crear informes HTML detallados
a partir de los resultados del benchmark.
"""

import os
import json
import time
from typing import Dict, List, Any
from datetime import datetime


def generate_report(results: Dict[str, Any], 
                   vis_dir: str, 
                   output_dir: str) -> str:
    """
    Genera un informe HTML con los resultados del benchmark.
    
    Args:
        results: Diccionario con los resultados del benchmark
        vis_dir: Directorio con las visualizaciones
        output_dir: Directorio donde guardar el informe
        
    Returns:
        Ruta al informe generado
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Nombre del archivo de informe
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_filename = f"benchmark_report_{timestamp}.html"
    report_path = os.path.join(output_dir, report_filename)
    
    # Extraer datos para el informe
    scenario_results = results.get("scenario_results", {})
    configs = results.get("configs", {})
    timestamp = results.get("timestamp", time.time())
    
    # Convertir timestamp a fecha legible
    date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    # Generar HTML
    html = _generate_html_report(scenario_results, configs, date_str, vis_dir)
    
    # Guardar informe
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return report_path


def _generate_html_report(scenario_results: Dict[str, Dict[str, Dict[str, Any]]],
                         configs: Dict[str, Dict[str, Any]],
                         date_str: str,
                         vis_dir: str) -> str:
    """
    Genera el contenido HTML del informe.
    
    Args:
        scenario_results: Resultados de todos los escenarios
        configs: Configuraciones de caché
        date_str: Fecha y hora del benchmark
        vis_dir: Directorio con las visualizaciones
        
    Returns:
        Contenido HTML del informe
    """
    # Ruta relativa a las imágenes
    vis_rel_path = os.path.relpath(vis_dir, os.path.dirname(vis_dir))
    
    # Iniciar HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Informe de Benchmark - SM-CACHE</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                background-color: #3498db;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 30px;
                border-radius: 5px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .chart {{
                max-width: 100%;
                height: auto;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .summary {{
                background-color: #e8f4f8;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .highlight {{
                font-weight: bold;
                color: #2980b9;
            }}
            footer {{
                text-align: center;
                margin-top: 30px;
                padding: 10px;
                background-color: #2c3e50;
                color: white;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Informe de Benchmark - Sistema de Caché Inteligente SM-CACHE</h1>
            <p>Fecha: {date_str}</p>
        </div>
        
        <div class="section">
            <h2>Resumen Ejecutivo</h2>
            <div class="summary">
                <p>Este informe presenta los resultados de las pruebas de rendimiento del sistema de caché inteligente SM-CACHE bajo diferentes configuraciones y escenarios de uso.</p>
                <p>Se evaluaron <span class="highlight">{len(configs)}</span> configuraciones diferentes en <span class="highlight">{len(scenario_results)}</span> escenarios de uso.</p>
                {_generate_executive_summary(scenario_results, configs)}
            </div>
        </div>
        
        <div class="section">
            <h2>Configuraciones Evaluadas</h2>
            <table>
                <tr>
                    <th>Configuración</th>
                    <th>Descripción</th>
                    <th>Parámetros</th>
                </tr>
                {_generate_config_table_rows(configs)}
            </table>
        </div>
        
        <div class="section">
            <h2>Resultados por Escenario</h2>
            {_generate_scenario_sections(scenario_results, configs, vis_rel_path)}
        </div>
        
        <div class="section">
            <h2>Comparación entre Escenarios</h2>
            <div class="chart-container">
                <h3>Tasas de Aciertos por Escenario</h3>
                <img class="chart" src="{vis_rel_path}/scenario_comparison_hit_rates.png" alt="Comparación de tasas de aciertos">
                
                <h3>Tiempos de Acceso por Escenario</h3>
                <img class="chart" src="{vis_rel_path}/scenario_comparison_access_times.png" alt="Comparación de tiempos de acceso">
                
                <h3>Comparación de Configuraciones</h3>
                <img class="chart" src="{vis_rel_path}/radar_comparison.png" alt="Comparación de configuraciones">
                
                <h3>Relación entre Tasa de Aciertos y Tiempo de Acceso</h3>
                <img class="chart" src="{vis_rel_path}/scatter_hit_rate_vs_time.png" alt="Relación entre tasa de aciertos y tiempo de acceso">
            </div>
        </div>
        
        <div class="section">
            <h2>Conclusiones y Recomendaciones</h2>
            {_generate_conclusions(scenario_results, configs)}
        </div>
        
        <footer>
            <p>Generado automáticamente por el sistema de benchmark de SM-CACHE</p>
        </footer>
    </body>
    </html>
    """
    
    return html


def _generate_executive_summary(scenario_results: Dict[str, Dict[str, Dict[str, Any]]],
                               configs: Dict[str, Dict[str, Any]]) -> str:
    """
    Genera el resumen ejecutivo del informe.
    
    Args:
        scenario_results: Resultados de todos los escenarios
        configs: Configuraciones de caché
        
    Returns:
        HTML del resumen ejecutivo
    """
    # Encontrar la mejor configuración general
    best_config = None
    best_avg_hit_rate = 0
    
    for config_name in configs.keys():
        # Calcular tasa de aciertos promedio para esta configuración
        hit_rates = [scenario_data[config_name]["hit_rate"] 
                    for scenario_data in scenario_results.values()]
        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        
        if avg_hit_rate > best_avg_hit_rate:
            best_avg_hit_rate = avg_hit_rate
            best_config = config_name
    
    # Encontrar el escenario con mejor rendimiento
    best_scenario = None
    best_scenario_hit_rate = 0
    
    for scenario_name, scenario_data in scenario_results.items():
        # Calcular tasa de aciertos promedio para este escenario
        hit_rates = [config_data["hit_rate"] 
                    for config_data in scenario_data.values()]
        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        
        if avg_hit_rate > best_scenario_hit_rate:
            best_scenario_hit_rate = avg_hit_rate
            best_scenario = scenario_name
    
    # Generar HTML
    html = f"""
    <p>La configuración con mejor rendimiento general fue <span class="highlight">{best_config}</span> con una tasa de aciertos promedio de <span class="highlight">{best_avg_hit_rate:.2f}%</span>.</p>
    <p>El escenario con mejor rendimiento fue <span class="highlight">{best_scenario}</span> con una tasa de aciertos promedio de <span class="highlight">{best_scenario_hit_rate:.2f}%</span>.</p>
    """
    
    return html


def _generate_config_table_rows(configs: Dict[str, Dict[str, Any]]) -> str:
    """
    Genera las filas de la tabla de configuraciones.
    
    Args:
        configs: Configuraciones de caché
        
    Returns:
        HTML de las filas de la tabla
    """
    html = ""
    
    for config_name, config_params in configs.items():
        # Generar descripción basada en parámetros
        description = "Configuración "
        if config_params.get("prefetch_enabled", False):
            description += "con prefetch "
        else:
            description += "sin prefetch "
            
        if config_params.get("use_advanced_learning", False):
            description += "y aprendizaje avanzado"
        else:
            description += "y aprendizaje básico"
            
        if config_params.get("persistence_enabled", False):
            description += ", con persistencia"
            
        if config_params.get("monitoring_enabled", False):
            description += ", con monitoreo"
        
        # Formatear parámetros
        params_str = "<ul>"
        for key, value in config_params.items():
            params_str += f"<li><strong>{key}:</strong> {value}</li>"
        params_str += "</ul>"
        
        # Añadir fila
        html += f"""
        <tr>
            <td>{config_name}</td>
            <td>{description}</td>
            <td>{params_str}</td>
        </tr>
        """
    
    return html


def _generate_scenario_sections(scenario_results: Dict[str, Dict[str, Dict[str, Any]]],
                               configs: Dict[str, Dict[str, Any]],
                               vis_rel_path: str) -> str:
    """
    Genera las secciones de resultados por escenario.
    
    Args:
        scenario_results: Resultados de todos los escenarios
        configs: Configuraciones de caché
        vis_rel_path: Ruta relativa a las visualizaciones
        
    Returns:
        HTML de las secciones de escenarios
    """
    html = ""
    
    for scenario_name, scenario_data in scenario_results.items():
        html += f"""
        <div class="section">
            <h3>{scenario_name}</h3>
            
            <table>
                <tr>
                    <th>Configuración</th>
                    <th>Tasa de Aciertos</th>
                    <th>Tiempo de Acceso</th>
                    <th>Operaciones</th>
                </tr>
                {_generate_scenario_table_rows(scenario_data)}
            </table>
            
            <div class="chart-container">
                <h4>Tasa de Aciertos</h4>
                <img class="chart" src="{vis_rel_path}/{scenario_name}_hit_rates.png" alt="Tasa de aciertos - {scenario_name}">
                
                <h4>Tiempo de Acceso</h4>
                <img class="chart" src="{vis_rel_path}/{scenario_name}_access_times.png" alt="Tiempo de acceso - {scenario_name}">
            </div>
        </div>
        """
    
    return html


def _generate_scenario_table_rows(scenario_data: Dict[str, Dict[str, Any]]) -> str:
    """
    Genera las filas de la tabla de resultados de un escenario.
    
    Args:
        scenario_data: Datos del escenario
        
    Returns:
        HTML de las filas de la tabla
    """
    html = ""
    
    for config_name, result in scenario_data.items():
        html += f"""
        <tr>
            <td>{config_name}</td>
            <td>{result["hit_rate"]:.2f}%</td>
            <td>{result["avg_access_time"]*1000:.2f} ms</td>
            <td>{result["operations"]}</td>
        </tr>
        """
    
    return html


def _generate_conclusions(scenario_results: Dict[str, Dict[str, Dict[str, Any]]],
                         configs: Dict[str, Dict[str, Any]]) -> str:
    """
    Genera las conclusiones y recomendaciones del informe.
    
    Args:
        scenario_results: Resultados de todos los escenarios
        configs: Configuraciones de caché
        
    Returns:
        HTML de las conclusiones
    """
    # Calcular métricas para cada configuración
    config_metrics = {}
    
    for config_name in configs.keys():
        hit_rates = []
        access_times = []
        
        for scenario_data in scenario_results.values():
            result = scenario_data[config_name]
            hit_rates.append(result["hit_rate"])
            access_times.append(result["avg_access_time"])
        
        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        avg_access_time = sum(access_times) / len(access_times)
        
        config_metrics[config_name] = {
            "avg_hit_rate": avg_hit_rate,
            "avg_access_time": avg_access_time
        }
    
    # Encontrar la mejor configuración para diferentes criterios
    best_hit_rate_config = max(config_metrics.items(), key=lambda x: x[1]["avg_hit_rate"])[0]
    best_access_time_config = min(config_metrics.items(), key=lambda x: x[1]["avg_access_time"])[0]
    
    # Generar recomendaciones basadas en los resultados
    html = f"""
    <p>Basado en los resultados del benchmark, se pueden extraer las siguientes conclusiones:</p>
    
    <ul>
        <li>La configuración <strong>{best_hit_rate_config}</strong> ofrece la mejor tasa de aciertos promedio ({config_metrics[best_hit_rate_config]["avg_hit_rate"]:.2f}%), lo que la hace ideal para aplicaciones donde la prioridad es minimizar las cargas desde la fuente de datos.</li>
        
        <li>La configuración <strong>{best_access_time_config}</strong> proporciona el tiempo de acceso más rápido ({config_metrics[best_access_time_config]["avg_access_time"]*1000:.2f} ms), siendo óptima para aplicaciones donde la latencia es crítica.</li>
    """
    
    # Analizar el impacto de características específicas
    prefetch_impact = _analyze_feature_impact(scenario_results, configs, "prefetch_enabled")
    advanced_learning_impact = _analyze_feature_impact(scenario_results, configs, "use_advanced_learning")
    
    html += f"""
        <li>El uso de prefetch {prefetch_impact["conclusion"]} La mejora promedio en la tasa de aciertos es de aproximadamente {prefetch_impact["hit_rate_diff"]:.2f}%.</li>
        
        <li>El aprendizaje avanzado {advanced_learning_impact["conclusion"]} La diferencia en la tasa de aciertos es de aproximadamente {advanced_learning_impact["hit_rate_diff"]:.2f}%.</li>
    </ul>
    
    <p><strong>Recomendaciones:</strong></p>
    
    <ul>
    """
    
    # Generar recomendaciones específicas para diferentes casos de uso
    if config_metrics[best_hit_rate_config]["avg_hit_rate"] > 80:
        html += f"""
        <li>Para aplicaciones con patrones de acceso predecibles, se recomienda usar la configuración <strong>{best_hit_rate_config}</strong> que aprovecha al máximo las capacidades predictivas del sistema.</li>
        """
    else:
        html += f"""
        <li>Para aplicaciones con patrones de acceso menos predecibles, considere aumentar el tamaño del caché o ajustar los algoritmos de predicción.</li>
        """
    
    if prefetch_impact["hit_rate_diff"] > 10:
        html += f"""
        <li>El prefetch muestra beneficios significativos en los escenarios probados. Se recomienda mantenerlo habilitado en la mayoría de los casos.</li>
        """
    else:
        html += f"""
        <li>El prefetch muestra beneficios moderados. Evalúe su uso según las características específicas de su aplicación.</li>
        """
    
    if advanced_learning_impact["hit_rate_diff"] > 5:
        html += f"""
        <li>Los algoritmos de aprendizaje avanzado muestran mejoras notables. Se recomienda su uso especialmente en aplicaciones con patrones de acceso complejos.</li>
        """
    else:
        html += f"""
        <li>Los algoritmos de aprendizaje avanzado muestran mejoras limitadas en los escenarios probados. Considere usar el algoritmo básico si los recursos son limitados.</li>
        """
    
    html += f"""
    </ul>
    
    <p><strong>Próximos pasos:</strong></p>
    
    <ul>
        <li>Realizar pruebas con conjuntos de datos más grandes para evaluar el rendimiento a escala.</li>
        <li>Probar el sistema en entornos distribuidos con múltiples nodos.</li>
        <li>Evaluar el impacto de diferentes políticas de evicción en escenarios específicos.</li>
        <li>Medir el consumo de recursos (CPU, memoria) de las diferentes configuraciones.</li>
    </ul>
    """
    
    return html


def _analyze_feature_impact(scenario_results: Dict[str, Dict[str, Dict[str, Any]]],
                           configs: Dict[str, Dict[str, Any]],
                           feature: str) -> Dict[str, Any]:
    """
    Analiza el impacto de una característica específica en el rendimiento.
    
    Args:
        scenario_results: Resultados de todos los escenarios
        configs: Configuraciones de caché
        feature: Característica a analizar
        
    Returns:
        Diccionario con análisis del impacto
    """
    # Separar configuraciones con y sin la característica
    with_feature = [name for name, params in configs.items() if params.get(feature, False)]
    without_feature = [name for name, params in configs.items() if not params.get(feature, False)]
    
    if not with_feature or not without_feature:
        return {
            "hit_rate_diff": 0,
            "access_time_diff": 0,
            "conclusion": "no pudo ser evaluado adecuadamente debido a la falta de configuraciones comparables."
        }
    
    # Calcular métricas promedio para cada grupo
    with_hit_rates = []
    with_access_times = []
    without_hit_rates = []
    without_access_times = []
    
    for scenario_data in scenario_results.values():
        for config in with_feature:
            if config in scenario_data:
                with_hit_rates.append(scenario_data[config]["hit_rate"])
                with_access_times.append(scenario_data[config]["avg_access_time"])
        
        for config in without_feature:
            if config in scenario_data:
                without_hit_rates.append(scenario_data[config]["hit_rate"])
                without_access_times.append(scenario_data[config]["avg_access_time"])
    
    # Calcular promedios
    avg_with_hit_rate = sum(with_hit_rates) / len(with_hit_rates) if with_hit_rates else 0
    avg_with_access_time = sum(with_access_times) / len(with_access_times) if with_access_times else 0
    avg_without_hit_rate = sum(without_hit_rates) / len(without_hit_rates) if without_hit_rates else 0
    avg_without_access_time = sum(without_access_times) / len(without_access_times) if without_access_times else 0
    
    # Calcular diferencias
    hit_rate_diff = avg_with_hit_rate - avg_without_hit_rate
    access_time_diff = avg_without_access_time - avg_with_access_time  # Positivo si con la característica es más rápido
    
    # Generar conclusión
    if hit_rate_diff > 0:
        if access_time_diff > 0:
            conclusion = f"mejora tanto la tasa de aciertos como el tiempo de acceso."
        else:
            conclusion = f"mejora la tasa de aciertos pero aumenta ligeramente el tiempo de acceso."
    else:
        if access_time_diff > 0:
            conclusion = f"reduce el tiempo de acceso pero disminuye ligeramente la tasa de aciertos."
        else:
            conclusion = f"no muestra beneficios claros en los escenarios probados."
    
    return {
        "hit_rate_diff": hit_rate_diff,
        "access_time_diff": access_time_diff * 1000,  # Convertir a ms
        "conclusion": conclusion
    }
