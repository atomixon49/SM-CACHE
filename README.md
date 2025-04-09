# Sistema de Caché Inteligente

Un sistema de caché avanzado que aprende patrones de uso, predice qué datos serán necesarios y gestiona automáticamente la memoria.

## Características

### Características básicas
- **Aprendizaje de patrones**: Analiza y aprende de los patrones de acceso a datos.
- **Predicción inteligente**: Anticipa qué datos serán necesarios en el futuro.
- **Prefetch automático**: Precarga datos que probablemente se necesitarán pronto.
- **Gestión automática de memoria**: Optimiza el uso de memoria eviccionando elementos según patrones de uso.
- **Soporte para TTL**: Tiempo de vida configurable para elementos del caché.
- **Carga asíncrona**: Posibilidad de cargar datos de forma asíncrona.
- **Thread-safe**: Diseñado para entornos multi-hilo.

### Características avanzadas
- **Algoritmos de aprendizaje avanzados**: Incluye modelos de predicción basados en cadenas de Markov, frecuencias y reglas de asociación.
- **Predicción de conjunto**: Combina múltiples algoritmos para mejorar la precisión de las predicciones.
- **Persistencia en disco**: Guarda y carga el estado del caché en disco para mantener los datos entre reinicios.
- **Guardado automático**: Guarda periódicamente el estado del caché para evitar pérdida de datos.
- **Caché distribuido**: Sincroniza el caché entre múltiples nodos para mejorar la escalabilidad y disponibilidad.
- **Monitoreo y métricas**: Proporciona información detallada sobre el rendimiento y uso del caché.

## Estructura del proyecto

```
cache_system/
├── __init__.py
├── intelligent_cache.py     # Clase principal del sistema
├── usage_tracker.py         # Seguimiento de patrones de uso
├── predictor.py             # Algoritmo de predicción básico
├── advanced_learning.py     # Algoritmos de aprendizaje avanzados
├── memory_manager.py        # Gestión de memoria
├── persistence.py           # Persistencia en disco
├── distributed.py           # Caché distribuido
├── monitoring.py            # Monitoreo y métricas
├── utils.py                 # Utilidades
└── tests/                   # Pruebas unitarias
    ├── __init__.py
    ├── test_cache.py
    ├── test_predictor.py
    ├── test_advanced_learning.py
    ├── test_memory_manager.py
    ├── test_persistence.py
    ├── test_distributed.py
    └── test_monitoring.py
```

## Instalación

No se requiere instalación especial, simplemente clone el repositorio:

```bash
git clone <url-del-repositorio>
cd cache-system
```

## Uso básico

```python
from cache_system import IntelligentCache

# Crear una instancia de caché básico
cache = IntelligentCache(
    max_size=1000,              # Número máximo de elementos
    max_memory_mb=100.0,        # Memoria máxima en MB
    ttl=3600,                   # Tiempo de vida en segundos (opcional)
    prefetch_enabled=True,      # Habilitar prefetch
    data_loader=load_function   # Función para cargar datos (opcional)
)

# O crear una instancia con características avanzadas
cache_advanced = IntelligentCache(
    max_size=1000,
    max_memory_mb=100.0,
    use_advanced_learning=True,     # Usar algoritmos de aprendizaje avanzados
    persistence_enabled=True,       # Habilitar persistencia en disco
    persistence_dir=".cache",       # Directorio para archivos de caché
    cache_name="my_cache",          # Nombre para identificar archivos
    auto_save_interval=300,         # Guardar cada 5 minutos
    distributed_enabled=True,       # Habilitar caché distribuido
    distributed_host="localhost",   # Host para este nodo
    distributed_port=5000,          # Puerto para este nodo
    monitoring_enabled=True,        # Habilitar monitoreo y métricas
    metrics_export_interval=60      # Exportar métricas cada minuto
)

# Operaciones básicas
cache.put("key1", "value1")     # Almacenar un valor
value = cache.get("key1")       # Obtener un valor
exists = cache.contains("key1") # Verificar si existe
cache.remove("key1")            # Eliminar un valor
cache.clear()                   # Limpiar todo el caché

# Operaciones de persistencia
cache_advanced.save()           # Guardar explícitamente el estado en disco

# Obtener estadísticas
stats = cache.get_stats()
print(stats)

# Detener el caché (limpia recursos y guarda el estado si tiene persistencia)
cache.stop()
cache_advanced.stop()
```

## Carga automática de datos

Si se proporciona un `data_loader`, el caché cargará automáticamente los datos cuando no estén disponibles:

```python
def load_data(key):
    # Simular carga desde base de datos o API
    print(f"Cargando datos para: {key}")
    return f"Datos para {key}"

cache = IntelligentCache(data_loader=load_data)

# Esto cargará automáticamente los datos si no están en caché
value = cache.get("nueva_clave")
```

## Ejemplo completo

Consulte el archivo `example.py` para ver un ejemplo completo de uso del sistema.

```bash
python example.py
```

## Ejecutar pruebas

```bash
python -m unittest discover cache_system/tests
```

## Características avanzadas

### Algoritmos de aprendizaje avanzados

El sistema incluye varios algoritmos de predicción avanzados:

- **Cadenas de Markov**: Modelan las transiciones entre claves para predecir la siguiente.
- **Predicción basada en frecuencias**: Utiliza ventanas temporales para predecir elementos frecuentes.
- **Reglas de asociación**: Identifica relaciones entre elementos que suelen aparecer juntos.
- **Predicción de conjunto**: Combina los resultados de múltiples algoritmos para mejorar la precisión.

```python
# Habilitar algoritmos avanzados
cache = IntelligentCache(use_advanced_learning=True)
```

### Persistencia en disco

El sistema puede guardar y cargar el estado del caché en disco:

```python
# Habilitar persistencia
cache = IntelligentCache(
    persistence_enabled=True,
    persistence_dir=".cache",       # Directorio para archivos
    cache_name="my_app_cache",     # Nombre identificativo
    auto_save_interval=300         # Guardar cada 5 minutos (o None para desactivar)
)

# Guardar explícitamente
cache.save()
```

### Caché distribuido

El sistema permite sincronizar el caché entre múltiples nodos:

```python
# Configuración para el Nodo 1
cache_nodo1 = IntelligentCache(
    distributed_enabled=True,
    distributed_host="192.168.1.10",  # IP del nodo actual
    distributed_port=5000,
    cluster_nodes=[("192.168.1.11", 5000), ("192.168.1.12", 5000)]  # Otros nodos
)

# Configuración para el Nodo 2
cache_nodo2 = IntelligentCache(
    distributed_enabled=True,
    distributed_host="192.168.1.11",
    distributed_port=5000,
    cluster_nodes=[("192.168.1.10", 5000), ("192.168.1.12", 5000)]
)

# Las operaciones se sincronizan automáticamente entre nodos
cache_nodo1.put("key1", "value1")  # Disponible en todos los nodos
value = cache_nodo2.get("key1")    # Obtiene el valor desde cualquier nodo
```

### Monitoreo y métricas

El sistema proporciona métricas detalladas sobre el rendimiento y uso del caché:

```python
# Habilitar monitoreo
cache = IntelligentCache(
    monitoring_enabled=True,
    metrics_export_interval=60,     # Exportar métricas cada minuto
    alert_thresholds={
        'performance.hit_rate': 50.0,  # Alertar si tasa de aciertos < 50%
        'memory_usage_percent': 90.0   # Alertar si uso de memoria > 90%
    }
)

# Obtener métricas
stats = cache.get_stats()
metricas = stats['metrics']

print(f"Tasa de aciertos: {metricas['performance']['hit_rate']:.2f}%")
print(f"Tiempo promedio de respuesta: {metricas['performance']['avg_get_time']*1000:.2f} ms")

# Exportar métricas manualmente
cache.metrics_monitor.export_metrics()
```

## Personalización

El sistema es altamente personalizable:

- **Estimación de tamaño**: Proporcione su propia función `size_estimator` para calcular el tamaño de los elementos.
- **Carga de datos**: Implemente su propio `data_loader` para integrar con su fuente de datos.
- **Serialización**: Use `CacheSerializer` para persistir el estado del caché.

## Uso en producción

Para utilizar el sistema de caché inteligente en entornos de producción, se recomienda seguir estas pautas:

### Instalación como paquete

```bash
# Desde el directorio raíz del proyecto
pip install -e .
```

### Configuración recomendada para producción

```python
# Función para cargar datos cuando no están en caché
def cargar_datos(clave):
    # Lógica para cargar datos desde la fuente original
    return datos_cargados

# Crear caché con todas las características habilitadas
cache = IntelligentCache(
    # Configuración básica
    max_size=10000,
    max_memory_mb=500,
    ttl=3600,  # 1 hora de tiempo de vida

    # Carga automática de datos
    prefetch_enabled=True,
    data_loader=cargar_datos,

    # Aprendizaje avanzado
    use_advanced_learning=True,

    # Persistencia
    persistence_enabled=True,
    persistence_dir="/ruta/almacenamiento",
    cache_name="mi_aplicacion_cache",
    auto_save_interval=300,  # Guardar cada 5 minutos

    # Caché distribuido
    distributed_enabled=True,
    distributed_host="192.168.1.10",  # IP del nodo actual
    distributed_port=5000,
    cluster_nodes=[("192.168.1.11", 5000), ("192.168.1.12", 5000)],  # Otros nodos

    # Monitoreo
    monitoring_enabled=True,
    metrics_export_interval=60,  # Exportar métricas cada minuto
    alert_thresholds={
        'performance.hit_rate': 50.0,  # Alertar si tasa de aciertos < 50%
        'memory_usage_percent': 90.0   # Alertar si uso de memoria > 90%
    }
)
```

### Integración con frameworks web

```python
# Ejemplo con Flask
from flask import Flask
from cache_system import IntelligentCache

app = Flask(__name__)
cache = IntelligentCache(max_size=1000)

@app.route('/datos/<id>')
def obtener_datos(id):
    # Intentar obtener del caché primero
    resultado = cache.get(id)
    if resultado is None:
        # Si no está en caché, obtener de la base de datos
        resultado = obtener_de_base_de_datos(id)
        # Guardar en caché para futuras solicitudes
        cache.put(id, resultado)
    return resultado
```

### Consideraciones para producción

1. **Memoria**: Ajusta `max_memory_mb` según los recursos disponibles en tu servidor.
2. **TTL**: Configura un tiempo de vida adecuado para tus datos para evitar información obsoleta.
3. **Persistencia**: Asegúrate de que el directorio de persistencia tenga suficiente espacio y permisos.
4. **Distribución**: Configura correctamente los firewalls para permitir la comunicación entre nodos.
5. **Monitoreo**: Revisa regularmente las métricas para optimizar el rendimiento.

## Licencia

[MIT](LICENSE)
