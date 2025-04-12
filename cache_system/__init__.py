"""
Sistema de Caché Inteligente
----------------------------
Un sistema de caché que aprende patrones de uso, predice qué datos serán necesarios
y gestiona automáticamente la memoria.

Incluye implementaciones optimizadas para mayor rendimiento y eficiencia.
"""

from .intelligent_cache import IntelligentCache
from .optimized_cache import FastCache

__all__ = ['IntelligentCache', 'FastCache']
