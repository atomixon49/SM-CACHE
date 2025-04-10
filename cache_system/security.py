"""
Sistema de seguridad y autenticación para SM-CACHE.
"""
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class CacheSecurity:
    """Gestiona la seguridad del sistema de caché."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self._api_keys: Dict[str, Dict[str, Any]] = {}
    
    def generate_api_key(self, client_id: str, permissions: list = None) -> str:
        """Genera una nueva API key para un cliente."""
        if permissions is None:
            permissions = ['read', 'write']
            
        api_key = secrets.token_urlsafe(32)
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        
        self._api_keys[hashed_key] = {
            'client_id': client_id,
            'permissions': permissions,
            'created_at': datetime.utcnow()
        }
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> bool:
        """Valida una API key."""
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        return hashed_key in self._api_keys
    
    def generate_access_token(self, client_id: str, expire_minutes: int = 30) -> str:
        """Genera un token JWT de acceso temporal."""
        expiration = datetime.utcnow() + timedelta(minutes=expire_minutes)
        
        payload = {
            'client_id': client_id,
            'exp': expiration,
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def validate_token(self, token: str) -> bool:
        """Valida un token JWT."""
        try:
            jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return True
        except jwt.InvalidTokenError:
            return False
    
    def check_permission(self, api_key: str, required_permission: str) -> bool:
        """Verifica si una API key tiene un permiso específico."""
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        if hashed_key not in self._api_keys:
            return False
            
        return required_permission in self._api_keys[hashed_key]['permissions']