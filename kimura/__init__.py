# kimura/__init__.py - Expose public API
from .session.worker import SecureClient
from .session.master import SecureServer
from .session.manager import SessionManager

__version__ = "0.1.0"
__all__ = ["SecureClient", "SecureServer", "SessionManager"]
