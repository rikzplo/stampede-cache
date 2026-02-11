"""Lua script loader and registry for Redis/Valkey.

Centralizes Lua script management with:
- One-time file loading at module import
- Per-client script registration (EVALSHA optimization)
- Clean API for script execution

Usage:
    from stampede.lua import LuaScripts

    scripts = LuaScripts(__file__, {
        "release_lock": "release_lock.lua",
        "check_cache": "check_cache_and_lock.lua",
    })

    result = scripts.call(redis_client, "release_lock", keys=[key], args=[owner])
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import redis
    from redis.commands.core import Script


class LuaScripts:
    """Lazy-loading Lua script registry with per-client caching.

    Scripts are loaded from disk once at instantiation.
    Each Redis client gets its own registered Script objects (cached by client id).
    """

    __slots__ = ("_cache", "_scripts")

    def __init__(self, module_file: str, scripts: dict[str, str]):
        """Initialize script loader.

        Args:
            module_file: Pass __file__ from your module for relative path resolution
            scripts: Dict mapping script names to .lua filenames
        """
        base_dir = Path(module_file).parent
        self._scripts: dict[str, str] = {
            name: (base_dir / filename).read_text()
            for name, filename in scripts.items()
        }
        self._cache: dict[int, dict[str, Script]] = {}

    def _get_registered(self, client: redis.Redis) -> dict[str, Script]:
        """Get or register scripts for a Redis client."""
        client_id = id(client)
        if client_id not in self._cache:
            self._cache[client_id] = {
                name: client.register_script(script)
                for name, script in self._scripts.items()
            }
        return self._cache[client_id]

    def call(
        self,
        client: redis.Redis,
        name: str,
        keys: list[str] | None = None,
        args: list[Any] | None = None,
    ) -> Any:
        """Execute a named script on the given client.

        Args:
            client: Redis client to execute on
            name: Script name (as defined in __init__)
            keys: KEYS array for the Lua script
            args: ARGV array for the Lua script

        Returns:
            Script execution result
        """
        return self._get_registered(client)[name](keys=keys or [], args=args or [])

    def get(self, client: redis.Redis, name: str) -> Script:
        """Get a registered Script object for direct use."""
        return self._get_registered(client)[name]

    def clear_cache(self) -> None:
        """Clear the script registration cache. For testing."""
        self._cache.clear()
