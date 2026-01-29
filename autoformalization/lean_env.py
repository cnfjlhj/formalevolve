"""
Lean Server Environment - Global Singleton.

Provides a globally shared Lean4 server instance to avoid frequent restarts.
Used for both compilation checking and BEq+ equivalence verification.
"""

import logging
import os
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

                           
_lean_server = None

               
 
                  
                                                                                                
                                                                                            
                                                        
 
                                                   
                                        
                                   
LEAN_VERSION = os.environ.get("LEAN_INTERACT_LEAN_VERSION", "v4.15.0").strip() or "v4.15.0"
LEAN_REQUIRE = os.environ.get("LEAN_INTERACT_REQUIRE", "mathlib").strip() or "mathlib"
DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "lean_interact")
DEFAULT_MAX_TOTAL_MEMORY = float(os.environ.get("LEAN_INTERACT_MAX_TOTAL_MEMORY", "0.90"))


def _pick_writable_cache_dir() -> str:
    candidates = []
    env_cache = os.environ.get("LEAN_INTERACT_CACHE_DIR", "").strip()
    if env_cache:
        candidates.append(env_cache)
    candidates.append(DEFAULT_CACHE_DIR)
    candidates.append(str(Path.cwd() / ".lean_interact_cache"))
    candidates.append("/tmp/lean_interact_cache_autoformal")

    for c in candidates:
        p = Path(c)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return str(p)
        except Exception:
            continue

    fallback = Path.cwd() / ".lean_interact_cache"
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


def get_lean_server():
    """
    Get the global singleton Lean server instance.

    Lazily initializes the server on first call. Subsequent calls return
    the same instance to avoid costly restarts.

    Returns:
        AutoLeanServer: The global Lean server instance

    Raises:
        ImportError: If lean_interact is not installed
        Exception: If server initialization fails
    """
    global _lean_server

    if _lean_server is None:
        logger.info("Initializing global Lean server...")

        try:
            from lean_interact import AutoLeanServer, LeanREPLConfig
            from lean_interact.project import TempRequireProject

            cache_dir = _pick_writable_cache_dir()
            os.environ["LEAN_INTERACT_CACHE_DIR"] = cache_dir

            config = LeanREPLConfig(
                project=TempRequireProject(
                    lean_version=LEAN_VERSION,
                    require=LEAN_REQUIRE
                ),
                cache_dir=cache_dir,
                verbose=False,
            )
            _lean_server = AutoLeanServer(config=config, max_total_memory=DEFAULT_MAX_TOTAL_MEMORY)
            logger.info(
                f"Lean server initialized (version={LEAN_VERSION}, max_total_memory={DEFAULT_MAX_TOTAL_MEMORY})"
            )

        except ImportError as e:
            logger.error(f"lean_interact not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Lean server: {e}")
            raise

    return _lean_server


def is_lean_server_available() -> bool:
    """
    Check if the Lean server is available without initializing it.

    Returns:
        bool: True if server is already initialized
    """
    return _lean_server is not None


def reset_lean_server():
    """
    Reset the global Lean server instance.

    This forces a new server to be created on the next get_lean_server() call.
    Use this if the server becomes unresponsive.
    """
    global _lean_server
    if _lean_server is not None:
        logger.info("Resetting Lean server...")
        _lean_server = None


def shutdown_lean_server():
    """
    Gracefully shutdown the Lean server.

    Call this when done with all Lean operations.
    """
    global _lean_server
    if _lean_server is not None:
        logger.info("Shutting down Lean server...")
        try:
                                                             
            if hasattr(_lean_server, 'close'):
                _lean_server.close()
            elif hasattr(_lean_server, 'shutdown'):
                _lean_server.shutdown()
        except Exception as e:
            logger.warning(f"Error during Lean server shutdown: {e}")
        finally:
            _lean_server = None


if __name__ == "__main__":
                     
    logging.basicConfig(level=logging.INFO)

    print("Testing lean_env module...")

    try:
        server = get_lean_server()
        print(f"Server initialized: {server}")
        print(f"Is available: {is_lean_server_available()}")

                                       
        from lean_interact import Command
        result = server.run(Command(cmd="#check Nat"), timeout=30)
        print(f"Test result: {result}")

        shutdown_lean_server()
        print("Server shutdown complete")

    except Exception as e:
        print(f"Test failed: {e}")
