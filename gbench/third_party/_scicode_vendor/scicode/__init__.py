# Vendored verbatim from github.com/scicode-bench/SciCode at commit
# e3158ea011d4235245a547460d3688d7ccbf9900, src/scicode/__init__.py (Apache-2.0, the SciCode
# authors). Everything below the blank line after this header is the upstream file.

from pathlib import Path

__version__ = "1.0.0"


keys_cfg_path = Path(__file__).resolve().parent.parent.parent / "keys.cfg"


__all__ = [
    "keys_cfg_path",
    "__version__",
]
