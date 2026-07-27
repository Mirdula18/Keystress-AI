"""
Console entrypoint: ``python -m keystress``.

Delegates to the application factory's ``main``. Also reachable as the ``keystress``
console script installed by ``pip install -e .``.
"""

from __future__ import annotations

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
