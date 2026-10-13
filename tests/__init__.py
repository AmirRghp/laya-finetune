"""Test package.

Present so ``from tests.conftest import ...`` resolves: without it pytest puts
only ``tests/`` on ``sys.path``, and ``tests`` is not importable as a package.
"""