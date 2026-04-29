import hashlib
import socket

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options to pytest.

    The `--update-trace-assets` flag, when set, causes integration tests that
    generate trace asset files to update the assets directory instead of
    asserting equality.
    """
    parser.addoption(
        "--update-trace-assets",
        action="store_true",
        default=False,
        help="Update trace asset files instead of asserting equality.",
    )


def _is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def _get_deterministic_port(test_name: str) -> int:
    """Generate a deterministic port number based on test name."""
    hash_value = int(hashlib.md5(test_name.encode()).hexdigest()[:4], 16)  # noqa: S324
    port = 6000 + (hash_value % 4000)

    original_port = port
    attempts = 0
    while not _is_port_available(port) and attempts < 50:
        port = original_port + attempts + 1
        attempts += 1

    if not _is_port_available(port):
        msg = f"Could not find an available port starting from {original_port}"
        raise RuntimeError(msg)

    return port


@pytest.fixture
def test_port(request: pytest.FixtureRequest) -> int:
    """Single fixture that provides a unique, deterministic port for each test."""
    return _get_deterministic_port(request.node.name)
