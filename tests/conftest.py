"""Fixtures for tests that run the vendored verl actor in-process (see tests/distill_helpers.py)."""

import socket

import pytest


@pytest.fixture(scope="session")
def torch_process_group():
    """verl's actor asks torch.distributed for its rank, so start a one-process group."""
    torch = pytest.importorskip("torch")
    if torch.distributed.is_initialized():
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    torch.distributed.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    yield
    torch.distributed.destroy_process_group()


@pytest.fixture(scope="session")
def cpu_memory_logging():
    """verl logs device memory around every actor call; torch has no such counters for the CPU."""
    device = pytest.importorskip("verl.utils.device")
    if device.get_device_name() != "cpu":
        yield
        return
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "verl.utils.debug.performance._get_current_mem_info", lambda *a, **k: ("0",) * 4
        )
        yield
