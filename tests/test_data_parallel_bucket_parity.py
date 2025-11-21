"""
python -m pytest tests/test_data_parallel_bucket_parity.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from picotron.data_parallel.data_parallel import (
    DataParallelBucket,
    DataParalleSyncronize,
)


class TinyNet(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.Tanh(),
            nn.Linear(8, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SpyBucketManager:
    """
    Minimal drop-in replacement that avoids initializing real distributed groups.

    The class only exposes the pieces DataParallelBucket touches in these tests:
    - `mark_param_as_ready`
    - `wait`
    - `reset`
    - `params_to_bucket_location` (for completeness)
    and it supplies `main_grad` views for every parameter.
    """

    params: List[torch.nn.Parameter]
    process_group: object
    bucket_size: int
    grad_type: torch.dtype
    wait_calls: int = 0
    reset_calls: int = 0

    def __init__(self, params, process_group, bucket_size, grad_type=torch.float32):
        self.params = [p for p in params if p.requires_grad]
        self.process_group = process_group
        self.bucket_size = bucket_size
        self.grad_type = grad_type
        self.wait_calls = 0
        self.reset_calls = 0
        self.params_with_grad_ready: List[torch.nn.Parameter] = []
        self.params_to_bucket_location = {
            param: (0, param.numel(), 0) for param in self.params
        }
        self.buckets = []
        for param in self.params:
            param.main_grad = torch.zeros_like(param, dtype=self.grad_type)

    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        self.params_with_grad_ready.append(param)

    def wait(self) -> None:
        self.wait_calls += 1

    def reset(self) -> None:
        self.reset_calls += 1
        self.params_with_grad_ready.clear()
        for param in self.params:
            param.main_grad.zero_()


@pytest.fixture(autouse=True)
def stub_process_group(monkeypatch):
    class DummyGroups:
        cp_dp_group = object()
        cp_dp_world_size = 1

    monkeypatch.setattr(
        "picotron.data_parallel.data_parallel.pgm.process_group_manager",
        DummyGroups(),
        raising=False,
    )


@pytest.fixture(autouse=True)
def spy_bucket_manager(monkeypatch):
    monkeypatch.setattr(
        "picotron.data_parallel.data_parallel.BucketManager",
        SpyBucketManager,
    )


def _build_model_pair() -> Sequence[TinyNet]:
    base = TinyNet()
    clone = TinyNet()
    clone.load_state_dict(base.state_dict())
    return base, clone


def _capture_param_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for name, param in module.named_parameters():
        state[name] = None if param.grad is None else param.grad.detach().clone()
    return state


def _drain_post_backward(dp_module):
    if dp_module._post_backward_callback_set:
        dp_module._post_backward()


def _run_backward_and_capture(dp_module, inputs, targets):
    dp_module.module.zero_grad(set_to_none=True)
    loss = F.mse_loss(dp_module(inputs), targets)
    loss.backward()
    _drain_post_backward(dp_module)
    return _capture_param_state(dp_module.module)


def _run_no_sync_sequence(dp_module, inputs_a, targets_a, inputs_b, targets_b):
    dp_module.module.zero_grad(set_to_none=True)
    with dp_module.no_sync():
        loss_a = F.mse_loss(dp_module(inputs_a), targets_a)
        loss_a.backward()

    loss_b = F.mse_loss(dp_module(inputs_b), targets_b)
    loss_b.backward()
    _drain_post_backward(dp_module)
    return _capture_param_state(dp_module.module)


def test_forward_outputs_match():
    torch.manual_seed(2024)
    model_ref, model_jovsa = _build_model_pair()
    dp_ref = DataParallelBucket(model_ref, bucket_cap_mb=1, grad_type=torch.float32)
    dp_sync = DataParalleSyncronize(model_jovsa)
    inputs = torch.randn(5, 6)

    out_ref = dp_ref(inputs)
    out_sync = dp_sync(inputs.clone())

    torch.testing.assert_close(out_ref, out_sync)


def test_gradients_match_reference():
    torch.manual_seed(1337)
    model_ref, model_jovsa = _build_model_pair()
    dp_ref = DataParallelBucket(model_ref, bucket_cap_mb=1, grad_type=torch.float32)
    dp_sync = DataParalleSyncronize(model_jovsa)
    inputs = torch.randn(4, 6)
    targets = torch.randn(4, 3)

    ref_state = _run_backward_and_capture(dp_ref, inputs, targets)
    sync_state = _run_backward_and_capture(dp_sync, inputs, targets)

    for name in ref_state:
        torch.testing.assert_close(
            ref_state[name],
            sync_state[name],
            msg=f"Gradient mismatch for {name}",
        )


def test_no_sync_sequences_identical():
    torch.manual_seed(42)
    model_ref, model_jovsa = _build_model_pair()
    dp_ref = DataParallelBucket(model_ref, bucket_cap_mb=1, grad_type=torch.float32)
    dp_sync = DataParalleSyncronize(model_jovsa)

    inputs_a = torch.randn(2, 6)
    targets_a = torch.randn(2, 3)
    inputs_b = torch.randn(2, 6)
    targets_b = torch.randn(2, 3)

    ref_state = _run_no_sync_sequence(
        dp_ref, inputs_a, targets_a, inputs_b, targets_b
    )
    sync_state = _run_no_sync_sequence(
        dp_sync, inputs_a, targets_a, inputs_b, targets_b
    )

    for name in ref_state:
        torch.testing.assert_close(
            ref_state[name],
            sync_state[name],
            msg=f"Gradient mismatch for {name}",
        )


def test_reset_clears_sync_gradients():
    torch.manual_seed(7)
    model = TinyNet()
    dp_sync = DataParalleSyncronize(model)

    inputs = torch.randn(4, 6)
    targets = torch.randn(4, 3)

    _run_backward_and_capture(dp_sync, inputs, targets)
    has_grad = any(param.grad is not None and torch.count_nonzero(param.grad) > 0 for param in model.parameters())
    assert has_grad

    dp_sync.reset()

    for param in model.parameters():
        if param.grad is not None:
            assert torch.count_nonzero(param.grad) == 0

