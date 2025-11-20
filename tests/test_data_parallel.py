"""
Distributed test suite for picotron.data_parallel.

You can run these tests in two ways:

1. Standard unittest runner (default):
   python tests/test_data_parallel.py
   - Each test uses torch.multiprocessing.spawn to create ranks locally.

2. torchrun (mirrors training-time launch, executes one scenario per rank):
   torchrun --nproc_per_node=2 tests/test_data_parallel.py

Environment variables:
  DP_TEST_MAX_PROCS   -> max world size runnable via unittest (default: 2)
  DP_TEST_MASTER_PORT -> TCP port for init_process_group (default: 12367)
"""

import argparse
import os
import sys
import unittest
from typing import Dict, Iterable, List, Tuple
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from picotron.data_parallel import data_parallel as dp_module  # noqa: E402
from picotron.data_parallel.data_parallel import (  # noqa: E402
    DataParallelBucket,
    DataParallelNaive,
)
from picotron.process_group_manager import setup_process_group_manager  # noqa: E402

MAX_LOCAL_PROCS = int(os.environ.get("DP_TEST_MAX_PROCS", 2))
TEST_MASTER_PORT = os.environ.get("DP_TEST_MASTER_PORT", "12367")


class ToyModel(nn.Module):
    def __init__(self, in_features: int = 8, hidden_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _init_distributed(rank: int, world_size: int, backend: str = "gloo") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", TEST_MASTER_PORT)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1, dp_size=world_size)


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _sync_model_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        if not param.is_cuda:
            tensor = param.data.contiguous()
            dist.broadcast(tensor, src=0)
            param.data.copy_(tensor)
        else:
            dist.broadcast(param.data, src=0)


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _build_batch(
    rank: int,
    batch_idx: int,
    batch_size: int = 2,
    in_features: int = 8,
    out_features: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_input = torch.arange(
        batch_size * in_features, dtype=torch.float32
    ).view(batch_size, in_features)
    inputs = base_input / (batch_idx + 1) + rank * 0.5

    base_target = torch.arange(
        batch_size * out_features, dtype=torch.float32
    ).view(batch_size, out_features)
    targets = (base_target + batch_idx).div(10.0) + rank * 0.1
    return inputs, targets


def _compute_reference_grads(
    state_dict: Dict[str, torch.Tensor],
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    reference = ToyModel()
    reference.load_state_dict(state_dict)
    reference.zero_grad()
    for inputs, targets in batches:
        outputs = reference(inputs)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
    return {name: param.grad.detach().clone() for name, param in reference.named_parameters()}


def _gather_mean(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.contiguous().view(-1)
    gather_list = [torch.zeros_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, flat)
    stacked = torch.stack(gather_list, dim=0)
    return stacked.mean(dim=0).view_as(tensor)


def _assert_param_grads_match_mean(
    module: nn.Module,
    local_reference_grads: Dict[str, torch.Tensor],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    for (name, param) in module.named_parameters():
        expected_mean = _gather_mean(local_reference_grads[name])
        diff = (param.grad - expected_mean).abs().max().item()
        torch.testing.assert_close(
            param.grad,
            expected_mean,
            rtol=rtol,
            atol=atol,
            msg=f"Mismatch in {name}; max diff={diff}",
        )


def _scenario_naive(rank: int, world_size: int) -> None:
    _init_distributed(rank, world_size)
    torch.manual_seed(1234)
    model = ToyModel()
    _sync_model_parameters(model)
    assert dp_module.pgm.process_group_manager.cp_dp_world_size == world_size, (
        f"cp_dp_world_size mismatch: "
        f"{dp_module.pgm.process_group_manager.cp_dp_world_size} vs {world_size}"
    )

    batches = [_build_batch(rank, batch_idx=0)]
    ref_grads = _compute_reference_grads(_clone_state_dict(model), batches)

    dp = DataParallelNaive(model)
    inputs, targets = batches[0]
    outputs = dp(inputs)
    loss = F.mse_loss(outputs, targets)
    loss.backward()

    for name, param in dp.module.named_parameters():
        averaged = _gather_mean(param.grad)
        torch.testing.assert_close(
            param.grad,
            averaged,
            msg=f"Hook sync mismatch for {name}",
            atol=1e-6,
            rtol=1e-5,
        )

    _assert_param_grads_match_mean(dp.module, ref_grads)
    _cleanup_distributed()


def _scenario_bucket(rank: int, world_size: int) -> None:
    _init_distributed(rank, world_size)
    torch.manual_seed(5678)
    model = ToyModel()
    _sync_model_parameters(model)

    batches = [_build_batch(rank, batch_idx=1)]
    ref_grads = _compute_reference_grads(_clone_state_dict(model), batches)

    dp = DataParallelBucket(model, bucket_cap_mb=1, grad_type=torch.float32)
    inputs, targets = batches[0]
    loss = F.mse_loss(dp(inputs), targets)
    loss.backward()

    _assert_param_grads_match_mean(dp.module, ref_grads)
    for param in dp.module.parameters():
        assert hasattr(param, "main_grad")
        torch.testing.assert_close(param.main_grad, param.grad.to(param.main_grad.dtype))

    dp.reset()
    for param in dp.module.parameters():
        assert torch.count_nonzero(param.main_grad) == 0
    for bucket in dp.bucket_manager.buckets:
        assert bucket.handle is None
        assert not bucket.params_with_grad_ready

    _cleanup_distributed()


def _scenario_no_sync(rank: int, world_size: int) -> None:
    _init_distributed(rank, world_size)
    torch.manual_seed(2222)
    model = ToyModel()
    _sync_model_parameters(model)

    batches = [_build_batch(rank, 0), _build_batch(rank, 2)]
    dp = DataParallelBucket(model, bucket_cap_mb=1, grad_type=torch.float32)
    state_dict = _clone_state_dict(model)

    queue_path = "picotron.data_parallel.data_parallel.Variable._execution_engine.queue_callback"
    with mock.patch(queue_path, wraps=dp_module.Variable._execution_engine.queue_callback) as queue_cb:
        with dp.no_sync():
            loss = F.mse_loss(dp(batches[0][0]), batches[0][1])
            loss.backward()
        assert queue_cb.call_count == 0
        queue_cb.reset_mock()

        loss = F.mse_loss(dp(batches[1][0]), batches[1][1])
        loss.backward()
        assert queue_cb.call_count == 1

    ref_grads = _compute_reference_grads(state_dict, batches)
    _assert_param_grads_match_mean(dp.module, ref_grads)
    _cleanup_distributed()


def _scenario_integration(rank: int, world_size: int) -> None:
    _init_distributed(rank, world_size)
    torch.manual_seed(4242)
    model = ToyModel()
    _sync_model_parameters(model)

    dp = DataParallelBucket(model, bucket_cap_mb=1, grad_type=torch.float32)
    optimizer = torch.optim.SGD(dp.module.parameters(), lr=0.05)

    micro_batches = [_build_batch(rank, idx) for idx in range(3)]
    optimizer.zero_grad()
    for inputs, targets in micro_batches:
        loss = F.mse_loss(dp(inputs), targets) / len(micro_batches)
        loss.backward()
    optimizer.step()

    for param in dp.module.parameters():
        broadcast_clone = param.data.clone()
        dist.broadcast(broadcast_clone, src=0)
        torch.testing.assert_close(param.data, broadcast_clone)

    _cleanup_distributed()


SCENARIOS = {
    "naive": _scenario_naive,
    "bucket": _scenario_bucket,
    "no_sync": _scenario_no_sync,
    "integration": _scenario_integration,
}


def _distributed_test_worker(rank: int, world_size: int, scenario: str) -> None:
    try:
        SCENARIOS[scenario](rank, world_size)
    except Exception as exc:
        print(f"Rank {rank} failed in scenario '{scenario}': {exc}")
        raise


class TestDataParallel(unittest.TestCase):
    def _run_or_skip(self, scenario: str, world_size: int = 2) -> None:
        if world_size > MAX_LOCAL_PROCS:
            self.skipTest(
                f"World size {world_size} exceeds DP_TEST_MAX_PROCS={MAX_LOCAL_PROCS}"
            )
        mp.spawn(
            _distributed_test_worker,
            args=(world_size, scenario),
            nprocs=world_size,
            join=True,
        )

    def test_data_parallel_naive_grad_sync(self):
        self._run_or_skip("naive", world_size=2)

    def test_bucket_grad_and_reset(self):
        self._run_or_skip("bucket", world_size=2)

    def test_bucket_no_sync_accumulation(self):
        self._run_or_skip("no_sync", world_size=2)

    def test_bucket_end_to_end_update(self):
        self._run_or_skip("integration", world_size=2)


if __name__ == "__main__":
    if "RANK" in os.environ:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--scenario",
            choices=SCENARIOS.keys(),
            required=False,
            default="bucket",
            help="Scenario to run when launched with torchrun.",
        )
        args = parser.parse_args()
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        SCENARIOS[args.scenario](rank, world_size)
    else:
        unittest.main()

