import argparse
import os
import sys
import unittest

"""
Instructions to run the tests:

1. Run with standard unittest (uses torch.multiprocessing):
   python tests/test_process_group_manager.py

2. Run with torchrun (simulates distributed environment):
   torchrun --nproc_per_node=4 tests/test_process_group_manager.py \
       --tp_size 2 --dp_size 2

   Note: Ensure that tp_size * cp_size * pp_size * dp_size == nproc_per_node
"""

# Ensure picotron is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from picotron.process_group_manager import (  # noqa: E402
    setup_process_group_manager
)
import picotron.process_group_manager as pgm  # noqa: E402


def check_pgm_state(tp_size, cp_size, pp_size, dp_size):
    """
    Verifies the state of the ProcessGroupManager singleton.
    """
    pgm_instance = pgm.process_group_manager
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}: Verifying PGM state...")

    # 1. Basic checks
    assert pgm_instance.global_rank == rank, \
        f"Rank mismatch: {pgm_instance.global_rank} != {rank}"
    assert pgm_instance.world_size == world_size, \
        f"World size mismatch: {pgm_instance.world_size} != {world_size}"
    expected_world_size = tp_size * cp_size * pp_size * dp_size
    assert world_size == expected_world_size, \
        f"World size config mismatch: {world_size} != {expected_world_size}"

    # 2. Grid checks
    # Verify ranks are within bounds
    assert 0 <= pgm_instance.tp_rank < tp_size, \
        f"TP rank {pgm_instance.tp_rank} out of bounds [0, {tp_size})"
    assert 0 <= pgm_instance.cp_rank < cp_size, \
        f"CP rank {pgm_instance.cp_rank} out of bounds [0, {cp_size})"
    assert 0 <= pgm_instance.pp_rank < pp_size, \
        f"PP rank {pgm_instance.pp_rank} out of bounds [0, {pp_size})"
    assert 0 <= pgm_instance.dp_rank < dp_size, \
        f"DP rank {pgm_instance.dp_rank} out of bounds [0, {dp_size})"

    # 3. Group size checks
    assert pgm_instance.tp_world_size == tp_size
    assert pgm_instance.cp_world_size == cp_size
    assert pgm_instance.pp_world_size == pp_size
    assert pgm_instance.dp_world_size == dp_size

    # 4. Pipeline Parallel specific checks
    if pp_size > 1:
        if pgm_instance.pp_rank == 0:
            assert pgm_instance.pp_is_first_stage
            assert not pgm_instance.pp_is_last_stage
            assert pgm_instance.pp_prev_rank is None
            assert pgm_instance.pp_next_rank is not None
        elif pgm_instance.pp_rank == pp_size - 1:
            assert not pgm_instance.pp_is_first_stage
            assert pgm_instance.pp_is_last_stage
            assert pgm_instance.pp_prev_rank is not None
            assert pgm_instance.pp_next_rank is None
        else:
            assert not pgm_instance.pp_is_first_stage
            assert not pgm_instance.pp_is_last_stage
            assert pgm_instance.pp_prev_rank is not None
            assert pgm_instance.pp_next_rank is not None
    else:
        assert pgm_instance.pp_is_first_stage
        assert pgm_instance.pp_is_last_stage
        assert pgm_instance.pp_prev_rank is None
        assert pgm_instance.pp_next_rank is None

    # 5. Context Parallel specific checks
    if cp_size > 1:
        # Verify ring topology logic
        expected_send = pgm_instance.cp_group_ids[
            (pgm_instance.cp_rank + 1) % cp_size
        ]
        expected_recv = pgm_instance.cp_group_ids[
            (pgm_instance.cp_rank - 1) % cp_size
        ]
        assert pgm_instance.cp_send_rank == expected_send, \
            (f"CP send rank mismatch: {pgm_instance.cp_send_rank} "
             f"!= {expected_send}")
        assert pgm_instance.cp_recv_rank == expected_recv, \
            (f"CP recv rank mismatch: {pgm_instance.cp_recv_rank} "
             f"!= {expected_recv}")

    # 6. Group existence checks
    assert pgm_instance.tp_group is not None
    assert pgm_instance.cp_group is not None
    assert pgm_instance.pp_group is not None
    assert pgm_instance.dp_group is not None
    assert pgm_instance.cp_dp_group is not None
    assert pgm_instance.world_group is not None

    print(f"Rank {rank}: PGM state verified successfully.")


def run_test_worker(rank, world_size, tp_size, cp_size, pp_size, dp_size):
    """
    Worker function for multiprocessing tests.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # Initialize process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        # Setup PGM
        setup_process_group_manager(tp_size, cp_size, pp_size, dp_size)

        # Verify
        check_pgm_state(tp_size, cp_size, pp_size, dp_size)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        raise e
    finally:
        dist.destroy_process_group()


class TestProcessGroupManager(unittest.TestCase):
    def test_tp_dp(self):
        print("\n=== Running test_tp_dp (TP=2, DP=2) ===")
        # TP=2, DP=2 -> World=4
        tp, cp, pp, dp = 2, 1, 1, 2
        world_size = tp * cp * pp * dp
        mp.spawn(
            run_test_worker,
            args=(world_size, tp, cp, pp, dp),
            nprocs=world_size,
            join=True
        )

    def test_pp_dp(self):
        print("\n=== Running test_pp_dp (PP=2, DP=2) ===")
        # PP=2, DP=2 -> World=4
        tp, cp, pp, dp = 1, 1, 2, 2
        world_size = tp * cp * pp * dp
        mp.spawn(
            run_test_worker,
            args=(world_size, tp, cp, pp, dp),
            nprocs=world_size,
            join=True
        )

    def test_complex_grid(self):
        print("\n=== Running test_complex_grid (TP=2, PP=2) ===")
        # TP=2, PP=2 -> World=4
        tp, cp, pp, dp = 2, 1, 2, 1
        world_size = tp * cp * pp * dp
        mp.spawn(
            run_test_worker,
            args=(world_size, tp, cp, pp, dp),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    # Check if running with torchrun (RANK environment variable set)
    if "RANK" in os.environ:
        parser = argparse.ArgumentParser()
        parser.add_argument("--tp_size", type=int, default=1)
        parser.add_argument("--cp_size", type=int, default=1)
        parser.add_argument("--pp_size", type=int, default=1)
        parser.add_argument("--dp_size", type=int, default=1)
        args = parser.parse_args()

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Replicate train.py initialization flow
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )

        setup_process_group_manager(
            args.tp_size, args.cp_size, args.pp_size, args.dp_size
        )

        check_pgm_state(
            args.tp_size, args.cp_size, args.pp_size, args.dp_size
        )

        dist.destroy_process_group()
        print(f"Rank {rank}: Torchrun test passed.")
    else:
        unittest.main()

