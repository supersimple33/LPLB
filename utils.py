from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Callable, Iterator

import torch


@contextmanager
def empty_suppress() -> Iterator[None]:
    yield


@contextmanager
def suppress_stdout_stderr() -> Iterator[None]:
    with open(os.devnull, 'w') as outnull_file, open(os.devnull, 'w') as errnull_file:
        old_stdout_fileno_undup = sys.stdout.fileno()
        old_stderr_fileno_undup = sys.stderr.fileno()

        old_stdout_fileno = os.dup(sys.stdout.fileno())
        old_stderr_fileno = os.dup(sys.stderr.fileno())

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        os.dup2(outnull_file.fileno(), old_stdout_fileno_undup)
        os.dup2(errnull_file.fileno(), old_stderr_fileno_undup)

        sys.stdout = outnull_file
        sys.stderr = errnull_file

        yield

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        os.dup2(old_stdout_fileno, old_stdout_fileno_undup)
        os.dup2(old_stderr_fileno, old_stderr_fileno_undup)

        os.close(old_stdout_fileno)
        os.close(old_stderr_fileno)


def bench_kineto(
    fn: Callable[[], None],
    kernel_names: str | list[str],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: str | None = None,
) -> float | list[float]:
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
        ) as prof:
            for _i in range(2):
                for _ in range(num_tests):
                    fn()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, (str, list))
    is_listd = isinstance(kernel_names, list)
    prof_lines = (
        prof.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all(isinstance(name, str) for name in kernel_names)
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) == 1, (
            f'Errors of the kernel {name} in the profiling table'
        )

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {'ms': 1e3, 'us': 1e6}
    kernel_times = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_times.append(float(time_str.replace(unit, '')) / scale)
                        break
                break
    return list(kernel_times) if is_listd else kernel_times[0]

SQUARE_4P2E = torch.tensor(
    [
        [2, 0, 1, 3],
        [3, 1, 0, 2],
    ]
).T

CUBE_8P2E = torch.tensor(
    [
        [3, 0, 1, 2, 7, 4, 5, 6],
        [6, 7, 4, 5, 0, 1, 2, 3],
    ]
).T
RING_8P = torch.tensor(
    [
        [1, 2, 3, 4, 5, 6, 7, 0],
    ]
).T
HYPERCUBE_16P2E = torch.tensor(
    [
        [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14],
        [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
).T


def torus_2d(m: int, n: int) -> torch.Tensor:
    return torch.tensor(
        [[(i + 1) % m * n + j, i * n + (j + 1) % n] for i in range(m) for j in range(n)]
    )
