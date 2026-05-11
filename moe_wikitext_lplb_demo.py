"""Tiny MoE language-model demo with LPLB.

Run with 4 GPUs, for example:

    torchrun --nproc_per_node=4 moe_wikitext_lplb_demo.py --steps 200

The script uses Hugging Face `datasets` and `transformers` for Wikitext loading
and tokenization. If the dataset or tokenizer cannot be downloaded, it falls
back to a small built-in corpus so the demo still runs offline.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path
from collections import Counter
from typing import cast

from datasets import load_dataset
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lplb import Planner


NUM_LOGICAL_EXPERTS = 64
TOP_K = 8
EP_SIZE = 4
DEFAULT_SEQ_LEN = 64
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_MOE_HIDDEN = 1024
FALLBACK_VOCAB_SIZE = 8_000

R2O_SQUARE_4P2E = torch.tensor(
    [
        [2, 0, 1, 3],
        [3, 1, 0, 2],
    ],
    dtype=torch.int32,
).T

FALLBACK_CORPUS = {
    'train': '''
Deep learning systems often trade generality for speed.
Mixture-of-experts models add conditional computation so that only a subset of experts
handles each token. This makes it possible to scale model capacity without scaling
per-token compute linearly.

Load balancing matters because a router that sends many tokens to the same expert can
create stragglers and waste hardware. A planner that redistributes expert assignments
based on recent workload history can reduce the imbalance.

Small training demos are useful because they let you validate routing logic, tensor
shapes, and distributed execution before committing to a larger run.

''',
    'valid': '''
Language models learn from token sequences and predict the next token at each step.
A small corpus is enough to verify whether the training loop, optimizer, and routing
mechanisms work together.
''',
    'test': '''
This fallback text exists so the demo still runs when network access is unavailable.
''',
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a tiny MoE LM with LPLB.')
    parser.add_argument('--cache-dir', type=Path, default=Path.home() / '.cache' / 'lplb_moe_demo')
    parser.add_argument('--seq-len', type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--refresh-interval', type=int, default=10)
    parser.add_argument('--max-train-tokens', type=int, default=120_000)
    parser.add_argument('--tokenizer-name', type=str, default='gpt2')
    parser.add_argument('--redundants-per-rank', type=int, default=2)
    parser.add_argument('--disable-load-balancing', action='store_true')
    parser.add_argument('--hidden-size', type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument('--moe-hidden-size', type=int, default=DEFAULT_MOE_HIDDEN)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--aux-loss-weight', type=float, default=0.1)
    parser.add_argument('--profile-interval', type=int, default=10)
    parser.add_argument('--profile-warmup', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--world-size', type=int, default=EP_SIZE)
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        return True, rank, local_rank

    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return False, 0, 0

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return True, rank, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def log(rank: int, message: str) -> None:
    if is_main_process(rank):
        print(message, flush=True)


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _fmt_time_s_to_ms_or_us(seconds: float) -> str:
    """Format a time given in seconds to a string in ms or µs.

    If the value is smaller than 0.01 ms, render in microseconds; otherwise render in milliseconds.
    """
    ms = seconds * 1e3
    if ms < 0.01:
        us = ms * 1e3
        if us < 0.01:
            return f"{us * 1e3:.2f}ns"
        return f"{us:.2f}µs"
    return f"{ms:.2f}ms"


def reduce_mean_scalar(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return float(tensor.item())


def build_redundancy_topology(group_size: int, num_redundants_per_rank: int) -> torch.Tensor:
    if num_redundants_per_rank < 0:
        raise ValueError('redundants-per-rank must be non-negative')
    if num_redundants_per_rank == 0:
        return torch.empty(group_size, 0, dtype=torch.int32)

    ranks = torch.arange(group_size, dtype=torch.int32)
    topology = torch.empty(group_size, num_redundants_per_rank, dtype=torch.int32)
    for column in range(num_redundants_per_rank):
        topology[:, column] = (ranks + column + 1) % group_size
    return topology


TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(text: str, max_vocab_size: int) -> dict[str, int]:
    counter: Counter[str] = Counter(tokenize(text))
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for token, _count in counter.most_common(max_vocab_size - len(special_tokens)):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    unk = vocab['<unk>']
    encoded = [vocab['<bos>']]
    encoded.extend(vocab.get(token, unk) for token in tokenize(text))
    encoded.append(vocab['<eos>'])
    return encoded


class SimpleTokenizer:
    def __init__(self, text: str, max_vocab_size: int) -> None:
        self.vocab = build_vocab(text, max_vocab_size)
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'

    def __len__(self) -> int:
        return len(self.vocab)

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        encoded = encode_text(text, self.vocab) if add_special_tokens else [
            self.vocab.get(token, self.vocab['<unk>']) for token in tokenize(text)
        ]
        return {'input_ids': encoded}


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, token_ids: list[int], seq_len: int, max_sequences: int | None = None) -> None:
        if len(token_ids) <= seq_len + 1:
            raise ValueError('the token stream is too short for the requested sequence length')

        samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        limit = len(token_ids) - seq_len - 1
        for start in range(0, limit, seq_len):
            x = torch.tensor(token_ids[start : start + seq_len], dtype=torch.long)
            y = torch.tensor(token_ids[start + 1 : start + seq_len + 1], dtype=torch.long)
            samples.append((x, y))
            if max_sequences is not None and len(samples) >= max_sequences:
                break

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, moe_hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, moe_hidden_size)
        self.fc2 = nn.Linear(moe_hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class LPLBMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_hidden_size: int,
        num_logical_experts: int,
        num_physical_experts: int,
        top_k: int,
        planner: Planner | None,
        disable_load_balancing: bool,
        static_log2phy: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_logical_experts = num_logical_experts
        self.num_physical_experts = num_physical_experts
        self.top_k = top_k
        self.planner = planner
        self.disable_load_balancing = disable_load_balancing

        self.router = nn.Linear(hidden_size, num_logical_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_size, moe_hidden_size) for _ in range(num_physical_experts)]
        )
        self.register_buffer(
            'workload_history',
            torch.zeros(num_logical_experts, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            'static_log2phy',
            static_log2phy if static_log2phy is not None else torch.empty(0, 2, dtype=torch.int32),
            persistent=False,
        )
        self._router_profile_sums = {
            'router_kernel_ms': 0.0,
            'planner_run_ms': 0.0,
            'router_topk_ms': 0.0,
            'router_assignment_ms': 0.0,
            'router_dispatch_prep_ms': 0.0,
        }

    def refresh_mapping(self, rank: int) -> None:
        if self.disable_load_balancing or self.planner is None:
            return
        workload_history = cast(torch.Tensor, self.workload_history)
        global_history = workload_history.clone()
        if dist.is_initialized():
            dist.all_reduce(global_history, op=dist.ReduceOp.SUM)
        phy2log, _log2phy, _logcnt = self.planner.update_redundancy_mapping(
            global_history.to(dtype=torch.int32)
        )
        if is_main_process(rank):
            max_history = int(global_history.max().item())
            print(
                f'[rank {rank}] refreshed LPLB mapping; global workload max = {max_history}',
                flush=True,
            )
        self.planner.phy2log = phy2log

    def router_aux_loss(self, topk_indices: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        flat_indices = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        importance = torch.zeros(
            self.num_logical_experts,
            device=topk_indices.device,
            dtype=flat_weights.dtype,
        )
        load = torch.zeros_like(importance)
        importance.scatter_add_(0, flat_indices, flat_weights)
        load.scatter_add_(0, flat_indices, torch.ones_like(flat_weights))
        denom = importance.sum().clamp_min(1.0) * load.sum().clamp_min(1.0)
        return self.num_logical_experts * (importance * load).sum() / denom

    def pop_router_profile(self) -> dict[str, float]:
        result = dict(self._router_profile_sums)
        result['router_events'] = 0.0
        for key in self._router_profile_sums:
            self._router_profile_sums[key] = 0.0
        return result

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_size = x.shape
        flat = x.reshape(batch_size * sequence_length, hidden_size)
        use_router_timing = flat.is_cuda

        if use_router_timing:
            maybe_cuda_synchronize(flat.device)
            router_start = time.perf_counter()

        router_logits = self.router(flat)

        if use_router_timing:
            maybe_cuda_synchronize(flat.device)
            self._router_profile_sums['router_kernel_ms'] += (time.perf_counter() - router_start) * 1e3

        if use_router_timing:
            maybe_cuda_synchronize(flat.device)
            topk_start = time.perf_counter()
        topk_logits, topk_logical = router_logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits.float(), dim=-1).to(dtype=flat.dtype)
        if use_router_timing:
            maybe_cuda_synchronize(flat.device)
            self._router_profile_sums['router_topk_ms'] += (time.perf_counter() - topk_start) * 1e3

        if use_router_timing:
            maybe_cuda_synchronize(flat.device)
            assign_start = time.perf_counter()

        if self.disable_load_balancing or self.planner is None:
            static_log2phy = cast(torch.Tensor, self.static_log2phy)
            if static_log2phy.numel() > 0:
                topk_physical = static_log2phy[topk_logical, 0]
            else:
                topk_physical = topk_logical
        else:
            counts = torch.bincount(topk_logical.reshape(-1), minlength=self.num_logical_experts)
            workload_history = cast(torch.Tensor, self.workload_history)
            workload_history.add_(counts)

            avail_counter = torch.zeros((), dtype=torch.int32, device=flat.device)
            if use_router_timing:
                maybe_cuda_synchronize(flat.device)
                planner_start = time.perf_counter()
            topk_physical = self.planner.run(topk_logical, avail_counter)
            if use_router_timing:
                maybe_cuda_synchronize(flat.device)
                self._router_profile_sums['planner_run_ms'] += (time.perf_counter() - planner_start) * 1e3

        if use_router_timing:
            maybe_cuda_synchronize(flat.device)
            self._router_profile_sums['router_assignment_ms'] += (time.perf_counter() - assign_start) * 1e3

        if use_router_timing:
            maybe_cuda_synchronize(flat.device)
            dispatch_start = time.perf_counter()

        token_indices = torch.arange(flat.size(0), device=flat.device).repeat_interleave(self.top_k)
        flat_inputs = flat.index_select(0, token_indices)
        flat_physical = topk_physical.reshape(-1)
        flat_weights = topk_weights.reshape(-1)

        if use_router_timing:
            maybe_cuda_synchronize(flat.device)
            self._router_profile_sums['router_dispatch_prep_ms'] += (time.perf_counter() - dispatch_start) * 1e3

        output = torch.zeros_like(flat)
        for expert_id in range(self.num_physical_experts):
            expert_mask = flat_physical == expert_id
            if expert_mask.any():
                expert_out = self.experts[expert_id](flat_inputs[expert_mask])
                weighted = expert_out * flat_weights[expert_mask].unsqueeze(-1)
                output.index_add_(0, token_indices[expert_mask], weighted)

        return output.view(batch_size, sequence_length, hidden_size), self.router_aux_loss(
            topk_logical, topk_weights
        )


class TinyMoeBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        moe_hidden_size: int,
        num_logical_experts: int,
        num_physical_experts: int,
        top_k: int,
        planner: Planner | None,
        disable_load_balancing: bool,
        static_log2phy: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.moe = LPLBMoE(
            hidden_size,
            moe_hidden_size,
            num_logical_experts,
            num_physical_experts,
            top_k,
            planner,
            disable_load_balancing,
            static_log2phy,
        )

    def refresh_mapping(self, rank: int) -> None:
        self.moe.refresh_mapping(rank)

    def pop_router_profile(self) -> dict[str, float]:
        return self.moe.pop_router_profile()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.norm1(x)
        attn_out, _attention_weights = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_out
        moe_out, aux_loss = self.moe(self.norm2(x))
        return x + moe_out, aux_loss


class TinyMoeLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        moe_hidden_size: int,
        num_logical_experts: int,
        num_physical_experts: int,
        top_k: int,
        planner: Planner | None,
        disable_load_balancing: bool,
        static_log2phy: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(seq_len, hidden_size)
        self.blocks = nn.ModuleList(
            [
                TinyMoeBlock(
                    hidden_size,
                    num_heads,
                    moe_hidden_size,
                    num_logical_experts,
                    num_physical_experts,
                    top_k,
                    planner,
                    disable_load_balancing,
                    static_log2phy,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def refresh_mapping(self, rank: int) -> None:
        for block in self.blocks:
            cast(TinyMoeBlock, block).refresh_mapping(rank)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        aux_loss = x.new_zeros(())
        for block in self.blocks:
            x, block_aux_loss = block(x)
            aux_loss = aux_loss + block_aux_loss
        x = self.final_norm(x)
        return self.lm_head(x), aux_loss

    def pop_router_profile(self) -> dict[str, float]:
        totals = {
            'router_topk_ms': 0.0,
            'router_assignment_ms': 0.0,
            'router_dispatch_prep_ms': 0.0,
            'router_events': 0.0,
        }
        for block in self.blocks:
            block_stats = cast(TinyMoeBlock, block).pop_router_profile()
            for key in totals:
                totals[key] += block_stats[key]
        return totals


def unwrap_model(model: nn.Module) -> TinyMoeLM:
    return model.module if isinstance(model, DDP) else model  # type: ignore[return-value]


def synchronize_shared_grads(model: nn.Module) -> None:
    if not dist.is_initialized():
        return

    base_model = unwrap_model(model)
    for _name, parameter in base_model.named_parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(dist.get_world_size())


def build_corpus(cache_dir: Path) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=str(cache_dir))
        return {
            'train': '\n'.join(dataset['train']['text']),
            'valid': '\n'.join(dataset['validation']['text']),
            'test': '\n'.join(dataset['test']['text']),
        }
    except Exception:
        return FALLBACK_CORPUS.copy()


def make_datasets(
    texts: dict[str, str],
    tokenizer_name: str,
    seq_len: int,
    max_train_tokens: int,
) -> tuple[PreTrainedTokenizerBase | SimpleTokenizer, SequenceDataset]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        tokenizer = SimpleTokenizer(texts['train'], FALLBACK_VOCAB_SIZE)

    def encode_corpus(text: str, limit: int) -> list[int]:
        token_ids: list[int] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            token_ids.extend(tokenizer(line, add_special_tokens=True)['input_ids'])
            if len(token_ids) >= limit:
                break
        return token_ids[:limit]

    train_tokens = encode_corpus(texts['train'], max_train_tokens)
    train_dataset = SequenceDataset(train_tokens, seq_len)
    return tokenizer, train_dataset


def make_dataloader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DistributedSampler | None]:
    sampler: DistributedSampler | None
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
    else:
        sampler = None

    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None and shuffle,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        ),
        sampler,
    )



def main() -> None:
    args = parse_args()
    distributed, rank, local_rank = setup_distributed()
    world_size = dist.get_world_size() if distributed else 1
    if world_size != args.world_size:
        raise RuntimeError(
            f'this demo expects {args.world_size} processes; received {world_size}'
        )

    if not torch.cuda.is_available():
        raise RuntimeError('this demo requires CUDA because LPLB runs on CUDA tensors')

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    disable_load_balancing = bool(args.disable_load_balancing or args.redundants_per_rank == 0)
    num_logical_experts = NUM_LOGICAL_EXPERTS
    num_physical_experts = num_logical_experts + args.redundants_per_rank * world_size
    local_experts_per_gpu = num_physical_experts // world_size

    texts = build_corpus(args.cache_dir)
    tokenizer, train_dataset = make_datasets(
        texts,
        args.tokenizer_name,
        args.seq_len,
        args.max_train_tokens,
    )

    train_loader, train_sampler = make_dataloader(
        train_dataset,
        args.batch_size,
        rank,
        world_size,
        shuffle=True,
    )

    r2o = build_redundancy_topology(world_size, args.redundants_per_rank).to(device)
    planner_group = cast(dist.ProcessGroup, dist.new_group()) if distributed else None
    planner: Planner | None = None
    static_log2phy: torch.Tensor | None = None
    if args.redundants_per_rank > 0:
        planner = Planner(
            r2o,
            num_physical_experts,
            num_logical_experts,
            ep_size=world_size,
            group=planner_group,
        )
        if disable_load_balancing:
            _static_phy2log, static_log2phy, _static_logcnt = planner.update_redundancy_mapping()

    model = TinyMoeLM(
        vocab_size=len(tokenizer),
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        moe_hidden_size=args.moe_hidden_size,
        num_logical_experts=num_logical_experts,
        num_physical_experts=num_physical_experts,
        top_k=TOP_K,
        planner=planner,
        disable_load_balancing=disable_load_balancing,
        static_log2phy=static_log2phy,
    ).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    base_model = unwrap_model(model)
    base_model.refresh_mapping(rank)

    log(
        rank,
        f'[rank {rank}] tokenizer={args.tokenizer_name} vocab={len(tokenizer)} '
        f'train_samples={len(train_dataset)}',
    )
    log(
        rank,
        f'[rank {rank}] world_size={world_size} logical_experts={num_logical_experts} '
        f'physical_experts={num_physical_experts} local_experts={local_experts_per_gpu} '
        f'redundants_per_rank={args.redundants_per_rank} load_balancing={not disable_load_balancing} '
        f'top_k={TOP_K}',
    )

    profile_enabled = args.profile_interval > 0
    profile_window = max(1, args.profile_interval)
    profile_sums = {
        'total': 0.0,
        'data': 0.0,
        'refresh': 0.0,
        'forward': 0.0,
        'backward': 0.0,
        'step': 0.0,
        'router_kernel': 0.0,
        'planner_run': 0.0,
        'router_topk': 0.0,
        'router_assignment': 0.0,
        'router_dispatch_prep': 0.0,
    }
    profile_count = 0

    data_iter = iter(train_loader)
    for step in range(args.steps):
        step_should_profile = profile_enabled and step >= args.profile_warmup
        step_total_start = time.perf_counter() if step_should_profile else None
        data_start = time.perf_counter() if step_should_profile else None
        if train_sampler is not None:
            train_sampler.set_epoch(step)

        if step > 0 and step % args.refresh_interval == 0:
            maybe_cuda_synchronize(device)
            refresh_start = time.perf_counter() if step_should_profile else None
            base_model.refresh_mapping(rank)
            if step_should_profile and refresh_start is not None:
                maybe_cuda_synchronize(device)
                profile_sums['refresh'] += time.perf_counter() - refresh_start

        try:
            input_ids, target_ids = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            input_ids, target_ids = next(data_iter)

        if step_should_profile and data_start is not None:
            maybe_cuda_synchronize(device)
            profile_sums['data'] += time.perf_counter() - data_start

        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        maybe_cuda_synchronize(device)
        forward_start = time.perf_counter() if step_should_profile else None
        logits, aux_loss = model(input_ids)
        if step_should_profile and forward_start is not None:
            maybe_cuda_synchronize(device)
            profile_sums['forward'] += time.perf_counter() - forward_start
            router_stats = base_model.pop_router_profile()
            profile_sums['router_kernel'] += router_stats.get('router_kernel_ms', 0.0) / 1e3
            profile_sums['planner_run'] += router_stats.get('planner_run_ms', 0.0) / 1e3
            profile_sums['router_topk'] += router_stats['router_topk_ms'] / 1e3
            profile_sums['router_assignment'] += router_stats['router_assignment_ms'] / 1e3
            profile_sums['router_dispatch_prep'] += router_stats['router_dispatch_prep_ms'] / 1e3

        backward_start = time.perf_counter() if step_should_profile else None
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        loss = lm_loss + args.aux_loss_weight * aux_loss
        loss.backward()
        if step_should_profile and backward_start is not None:
            maybe_cuda_synchronize(device)
            profile_sums['backward'] += time.perf_counter() - backward_start

        step_start = time.perf_counter() if step_should_profile else None
        optimizer.step()
        if step_should_profile and step_start is not None:
            maybe_cuda_synchronize(device)
            profile_sums['step'] += time.perf_counter() - step_start

        if step_should_profile and step_total_start is not None:
            maybe_cuda_synchronize(device)
            profile_sums['total'] += time.perf_counter() - step_total_start
            profile_count += 1

        if is_main_process(rank) and step % 10 == 0:
            ppl = math.exp(min(float(lm_loss.item()), 20.0))
            print(
                # step, loss, LM loss, aux loss, perplexity
                f'step={step:04d} loss={float(loss.item()):.4f} '
                f'lm={float(lm_loss.item()):.4f} aux={float(aux_loss.item()):.4f} ppl={ppl:.2f}',
                flush=True,
            )

    # Print aggregated profiling summary
    if profile_enabled and profile_count > 0:
        profile_values = torch.tensor(
            [
                profile_sums['total'],
                profile_sums['data'],
                profile_sums['refresh'],
                profile_sums['forward'],
                profile_sums['backward'],
                profile_sums['step'],
                profile_sums['router_kernel'],
                profile_sums['planner_run'],
                profile_sums['router_topk'],
                profile_sums['router_assignment'],
                profile_sums['router_dispatch_prep'],
            ],
            device=device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(profile_values, op=dist.ReduceOp.SUM)
            profile_values.div_(dist.get_world_size())

        if is_main_process(rank):
            avg = profile_values / profile_count
            total_s = float(avg[0].item())
            tokens_per_sec = (args.batch_size * world_size * args.seq_len) / max(total_s, 1e-12)

            def _fmt_idx(i: int) -> str:
                return _fmt_time_s_to_ms_or_us(float(avg[i].item()))

            router_total_s = sum(float(avg[i].item()) for i in range(6, 11))

            print(
                '\n[profile aggregate]\n'
                f'  steps profiled: {profile_count}\n'
                f'  total time: {_fmt_time_s_to_ms_or_us(float(avg[0].item()))}/step\n'
                f'  data loading: {_fmt_idx(1)}\n'
                f'  refresh mapping: {_fmt_idx(2)}\n'
                f'  forward pass: {_fmt_idx(3)}\n'
                f'  backward pass: {_fmt_idx(4)}\n'
                f'  optimizer step: {_fmt_idx(5)}\n'
                f'  router kernel (self.router): {_fmt_idx(6)}\n'
                f'  planner.run: {_fmt_idx(7)}\n'
                f'  router topk+weights: {_fmt_idx(8)}\n'
                f'  router assignment: {_fmt_idx(9)}\n'
                f'  router dispatch prep: {_fmt_idx(10)}\n'
                f'  router total: {_fmt_time_s_to_ms_or_us(router_total_s)}\n'
                f'  throughput: {tokens_per_sec:.2f} tokens/sec',
                flush=True,
            )

    cleanup_distributed()


if __name__ == '__main__':
    main()