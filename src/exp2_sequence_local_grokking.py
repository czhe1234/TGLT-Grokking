#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 2: sequence-form modular arithmetic with local mode transport.

The circular candidate strip used by the earlier Exp2 model exposes all output
residues as spatial candidates.  This version uses the standard classification
form instead: two operands occupy separated positions in an open sequence and
only the final query position is read out.  Cross-position communication is
restricted to repeated radius-r exponential mode sweeps.  There is no global
attention, pooling, convolution, or MLP/FFN.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import os
from tqdm import tqdm
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TASKS = {"add", "sub", "affine", "quadratic", "mul", "mixed", "dihedral"}
RESIDUE_ENCODINGS = {
    "learned",
    "fixed_fourier",
    "fixed_one_hot",
    "fixed_random",
    "fixed_permuted_fourier",
}
DIAGNOSTIC_FIELD = {
    "lambdas": (0.85, 0.60),
    "alphas": (0.60, 0.40),
    "freqs": (1, 2),
    "etas": (0.05, -0.04),
    "tau": 5.0,
}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def parse_float_tuple(value: str | Sequence[float]) -> Tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    return tuple(float(item) for item in value)


def save_json(obj: object, path: Path) -> None:
    def convert(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, torch.Tensor):
            return convert(value.detach().cpu().tolist())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(convert(obj), handle, indent=2, ensure_ascii=False)
    tmp.replace(path)


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    tmp.replace(path)


def atomic_torch_save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_all_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def modular_label(a: torch.Tensor, b: torch.Tensor, p: int, task: str) -> torch.Tensor:
    if task == "add":
        return (a + b) % p
    if task == "sub":
        return (a - b) % p
    if task == "affine":
        return (2 * a + 3 * b + 1) % p
    if task == "quadratic":
        return (a * a + b) % p
    if task == "mul":
        return (a * b) % p
    if task == "mixed":
        return (a * b + a + b) % p
    if task == "dihedral":
        if int(p) % 2 != 0:
            raise ValueError("dihedral task requires an even class count p=2n")
        n = int(p) // 2
        flip_a, rotation_a = torch.div(a, n, rounding_mode="floor"), a % n
        flip_b, rotation_b = torch.div(b, n, rounding_mode="floor"), b % n
        output_flip = (flip_a + flip_b) % 2
        signed_rotation_b = torch.where(flip_a == 0, rotation_b, -rotation_b)
        output_rotation = (rotation_a + signed_rotation_b) % n
        return output_flip * n + output_rotation
    raise ValueError(f"unknown task: {task!r}")


def resolve_positions(seq_len: int, pos_a: int, pos_b: int, query_pos: int) -> Tuple[int, int, int]:
    positions = [int(pos_a), int(pos_b), int(query_pos)]
    positions = [position if position >= 0 else int(seq_len) + position for position in positions]
    if any(position < 0 or position >= int(seq_len) for position in positions):
        raise ValueError(f"positions {positions} are outside sequence length {seq_len}")
    if len(set(positions)) != 3:
        raise ValueError("operand and query positions must be distinct")
    return positions[0], positions[1], positions[2]


def build_sequence_data(
    p: int,
    task: str,
    seq_len: int,
    pos_a: int,
    pos_b: int,
    query_pos: int,
    random_labels: bool = False,
    random_label_seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Return token ids, role ids, labels, and resolved sequence positions.

    Token zero is the empty token. Residue c is represented by token c+1, so
    the query token is always zero and cannot contain the target label.
    Roles are 0=gap, 1=operand A, 2=operand B, 3=query.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    pa, pb, pq = resolve_positions(seq_len, pos_a, pos_b, query_pos)
    a = torch.arange(int(p)).repeat_interleave(int(p))
    b = torch.arange(int(p)).repeat(int(p))
    tokens = torch.zeros(int(p) * int(p), int(seq_len), dtype=torch.long)
    roles = torch.zeros_like(tokens)
    tokens[:, pa] = a + 1
    tokens[:, pb] = b + 1
    roles[:, pa] = 1
    roles[:, pb] = 2
    roles[:, pq] = 3
    labels = modular_label(a, b, int(p), task).long()
    if random_labels:
        generator = torch.Generator().manual_seed(int(random_label_seed))
        labels = torch.randint(int(p), labels.shape, generator=generator)
    return tokens, roles, labels, {"pos_a": pa, "pos_b": pb, "query_pos": pq}


def make_split(n: int, train_fraction: float, split_seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(split_seed))
    permutation = torch.randperm(int(n), generator=generator)
    n_train = max(1, min(int(n) - 1, int(round(float(train_fraction) * int(n)))))
    return permutation[:n_train], permutation[n_train:]


def diagnostic_teacher_field(p: int, labels: torch.Tensor) -> torch.Tensor:
    """Smooth class-space field used only for post-training diagnostics."""
    classes = torch.arange(int(p), device=labels.device, dtype=torch.float32).view(1, -1)
    centers = labels.float().view(-1, 1)
    delta = torch.remainder(classes - centers, float(p))
    distance = torch.minimum(delta, float(p) - delta)
    field = torch.zeros_like(distance)
    for alpha, lam in zip(DIAGNOSTIC_FIELD["alphas"], DIAGNOSTIC_FIELD["lambdas"]):
        field = field + float(alpha) * float(lam) ** distance
    for frequency, amplitude in zip(DIAGNOSTIC_FIELD["freqs"], DIAGNOSTIC_FIELD["etas"]):
        phase = 2.0 * math.pi * float(frequency) * (classes - centers) / float(p)
        field = field + float(amplitude) * torch.cos(phase)
    return float(DIAGNOSTIC_FIELD["tau"]) * field


def xi_spread_lambdas(M: int, xi_min: float, xi_max: float) -> torch.Tensor:
    if M <= 0:
        raise ValueError("M must be positive")
    if xi_min <= 0 or xi_max < xi_min:
        raise ValueError("require 0 < xi_min <= xi_max")
    if M == 1:
        xis = torch.tensor([float(xi_max)])
    else:
        xis = torch.exp(torch.linspace(math.log(float(xi_min)), math.log(float(xi_max)), int(M)))
    return torch.exp(-1.0 / xis)


def fixed_phase_features(count: int, d: int) -> torch.Tensor:
    """Return a fixed real Fourier code with shape [count, d]."""
    if int(d) < 2 or int(d) % 2 != 0:
        raise ValueError("fixed Fourier features require an even d >= 2")
    values = torch.arange(int(count), dtype=torch.float32).view(-1, 1)
    frequencies = torch.arange(1, int(d) // 2 + 1, dtype=torch.float32).view(1, -1)
    phase = 2.0 * math.pi * values * frequencies / float(count)
    output = torch.empty(int(count), int(d), dtype=torch.float32)
    output[:, 0::2] = torch.cos(phase)
    output[:, 1::2] = torch.sin(phase)
    return output / math.sqrt(float(d) / 2.0)


def fixed_role_features(d: int) -> torch.Tensor:
    """Return deterministic role codes for gap, operand A/B, and query."""
    output = torch.zeros(4, int(d), dtype=torch.float32)
    for role in range(1, 4):
        output[role, (role - 1) % int(d)] = 1.0
        output[role, (role + 2) % int(d)] = -0.5
    return output


def fixed_circular_features(count: int, d: int) -> torch.Tensor:
    """Return the first real Fourier character repeated across d channels."""
    if int(d) < 2 or int(d) % 2 != 0:
        raise ValueError("fixed circular features require an even d >= 2")
    values = torch.arange(int(count), dtype=torch.float32).view(-1, 1)
    phase = 2.0 * math.pi * values / float(count)
    output = torch.empty(int(count), int(d), dtype=torch.float32)
    output[:, 0::2] = torch.cos(phase)
    output[:, 1::2] = torch.sin(phase)
    return output / math.sqrt(float(d) / 2.0)


def fixed_residue_encoding_table(
    p: int,
    d: int,
    encoding: str,
    seed: int,
    target_norm: float = 0.0,
) -> torch.Tensor:
    """Build a fixed residue table while reserving token zero for padding."""
    p = int(p)
    d = int(d)
    encoding = str(encoding)
    if encoding not in RESIDUE_ENCODINGS - {"learned"}:
        raise ValueError(f"unsupported fixed residue encoding: {encoding}")
    norm = math.sqrt(float(d)) if float(target_norm) <= 0 else float(target_norm)
    table = torch.zeros(p + 1, d, dtype=torch.float32)
    if encoding == "fixed_fourier":
        code = fixed_circular_features(p, d) * norm
    elif encoding == "fixed_one_hot":
        if d < p:
            raise ValueError(f"fixed_one_hot requires d >= p, got d={d}, p={p}")
        code = torch.zeros(p, d, dtype=torch.float32)
        code[torch.arange(p), torch.arange(p)] = norm
    elif encoding == "fixed_random":
        generator = torch.Generator().manual_seed(int(seed))
        code = torch.randn(p, d, generator=generator)
        code = F.normalize(code, dim=1) * norm
    else:
        generator = torch.Generator().manual_seed(int(seed))
        permutation = torch.randperm(p, generator=generator)
        identity = torch.arange(p)
        if torch.equal(permutation, identity):
            permutation = torch.roll(permutation, shifts=1)
        code = fixed_circular_features(p, d)[permutation] * norm
    table[1:] = code
    return table


def shift_open(x: torch.Tensor, displacement: int) -> torch.Tensor:
    """Shift along the sequence with zero fill and no wrap-around.

    Positive displacement moves information to the right: out[i] = x[i-s].
    Negative displacement moves information to the left: out[i] = x[i+s].
    """
    length = int(x.shape[1])
    amount = abs(int(displacement))
    if amount == 0:
        return x
    if amount >= length:
        return torch.zeros_like(x)
    zeros = torch.zeros_like(x[:, :amount])
    if displacement > 0:
        return torch.cat([zeros, x[:, :-amount]], dim=1)
    return torch.cat([x[:, amount:], zeros], dim=1)


class OpenLocalExponentialSweep(nn.Module):
    """Repeated radius-r exponential sweeps on an open sequence."""

    def __init__(
        self,
        M: int,
        r: int,
        T: int,
        lambdas: Sequence[float],
        learnable_lambda: bool,
    ) -> None:
        super().__init__()
        self.M = int(M)
        self.r = int(r)
        self.T = int(T)
        self.H = self.r * self.T
        values = torch.tensor(tuple(float(value) for value in lambdas), dtype=torch.float32)
        if values.numel() != self.M:
            raise ValueError(f"expected {self.M} lambdas, got {values.numel()}")
        values = values.clamp(0.02, 0.995)
        raw = torch.log(values / (0.999 - values))
        self.raw_lambda = nn.Parameter(raw, requires_grad=bool(learnable_lambda))
        self.learnable_lambda = bool(learnable_lambda)

    @property
    def lambdas(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_lambda) * 0.999

    def _drive(self, source: torch.Tensor, direction: int) -> torch.Tensor:
        lam = self.lambdas.to(source.dtype).view(1, 1, self.M, 1)
        drive = torch.zeros_like(source)
        for distance in range(1, self.r + 1):
            drive = drive + lam.pow(distance) * shift_open(source, direction * distance)
        return drive

    def forward(self, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if source.ndim != 4:
            raise ValueError(f"source must be [batch, position, mode, channel], got {source.shape}")
        lam_r = self.lambdas.to(source.dtype).pow(self.r).view(1, 1, self.M, 1)
        drive_right = self._drive(source, direction=+1)
        drive_left = self._drive(source, direction=-1)
        right = torch.zeros_like(source)
        left = torch.zeros_like(source)
        for _ in range(self.T):
            right = drive_right + lam_r * shift_open(right, +self.r)
            left = drive_left + lam_r * shift_open(left, -self.r)
        return right, left

    def closed_form(self, source: torch.Tensor, direction: int) -> torch.Tensor:
        lam = self.lambdas.to(source.dtype).view(1, 1, self.M, 1)
        result = torch.zeros_like(source)
        power = torch.ones_like(lam)
        for distance in range(1, self.H + 1):
            power = power * lam
            result = result + power * shift_open(source, direction * distance)
        return result


class SequenceModeBlock(nn.Module):
    """Pointwise visible update driven only by locally propagated modes."""

    def __init__(self, d: int, d_a: int, M: int, sweep: OpenLocalExponentialSweep, local_coupling: float = 1.0) -> None:
        super().__init__()
        self.d = int(d)
        self.d_a = int(d_a)
        self.M = int(M)
        self.sweep = sweep
        self.local_coupling = float(local_coupling)
        self.drive = nn.Linear(self.d, self.M * self.d_a)
        self.visible_factor = nn.Parameter(torch.empty(self.M, self.d, self.d))
        self.mode_factor = nn.Parameter(torch.empty(self.M, 2 * self.d_a, self.d))
        self.mode_logits = nn.Parameter(torch.zeros(self.M))
        self.out = nn.Linear(self.d, self.d)
        self.norm = nn.LayerNorm(self.d)
        for mode in range(self.M):
            nn.init.xavier_uniform_(self.visible_factor[mode])
            nn.init.xavier_uniform_(self.mode_factor[mode])

    def forward(
        self,
        hidden: torch.Tensor,
        mode_mask: Optional[torch.Tensor] = None,
        renormalize_mode_mask: bool = False,
        stats: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        source = self.drive(hidden).view(batch, length, self.M, self.d_a)
        right, left = self.sweep(source)
        if mode_mask is not None:
            mask = mode_mask.to(hidden).view(1, 1, self.M, 1)
            right = right * mask
            left = left * mask
        mode_state = torch.cat([right, left], dim=-1)
        visible_term = torch.einsum("bpd,mdo->bpmo", hidden, self.visible_factor)
        mode_term = torch.einsum("bpma,mao->bpmo", mode_state, self.mode_factor)
        gates = torch.softmax(self.mode_logits, dim=0)
        effective_gates = gates
        if mode_mask is not None:
            effective_gates = gates * mode_mask.to(hidden).view(self.M)
            if renormalize_mode_mask:
                effective_gates = effective_gates / effective_gates.sum().clamp_min(1e-12)
        contribution = self.local_coupling * visible_term * mode_term * effective_gates.view(1, 1, self.M, 1)
        visible_contribution = F.linear(contribution, self.out.weight, bias=None)
        if stats is not None:
            coupling = (hidden.unsqueeze(2) * visible_contribution).sum(dim=-1)
            denominator = (
                hidden.norm(dim=-1).unsqueeze(2)
                * visible_contribution.norm(dim=-1)
            ).clamp_min(1e-12)
            stats.append({
                "gates": effective_gates.detach(),
                "mode_rms": mode_state.detach().square().mean(dim=(0, 1, 3)).sqrt(),
                "injection_rms": visible_contribution.detach().square().mean(dim=(0, 1, 3)).sqrt(),
                "C_raw": coupling.detach().mean(dim=(0, 1)),
                "C_abs": coupling.detach().abs().mean(dim=(0, 1)),
                "C_cos": (coupling / denominator).detach().mean(dim=(0, 1)),
                "C_raw_position": coupling.detach().mean(dim=0),
                "C_abs_position": coupling.detach().abs().mean(dim=0),
                "C_cos_position": (coupling / denominator).detach().mean(dim=0),
                "mode_state": mode_state.detach(),
            })
        update = contribution.sum(dim=2)
        return self.norm(hidden + self.out(update))


class LowRankSequenceModeBlock(nn.Module):
    """Low-rank parameterization of the same local bilinear mode update."""

    def __init__(
        self,
        d: int,
        d_a: int,
        M: int,
        sweep: OpenLocalExponentialSweep,
        rank: int,
        local_coupling: float = 1.0,
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.d_a = int(d_a)
        self.M = int(M)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("factor rank must be positive")
        self.sweep = sweep
        self.local_coupling = float(local_coupling)
        self.drive = nn.Linear(self.d, self.M * self.d_a)
        self.visible_left = nn.Parameter(torch.empty(self.M, self.d, self.rank))
        self.visible_right = nn.Parameter(torch.empty(self.M, self.rank, self.d))
        self.mode_left = nn.Parameter(torch.empty(self.M, 2 * self.d_a, self.rank))
        self.mode_right = nn.Parameter(torch.empty(self.M, self.rank, self.d))
        self.mode_logits = nn.Parameter(torch.zeros(self.M))
        self.out_in = nn.Parameter(torch.empty(self.rank, self.d))
        self.out_out = nn.Parameter(torch.empty(self.d, self.rank))
        self.out_bias = nn.Parameter(torch.zeros(self.d))
        self.norm = nn.LayerNorm(self.d)
        for parameter in (
            self.visible_left,
            self.visible_right,
            self.mode_left,
            self.mode_right,
            self.out_in,
            self.out_out,
        ):
            nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        hidden: torch.Tensor,
        mode_mask: Optional[torch.Tensor] = None,
        renormalize_mode_mask: bool = False,
        stats: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        source = self.drive(hidden).view(batch, length, self.M, self.d_a)
        right, left = self.sweep(source)
        if mode_mask is not None:
            mask = mode_mask.to(hidden).view(1, 1, self.M, 1)
            right = right * mask
            left = left * mask
        mode_state = torch.cat([right, left], dim=-1)
        visible_term = torch.einsum(
            "bpd,mdk,mko->bpmo",
            hidden,
            self.visible_left,
            self.visible_right,
        )
        mode_term = torch.einsum(
            "bpma,mak,mko->bpmo",
            mode_state,
            self.mode_left,
            self.mode_right,
        )
        gates = torch.softmax(self.mode_logits, dim=0)
        effective_gates = gates
        if mode_mask is not None:
            effective_gates = gates * mode_mask.to(hidden).view(self.M)
            if renormalize_mode_mask:
                effective_gates = effective_gates / effective_gates.sum().clamp_min(1e-12)
        contribution = self.local_coupling * visible_term * mode_term * effective_gates.view(1, 1, self.M, 1)
        compressed = F.linear(contribution, self.out_in)
        visible_contribution = F.linear(compressed, self.out_out, bias=None)
        if stats is not None:
            coupling = (hidden.unsqueeze(2) * visible_contribution).sum(dim=-1)
            denominator = (
                hidden.norm(dim=-1).unsqueeze(2)
                * visible_contribution.norm(dim=-1)
            ).clamp_min(1e-12)
            stats.append({
                "gates": effective_gates.detach(),
                "mode_rms": mode_state.detach().square().mean(dim=(0, 1, 3)).sqrt(),
                "injection_rms": visible_contribution.detach().square().mean(dim=(0, 1, 3)).sqrt(),
                "C_raw": coupling.detach().mean(dim=(0, 1)),
                "C_abs": coupling.detach().abs().mean(dim=(0, 1)),
                "C_cos": (coupling / denominator).detach().mean(dim=(0, 1)),
                "C_raw_position": coupling.detach().mean(dim=0),
                "C_abs_position": coupling.detach().abs().mean(dim=0),
                "C_cos_position": (coupling / denominator).detach().mean(dim=0),
                "mode_state": mode_state.detach(),
            })
        update = visible_contribution.sum(dim=2) + self.out_bias
        return self.norm(hidden + update)


class ScalarSequenceModeBlock(nn.Module):
    """Tiny local block: each propagated mode produces one scalar field."""

    def __init__(
        self,
        d: int,
        d_a: int,
        M: int,
        sweep: OpenLocalExponentialSweep,
        rank: int,
        local_coupling: float = 1.0,
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.d_a = int(d_a)
        self.M = int(M)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("factor rank must be positive")
        self.sweep = sweep
        self.local_coupling = float(local_coupling)
        self.drive = nn.Linear(self.d, self.M * self.d_a)
        self.mode_readout = nn.Parameter(torch.empty(self.M, 2 * self.d_a))
        self.visible_scale = nn.Parameter(torch.empty(self.M, self.d))
        self.mode_logits = nn.Parameter(torch.zeros(self.M))
        self.out_in = nn.Parameter(torch.empty(self.rank, self.d))
        self.out_out = nn.Parameter(torch.empty(self.d, self.rank))
        self.out_bias = nn.Parameter(torch.zeros(self.d))
        self.norm = nn.LayerNorm(self.d)
        nn.init.xavier_uniform_(self.mode_readout)
        nn.init.normal_(self.visible_scale, mean=1.0, std=0.05)
        nn.init.xavier_uniform_(self.out_in)
        nn.init.xavier_uniform_(self.out_out)

    def forward(
        self,
        hidden: torch.Tensor,
        mode_mask: Optional[torch.Tensor] = None,
        renormalize_mode_mask: bool = False,
        stats: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        source = self.drive(hidden).view(batch, length, self.M, self.d_a)
        right, left = self.sweep(source)
        if mode_mask is not None:
            mask = mode_mask.to(hidden).view(1, 1, self.M, 1)
            right = right * mask
            left = left * mask
        mode_state = torch.cat([right, left], dim=-1)
        scalar_field = torch.einsum("bpma,ma->bpm", mode_state, self.mode_readout)
        gates = torch.softmax(self.mode_logits, dim=0)
        effective_gates = gates
        if mode_mask is not None:
            effective_gates = gates * mode_mask.to(hidden).view(self.M)
            if renormalize_mode_mask:
                effective_gates = effective_gates / effective_gates.sum().clamp_min(1e-12)
        contribution = self.local_coupling * (
            hidden.unsqueeze(2)
            * self.visible_scale.view(1, 1, self.M, self.d)
            * scalar_field.unsqueeze(-1)
            * effective_gates.view(1, 1, self.M, 1)
        )
        compressed = F.linear(contribution, self.out_in)
        visible_contribution = F.linear(compressed, self.out_out, bias=None)
        if stats is not None:
            coupling = (hidden.unsqueeze(2) * visible_contribution).sum(dim=-1)
            denominator = (
                hidden.norm(dim=-1).unsqueeze(2)
                * visible_contribution.norm(dim=-1)
            ).clamp_min(1e-12)
            stats.append({
                "gates": effective_gates.detach(),
                "mode_rms": mode_state.detach().square().mean(dim=(0, 1, 3)).sqrt(),
                "injection_rms": visible_contribution.detach().square().mean(dim=(0, 1, 3)).sqrt(),
                "C_raw": coupling.detach().mean(dim=(0, 1)),
                "C_abs": coupling.detach().abs().mean(dim=(0, 1)),
                "C_cos": (coupling / denominator).detach().mean(dim=(0, 1)),
                "C_raw_position": coupling.detach().mean(dim=0),
                "C_abs_position": coupling.detach().abs().mean(dim=0),
                "C_cos_position": (coupling / denominator).detach().mean(dim=0),
                "mode_state": mode_state.detach(),
            })
        update = visible_contribution.sum(dim=2) + self.out_bias
        return self.norm(hidden + update)


class SequenceLocalModeModel(nn.Module):
    """Open-sequence, query-only classifier with no non-mode communication."""

    def __init__(
        self,
        p: int,
        seq_len: int,
        query_pos: int,
        d: int,
        d_a: int,
        M: int,
        r: int,
        T: int,
        layers: int,
        lambdas: Sequence[float],
        learnable_lambda: bool,
        local_coupling: float = 1.0,
        block_type: str = "full",
        factor_rank: int = 4,
        representation: str = "learned",
        learned_position: bool = True,
        residue_encoding: str = "learned",
        residue_encoding_seed: int = 0,
        fixed_residue_norm: float = 0.0,
    ) -> None:
        super().__init__()
        self.p = int(p)
        self.seq_len = int(seq_len)
        self.query_pos = int(query_pos)
        self.d = int(d)
        self.d_a = int(d_a)
        self.M = int(M)
        self.r = int(r)
        self.T = int(T)
        self.layers = int(layers)
        self.local_coupling = float(local_coupling)
        if self.local_coupling < 0:
            raise ValueError("local_coupling must be non-negative")
        self.block_type = str(block_type)
        self.factor_rank = int(factor_rank)
        self.representation = str(representation)
        self.learned_position = bool(learned_position)
        self.residue_encoding = str(residue_encoding)
        self.residue_encoding_seed = int(residue_encoding_seed)
        self.fixed_residue_norm = float(fixed_residue_norm)
        if self.block_type not in {"full", "low_rank", "scalar"}:
            raise ValueError("block_type must be full, low_rank, or scalar")
        if self.representation not in {"learned", "fixed_fourier"}:
            raise ValueError("representation must be learned or fixed_fourier")
        self.sweep = OpenLocalExponentialSweep(M, r, T, lambdas, learnable_lambda)
        if self.representation == "learned":
            if self.residue_encoding not in RESIDUE_ENCODINGS:
                raise ValueError(
                    f"residue_encoding must be one of {sorted(RESIDUE_ENCODINGS)}"
                )
            # Always construct the same trainable table first so all shared
            # parameters receive identical same-seed initialization.
            self.residue = nn.Embedding(self.p + 1, self.d, padding_idx=0)
            if self.residue_encoding != "learned":
                residue_table = fixed_residue_encoding_table(
                    self.p,
                    self.d,
                    self.residue_encoding,
                    self.residue_encoding_seed,
                    self.fixed_residue_norm,
                )
                with torch.no_grad():
                    self.residue.weight.copy_(residue_table)
                self.residue.weight.requires_grad_(False)
            self.role = nn.Embedding(4, self.d)
        else:
            if self.residue_encoding != "learned":
                raise ValueError(
                    "residue_encoding is only independent when representation=learned"
                )
            residue_table = torch.zeros(self.p + 1, self.d, dtype=torch.float32)
            residue_table[1:] = fixed_phase_features(self.p, self.d)
            self.register_buffer("residue_table", residue_table)
            self.register_buffer("role_table", fixed_role_features(self.d))
            self.register_buffer("class_table", fixed_phase_features(self.p, self.d))
            self.decoder_scale = nn.Parameter(torch.ones(self.d))
            self.logit_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(float(self.d)))))
        if self.learned_position:
            self.position = nn.Parameter(torch.empty(self.seq_len, self.d))
            nn.init.normal_(self.position, std=1.0 / math.sqrt(float(self.d)))
        else:
            self.register_buffer(
                "position",
                0.25 * fixed_phase_features(self.seq_len, self.d),
            )
        if self.block_type == "full":
            block_factory = lambda: SequenceModeBlock(self.d, self.d_a, self.M, self.sweep, self.local_coupling)
        elif self.block_type == "low_rank":
            block_factory = lambda: LowRankSequenceModeBlock(
                self.d,
                self.d_a,
                self.M,
                self.sweep,
                self.factor_rank,
                self.local_coupling,
            )
        else:
            block_factory = lambda: ScalarSequenceModeBlock(
                self.d,
                self.d_a,
                self.M,
                self.sweep,
                self.factor_rank,
                self.local_coupling,
            )
        self.blocks = nn.ModuleList([block_factory() for _ in range(self.layers)])
        self.final_norm = nn.LayerNorm(self.d)
        if self.representation == "learned":
            self.classifier = nn.Linear(self.d, self.p)

    def residue_features(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.representation == "learned":
            return self.residue(tokens)
        return F.embedding(tokens, self.residue_table)

    def role_features(self, roles: torch.Tensor) -> torch.Tensor:
        if self.representation == "learned":
            return self.role(roles)
        return F.embedding(roles, self.role_table)

    def residue_code_weight(self) -> torch.Tensor:
        if self.representation == "learned":
            return self.residue.weight[1:]
        return self.residue_table[1:]

    def forward(
        self,
        tokens: torch.Tensor,
        roles: torch.Tensor,
        mode_mask: Optional[torch.Tensor] = None,
        renormalize_mode_mask: bool = False,
        stats: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        hidden = self.residue_features(tokens) + self.role_features(roles) + self.position.unsqueeze(0)
        for block in self.blocks:
            hidden = block(
                hidden,
                mode_mask=mode_mask,
                renormalize_mode_mask=renormalize_mode_mask,
                stats=stats,
            )
        query = self.final_norm(hidden[:, self.query_pos])
        if self.representation == "learned":
            return self.classifier(query)
        scaled_query = query * self.decoder_scale
        return torch.exp(self.logit_scale) * F.linear(scaled_query, self.class_table)

    def architecture_summary(self) -> str:
        lambdas = ", ".join(f"{float(value):.4f}" for value in self.sweep.lambdas.detach().cpu())
        return "\n".join([
            "Exp2 sequence-form local-mode model",
            "===================================",
            "cross-position path : open-boundary radius-r exponential sweeps only",
            "global modules      : none",
            "MLP/FFN             : none",
            "output              : query position only",
            f"sequence            : length={self.seq_len}, query={self.query_pos}",
            f"mode geometry       : M={self.M}, r={self.r}, T={self.T}, H/layer={self.r * self.T}",
            f"local coupling      : {self.local_coupling:.6g}",
            f"visible dimensions  : d={self.d}, d_a={self.d_a}, layers={self.layers}",
            f"parameterization    : block={self.block_type}, rank={self.factor_rank}",
            f"representation      : {self.representation}, residue={self.residue_encoding}, "
            f"learned_position={self.learned_position}",
            f"lambdas             : [{lambdas}], learnable={self.sweep.learnable_lambda}",
        ])


FORBIDDEN_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.MultiheadAttention,
    nn.TransformerEncoder,
    nn.TransformerEncoderLayer,
    nn.ReLU,
    nn.GELU,
    nn.Sequential,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveMaxPool1d,
)


def assert_no_forbidden_modules(model: nn.Module) -> None:
    for module in model.modules():
        if module is not model and isinstance(module, FORBIDDEN_MODULES):
            raise TypeError(f"forbidden module: {type(module).__name__}")


@torch.no_grad()
def evaluate(
    model: SequenceLocalModeModel,
    tokens: torch.Tensor,
    roles: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 4096,
    mode_mask: Optional[torch.Tensor] = None,
    renormalize_mode_mask: bool = False,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    loss_sum = 0.0
    margins: List[torch.Tensor] = []
    for start in range(0, labels.numel(), int(chunk_size)):
        stop = start + int(chunk_size)
        logits = model(
            tokens[start:stop],
            roles[start:stop],
            mode_mask=mode_mask,
            renormalize_mode_mask=renormalize_mode_mask,
        )
        target = labels[start:stop]
        correct += int((logits.argmax(dim=-1) == target).sum())
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
        margins.append(true_class_margin(logits, target).float().cpu())
    count = max(1, int(labels.numel()))
    all_margins = torch.cat(margins)
    return {
        "acc": correct / count,
        "ce": loss_sum / count,
        "true_margin_mean": float(all_margins.mean()),
        "true_margin_positive_fraction": float((all_margins > 0).float().mean()),
        "true_margin_q10": float(torch.quantile(all_margins, 0.10)),
        "true_margin_q50": float(torch.quantile(all_margins, 0.50)),
    }


@torch.no_grad()
def collect_logits(
    model: SequenceLocalModeModel,
    tokens: torch.Tensor,
    roles: torch.Tensor,
    chunk_size: int = 4096,
    mode_mask: Optional[torch.Tensor] = None,
    renormalize_mode_mask: bool = False,
) -> torch.Tensor:
    model.eval()
    pieces = []
    for start in range(0, tokens.shape[0], int(chunk_size)):
        stop = start + int(chunk_size)
        pieces.append(model(
            tokens[start:stop],
            roles[start:stop],
            mode_mask=mode_mask,
            renormalize_mode_mask=renormalize_mode_mask,
        ))
    return torch.cat(pieces, dim=0)


@torch.no_grad()
def empirical_mode_correlation_length(mode_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit the paper's open-strip Gamma_m(r) over r=1..floor(n/4)."""
    length = int(mode_state.shape[1])
    max_lag = max(2, length // 4)
    gamma = []
    for lag in range(max_lag + 1):
        left = mode_state[:, :length - lag]
        right = mode_state[:, lag:]
        gamma.append((left * right).sum(dim=-1).mean(dim=(0, 1)))
    gamma_tensor = torch.stack(gamma, dim=1)
    normalized = gamma_tensor / gamma_tensor[:, :1].abs().clamp_min(1e-12)
    distances = torch.arange(1, max_lag + 1, device=mode_state.device, dtype=mode_state.dtype)
    centered = distances - distances.mean()
    log_correlation = normalized[:, 1:].abs().clamp_min(1e-12).log()
    slopes = (log_correlation * centered.view(1, -1)).sum(dim=1) / centered.square().sum()
    xi = torch.full_like(slopes, float("nan"))
    valid = slopes < -1e-8
    xi[valid] = -1.0 / slopes[valid]
    return xi, normalized


def centered_flat_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float().reshape(-1)
    second = second.float().reshape(-1)
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().mean().sqrt() * second.square().mean().sqrt()
    if float(denominator) < 1e-12:
        return float("nan")
    return float((first * second).mean() / denominator)


def field_metrics(logits: torch.Tensor, field: torch.Tensor) -> Dict[str, float]:
    centered_logits = logits.float() - logits.float().mean(dim=-1, keepdim=True)
    centered_field = field.float() - field.float().mean(dim=-1, keepdim=True)
    return {
        "field_mse": float(F.mse_loss(logits.float(), field.float())),
        "field_centered_mse": float(F.mse_loss(centered_logits, centered_field)),
        "field_corr": centered_flat_correlation(logits, field),
    }


def true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_value = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    false_logits = logits.clone()
    false_logits.scatter_(1, labels.view(-1, 1), float("-inf"))
    return true_value - torch.logsumexp(false_logits, dim=1)


@torch.no_grad()
def observe(
    model: SequenceLocalModeModel,
    probe_tokens: torch.Tensor,
    probe_roles: torch.Tensor,
    test_tokens: torch.Tensor,
    test_roles: torch.Tensor,
    test_labels: torch.Tensor,
    base_test_acc: float,
    base_mode_mask: Optional[torch.Tensor] = None,
    renormalize_mode_mask: bool = False,
) -> Dict[str, float]:
    model.eval()
    stats: List[Dict[str, torch.Tensor]] = []
    model(
        probe_tokens,
        probe_roles,
        mode_mask=base_mode_mask,
        renormalize_mode_mask=renormalize_mode_mask,
        stats=stats,
    )
    output: Dict[str, float] = {}
    full_logits = collect_logits(
        model,
        test_tokens,
        test_roles,
        mode_mask=base_mode_mask,
        renormalize_mode_mask=renormalize_mode_mask,
    )
    teacher_field = diagnostic_teacher_field(model.p, test_labels)
    teacher_fft = torch.fft.rfft(teacher_field.float(), dim=-1)
    target_frequencies = [
        int(frequency)
        for frequency in DIAGNOSTIC_FIELD["freqs"]
        if int(frequency) < teacher_fft.shape[1]
    ]
    control_frequencies = [
        frequency
        for frequency in range(1, teacher_fft.shape[1])
        if frequency not in target_frequencies
    ][:len(target_frequencies)]
    full_margin = true_class_margin(full_logits, test_labels)
    lambdas = model.sweep.lambdas.detach().cpu()
    for mode in range(model.M):
        output[f"lambda/value_m{mode}"] = float(lambdas[mode])
        output[f"lambda/xi_m{mode}"] = float(-1.0 / math.log(float(lambdas[mode])))
        mask = (
            base_mode_mask.clone()
            if base_mode_mask is not None
            else torch.ones(model.M, device=test_tokens.device)
        )
        mask[mode] = 0.0
        ablated_logits = collect_logits(
            model,
            test_tokens,
            test_roles,
            mode_mask=mask,
            renormalize_mode_mask=renormalize_mode_mask,
        )
        ablated_accuracy = float((ablated_logits.argmax(dim=-1) == test_labels).float().mean())
        output[f"mode/test_drop_m{mode}"] = base_test_acc - ablated_accuracy
        effect = full_logits - ablated_logits
        effect = effect - effect.mean(dim=-1, keepdim=True)
        output[f"mode/effect_field_corr_m{mode}"] = centered_flat_correlation(effect, teacher_field)
        effect_fft = torch.fft.rfft(effect.float(), dim=-1)
        if target_frequencies:
            ef = effect_fft[:, target_frequencies]
            tf = teacher_fft[:, target_frequencies]
            numerator = (ef * tf.conj()).real.sum()
            denominator = ef.abs().square().sum().sqrt() * tf.abs().square().sum().sqrt()
            output[f"mode/Phi_phase_m{mode}"] = float(numerator / denominator.clamp_min(1e-12))
        non_dc_power = effect_fft[:, 1:].abs().square().sum().clamp_min(1e-12)
        target_mass = effect_fft[:, target_frequencies].abs().square().sum() / non_dc_power
        control_mass = effect_fft[:, control_frequencies].abs().square().sum() / non_dc_power
        output[f"mode/Phi_target_mass_m{mode}"] = float(target_mass)
        output[f"mode/Phi_alignment_m{mode}"] = float(target_mass - control_mass)
        ablated_margin = true_class_margin(ablated_logits, test_labels)
        output[f"mode/true_margin_contribution_m{mode}"] = float((full_margin - ablated_margin).mean())
        for layer, layer_stats in enumerate(stats):
            output[f"mode/gate_l{layer}_m{mode}"] = float(layer_stats["gates"][mode])
            output[f"mode/state_rms_l{layer}_m{mode}"] = float(layer_stats["mode_rms"][mode])
            output[f"mode/injection_rms_l{layer}_m{mode}"] = float(layer_stats["injection_rms"][mode])
            output[f"mode/C_raw_l{layer}_m{mode}"] = float(layer_stats["C_raw"][mode])
            output[f"mode/C_abs_l{layer}_m{mode}"] = float(layer_stats["C_abs"][mode])
            output[f"mode/C_cos_l{layer}_m{mode}"] = float(layer_stats["C_cos"][mode])
            query = model.query_pos
            output[f"mode/C_raw_query_l{layer}_m{mode}"] = float(
                layer_stats["C_raw_position"][query, mode]
            )
            output[f"mode/C_abs_query_l{layer}_m{mode}"] = float(
                layer_stats["C_abs_position"][query, mode]
            )
            output[f"mode/C_cos_query_l{layer}_m{mode}"] = float(
                layer_stats["C_cos_position"][query, mode]
            )
    for layer, layer_stats in enumerate(stats):
        xi_empirical, gamma = empirical_mode_correlation_length(layer_stats["mode_state"].float())
        for mode in range(model.M):
            output[f"mode/xi_empirical_l{layer}_m{mode}"] = float(xi_empirical[mode])
            for lag in range(1, gamma.shape[1]):
                output[f"mode/gamma_l{layer}_m{mode}_r{lag}"] = float(gamma[mode, lag])
    zero_mask = torch.zeros(model.M, device=test_tokens.device)
    output["mode/test_acc_all_off"] = evaluate(
        model, test_tokens, test_roles, test_labels, mode_mask=zero_mask
    )["acc"]

    # Fourier diagnostics are over residue identity, not sequence position.
    residue_code = model.residue_code_weight().detach().float()
    power = torch.fft.rfft(residue_code, dim=0).abs().square().sum(dim=1)
    share = power / power.sum().clamp_min(1e-12)
    output["residue_code/spectral_entropy"] = float(
        -(share * share.clamp_min(1e-12).log()).sum()
    )
    output["residue_code/low_frequency_mass_k1_to_k4"] = float(
        share[1:min(5, share.numel())].sum()
    )
    output["residue_code/first_frequency_mass"] = float(
        share[1] if share.numel() > 1 else 0.0
    )
    if share.numel() > 1:
        strongest_mass, strongest_index = share[1:].max(dim=0)
        output["residue_code/strongest_nonzero_frequency_mass"] = float(strongest_mass)
        output["residue_code/dominant_nonzero_frequency"] = int(strongest_index.item()) + 1
    else:
        output["residue_code/strongest_nonzero_frequency_mass"] = 0.0
        output["residue_code/dominant_nonzero_frequency"] = 0
    centered_code = residue_code - residue_code.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered_code)
    variances = singular_values.square()
    output["residue_code/effective_dimension"] = float(
        variances.sum().square() / variances.square().sum().clamp_min(1e-12)
    )
    return output


@dataclass
class RunConfig:
    out_dir: str = "exp2_no_mlp/current/results/01_classic_grokking/20260715_smoke_sequence_p59_M4"
    device: str = "cuda:0"
    seed: int = 0
    split_seed: int = 0
    loader_seed: int = 0
    p: int = 59
    task: str = "add"
    random_labels: bool = False
    random_label_seed: int = 0
    train_fraction: float = 0.2
    seq_len: int = 32
    pos_a: int = 0
    pos_b: int = 15
    query_pos: int = -1
    d: int = 64
    d_a: int = 8
    M: int = 4
    r: int = 2
    T: int = 8
    layers: int = 2
    local_coupling: float = 1.0
    block_type: str = "full"
    factor_rank: int = 4
    representation: str = "learned"
    learned_position: bool = True
    residue_encoding: str = "learned"
    residue_encoding_seed: int = 0
    fixed_residue_norm: float = 0.0
    lambda_values: Tuple[float, ...] = ()
    xi_min: float = 0.7
    xi_max: float = 32.0
    learnable_lambda: bool = False
    steps: int = 30000
    batch_size: int = 0
    lr: float = 1e-3
    wd: float = 0.3
    ce_weight: float = 1.0
    alpha_field: float = 0.0
    field_temperature: float = 1.0
    field_loss: str = "kl"
    warmup_steps: int = 100
    schedule: str = "constant"
    min_lr_fraction: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    obs_every: int = 1000
    checkpoint_every: int = 5000
    resume_checkpoint: str = ""
    intervention_step: int = 0
    trainable_scope_after: str = "all"
    lambda_override_after: Tuple[float, ...] = ()
    mode_mask_after: Tuple[float, ...] = ()
    renormalize_mode_mask_after: bool = False


def make_optimizer(model: SequenceLocalModeModel, cfg: RunConfig) -> torch.optim.AdamW:
    no_decay_ids = {id(model.sweep.raw_lambda)}
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            no_decay_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in no_decay_ids or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": float(cfg.wd)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=float(cfg.lr), betas=(0.9, 0.98), eps=1e-8)


def is_mode_parameter(name: str) -> bool:
    """The sweep and mode blocks form the complete cross-position subsystem."""
    return name.startswith("sweep.") or name.startswith("blocks.")


def is_mode_core_parameter(name: str) -> bool:
    """Parameters that create, transport, couple, or inject mode states."""
    return name.startswith("sweep.") or (
        name.startswith("blocks.") and ".norm." not in name
    )


def is_mode_readout_parameter(name: str) -> bool:
    return name.startswith("blocks.") and any(
        token in name
        for token in (
            ".visible_factor", ".mode_factor", ".visible_left", ".visible_right",
            ".mode_left", ".mode_right", ".mode_readout", ".visible_scale",
            ".mode_logits", ".out.", ".out_in", ".out_out", ".out_bias",
        )
    )


def is_shared_mode_out_parameter(name: str) -> bool:
    return name.startswith("blocks.") and any(
        token in name for token in (".out.", ".out_in", ".out_out", ".out_bias")
    )


def apply_trainable_scope(model: SequenceLocalModeModel, scope: str) -> None:
    if scope == "all":
        return
    valid = {
        "mode_only",
        "nonmode_only",
        "mode_core_only",
        "visible_only",
        "residue_only",
        "freeze_residue",
        "freeze_mode_readout",
        "freeze_shared_mode_out",
        "freeze_classifier",
    }
    if scope not in valid:
        raise ValueError(f"unknown trainable scope: {scope}")
    for name, parameter in model.named_parameters():
        if scope in {"mode_only", "nonmode_only"}:
            keep = is_mode_parameter(name)
            if scope == "nonmode_only":
                keep = not keep
        elif scope in {"mode_core_only", "visible_only"}:
            keep = is_mode_core_parameter(name)
            if scope == "visible_only":
                keep = not keep
        elif scope == "residue_only":
            keep = name.startswith("residue.")
        elif scope == "freeze_residue":
            keep = not name.startswith("residue.")
        elif scope == "freeze_mode_readout":
            keep = not is_mode_readout_parameter(name)
        elif scope == "freeze_shared_mode_out":
            keep = not is_shared_mode_out_parameter(name)
        else:
            keep = not name.startswith("classifier.")
        if not keep:
            parameter.grad = None


@torch.no_grad()
def override_lambdas(model: SequenceLocalModeModel, values: Sequence[float]) -> None:
    target = torch.tensor(tuple(float(value) for value in values), device=model.sweep.raw_lambda.device)
    if target.numel() != model.M:
        raise ValueError(f"lambda override requires {model.M} values, got {target.numel()}")
    target = target.clamp(0.02, 0.995)
    model.sweep.raw_lambda.copy_(torch.log(target / (0.999 - target)))


def learning_rate(step: int, cfg: RunConfig) -> float:
    if cfg.warmup_steps > 0 and step <= cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    if cfg.schedule == "constant":
        return cfg.lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    factor = cfg.min_lr_fraction + 0.5 * (1.0 - cfg.min_lr_fraction) * (
        1.0 + math.cos(math.pi * min(1.0, max(0.0, progress)))
    )
    return cfg.lr * factor


def classic_metrics(history: List[Dict[str, object]]) -> Dict[str, object]:
    train_threshold = 0.98
    low_test_threshold = 0.20
    final_test_threshold = 0.90
    required_plateau = 10000
    train_onset: Optional[int] = None
    generalization_onset: Optional[int] = None
    for row in history:
        if train_onset is None and float(row["train_acc"]) >= train_threshold:
            train_onset = int(row["step"])
        if (
            train_onset is not None
            and int(row["step"]) >= train_onset
            and float(row["test_acc"]) >= final_test_threshold
        ):
            generalization_onset = int(row["step"])
            break
    low_end: Optional[int] = None
    if train_onset is not None:
        for row in history:
            if int(row["step"]) >= train_onset and float(row["test_acc"]) < low_test_threshold:
                low_end = int(row["step"])
            elif low_end is not None and int(row["step"]) > low_end:
                break
    plateau = max(0, (low_end or train_onset or 0) - (train_onset or 0))
    success = (
        train_onset is not None
        and generalization_onset is not None
        and plateau >= required_plateau
        and generalization_onset > train_onset
    )
    return {
        "train98_step": train_onset,
        "test90_step": generalization_onset,
        "low_test_plateau_steps": plateau,
        "classic_grokking_success": success,
    }


def train_one_run(cfg: RunConfig) -> Dict[str, object]:
    if cfg.schedule not in {"constant", "cosine"}:
        raise ValueError("schedule must be constant or cosine")
    if cfg.block_type not in {"full", "low_rank", "scalar"}:
        raise ValueError("block_type must be full, low_rank, or scalar")
    if cfg.representation not in {"learned", "fixed_fourier"}:
        raise ValueError("representation must be learned or fixed_fourier")
    if cfg.residue_encoding not in RESIDUE_ENCODINGS:
        raise ValueError(f"residue_encoding must be one of {sorted(RESIDUE_ENCODINGS)}")
    if cfg.representation != "learned" and cfg.residue_encoding != "learned":
        raise ValueError("independent residue_encoding requires representation=learned")
    if cfg.factor_rank <= 0:
        raise ValueError("factor_rank must be positive")
    if cfg.representation == "fixed_fourier" and (cfg.d < 2 or cfg.d % 2 != 0):
        raise ValueError("fixed_fourier requires an even d >= 2")
    valid_scopes = {
        "all", "mode_only", "nonmode_only", "mode_core_only", "visible_only",
        "residue_only", "freeze_residue", "freeze_mode_readout",
        "freeze_shared_mode_out", "freeze_classifier",
    }
    if cfg.trainable_scope_after not in valid_scopes:
        raise ValueError(f"trainable_scope_after must be one of {sorted(valid_scopes)}")
    if cfg.lambda_override_after and cfg.intervention_step <= 0:
        raise ValueError("lambda_override_after requires a positive intervention_step")
    if cfg.mode_mask_after and cfg.intervention_step <= 0:
        raise ValueError("mode_mask_after requires a positive intervention_step")
    if cfg.mode_mask_after:
        if len(cfg.mode_mask_after) != cfg.M:
            raise ValueError("mode_mask_after must contain exactly M values")
        if any(float(value) not in {0.0, 1.0} for value in cfg.mode_mask_after):
            raise ValueError("mode_mask_after values must be binary")
    if cfg.field_loss not in {"kl", "mse"}:
        raise ValueError("field_loss must be kl or mse")
    if cfg.ce_weight <= 0 and cfg.alpha_field <= 0:
        raise ValueError("at least one of ce_weight and alpha_field must be positive")
    run_dir = Path(cfg.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if not cfg.device.startswith("cuda") or torch.cuda.is_available() else "cpu")

    tokens, roles, labels, positions = build_sequence_data(
        cfg.p,
        cfg.task,
        cfg.seq_len,
        cfg.pos_a,
        cfg.pos_b,
        cfg.query_pos,
        random_labels=cfg.random_labels,
        random_label_seed=cfg.random_label_seed,
    )
    train_idx, test_idx = make_split(labels.numel(), cfg.train_fraction, cfg.split_seed)
    if torch.isin(train_idx, test_idx).any():
        raise AssertionError("train and test splits overlap")
    tr_tokens = tokens[train_idx].to(device)
    tr_roles = roles[train_idx].to(device)
    tr_labels = labels[train_idx].to(device)
    te_tokens = tokens[test_idx].to(device)
    te_roles = roles[test_idx].to(device)
    te_labels = labels[test_idx].to(device)
    tr_fields = diagnostic_teacher_field(cfg.p, tr_labels)
    te_fields = diagnostic_teacher_field(cfg.p, te_labels)

    if cfg.lambda_values:
        lambdas = torch.tensor(cfg.lambda_values, dtype=torch.float32)
        if lambdas.numel() != cfg.M:
            raise ValueError("lambda_values must contain exactly M values")
    else:
        lambdas = xi_spread_lambdas(cfg.M, cfg.xi_min, cfg.xi_max)

    set_seed(cfg.seed)
    model = SequenceLocalModeModel(
        p=cfg.p,
        seq_len=cfg.seq_len,
        query_pos=positions["query_pos"],
        d=cfg.d,
        d_a=cfg.d_a,
        M=cfg.M,
        r=cfg.r,
        T=cfg.T,
        layers=cfg.layers,
        local_coupling=cfg.local_coupling,
        lambdas=lambdas,
        learnable_lambda=cfg.learnable_lambda,
        block_type=cfg.block_type,
        factor_rank=cfg.factor_rank,
        representation=cfg.representation,
        learned_position=cfg.learned_position,
        residue_encoding=cfg.residue_encoding,
        residue_encoding_seed=cfg.residue_encoding_seed,
        fixed_residue_norm=cfg.fixed_residue_norm,
    ).to(device)
    assert_no_forbidden_modules(model)
    optimizer = make_optimizer(model, cfg)
    loader_generator = torch.Generator().manual_seed(int(cfg.loader_seed))

    save_json(asdict(cfg), run_dir / "config.json")
    save_json({
        "train_idx": train_idx,
        "test_idx": test_idx,
        **positions,
        "query_tokens_unique": torch.unique(tokens[:, positions["query_pos"]]),
    }, run_dir / "split.json")
    with open(run_dir / "architecture.txt", "w", encoding="utf-8") as handle:
        handle.write(model.architecture_summary())
        handle.write(
            f"\n\n{model!r}\n\ntrainable_parameters={count_params(model)}\n"
            f"total_parameters={count_all_params(model)}\n"
        )
    print(model.architecture_summary())
    print(f"parameters={count_params(model):,} train={train_idx.numel()} test={test_idx.numel()}")

    history: List[Dict[str, object]] = []
    start_step = 0
    if cfg.resume_checkpoint:
        checkpoint = torch.load(cfg.resume_checkpoint, map_location=device, weights_only=False)
        saved_config = checkpoint.get("config", {})
        current_config = asdict(cfg)
        invariant_keys = (
            "p", "task", "random_labels", "random_label_seed", "train_fraction",
            "seq_len", "pos_a", "pos_b", "query_pos", "d", "d_a", "M", "r", "T",
            "layers", "local_coupling", "block_type", "factor_rank", "representation", "learned_position",
            "residue_encoding", "residue_encoding_seed", "fixed_residue_norm",
            "lambda_values", "xi_min", "xi_max", "learnable_lambda", "lr", "wd",
            "ce_weight", "alpha_field", "field_temperature", "field_loss",
            "warmup_steps", "schedule", "min_lr_fraction", "seed", "split_seed", "loader_seed",
            "mode_mask_after", "renormalize_mode_mask_after",
        )
        for key in invariant_keys:
            saved_value = saved_config.get(key)
            current_value = current_config.get(key)
            if isinstance(saved_value, list) and isinstance(current_value, tuple):
                saved_value = tuple(saved_value)
            if key in saved_config and saved_value != current_value:
                raise ValueError(
                    f"resume config mismatch for {key}: saved={saved_value!r}, current={current_value!r}"
                )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint.get("history", []))
        start_step = int(checkpoint["step"])
        if "loader_state" in checkpoint:
            loader_generator.set_state(checkpoint["loader_state"].cpu())
        if start_step >= cfg.steps:
            raise ValueError(f"resume step {start_step} must be below target steps {cfg.steps}")
        if "rng_state" in checkpoint:
            state = checkpoint["rng_state"]
            random.setstate(state["python"])
            torch.set_rng_state(state["torch"].cpu())
            if state["cuda"]:
                torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
        print(f"resumed {cfg.resume_checkpoint} at step {start_step}")

    elapsed_before = float(history[-1].get("wall_sec", 0.0)) if history else 0.0
    start_time = time.time() - elapsed_before
    if start_step == 0:
        tr0 = evaluate(model, tr_tokens, tr_roles, tr_labels)
        te0 = evaluate(model, te_tokens, te_roles, te_labels)
        row0 = {"step": 0, "lr": 0.0, "train_acc": tr0["acc"], "test_acc": te0["acc"],
                "train_ce": tr0["ce"], "test_ce": te0["ce"],
                "train_n": int(tr_labels.numel()), "test_n": int(te_labels.numel()),
                "train_correct": round(tr0["acc"] * tr_labels.numel()),
                "test_correct": round(te0["acc"] * te_labels.numel()), "wall_sec": time.time() - start_time}
        history.append(row0)
        write_csv(history, run_dir / "history.csv")
        save_json(history, run_dir / "history.json")
        atomic_torch_save({"step": 0, "config": asdict(cfg), "model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(), "loader_state": loader_generator.get_state(),
                          "rng_state": rng_state(), "history": history}, run_dir / "checkpoint_0000000.pt")
    lambda_override_applied = False
    intervention_mode_mask = (
        torch.tensor(cfg.mode_mask_after, device=device, dtype=torch.float32)
        if cfg.mode_mask_after
        else None
    )
    for step in tqdm(range(start_step + 1, cfg.steps + 1), initial=start_step, total=cfg.steps, desc=run_dir.name, mininterval=10.0, file=__import__("sys").stdout):
        model.train()
        if (
            cfg.lambda_override_after
            and step > cfg.intervention_step
            and not lambda_override_applied
        ):
            override_lambdas(model, cfg.lambda_override_after)
            lambda_override_applied = True
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if cfg.batch_size > 0 and cfg.batch_size < tr_labels.numel():
            selected = torch.randint(
                tr_labels.numel(), (cfg.batch_size,), generator=loader_generator
            ).to(device)
            batch_tokens = tr_tokens[selected]
            batch_roles = tr_roles[selected]
            batch_labels = tr_labels[selected]
            batch_fields = tr_fields[selected]
        else:
            batch_tokens, batch_roles, batch_labels = tr_tokens, tr_roles, tr_labels
            batch_fields = tr_fields
        optimizer.zero_grad(set_to_none=True)
        active_mode_mask = (
            intervention_mode_mask
            if cfg.intervention_step > 0 and step > cfg.intervention_step
            else None
        )
        active_mask_renormalization = bool(
            active_mode_mask is not None and cfg.renormalize_mode_mask_after
        )
        logits = model(
            batch_tokens,
            batch_roles,
            mode_mask=active_mode_mask,
            renormalize_mode_mask=active_mask_renormalization,
        )
        hard_ce = F.cross_entropy(logits, batch_labels)
        loss = float(cfg.ce_weight) * hard_ce
        field_loss_value = logits.new_zeros(())
        if cfg.alpha_field > 0:
            if cfg.field_loss == "kl":
                temperature = float(cfg.field_temperature)
                field_loss_value = F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    F.softmax(batch_fields / temperature, dim=-1),
                    reduction="batchmean",
                ) * temperature * temperature
            else:
                field_loss_value = F.mse_loss(logits, batch_fields)
            loss = loss + float(cfg.alpha_field) * field_loss_value
        loss.backward()
        active_scope = (
            cfg.trainable_scope_after
            if cfg.intervention_step > 0 and step > cfg.intervention_step
            else "all"
        )
        apply_trainable_scope(model, active_scope)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True))
        optimizer.step()

        if step != 1 and step != cfg.steps and step % cfg.eval_every != 0:
            continue
        train_metrics = evaluate(
            model,
            tr_tokens,
            tr_roles,
            tr_labels,
            mode_mask=active_mode_mask,
            renormalize_mode_mask=active_mask_renormalization,
        )
        test_metrics = evaluate(
            model,
            te_tokens,
            te_roles,
            te_labels,
            mode_mask=active_mode_mask,
            renormalize_mode_mask=active_mask_renormalization,
        )
        train_field_metrics: Dict[str, float] = {}
        test_field_metrics: Dict[str, float] = {}
        if cfg.alpha_field > 0:
            train_field_metrics = field_metrics(
                collect_logits(
                    model,
                    tr_tokens,
                    tr_roles,
                    mode_mask=active_mode_mask,
                    renormalize_mode_mask=active_mask_renormalization,
                ), tr_fields
            )
            test_field_metrics = field_metrics(
                collect_logits(
                    model,
                    te_tokens,
                    te_roles,
                    mode_mask=active_mode_mask,
                    renormalize_mode_mask=active_mask_renormalization,
                ), te_fields
            )
        with torch.no_grad():
            weight_norm = math.sqrt(sum(
                float(parameter.square().sum())
                for name, parameter in model.named_parameters()
                if not name.endswith(".bias") and "raw_lambda" not in name
            ))
        row: Dict[str, object] = {
            "step": step,
            "lr": lr,
            "loss": float(loss.detach()),
            "loss/hard_ce": float(hard_ce.detach()),
            "loss/field": float(field_loss_value.detach()),
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
            "intervention/trainable_scope": active_scope,
            "intervention/mode_mask_renormalized": active_mask_renormalization,
            "train_n": int(tr_labels.numel()),
            "test_n": int(te_labels.numel()),
            "train_correct": round(train_metrics["acc"] * tr_labels.numel()),
            "test_correct": round(test_metrics["acc"] * te_labels.numel()),
            "train_acc": train_metrics["acc"],
            "train_ce": train_metrics["ce"],
            "train_true_margin_mean": train_metrics["true_margin_mean"],
            "train_true_margin_positive_fraction": train_metrics["true_margin_positive_fraction"],
            "train_true_margin_q10": train_metrics["true_margin_q10"],
            "train_true_margin_q50": train_metrics["true_margin_q50"],
            "test_acc": test_metrics["acc"],
            "test_ce": test_metrics["ce"],
            "test_true_margin_mean": test_metrics["true_margin_mean"],
            "test_true_margin_positive_fraction": test_metrics["true_margin_positive_fraction"],
            "test_true_margin_q10": test_metrics["true_margin_q10"],
            "test_true_margin_q50": test_metrics["true_margin_q50"],
            **{f"train/{key}": value for key, value in train_field_metrics.items()},
            **{f"test/{key}": value for key, value in test_field_metrics.items()},
            "wall_sec": time.time() - start_time,
        }
        if step == 1 or step == cfg.steps or step % cfg.obs_every == 0:
            row.update(observe(
                model,
                te_tokens[:min(256, te_tokens.shape[0])],
                te_roles[:min(256, te_roles.shape[0])],
                te_tokens,
                te_roles,
                te_labels,
                base_test_acc=test_metrics["acc"],
                base_mode_mask=active_mode_mask,
                renormalize_mode_mask=active_mask_renormalization,
            ))
        history.append(row)
        write_csv(history, run_dir / "history.csv")
        save_json(history, run_dir / "history.json")
        if cfg.checkpoint_every > 0 and (step % cfg.checkpoint_every == 0 or step == cfg.steps):
            atomic_torch_save({
                "step": step,
                "config": asdict(cfg),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loader_state": loader_generator.get_state(),
                "history": history,
                "rng_state": rng_state(),
            }, run_dir / f"checkpoint_{step:07d}.pt")
        print(
            f"[{step:6d}] train={train_metrics['acc']:.3f} test={test_metrics['acc']:.3f} "
            f"train_ce={train_metrics['ce']:.4f} test_ce={test_metrics['ce']:.4f} "
            f"norm={weight_norm:.2f}"
        )

    summary: Dict[str, object] = {
        "final": history[-1],
        "best_test_acc": max(float(row["test_acc"]) for row in history),
        "n_params": count_params(model),
        "n_total_params": count_all_params(model),
        "residue_encoding": cfg.residue_encoding,
        "positions": positions,
        "horizon_per_layer": cfg.r * cfg.T,
        "maximum_receptive_distance": cfg.layers * cfg.r * cfg.T,
        "random_labels": cfg.random_labels,
        "intervention_step": cfg.intervention_step,
        "trainable_scope_after": cfg.trainable_scope_after,
        "lambda_override_after": cfg.lambda_override_after,
        "mode_mask_after": cfg.mode_mask_after,
        "renormalize_mode_mask_after": cfg.renormalize_mode_mask_after,
        "ce_weight": cfg.ce_weight,
        "alpha_field": cfg.alpha_field,
        "field_temperature": cfg.field_temperature,
        "field_loss": cfg.field_loss,
        **classic_metrics(history),
    }
    save_json(summary, run_dir / "summary.json")
    plot_history(history, run_dir / "curves.png")
    return summary


def plot_history(history: List[Dict[str, object]], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = [int(row["step"]) for row in history]
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(steps, [float(row["train_acc"]) for row in history], label="train")
    axes[0].plot(steps, [float(row["test_acc"]) for row in history], label="test")
    axes[0].axhline(0.2, color="gray", linestyle=":", linewidth=1)
    axes[0].axhline(0.9, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend()
    axes[1].plot(steps, [float(row["train_ce"]) for row in history], label="train")
    axes[1].plot(steps, [float(row["test_ce"]) for row in history], label="test")
    axes[1].set_ylabel("cross entropy")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[2].plot(steps, [float(row["weight_norm"]) for row in history])
    axes[2].set_ylabel("weight norm")
    axes[2].set_xlabel("training step")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def selftest() -> None:
    print("== sequence local-mode selftest ==")
    set_seed(0)
    source = torch.randn(2, 17, 3, 4)
    lambdas = (0.35, 0.75, 0.95)
    sweep = OpenLocalExponentialSweep(3, r=2, T=4, lambdas=lambdas, learnable_lambda=True)
    right, left = sweep(source)
    assert torch.allclose(right, sweep.closed_form(source, +1), atol=1e-5)
    assert torch.allclose(left, sweep.closed_form(source, -1), atol=1e-5)
    assert float(right[:, 0].abs().max()) == 0.0
    assert float(left[:, -1].abs().max()) == 0.0
    (right.sum() + left.sum()).backward()
    assert sweep.raw_lambda.grad is not None and float(sweep.raw_lambda.grad.abs().sum()) > 0
    print("  open local sweeps equal finite-horizon closed form       PASS")

    tokens, roles, labels, positions = build_sequence_data(
        p=11, task="add", seq_len=16, pos_a=0, pos_b=7, query_pos=-1
    )
    query = positions["query_pos"]
    assert torch.count_nonzero(tokens[:, query]) == 0
    assert torch.unique(roles[:, query]).tolist() == [3]
    assert labels.unique().numel() == 11
    train_idx, test_idx = make_split(labels.numel(), 0.2, 0)
    assert not torch.isin(train_idx, test_idx).any()
    print("  query contains no answer and train/test are disjoint      PASS")

    dihedral_p = 10
    reflection = torch.tensor([5])
    rotation = torch.tensor([1])
    assert int(modular_label(reflection, rotation, dihedral_p, "dihedral")) == 9
    assert int(modular_label(rotation, reflection, dihedral_p, "dihedral")) == 6
    _, _, dihedral_labels, _ = build_sequence_data(
        p=dihedral_p, task="dihedral", seq_len=16, pos_a=0, pos_b=7, query_pos=-1
    )
    assert torch.equal(
        torch.bincount(dihedral_labels, minlength=dihedral_p),
        torch.full((dihedral_p,), dihedral_p, dtype=torch.long),
    )
    print("  dihedral labels implement a balanced noncommutative group PASS")

    model = SequenceLocalModeModel(
        p=11, seq_len=16, query_pos=query, d=24, d_a=4, M=2,
        r=1, T=4, layers=1, lambdas=(0.6, 0.9), learnable_lambda=False,
    )
    assert_no_forbidden_modules(model)
    first = tokens[:1].clone()
    second = first.clone()
    second[:, positions["pos_a"]] = (second[:, positions["pos_a"]] % 11) + 1
    zero_modes = torch.zeros(model.M)
    with torch.no_grad():
        off_first = model(first, roles[:1], mode_mask=zero_modes)
        off_second = model(second, roles[:1], mode_mask=zero_modes)
        short_first = model(first, roles[:1])
        short_second = model(second, roles[:1])
        all_modes = torch.ones(model.M)
        renormalized_all = model(
            first,
            roles[:1],
            mode_mask=all_modes,
            renormalize_mode_mask=True,
        )
    assert torch.equal(off_first, off_second)
    assert torch.equal(short_first, short_second), "distant operand leaked beyond local horizon"
    assert torch.allclose(short_first, renormalized_all, atol=1e-6)
    print("  mode-off and insufficient-horizon query are independent  PASS")

    long_model = SequenceLocalModeModel(
        p=11, seq_len=16, query_pos=query, d=24, d_a=4, M=2,
        r=1, T=15, layers=1, lambdas=(0.6, 0.9), learnable_lambda=False,
    )
    with torch.no_grad():
        long_first = long_model(first, roles[:1])
        long_second = long_model(second, roles[:1])
    assert float((long_first - long_second).abs().max()) > 1e-7
    print("  sufficient local sweeps make the distant operand visible PASS")

    for parameter in long_model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    apply_trainable_scope(long_model, "mode_only")
    for name, parameter in long_model.named_parameters():
        if parameter.requires_grad:
            assert (parameter.grad is not None) == is_mode_parameter(name), name
    for parameter in long_model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    apply_trainable_scope(long_model, "nonmode_only")
    for name, parameter in long_model.named_parameters():
        if parameter.requires_grad:
            assert (parameter.grad is not None) != is_mode_parameter(name), name
    for parameter in long_model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    apply_trainable_scope(long_model, "visible_only")
    for name, parameter in long_model.named_parameters():
        if parameter.requires_grad:
            assert (parameter.grad is not None) != is_mode_core_parameter(name), name
    for parameter in long_model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    apply_trainable_scope(long_model, "freeze_shared_mode_out")
    for name, parameter in long_model.named_parameters():
        if parameter.requires_grad:
            assert (parameter.grad is None) == is_shared_mode_out_parameter(name), name
    for parameter in long_model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    apply_trainable_scope(long_model, "freeze_residue")
    for name, parameter in long_model.named_parameters():
        if parameter.requires_grad:
            assert (parameter.grad is None) == name.startswith("residue."), name
    override_lambdas(long_model, (0.4, 0.8))
    assert torch.allclose(long_model.sweep.lambdas, torch.tensor((0.4, 0.8)), atol=1e-6)
    print("  causal parameter scopes exactly partition the model      PASS")

    lightweight_specs = (
        ("low_rank", "learned", 16, 2, 2, 4, 2, True, 4000),
        ("scalar", "learned", 16, 2, 2, 4, 2, True, 3000),
        ("scalar", "fixed_fourier", 4, 1, 1, 2, 1, False, 200),
    )
    for block_type, representation, d, d_a, rank, modes, layers, learned_position, limit in lightweight_specs:
        lightweight = SequenceLocalModeModel(
            p=11,
            seq_len=16,
            query_pos=query,
            d=d,
            d_a=d_a,
            M=modes,
            r=2,
            T=4,
            layers=layers,
            lambdas=xi_spread_lambdas(modes, 0.7, 16.0),
            learnable_lambda=False,
            block_type=block_type,
            factor_rank=rank,
            representation=representation,
            learned_position=learned_position,
        )
        assert_no_forbidden_modules(lightweight)
        logits = lightweight(tokens[:16], roles[:16])
        loss = F.cross_entropy(logits, labels[:16])
        loss.backward()
        assert any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
            for parameter in lightweight.parameters()
            if parameter.requires_grad
        )
        assert count_params(lightweight) < limit
    print("  low-rank, scalar, and fixed-Fourier variants backpropagate PASS")

    encoding_models: Dict[str, SequenceLocalModeModel] = {}
    encoding_lambdas = xi_spread_lambdas(4, 0.7, 16.0)
    for encoding in sorted(RESIDUE_ENCODINGS):
        encoded = SequenceLocalModeModel(
            p=59,
            seq_len=16,
            query_pos=15,
            d=64,
            d_a=8,
            M=4,
            r=2,
            T=4,
            layers=2,
            lambdas=encoding_lambdas,
            learnable_lambda=False,
            residue_encoding=encoding,
            residue_encoding_seed=123,
        )
        assert_no_forbidden_modules(encoded)
        weight = encoded.residue.weight.detach()
        assert weight.shape == (60, 64)
        assert torch.count_nonzero(weight[0]) == 0
        if encoding == "learned":
            assert encoded.residue.weight.requires_grad
        else:
            assert not encoded.residue.weight.requires_grad
            expected = torch.full((59,), 8.0)
            assert torch.allclose(weight[1:].norm(dim=1), expected, atol=1e-5)
        encoding_models[encoding] = encoded
    correct = encoding_models["fixed_fourier"].residue.weight[1:].detach()
    permuted = encoding_models["fixed_permuted_fourier"].residue.weight[1:].detach()
    assert not torch.allclose(correct, permuted)
    assert torch.allclose(
        torch.linalg.svdvals(correct),
        torch.linalg.svdvals(permuted),
        atol=1e-5,
    )
    assert count_params(encoding_models["learned"]) == 62787
    for encoding, encoded in encoding_models.items():
        assert count_all_params(encoded) == 62791, encoding
        if encoding != "learned":
            assert count_params(encoded) == 58947, encoding
    same_seed_models: Dict[str, SequenceLocalModeModel] = {}
    for encoding in sorted(RESIDUE_ENCODINGS):
        set_seed(321)
        same_seed_models[encoding] = SequenceLocalModeModel(
            p=59,
            seq_len=16,
            query_pos=15,
            d=64,
            d_a=8,
            M=4,
            r=2,
            T=4,
            layers=2,
            lambdas=encoding_lambdas,
            learnable_lambda=False,
            residue_encoding=encoding,
            residue_encoding_seed=123,
        )
    reference_parameters = dict(same_seed_models["learned"].named_parameters())
    for encoding, encoded in same_seed_models.items():
        for name, parameter in encoded.named_parameters():
            if name.startswith("residue."):
                continue
            assert torch.equal(parameter, reference_parameters[name]), (encoding, name)
    print("  residue encoding controls preserve size and fixed norms   PASS")
    print("  same-seed shared parameters are exactly identical         PASS")
    print("  forbidden global and MLP/FFN modules are absent           PASS")
    print("== selftest complete ==")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequence-form local-mode Exp2 grokking search")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--out_dir", default=RunConfig.out_dir)
    parser.add_argument("--device", default=RunConfig.device)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--split_seed", type=int, default=RunConfig.split_seed)
    parser.add_argument("--loader_seed", type=int, default=RunConfig.loader_seed)
    parser.add_argument("--p", type=int, default=RunConfig.p)
    parser.add_argument("--task", choices=sorted(TASKS), default=RunConfig.task)
    parser.add_argument("--random_labels", type=str_to_bool, default=RunConfig.random_labels)
    parser.add_argument("--random_label_seed", type=int, default=RunConfig.random_label_seed)
    parser.add_argument("--train_fraction", type=float, default=RunConfig.train_fraction)
    parser.add_argument("--seq_len", type=int, default=RunConfig.seq_len)
    parser.add_argument("--pos_a", type=int, default=RunConfig.pos_a)
    parser.add_argument("--pos_b", type=int, default=RunConfig.pos_b)
    parser.add_argument("--query_pos", type=int, default=RunConfig.query_pos)
    parser.add_argument("--d", type=int, default=RunConfig.d)
    parser.add_argument("--d_a", type=int, default=RunConfig.d_a)
    parser.add_argument("--M", type=int, default=RunConfig.M)
    parser.add_argument("--r", type=int, default=RunConfig.r)
    parser.add_argument("--T", type=int, default=RunConfig.T)
    parser.add_argument("--layers", type=int, default=RunConfig.layers)
    parser.add_argument("--local_coupling", type=float, default=RunConfig.local_coupling)
    parser.add_argument(
        "--block_type",
        choices=["full", "low_rank", "scalar"],
        default=RunConfig.block_type,
    )
    parser.add_argument("--factor_rank", type=int, default=RunConfig.factor_rank)
    parser.add_argument(
        "--representation",
        choices=["learned", "fixed_fourier"],
        default=RunConfig.representation,
    )
    parser.add_argument(
        "--learned_position",
        type=str_to_bool,
        default=RunConfig.learned_position,
    )
    parser.add_argument(
        "--residue_encoding",
        choices=sorted(RESIDUE_ENCODINGS),
        default=RunConfig.residue_encoding,
    )
    parser.add_argument(
        "--residue_encoding_seed",
        type=int,
        default=RunConfig.residue_encoding_seed,
    )
    parser.add_argument(
        "--fixed_residue_norm",
        type=float,
        default=RunConfig.fixed_residue_norm,
    )
    parser.add_argument("--lambda_values", type=parse_float_tuple, default=RunConfig.lambda_values)
    parser.add_argument("--xi_min", type=float, default=RunConfig.xi_min)
    parser.add_argument("--xi_max", type=float, default=RunConfig.xi_max)
    parser.add_argument("--learnable_lambda", type=str_to_bool, default=RunConfig.learnable_lambda)
    parser.add_argument("--steps", type=int, default=RunConfig.steps)
    parser.add_argument("--batch_size", type=int, default=RunConfig.batch_size)
    parser.add_argument("--lr", type=float, default=RunConfig.lr)
    parser.add_argument("--wd", type=float, default=RunConfig.wd)
    parser.add_argument("--ce_weight", type=float, default=RunConfig.ce_weight)
    parser.add_argument("--alpha_field", type=float, default=RunConfig.alpha_field)
    parser.add_argument("--field_temperature", type=float, default=RunConfig.field_temperature)
    parser.add_argument("--field_loss", choices=["kl", "mse"], default=RunConfig.field_loss)
    parser.add_argument("--warmup_steps", type=int, default=RunConfig.warmup_steps)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default=RunConfig.schedule)
    parser.add_argument("--min_lr_fraction", type=float, default=RunConfig.min_lr_fraction)
    parser.add_argument("--grad_clip", type=float, default=RunConfig.grad_clip)
    parser.add_argument("--eval_every", type=int, default=RunConfig.eval_every)
    parser.add_argument("--obs_every", type=int, default=RunConfig.obs_every)
    parser.add_argument("--checkpoint_every", type=int, default=RunConfig.checkpoint_every)
    parser.add_argument("--resume_checkpoint", default=RunConfig.resume_checkpoint)
    parser.add_argument("--intervention_step", type=int, default=RunConfig.intervention_step)
    parser.add_argument(
        "--trainable_scope_after",
        choices=[
            "all", "mode_only", "nonmode_only", "mode_core_only", "visible_only",
            "residue_only", "freeze_residue", "freeze_mode_readout",
            "freeze_shared_mode_out", "freeze_classifier",
        ],
        default=RunConfig.trainable_scope_after,
    )
    parser.add_argument(
        "--lambda_override_after",
        type=parse_float_tuple,
        default=RunConfig.lambda_override_after,
    )
    parser.add_argument(
        "--mode_mask_after",
        type=parse_float_tuple,
        default=RunConfig.mode_mask_after,
    )
    parser.add_argument(
        "--renormalize_mode_mask_after",
        type=str_to_bool,
        default=RunConfig.renormalize_mode_mask_after,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    config = RunConfig(**{key: value for key, value in vars(args).items() if key != "selftest"})
    summary = train_one_run(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
