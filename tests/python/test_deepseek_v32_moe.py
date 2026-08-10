# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from xllm.python.models import deepseek_v32


class _Gate(nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self._calls.append("gate")
        return hidden.float()


class _SharedExperts(nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self._calls.append("shared")
        return hidden + 3


class _FakeEvent:
    def __init__(self, calls: list[str], name: str) -> None:
        self._calls = calls
        self._name = name

    def record(self, stream: "_FakeStream") -> None:
        self._calls.append(f"record_{self._name}_{stream.name}")


class _FakeStream:
    def __init__(self, calls: list[str], name: str) -> None:
        self._calls = calls
        self.name = name

    def wait_event(self, event: _FakeEvent) -> None:
        self._calls.append(f"wait_{self.name}_{event._name}")

    def wait_stream(self, stream: "_FakeStream") -> None:
        self._calls.append(f"wait_{self.name}_{stream.name}")


class _FakeNpu:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self.current = _FakeStream(calls, "current")
        self.shared = _FakeStream(calls, "shared")
        self._event_count = 0

    def Stream(self, device: torch.device) -> _FakeStream:
        del device
        self._calls.append("create_shared_stream")
        return self.shared

    def Event(self) -> _FakeEvent:
        name = f"start_{self._event_count}"
        self._event_count += 1
        self._calls.append(f"create_{name}_event")
        return _FakeEvent(self._calls, name)

    def current_stream(self) -> _FakeStream:
        return self.current

    @contextmanager
    def stream(self, stream: _FakeStream):
        self._calls.append(f"enter_{stream.name}")
        try:
            yield
        finally:
            self._calls.append(f"exit_{stream.name}")


def _make_moe(
    calls: list[str], *, parallel: bool, tp_size: int = 1
) -> deepseek_v32.DeepseekV3MoE:
    moe = deepseek_v32.DeepseekV3MoE.__new__(deepseek_v32.DeepseekV3MoE)
    nn.Module.__init__(moe)
    moe.cfg = SimpleNamespace(tp_size=tp_size, norm_topk_prob=True)
    moe.gate = _Gate(calls)
    moe.shared_experts = _SharedExperts(calls)
    moe.experts_w13 = nn.Parameter(
        torch.empty(1), requires_grad=False
    )
    moe.experts_w2 = nn.Parameter(
        torch.empty(1), requires_grad=False
    )
    moe.register_buffer("experts_w13_scale_compute", torch.empty(1))
    moe.register_buffer("experts_w2_scale_compute", torch.empty(1))
    moe.register_buffer("e_score_correction_bias", torch.empty(1))
    moe.topk = 1
    moe.topk_group = 1
    moe.n_group = 1
    moe.routed_scaling = 1.0
    moe._expert_parallel_enabled = parallel
    moe._shared_expert_start_event = None
    return moe


def _grouped_moe(calls: list[str], hidden: torch.Tensor) -> torch.Tensor:
    calls.append("routed")
    return hidden + 2


def test_moe_serial_fallback_combines_experts_before_tp_reduce() -> None:
    calls: list[str] = []
    moe = _make_moe(calls, parallel=False, tp_size=2)
    all_reduce = MagicMock()
    hidden = torch.ones(2, 4)

    with (
        patch.object(
            deepseek_v32.kernels,
            "grouped_moe",
            side_effect=lambda hidden, *_args: _grouped_moe(calls, hidden),
            create=True,
        ),
        patch.object(
            deepseek_v32.distributed,
            "all_reduce_",
            all_reduce,
            create=True,
        ),
    ):
        output = moe(hidden)

    assert calls == ["gate", "routed", "shared"]
    assert torch.equal(output, hidden * 2 + 5)
    all_reduce.assert_called_once_with(output)


def test_moe_parallel_queues_shared_before_gate_and_waits_before_add() -> None:
    calls: list[str] = []
    moe = _make_moe(calls, parallel=True)
    fake_npu = _FakeNpu(calls)
    hidden = torch.ones(2, 4)
    deepseek_v32._SHARED_EXPERT_STREAMS.clear()

    with (
        patch.object(torch, "npu", fake_npu, create=True),
        patch.object(
            deepseek_v32.kernels,
            "grouped_moe",
            side_effect=lambda hidden, *_args: _grouped_moe(calls, hidden),
            create=True,
        ),
    ):
        output = moe(hidden)

    assert torch.equal(output, hidden * 2 + 5)
    assert calls.index("shared") < calls.index("gate")
    assert calls.index("routed") < calls.index("wait_current_shared")
    assert calls.count("create_shared_stream") == 1

    with (
        patch.object(torch, "npu", fake_npu, create=True),
        patch.object(
            deepseek_v32.kernels,
            "grouped_moe",
            side_effect=lambda hidden, *_args: _grouped_moe(calls, hidden),
            create=True,
        ),
    ):
        moe(hidden)

    assert calls.count("create_shared_stream") == 1
