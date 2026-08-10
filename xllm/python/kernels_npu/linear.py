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

"""NPU weight preparation for linear layers."""

from __future__ import annotations

import os

import torch
import torch_npu

_FRACTAL_NZ_FORMAT = 29
_TRUE_ENV_VALUES = frozenset(("1", "true", "on"))
_FALSE_ENV_VALUES = frozenset(("0", "false", "off"))


def _parse_weight_nz_enabled(value: str | None) -> bool:
    """Parse the process-level FRACTAL_NZ switch."""
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    raise ValueError(
        "XLLM_W8A8_WEIGHT_NZ must be one of 1, true, on, 0, false, or off; "
        f"got {value!r}"
    )


# Read once when the NPU kernel package is loaded. Set the variable before
# process startup; changing the environment after import has no effect.
WEIGHT_NZ_ENABLED = _parse_weight_nz_enabled(
    os.environ.get("XLLM_W8A8_WEIGHT_NZ")
)


def prepare_row_parallel_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Lay out a row-parallel weight for the NPU matmul kernels.

    The matmul kernels read ``[K, N]`` in the fractal-NZ format, so the
    checkpoint layout is transposed and cast. A weight still on the host is
    returned untouched: the format cast needs device memory and runs after the
    weight is moved.

    Args:
        weight: Row-parallel weight of shape ``[N, K]``.

    Returns:
        The weight and whether it was transposed to ``[K, N]``.
    """
    if weight.device.type == "cpu":
        return weight, False
    transposed = weight.transpose(0, 1).contiguous()
    return torch_npu.npu_format_cast(transposed, _FRACTAL_NZ_FORMAT), True


def prepare_quant_weight(
    weight: torch.Tensor,
) -> torch.Tensor:
    """Pack an INT8 ``[out, in]`` weight for quantized matmul.

    ``torch_npu.npu_quant_matmul`` reads the right-hand operand as ``[K, N]``
    and benefits substantially from the NPU Fractal-NZ layout. Quantized Python
    linear layers use the same layout as the checkpoint before this helper is
    called, so keep the transpose and format conversion together here.  The
    CPU branch is intentionally a plain transpose for unit tests and weight
    loading without an initialized NPU.
    """
    transposed = weight.transpose(0, 1).contiguous()
    if weight.device.type == "cpu" or not WEIGHT_NZ_ENABLED:
        return transposed
    return torch_npu.npu_format_cast(transposed, _FRACTAL_NZ_FORMAT)


__all__ = [
    "prepare_row_parallel_weight",
    "prepare_quant_weight",
]
