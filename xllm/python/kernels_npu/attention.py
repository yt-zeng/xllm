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

"""NPU paged-attention support kernels.

Both operators are capturable: they read only their arguments, so a decode
graph can replay them without re-planning.
"""

from __future__ import annotations

import torch

reshape_paged_cache = torch.ops.xllm_ops.reshape_paged_cache
update_decode_graph_metadata = torch.ops.xllm_ops.update_decode_graph_metadata


def batch_matmul_transpose(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Batch matmul with the NPU-optimized transposed-weight kernel.

    MLA's recovered value path repeatedly computes ``[H, T, L] @ [H, L, V]``.
    The dedicated NPU operator avoids materializing a transpose and is the
    same path used by vLLM-Ascend for ``W_UV``.
    """
    if x.device.type != "npu":
        return torch.bmm(x, weight)
    return torch.ops.npu.npu_transpose_batchmatmul(
        x, weight, perm_y=(1, 0, 2)
    )

__all__ = [
    "reshape_paged_cache",
    "update_decode_graph_metadata",
    "batch_matmul_transpose",
]
