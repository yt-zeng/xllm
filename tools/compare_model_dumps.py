#!/usr/bin/env python3
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

"""Compare Python and C++ hidden-state or logits tensor dumps."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch


def _load(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor")
    return tensor


def _compare(
    name: str,
    python_path: Path,
    cpp_path: Path,
    topk: int,
    require_bitwise: bool,
    max_abs_threshold: float | None,
) -> bool:
    python_tensor = _load(python_path)
    cpp_tensor = _load(cpp_path)
    print(f"[{name}]")
    print(f"python: shape={tuple(python_tensor.shape)} dtype={python_tensor.dtype}")
    print(f"cpp:    shape={tuple(cpp_tensor.shape)} dtype={cpp_tensor.dtype}")
    if python_tensor.shape != cpp_tensor.shape:
        print("shape_equal: False")
        return False

    print("shape_equal: True")
    dtype_equal = python_tensor.dtype == cpp_tensor.dtype
    print(f"dtype_equal: {dtype_equal}")
    bitwise_equal = dtype_equal and torch.equal(python_tensor, cpp_tensor)
    print(f"bitwise_equal: {bitwise_equal}")
    python_float = python_tensor.float()
    cpp_float = cpp_tensor.float()
    difference = python_float - cpp_float
    max_abs = difference.abs().max().item()
    mean_abs = difference.abs().mean().item()
    cpp_norm = cpp_float.norm().item()
    relative_l2 = difference.norm().item() / cpp_norm if cpp_norm else math.inf
    python_flat = python_float.flatten().double()
    cpp_flat = cpp_float.flatten().double()
    denominator = python_flat.norm() * cpp_flat.norm()
    cosine = (
        torch.dot(python_flat, cpp_flat) / denominator
        if denominator.item()
        else torch.tensor(float("nan"))
    )
    print(f"max_abs: {max_abs:.10g}")
    print(f"mean_abs: {mean_abs:.10g}")
    print(f"relative_l2: {relative_l2:.10g}")
    print(f"cosine: {cosine.item():.10g}")

    if name == "logits" and python_tensor.size(-1) > 0:
        selected_topk = min(topk, python_tensor.size(-1))
        python_rows = python_float.reshape(-1, python_tensor.size(-1))
        cpp_rows = cpp_float.reshape(-1, cpp_tensor.size(-1))
        python_topk = python_rows.topk(selected_topk, dim=-1).indices
        cpp_topk = cpp_rows.topk(selected_topk, dim=-1).indices
        top1_match = (python_topk[:, 0] == cpp_topk[:, 0]).float().mean()
        overlap = []
        for python_row, cpp_row in zip(python_topk, cpp_topk):
            overlap.append(
                len(set(python_row.tolist()) & set(cpp_row.tolist()))
                / selected_topk
            )
        print(f"top1_match: {top1_match.item():.6f}")
        print(f"top{selected_topk}_overlap: {sum(overlap) / len(overlap):.6f}")
        print(f"python_top{selected_topk}: {python_topk.tolist()}")
        print(f"cpp_top{selected_topk}: {cpp_topk.tolist()}")
    has_acceptance_criteria = require_bitwise or max_abs_threshold is not None
    valid = not require_bitwise or bitwise_equal
    if max_abs_threshold is not None:
        valid &= max_abs <= max_abs_threshold
    if has_acceptance_criteria:
        print(f"result: {'PASS' if valid else 'FAIL'}")
    else:
        print("result: REPORT")
    return valid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-logits", type=Path)
    parser.add_argument("--cpp-logits", type=Path)
    parser.add_argument("--python-hidden", type=Path)
    parser.add_argument("--cpp-hidden", type=Path)
    parser.add_argument("--python-layer-dir", type=Path)
    parser.add_argument("--cpp-layer-dir", type=Path)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--max-abs-threshold", type=float)
    return parser.parse_args()


def _layer_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name == "embedding.pt":
        return (0, 0, "")
    layer_match = re.match(r"layer_(\d+)", path.stem)
    if layer_match:
        return (1, int(layer_match.group(1)), path.stem)
    if path.name == "final_hidden.pt":
        return (2, 0, "")
    return (3, 0, path.stem)


def _compare_layer_directories(args: argparse.Namespace) -> bool:
    if not args.python_layer_dir or not args.cpp_layer_dir:
        raise ValueError("both layer dump directories are required")
    python_files = {
        path.name: path for path in args.python_layer_dir.glob("*.pt")
    }
    cpp_files = {path.name: path for path in args.cpp_layer_dir.glob("*.pt")}
    if python_files.keys() != cpp_files.keys():
        print(f"python layer files: {sorted(python_files)}")
        print(f"cpp layer files: {sorted(cpp_files)}")
        return False
    valid = True
    for file_name in sorted(
        python_files, key=lambda name: _layer_sort_key(Path(name))
    ):
        valid &= _compare(
            file_name.removesuffix(".pt"),
            python_files[file_name],
            cpp_files[file_name],
            args.topk,
            args.require_bitwise,
            args.max_abs_threshold,
        )
    _print_quantization_diagnosis(python_files, cpp_files)
    return valid


def _bitwise_equal(
    python_files: dict[str, Path], cpp_files: dict[str, Path], name: str
) -> bool:
    python_tensor = _load(python_files[name])
    cpp_tensor = _load(cpp_files[name])
    return python_tensor.dtype == cpp_tensor.dtype and torch.equal(
        python_tensor, cpp_tensor
    )


def _values_equal(
    python_files: dict[str, Path], cpp_files: dict[str, Path], name: str
) -> bool:
    return torch.equal(
        _load(python_files[name]).float(), _load(cpp_files[name]).float()
    )


def _print_quantization_diagnosis(
    python_files: dict[str, Path], cpp_files: dict[str, Path]
) -> None:
    stages = (
        (
            "layer_0_mla_input_norm",
            None,
            "input RMSNorm",
        ),
        (
            "layer_0_mla_q_a_norm",
            "layer_0_mla_q_a_out.pt",
            "q_a RMSNorm",
        ),
    )
    messages = []
    for prefix, input_name, label in stages:
        names = {
            "bf16": f"{prefix}_bf16.pt",
            "scale": f"{prefix}_quant_scale.pt",
            "offset": f"{prefix}_quant_offset.pt",
            "quant": f"{prefix}_quant.pt",
            "quant_ref": f"{prefix}_quant_ref.pt",
        }
        required_names = set(names.values())
        if input_name is not None:
            required_names.add(input_name)
        if not required_names.issubset(python_files) or not required_names.issubset(
            cpp_files
        ):
            continue
        input_equal = input_name is None or _bitwise_equal(
            python_files, cpp_files, input_name
        )
        bf16_equal = _bitwise_equal(python_files, cpp_files, names["bf16"])
        scale_equal = _bitwise_equal(python_files, cpp_files, names["scale"])
        offset_equal = _bitwise_equal(python_files, cpp_files, names["offset"])
        offset_values_equal = _values_equal(
            python_files, cpp_files, names["offset"]
        )
        quant_equal = _bitwise_equal(python_files, cpp_files, names["quant"])
        quant_ref_equal = _bitwise_equal(
            python_files, cpp_files, names["quant_ref"]
        )
        if not input_equal:
            diagnosis = "input already differs; first locate the upstream projection"
        elif not bf16_equal:
            diagnosis = "BF16 RMSNorm differs"
        elif not scale_equal:
            diagnosis = "quant scale/offset differs"
        elif not offset_equal and not offset_values_equal:
            diagnosis = "quant offset value differs"
        elif not quant_ref_equal:
            diagnosis = "standalone quantization rounding/kernel differs"
        elif not quant_equal:
            offset_note = (
                "; offset values match but dtype differs"
                if not offset_equal
                else ""
            )
            diagnosis = (
                "standalone quantization matches; fused NormQuant differs"
                + offset_note
            )
        else:
            diagnosis = "BF16 RMSNorm and quantization are bitwise aligned"
        messages.append(f"{label}: {diagnosis}")
    if messages:
        print("[quantization_diagnosis]")
        for message in messages:
            print(message)


def main() -> int:
    args = _parse_args()
    compared = False
    valid = True
    if args.python_layer_dir or args.cpp_layer_dir:
        valid &= _compare_layer_directories(args)
        compared = True
    if args.python_hidden or args.cpp_hidden:
        if not args.python_hidden or not args.cpp_hidden:
            raise ValueError("both hidden dump paths are required")
        valid &= _compare(
            "hidden",
            args.python_hidden,
            args.cpp_hidden,
            args.topk,
            args.require_bitwise,
            args.max_abs_threshold,
        )
        compared = True
    if args.python_logits or args.cpp_logits:
        if not args.python_logits or not args.cpp_logits:
            raise ValueError("both logits dump paths are required")
        valid &= _compare(
            "logits",
            args.python_logits,
            args.cpp_logits,
            args.topk,
            args.require_bitwise,
            args.max_abs_threshold,
        )
        compared = True
    if not compared:
        raise ValueError("provide hidden or logits dump path pairs")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
