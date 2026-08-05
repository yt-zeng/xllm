# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed-load profiling tests for DeepSeek-V3.2 target and MTP serving.

The server must be started before running the test.  MTP is enabled by the
same xLLM service with ``--draft_model``; therefore the MTP case reuses the
target API URL and model by default.  Separate MTP variables are only needed
when comparing two independently started services:

    XLLM_DEEPSEEK_V32_API_URL=http://127.0.0.1:8199/v1 \
    XLLM_DEEPSEEK_V32_MODEL=DeepSeek-V3.2 \
    XLLM_DEEPSEEK_V32_PROFILE_VARIANT=mtp \
    XLLM_DEEPSEEK_V32_MTP_API_URL=http://127.0.0.1:8200/v1 \
    XLLM_DEEPSEEK_V32_MTP_MODEL=DeepSeek-V3.2 \
    pytest -s tests/python/test_deepseek_v32_profiling.py

Set ``XLLM_DEEPSEEK_V32_PROFILE_BACKEND=online`` and
``XLLM_DEEPSEEK_V32_START_PROFILE=1`` to call the server's online profiling
endpoints around the workload.  NPU runs should leave the backend as
``msprof`` (the default) and start the server under msprof instead.  Set
``XLLM_DEEPSEEK_V32_PROFILE_OUTPUT_DIR`` to save the measured report as JSON.
The test does not require those endpoints because NPU deployments may use an
external profiler such as msprof instead.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

import pytest


@dataclass(frozen=True)
class _ModelSpec:
    name: str
    api_url_env: str
    model_env: str


_MODEL_SPECS = (
    _ModelSpec(
        "deepseek_v32",
        "XLLM_DEEPSEEK_V32_API_URL",
        "XLLM_DEEPSEEK_V32_MODEL",
    ),
    _ModelSpec(
        "deepseek_v32_mtp",
        "XLLM_DEEPSEEK_V32_MTP_API_URL",
        "XLLM_DEEPSEEK_V32_MTP_MODEL",
    ),
)


def _resolve_service(spec: _ModelSpec) -> tuple[str | None, str | None]:
    common_api_url = os.getenv("XLLM_DEEPSEEK_V32_API_URL")
    common_model = os.getenv("XLLM_DEEPSEEK_V32_MODEL")
    profile_variant = os.getenv("XLLM_DEEPSEEK_V32_PROFILE_VARIANT", "").lower()
    has_variant_service = bool(
        os.getenv(spec.api_url_env) or os.getenv(spec.model_env)
    )
    if (common_api_url or common_model) and not has_variant_service:
        if profile_variant != spec.name:
            return None, None
    return (
        os.getenv(spec.api_url_env) or common_api_url,
        os.getenv(spec.model_env) or common_model,
    )


def _read_metric(base_url: str, metric_name: str) -> float | None:
    metric_url = f"{base_url.rstrip('/')}/vars/{quote(metric_name, safe='')}"
    try:
        with urlopen(metric_url, timeout=10) as response:
            value = response.read().decode("utf-8").strip()
    except (HTTPError, OSError, TimeoutError, URLError):
        return None
    try:
        return float(value)
    except ValueError:
        pattern = rf"{re.escape(metric_name)}\s*:\s*(?:<[^>]+>)*"
        match = re.search(pattern + r"([-+]?\d+(?:\.\d+)?)", value)
        return float(match.group(1)) if match else None


def _post_profile(base_url: str, action: str) -> bool:
    request = Request(
        f"{base_url.rstrip('/')}/{action}_profile",
        data=b"",
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            response.read()
        return True
    except (HTTPError, OSError, TimeoutError, URLError):
        return False


def _request_body(prompt_chars: int, max_tokens: int) -> dict[str, Any]:
    prompt = (
        "Solve the following arithmetic problem and explain the result briefly: "
        "A warehouse has 48 boxes with 16 items in each box. How many items "
        "are there in total? "
    )
    repeated_prompt = (prompt * ((prompt_chars + len(prompt) - 1) // len(prompt)))[
        :prompt_chars
    ]
    return {
        "prompt": repeated_prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "ignore_eos": True,
        "stream": False,
    }


def _send_request(
    endpoint: str,
    model: str,
    request_body: dict[str, Any],
    timeout_seconds: float,
) -> float:
    payload = dict(request_body)
    payload["model"] = model
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start_time = time.perf_counter()
    with urlopen(request, timeout=timeout_seconds) as response:
        response.read()
    return (time.perf_counter() - start_time) * 1000.0


def _run_batch(
    api_url: str,
    model: str,
    request_body: dict[str, Any],
    request_count: int,
    batch_size: int,
    timeout_seconds: float,
) -> list[float]:
    endpoint = f"{api_url.rstrip('/')}/completions"
    latencies: list[float] = []
    for batch_start in range(0, request_count, batch_size):
        current_batch_size = min(batch_size, request_count - batch_start)
        with ThreadPoolExecutor(max_workers=current_batch_size) as executor:
            futures = [
                executor.submit(
                    _send_request,
                    endpoint,
                    model,
                    request_body,
                    timeout_seconds,
                )
                for _ in range(current_batch_size)
            ]
            latencies.extend(future.result() for future in futures)
    return latencies


def _metric_delta(
    before: dict[str, float | None], after: dict[str, float | None], name: str
) -> float | None:
    if before.get(name) is None or after.get(name) is None:
        return None
    return after[name] - before[name]  # type: ignore[operator]


def _write_report(spec: _ModelSpec, report: dict[str, Any]) -> None:
    output_dir = os.getenv("XLLM_DEEPSEEK_V32_PROFILE_OUTPUT_DIR")
    if not output_dir:
        return
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / f"{spec.name}_profiling.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Profiling report: {report_path}")


@pytest.mark.parametrize("spec", _MODEL_SPECS, ids=lambda item: item.name)
def test_deepseek_v32_fixed_load_profiling(spec: _ModelSpec) -> None:
    api_url, model = _resolve_service(spec)
    if not api_url or not model:
        pytest.skip(
            f"set XLLM_DEEPSEEK_V32_API_URL and XLLM_DEEPSEEK_V32_MODEL "
            f"and XLLM_DEEPSEEK_V32_PROFILE_VARIANT={spec.name}, or set "
            f"the variant-specific {spec.api_url_env}/{spec.model_env} to run"
        )

    request_count = int(os.getenv("XLLM_DEEPSEEK_V32_PROFILE_REQUESTS", "8"))
    batch_size = int(os.getenv("XLLM_DEEPSEEK_V32_PROFILE_BATCH_SIZE", "1"))
    prompt_chars = int(os.getenv("XLLM_DEEPSEEK_V32_PROFILE_PROMPT_CHARS", "2048"))
    max_tokens = int(os.getenv("XLLM_DEEPSEEK_V32_PROFILE_MAX_TOKENS", "128"))
    timeout_seconds = float(
        os.getenv("XLLM_DEEPSEEK_V32_PROFILE_TIMEOUT", "300")
    )
    if request_count <= 0 or batch_size <= 0 or prompt_chars <= 0 or max_tokens <= 0:
        raise ValueError("profiling workload parameters must be positive")

    api_parts = urlsplit(api_url)
    base_url = f"{api_parts.scheme}://{api_parts.netloc}"
    request_body = _request_body(prompt_chars, max_tokens)

    # Remove graph compilation and cache allocation from the measured window.
    _run_batch(api_url, model, request_body, 1, 1, timeout_seconds)

    metric_names = (
        "num_processing_tokens_total_prompt",
        "num_processing_tokens_total_generated",
        "speculative_num_draft_tokens_total",
        "speculative_num_accepted_tokens_total",
    )
    before = {name: _read_metric(base_url, name) for name in metric_names}
    profile_started = False
    profile_backend = os.getenv(
        "XLLM_DEEPSEEK_V32_PROFILE_BACKEND", "msprof"
    ).lower()
    start_online_profile = (
        profile_backend == "online"
        and os.getenv("XLLM_DEEPSEEK_V32_START_PROFILE", "0") == "1"
    )
    if (
        os.getenv("XLLM_DEEPSEEK_V32_START_PROFILE", "0") == "1"
        and profile_backend != "online"
    ):
        pytest.fail(
            "online profiling was requested, but the profiling backend is "
            f"{profile_backend!r}; use PROFILE_BACKEND=online only for a "
            "supported online profiler"
        )
    if start_online_profile:
        profile_started = _post_profile(base_url, "start")
        if not profile_started:
            pytest.fail("online profiling was requested but /start_profile failed")

    start_time = time.perf_counter()
    request_error: BaseException | None = None
    stop_profile_failed = False
    try:
        latencies = _run_batch(
            api_url,
            model,
            request_body,
            request_count,
            batch_size,
            timeout_seconds,
        )
    except BaseException as error:
        request_error = error
    finally:
        stop_profile_failed = profile_started and not _post_profile(
            base_url, "stop"
        )
    if request_error is not None:
        raise request_error.with_traceback(request_error.__traceback__)
    if stop_profile_failed:
        pytest.fail("online profiling started but /stop_profile failed")
    duration_seconds = time.perf_counter() - start_time
    after = {name: _read_metric(base_url, name) for name in metric_names}

    prompt_tokens = _metric_delta(before, after, metric_names[0])
    generated_tokens = _metric_delta(before, after, metric_names[1])
    draft_tokens = _metric_delta(before, after, metric_names[2])
    accepted_tokens = _metric_delta(before, after, metric_names[3])
    if spec.name == "deepseek_v32_mtp":
        if draft_tokens is None or accepted_tokens is None:
            pytest.fail(
                "MTP metrics are unavailable; verify that the server exposes "
                "speculative_num_draft_tokens_total and "
                "speculative_num_accepted_tokens_total"
            )
        if draft_tokens <= 0 or not 0 <= accepted_tokens <= draft_tokens:
            pytest.fail(
                "MTP speculative counters are invalid: "
                f"draft={draft_tokens}, accepted={accepted_tokens}"
            )
    report: dict[str, Any] = {
        "model_variant": spec.name,
        "requests": request_count,
        "batch_size": batch_size,
        "prompt_chars": prompt_chars,
        "max_tokens": max_tokens,
        "duration_seconds": duration_seconds,
        "request_latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": sorted(latencies)[len(latencies) // 2],
            "max": max(latencies),
        },
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prompt_tokens_per_second": (
            prompt_tokens / duration_seconds if prompt_tokens is not None else None
        ),
        "generated_tokens_per_second": (
            generated_tokens / duration_seconds
            if generated_tokens is not None
            else None
        ),
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance_rate": (
            accepted_tokens / draft_tokens
            if draft_tokens is not None and draft_tokens > 0
            else None
        ),
    }
    print(json.dumps(report, indent=2))
    _write_report(spec, report)
    assert len(latencies) == request_count
