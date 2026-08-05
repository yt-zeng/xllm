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

"""EvalScope integration test for the MTP acceptance rate.

Run against an already-started xLLM server with:

  XLLM_MTP_API_URL=http://127.0.0.1:8199/v1 \
  XLLM_MTP_MODEL=Qwen3-4B \
  pytest -q tests/python/test_mtp_acceptance_rate.py

The test uses a fixed GSM8K JSONL fixture by default. Set
``XLLM_MTP_REQUESTS_FILE`` to use another fixed request file, or set
``XLLM_MTP_USE_EVALSCOPE=1`` to use the legacy EvalScope path. Fixed requests
are sent in file order, in sequential batches of
``XLLM_MTP_EVAL_BATCH_SIZE``; the fixed-request default is batch size 1 and
the next batch is not submitted until the previous batch completes. Fixed
requests also ignore EOS by default so ``max_tokens=1024`` measures the full
1K output workload. ``XLLM_MTP_EVAL_LIMIT`` controls the number of fixed
requests. If a custom fixture has one request and the limit is larger, that
request is repeated with unique request IDs to create a deterministic
multi-request workload.

The server must be started with ``--draft_model`` and
``--num_speculative_tokens > 0``. The acceptance rate is calculated from
the existing cumulative accepted/draft token metrics.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest


def _read_metric(base_url: str, metric_name: str) -> float:
    metric_url = f"{base_url.rstrip('/')}/vars/{quote(metric_name, safe='')}"
    with urlopen(metric_url, timeout=10) as response:
        metric = response.read().decode("utf-8")
    try:
        return float(metric.strip())
    except ValueError:
        pass
    pattern = rf"{re.escape(metric_name)}\s*:\s*(?:<[^>]+>)*"
    match = re.search(pattern + r"([-+]?\d+(?:\.\d+)?)", metric)
    if match:
        return float(match.group(1))
    raise AssertionError(f"metric {metric_name!r} was not found in /vars")


def _read_optional_metric(
    base_url: str, metric_name: str
) -> float | None:
    try:
        return _read_metric(base_url, metric_name)
    except (AssertionError, OSError, ValueError):
        return None


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _print_benchmark_result(
    successful_requests: int,
    failed_requests: int,
    duration_seconds: float,
    input_tokens: float,
    generated_tokens: float,
    ttft_mean: float | None,
    ttft_p99: float | None,
    tpot_mean: float | None,
    tpot_p99: float | None,
    itl_mean: float | None,
    itl_p99: float | None,
    peak_output_throughput: float,
    peak_concurrent_requests: float,
    draft_tokens: float,
    accepted_tokens: float,
) -> None:
    """Print the MTP result in the standard serving benchmark format."""
    acceptance_rate = accepted_tokens / draft_tokens * 100.0
    speculative_tokens = int(
        os.getenv("XLLM_MTP_NUM_SPECULATIVE_TOKENS", "1")
    )
    if speculative_tokens <= 0:
        raise ValueError("XLLM_MTP_NUM_SPECULATIVE_TOKENS must be positive")
    drafts = draft_tokens / speculative_tokens
    acceptance_length = 1.0 + accepted_tokens / drafts

    print("============ Serving Benchmark Result ============")
    print(f"Successful requests:                     {successful_requests}")
    print(f"Failed requests:                         {failed_requests}")
    print("Request rate configured (RPS):           N/A")
    print(f"Benchmark duration (s):                  {duration_seconds:.2f}")
    print(f"Total input tokens:                      {input_tokens:.0f}")
    print(f"Total generated tokens:                  {generated_tokens:.0f}")
    print(
        "Request throughput (req/s):              "
        f"{successful_requests / duration_seconds:.2f}"
    )
    print(
        "Output token throughput (tok/s):         "
        f"{generated_tokens / duration_seconds:.2f}"
    )
    print(
        "Peak output token throughput (tok/s):    "
        f"{peak_output_throughput:.2f}"
    )
    print(f"Peak concurrent requests:                {peak_concurrent_requests:.2f}")
    print(
        "Total token throughput (tok/s):          "
        f"{(input_tokens + generated_tokens) / duration_seconds:.2f}"
    )
    print("---------------Time to First Token----------------")
    print(f"Mean TTFT (ms):                          {_format_metric(ttft_mean)}")
    print("Median TTFT (ms):                        N/A")
    print(f"P99 TTFT (ms):                           {_format_metric(ttft_p99)}")
    print("-----Time per Output Token (excl. 1st token)------")
    print(f"Mean TPOT (ms):                          {_format_metric(tpot_mean)}")
    print("Median TPOT (ms):                        N/A")
    print(f"P99 TPOT (ms):                           {_format_metric(tpot_p99)}")
    print("---------------Inter-token Latency----------------")
    print(f"Mean ITL (ms):                           {_format_metric(itl_mean)}")
    print("Median ITL (ms):                         N/A")
    print(f"P99 ITL (ms):                            {_format_metric(itl_p99)}")
    print("---------------Speculative Decoding---------------")
    print(f"Acceptance rate (%):                     {acceptance_rate:.2f}")
    print(f"Acceptance length:                       {acceptance_length:.2f}")
    print(f"Drafts:                                  {drafts:.0f}")
    print(f"Draft tokens:                            {draft_tokens:.0f}")
    print(f"Accepted tokens:                         {accepted_tokens:.0f}")
    print("Per-position acceptance (%):             N/A")
    print("==================================================")


def _sample_peak_metrics(
    base_url: str,
    stop_event: threading.Event,
    peak_metrics: dict[str, float],
) -> None:
    """Sample runtime metrics while benchmark requests are running."""
    previous_time = time.monotonic()
    try:
        previous_tokens = _read_metric(
            base_url, "num_processing_tokens_total_generated"
        )
    except (AssertionError, HTTPError, OSError, TimeoutError, URLError, ValueError):
        return

    while not stop_event.wait(0.1):
        current_time = time.monotonic()
        try:
            current_tokens = _read_metric(
                base_url, "num_processing_tokens_total_generated"
            )
            current_concurrent = _read_metric(
                base_url, "num_concurrent_requests"
            )
        except (AssertionError, OSError, ValueError):
            continue

        elapsed = current_time - previous_time
        if elapsed > 0.0:
            peak_metrics["output_throughput"] = max(
                peak_metrics["output_throughput"],
                (current_tokens - previous_tokens) / elapsed,
            )
        peak_metrics["concurrent_requests"] = max(
            peak_metrics["concurrent_requests"], current_concurrent
        )
        previous_time = current_time
        previous_tokens = current_tokens


def _load_fixed_requests(
    request_file: str | None,
) -> list[dict[str, object]] | None:
    if not request_file:
        return None

    requests: list[dict[str, object]] = []
    with open(request_file, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {request_file}:{line_number}"
                ) from error
            if not isinstance(request, dict):
                raise ValueError(
                    f"request in {request_file}:{line_number} must be an object"
                )
            if "prompt" not in request and "messages" not in request:
                raise ValueError(
                    f"request in {request_file}:{line_number} must contain "
                    "'prompt' or 'messages'"
                )
            prompt_repeat_to_chars = request.pop("prompt_repeat_to_chars", None)
            if prompt_repeat_to_chars is not None:
                prompt = request.get("prompt")
                if not isinstance(prompt, str):
                    raise ValueError(
                        f"request in {request_file}:{line_number} with "
                        "prompt_repeat_to_chars must contain a string prompt"
                    )
                if not isinstance(prompt_repeat_to_chars, int):
                    raise ValueError(
                        f"request in {request_file}:{line_number} has an "
                        "invalid prompt_repeat_to_chars value"
                    )
                if prompt_repeat_to_chars <= 0 or not prompt:
                    raise ValueError(
                        f"request in {request_file}:{line_number} has an "
                        "invalid prompt length configuration"
                    )
                repeat_count = (
                    prompt_repeat_to_chars + len(prompt) - 1
                ) // len(prompt)
                request["prompt"] = (prompt * repeat_count)[
                    :prompt_repeat_to_chars
                ]
            requests.append(request)

    if not requests:
        raise ValueError(f"no requests found in {request_file}")
    return requests


def _limit_fixed_requests(
    requests: list[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    if limit <= 0:
        raise ValueError("XLLM_MTP_EVAL_LIMIT must be positive")
    if len(requests) >= limit:
        return requests[:limit]
    if len(requests) != 1:
        raise ValueError(
            "XLLM_MTP_EVAL_LIMIT exceeds the number of fixed requests; "
            "provide enough requests in XLLM_MTP_REQUESTS_FILE"
        )

    request = requests[0]
    request_id = str(request.get("id", "fixed-request"))
    return [
        dict(request, id=f"{request_id}-{request_index:05d}")
        for request_index in range(limit)
    ]


def _send_fixed_request(
    endpoint: str,
    model: str,
    request_body: dict[str, object],
    timeout_seconds: float,
) -> None:
    payload = dict(request_body)
    payload["model"] = model
    payload.setdefault("temperature", 0.0)
    payload.setdefault("top_p", 1.0)
    payload.setdefault("do_sample", False)
    payload.setdefault("seed", 42)
    payload.setdefault("ignore_eos", True)
    payload.setdefault("stream", False)
    payload.setdefault("max_tokens", 256)
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        response.read()


def _run_fixed_requests(
    api_url: str,
    model: str,
    requests: list[dict[str, object]],
    batch_size: int,
    timeout_seconds: float,
) -> None:
    """Run a fixed request file in deterministic sequential batches.

    Waiting for each batch before submitting the next one makes the request
    admission pattern reproducible across Python and native server runs.
    """
    if batch_size <= 0:
        raise ValueError("XLLM_MTP_EVAL_BATCH_SIZE must be positive")

    endpoint = f"{api_url.rstrip('/')}/completions"
    for batch_start in range(0, len(requests), batch_size):
        batch = requests[batch_start : batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(
                    _send_fixed_request,
                    endpoint,
                    model,
                    request_body,
                    timeout_seconds,
                )
                for request_body in batch
            ]
            for request_index, future in enumerate(futures, start=batch_start):
                try:
                    future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"fixed MTP request {request_index} failed"
                    ) from error


def test_mtp_acceptance_rate_after_evalscope() -> None:
    api_url = os.getenv("XLLM_MTP_API_URL")
    model = os.getenv("XLLM_MTP_MODEL")
    if not api_url or not model:
        pytest.skip("set XLLM_MTP_API_URL and XLLM_MTP_MODEL to run MTP integration")
    use_evalscope = os.getenv("XLLM_MTP_USE_EVALSCOPE", "0") == "1"
    request_file = os.getenv("XLLM_MTP_REQUESTS_FILE")
    if request_file is None and not use_evalscope:
        request_file = str(
            Path(__file__).resolve().parent / "data" / "gsm8k_fixed_requests.jsonl"
        )
    fixed_requests = _load_fixed_requests(request_file)
    if fixed_requests is not None:
        fixed_requests = _limit_fixed_requests(
            fixed_requests,
            int(os.getenv("XLLM_MTP_EVAL_LIMIT", "10")),
        )
    if fixed_requests is None and shutil.which("evalscope") is None:
        pytest.skip("evalscope is not installed")

    api_parts = urlsplit(api_url)
    base_url = f"{api_parts.scheme}://{api_parts.netloc}"
    evalscope_command = None
    if fixed_requests is None:
        work_dir = os.getenv("XLLM_MTP_EVAL_WORK_DIR", "/tmp/evalscope_mtp")
        evalscope_command = [
            "evalscope",
            "eval",
            "--model",
            model,
            "--api-url",
            api_url,
            "--api-key",
            os.getenv("XLLM_MTP_API_KEY", "dummy"),
            "--datasets",
            os.getenv("XLLM_MTP_DATASET", "gsm8k"),
            "--limit",
            os.getenv("XLLM_MTP_EVAL_LIMIT", "10"),
            "--eval-batch-size",
            os.getenv("XLLM_MTP_EVAL_BATCH_SIZE", "4"),
            "--timeout",
            os.getenv("XLLM_MTP_EVAL_TIMEOUT", "180"),
            "--work-dir",
            work_dir,
            "--generation-config",
            os.getenv(
                "XLLM_MTP_GENERATION_CONFIG",
                '{"temperature":0.0,"top_p":1.0,"do_sample":false,"seed":42}',
            ),
        ]

    counter_metric_names = [
        "server_request_total_ok",
        "server_request_total_fail",
        "num_processing_tokens_total_prompt",
        "num_processing_tokens_total_generated",
        "speculative_num_draft_tokens_total",
        "speculative_num_accepted_tokens_total",
    ]
    optional_metric_names = [
        "num_concurrent_requests",
        "time_to_first_token_latency_milliseconds_latency",
        "time_to_first_token_latency_milliseconds_latency_99",
        "speculative_per_token_latency_milliseconds_latency",
        "speculative_per_token_latency_milliseconds_latency_99",
        "inter_token_latency_milliseconds_latency",
        "inter_token_latency_milliseconds_latency_99",
    ]
    before = {
        metric_name: _read_metric(base_url, metric_name)
        for metric_name in counter_metric_names
    }
    before_concurrent = _read_optional_metric(
        base_url, "num_concurrent_requests"
    )
    peak_metrics = {
        "output_throughput": 0.0,
        "concurrent_requests": before_concurrent or 0.0,
    }
    stop_sampling = threading.Event()
    sampler = threading.Thread(
        target=_sample_peak_metrics,
        args=(base_url, stop_sampling, peak_metrics),
        daemon=True,
    )
    start_time = time.monotonic()
    sampler.start()
    try:
        if fixed_requests is not None:
            _run_fixed_requests(
                api_url,
                model,
                fixed_requests,
                int(os.getenv("XLLM_MTP_EVAL_BATCH_SIZE", "1")),
                float(os.getenv("XLLM_MTP_EVAL_TIMEOUT", "180")),
            )
            result = None
        else:
            assert evalscope_command is not None
            result = subprocess.run(
                evalscope_command,
                check=True,
                timeout=600,
                capture_output=True,
                text=True,
            )
    finally:
        stop_sampling.set()
        sampler.join(timeout=10)
    if result is not None and result.stdout:
        print(result.stdout, end="")
    if result is not None and result.stderr:
        print(result.stderr, end="")
    after = {
        metric_name: _read_metric(base_url, metric_name)
        for metric_name in counter_metric_names
    }
    after.update(
        {
            metric_name: _read_optional_metric(base_url, metric_name)
            for metric_name in optional_metric_names
        }
    )

    successful_requests = (
        after["server_request_total_ok"] - before["server_request_total_ok"]
    )
    failed_requests = (
        after["server_request_total_fail"]
        - before["server_request_total_fail"]
    )
    input_tokens = (
        after["num_processing_tokens_total_prompt"]
        - before["num_processing_tokens_total_prompt"]
    )
    generated_tokens = (
        after["num_processing_tokens_total_generated"]
        - before["num_processing_tokens_total_generated"]
    )
    draft_delta = (
        after["speculative_num_draft_tokens_total"]
        - before["speculative_num_draft_tokens_total"]
    )
    accepted_delta = (
        after["speculative_num_accepted_tokens_total"]
        - before["speculative_num_accepted_tokens_total"]
    )
    if fixed_requests is not None:
        assert successful_requests == len(fixed_requests)
        assert failed_requests == 0.0
    assert draft_delta > 0.0
    assert 0.0 <= accepted_delta <= draft_delta
    acceptance_rate = accepted_delta / draft_delta
    assert 0.0 <= acceptance_rate <= 1.0

    _print_benchmark_result(
        successful_requests=int(successful_requests),
        failed_requests=int(failed_requests),
        duration_seconds=time.monotonic() - start_time,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        ttft_mean=after.get(
            "time_to_first_token_latency_milliseconds_latency"
        ),
        ttft_p99=after.get(
            "time_to_first_token_latency_milliseconds_latency_99"
        ),
        tpot_mean=after.get(
            "speculative_per_token_latency_milliseconds_latency"
        ),
        tpot_p99=after.get(
            "speculative_per_token_latency_milliseconds_latency_99"
        ),
        itl_mean=after.get("inter_token_latency_milliseconds_latency"),
        itl_p99=after.get("inter_token_latency_milliseconds_latency_99"),
        peak_output_throughput=peak_metrics["output_throughput"],
        peak_concurrent_requests=peak_metrics["concurrent_requests"],
        draft_tokens=draft_delta,
        accepted_tokens=accepted_delta,
    )
