"""Unit tests for scripts/benchmark_latency.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import requests


def _load_module():
    script_path = Path("scripts/benchmark_latency.py")
    module_name = "benchmark_latency_script"
    spec = importlib.util.spec_from_file_location(
        module_name, script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_percentile_and_summary_calculation() -> None:
    module = _load_module()

    values = [100.0, 200.0, 300.0, 400.0]
    assert module.percentile(values, 0.5) == 250.0
    assert round(module.percentile(values, 0.9), 2) == 370.0
    assert round(module.percentile(values, 0.95), 2) == 385.0

    summary = module.summarize_values(values)
    assert summary["count"] == 4
    assert summary["p50_ms"] == 250.0
    assert summary["p90_ms"] == 370.0
    assert summary["p95_ms"] == 385.0
    assert summary["max_ms"] == 400.0


def test_extract_backend_latency_supports_compare_trace_shape() -> None:
    module = _load_module()

    payload = {
        "mode": "compare",
        "standard": {
            "trace": [
                {"step": "timing_summary", "total_ms": 50},
                {"step": "compare_timing", "compare_total_ms": 123},
            ]
        },
        "advanced": {"trace": []},
    }

    assert module._extract_backend_latency_ms(payload) == 123


def test_iter_sse_events_parses_frames() -> None:
    module = _load_module()
    lines = [
        "event: start",
        'data: {"type":"start"}',
        "",
        "event: generation_delta",
        'data: {"type":"generation_delta","delta":"hello"}',
        "",
        "event: final",
        'data: {"type":"final","response":{"mode":"standard","latency_ms":99}}',
        "",
    ]

    events = list(module._iter_sse_events(lines))
    assert events[0][0] == "start"
    assert events[1][0] == "generation_delta"
    assert events[2][0] == "final"
    assert events[2][1]["response"]["latency_ms"] == 99


def test_run_non_stream_attempt_with_mock_response(monkeypatch) -> None:
    module = _load_module()

    class _Response:
        status_code = 200
        text = ""

        def json(self):
            return {
                "mode": "standard",
                "answer": "ok",
                "citations": [],
                "status": "answered",
                "latency_ms": 77,
                "trace": [],
            }

    def _fake_post(*args, **kwargs):
        _ = args
        _ = kwargs
        return _Response()

    monkeypatch.setattr(module.requests, "post", _fake_post)
    case = module.QueryCase(query_id="q1", query="test", query_set="normal")

    result = module._run_non_stream_attempt(
        case=case,
        api_base_url="http://127.0.0.1:8000/api/v1",
        mode="standard",
        timeout_s=10.0,
        run_index=1,
        attempt_index=1,
        model=None,
    )

    assert result.ok is True
    assert result.backend_latency_ms == 77
    assert isinstance(result.client_latency_ms, int)
    assert result.client_latency_ms >= 0


def test_run_non_stream_attempt_handles_connection_error(monkeypatch) -> None:
    module = _load_module()

    def _fake_post(*args, **kwargs):
        _ = args
        _ = kwargs
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(module.requests, "post", _fake_post)
    case = module.QueryCase(query_id="q1", query="test", query_set="normal")

    result = module._run_non_stream_attempt(
        case=case,
        api_base_url="http://127.0.0.1:8000/api/v1",
        mode="standard",
        timeout_s=10.0,
        run_index=1,
        attempt_index=1,
        model=None,
    )

    assert result.ok is False
    assert "connection refused" in (result.error or "")
