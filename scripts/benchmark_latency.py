#!/usr/bin/env python3
"""Reproducible latency benchmark for /query and /query/stream endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests


SAMPLE_QUERY_SETS: dict[str, list[str]] = {
    "simple_extractive": [
        "What are the three query modes supported by this system?",
        "Which mode is the baseline retrieval workflow?",
        "What is the default max loop count for advanced mode?",
    ],
    "normal": [
        "Compare standard mode and advanced mode in terms of cost and reliability.",
        "When should the assistant abstain due to insufficient evidence?",
        "What are key differences between standard and compare mode outputs?",
    ],
    "complex": [
        "Given conflicting sources and weak evidence, explain how advanced mode should decide retry, refine, or abstain.",
        "Analyze trade-offs between retrieval depth, rerank cost, and groundedness checks in compare mode.",
        "If query rewrite changes intent, describe safe fallback behavior and stop conditions.",
    ],
}


def clamp_non_negative_int(value: Any, default: int = 0) -> int:
    """Convert value to non-negative integer."""
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def elapsed_ms(started: float) -> int:
    """Return elapsed milliseconds from perf_counter start."""
    return clamp_non_negative_int((time.perf_counter() - started) * 1000, 0)


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile where p is in [0, 1]."""
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 1:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    """Compute p50/p90/p95/max summary for one metric series."""
    if not values:
        return {
            "count": 0,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "p50_ms": round(percentile(values, 0.50) or 0.0, 2),
        "p90_ms": round(percentile(values, 0.90) or 0.0, 2),
        "p95_ms": round(percentile(values, 0.95) or 0.0, 2),
        "max_ms": round(max(values), 2),
    }


@dataclass(frozen=True)
class QueryCase:
    """One benchmark query case."""

    query_id: str
    query: str
    query_set: str


@dataclass
class AttemptResult:
    """One measured benchmark attempt."""

    attempt_index: int
    run_index: int
    mode: str
    stream: bool
    query_id: str
    query_set: str
    query: str
    ok: bool
    status_code: int | None = None
    error: str | None = None
    client_latency_ms: int | None = None
    backend_latency_ms: int | None = None
    time_to_first_event_ms: int | None = None
    time_to_first_token_ms: int | None = None
    total_stream_ms: int | None = None


def _safe_error_message(message: str, limit: int = 240) -> str:
    collapsed = " ".join(str(message).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse SSE lines into `(event_name, payload)` tuples."""
    event_name = "message"
    data_lines: list[str] = []

    def _flush() -> tuple[str, dict[str, Any]] | None:
        if not data_lines:
            return None
        raw = "\n".join(data_lines)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return event_name, payload
        return None

    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            parsed = _flush()
            if parsed is not None:
                yield parsed
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", maxsplit=1)[1].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", maxsplit=1)[1].lstrip())

    parsed = _flush()
    if parsed is not None:
        yield parsed


def _extract_backend_latency_ms(payload: dict[str, Any]) -> int | None:
    """Extract backend latency_ms when present across mode payload shapes."""
    latency_raw = payload.get("latency_ms")
    if isinstance(latency_raw, (int, float)):
        return clamp_non_negative_int(latency_raw, 0)

    mode = str(payload.get("mode") or "").strip().lower()
    if mode != "compare":
        return None

    for branch in ("standard", "advanced"):
        branch_payload = payload.get(branch)
        if not isinstance(branch_payload, dict):
            continue
        branch_latency = branch_payload.get("latency_ms")
        if isinstance(branch_latency, (int, float)):
            return clamp_non_negative_int(branch_latency, 0)
        trace = branch_payload.get("trace")
        if isinstance(trace, list):
            for step in reversed(trace):
                if not isinstance(step, dict):
                    continue
                compare_total = step.get("compare_total_ms")
                if isinstance(compare_total, (int, float)):
                    return clamp_non_negative_int(compare_total, 0)
    return None


def _load_query_cases_from_json_payload(payload: Any, source: str) -> list[QueryCase]:
    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        items = payload["queries"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unsupported query file format for {source}.")

    cases: list[QueryCase] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, str):
            query_text = item.strip()
            query_id = f"file_{idx:02d}"
        elif isinstance(item, dict):
            raw_query = item.get("query") or item.get("text")
            if not isinstance(raw_query, str):
                continue
            query_text = raw_query.strip()
            raw_id = item.get("id") or item.get("label")
            query_id = str(raw_id).strip() if raw_id else f"file_{idx:02d}"
        else:
            continue
        if not query_text:
            continue
        cases.append(QueryCase(query_id=query_id, query=query_text, query_set="file"))
    return cases


def load_query_cases(query_file: str | None, query_sets: list[str]) -> list[QueryCase]:
    """Load query cases from file or built-in sets."""
    if query_file:
        path = Path(query_file)
        if not path.exists():
            raise FileNotFoundError(f"Query file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            lines: list[Any] = []
            for raw in path.read_text(encoding="utf-8").splitlines():
                stripped = raw.strip()
                if not stripped:
                    continue
                lines.append(json.loads(stripped))
            return _load_query_cases_from_json_payload(lines, str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _load_query_cases_from_json_payload(payload, str(path))

    selected_sets = query_sets or ["simple_extractive", "normal", "complex"]
    cases: list[QueryCase] = []
    for set_name in selected_sets:
        queries = SAMPLE_QUERY_SETS.get(set_name, [])
        for idx, query in enumerate(queries, start=1):
            cases.append(
                QueryCase(
                    query_id=f"{set_name}_{idx:02d}",
                    query=query,
                    query_set=set_name,
                )
            )
    return cases


def _request_payload(case: QueryCase, mode: str, model: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": case.query,
        "mode": mode,
        "chat_history": [],
    }
    if model:
        payload["model"] = model
    return payload


def _run_non_stream_attempt(
    *,
    case: QueryCase,
    api_base_url: str,
    mode: str,
    timeout_s: float,
    run_index: int,
    attempt_index: int,
    model: str | None,
) -> AttemptResult:
    started = time.perf_counter()
    url = f"{api_base_url.rstrip('/')}/query"
    payload = _request_payload(case, mode, model)
    try:
        response = requests.post(url, json=payload, timeout=(5.0, timeout_s))
    except requests.RequestException as exc:
        return AttemptResult(
            attempt_index=attempt_index,
            run_index=run_index,
            mode=mode,
            stream=False,
            query_id=case.query_id,
            query_set=case.query_set,
            query=case.query,
            ok=False,
            error=_safe_error_message(str(exc)),
            client_latency_ms=elapsed_ms(started),
        )

    client_ms = elapsed_ms(started)
    status_code = int(response.status_code)
    if status_code >= 400:
        message = response.text
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = body.get("detail")
                if isinstance(detail, str):
                    message = detail
        except ValueError:
            pass
        return AttemptResult(
            attempt_index=attempt_index,
            run_index=run_index,
            mode=mode,
            stream=False,
            query_id=case.query_id,
            query_set=case.query_set,
            query=case.query,
            ok=False,
            status_code=status_code,
            error=_safe_error_message(message),
            client_latency_ms=client_ms,
        )

    try:
        body = response.json()
    except ValueError:
        return AttemptResult(
            attempt_index=attempt_index,
            run_index=run_index,
            mode=mode,
            stream=False,
            query_id=case.query_id,
            query_set=case.query_set,
            query=case.query,
            ok=False,
            status_code=status_code,
            error="Invalid JSON response from /query endpoint.",
            client_latency_ms=client_ms,
        )

    if not isinstance(body, dict):
        return AttemptResult(
            attempt_index=attempt_index,
            run_index=run_index,
            mode=mode,
            stream=False,
            query_id=case.query_id,
            query_set=case.query_set,
            query=case.query,
            ok=False,
            status_code=status_code,
            error="Unexpected /query response shape.",
            client_latency_ms=client_ms,
        )

    backend_ms = _extract_backend_latency_ms(body)
    return AttemptResult(
        attempt_index=attempt_index,
        run_index=run_index,
        mode=mode,
        stream=False,
        query_id=case.query_id,
        query_set=case.query_set,
        query=case.query,
        ok=True,
        status_code=status_code,
        client_latency_ms=client_ms,
        backend_latency_ms=backend_ms,
    )


def _run_stream_attempt(
    *,
    case: QueryCase,
    api_base_url: str,
    mode: str,
    timeout_s: float,
    run_index: int,
    attempt_index: int,
    model: str | None,
) -> AttemptResult:
    started = time.perf_counter()
    url = f"{api_base_url.rstrip('/')}/query/stream"
    payload = _request_payload(case, mode, model)
    try:
        response = requests.post(
            url, json=payload, stream=True, timeout=(5.0, timeout_s)
        )
    except requests.RequestException as exc:
        return AttemptResult(
            attempt_index=attempt_index,
            run_index=run_index,
            mode=mode,
            stream=True,
            query_id=case.query_id,
            query_set=case.query_set,
            query=case.query,
            ok=False,
            error=_safe_error_message(str(exc)),
            client_latency_ms=elapsed_ms(started),
            total_stream_ms=elapsed_ms(started),
        )

    first_event_ms: int | None = None
    first_token_ms: int | None = None
    final_event_payload: dict[str, Any] | None = None
    stream_error: str | None = None
    status_code = int(response.status_code)

    try:
        if status_code >= 400:
            message = response.text
            try:
                body = response.json()
                if isinstance(body, dict):
                    detail = body.get("detail")
                    if isinstance(detail, str):
                        message = detail
            except ValueError:
                pass
            return AttemptResult(
                attempt_index=attempt_index,
                run_index=run_index,
                mode=mode,
                stream=True,
                query_id=case.query_id,
                query_set=case.query_set,
                query=case.query,
                ok=False,
                status_code=status_code,
                error=_safe_error_message(message),
                client_latency_ms=elapsed_ms(started),
                total_stream_ms=elapsed_ms(started),
            )

        for event_name, event_payload in _iter_sse_events(
            response.iter_lines(decode_unicode=True)
        ):
            event_elapsed = elapsed_ms(started)
            if first_event_ms is None:
                first_event_ms = event_elapsed
            if event_name in {"generation_delta", "generation"}:
                delta_raw = event_payload.get("delta")
                if isinstance(delta_raw, str) and delta_raw and first_token_ms is None:
                    first_token_ms = event_elapsed
            if event_name == "error":
                err_payload = event_payload.get("error")
                if isinstance(err_payload, dict):
                    message = err_payload.get("message")
                    if isinstance(message, str):
                        stream_error = message
                    else:
                        stream_error = "Stream returned error event."
                else:
                    stream_error = "Stream returned error event."
                break
            if event_name == "final":
                final_event_payload = event_payload
    finally:
        response.close()

    total_stream_ms = elapsed_ms(started)
    if stream_error is not None:
        return AttemptResult(
            attempt_index=attempt_index,
            run_index=run_index,
            mode=mode,
            stream=True,
            query_id=case.query_id,
            query_set=case.query_set,
            query=case.query,
            ok=False,
            status_code=status_code,
            error=_safe_error_message(stream_error),
            client_latency_ms=total_stream_ms,
            time_to_first_event_ms=first_event_ms,
            time_to_first_token_ms=first_token_ms,
            total_stream_ms=total_stream_ms,
        )

    if final_event_payload is None:
        return AttemptResult(
            attempt_index=attempt_index,
            run_index=run_index,
            mode=mode,
            stream=True,
            query_id=case.query_id,
            query_set=case.query_set,
            query=case.query,
            ok=False,
            status_code=status_code,
            error="Stream ended without final event.",
            client_latency_ms=total_stream_ms,
            time_to_first_event_ms=first_event_ms,
            time_to_first_token_ms=first_token_ms,
            total_stream_ms=total_stream_ms,
        )

    raw_response = final_event_payload.get("response")
    backend_ms_raw = final_event_payload.get("latency_ms")
    backend_latency_ms: int | None = None
    if isinstance(backend_ms_raw, (int, float)):
        backend_latency_ms = clamp_non_negative_int(backend_ms_raw, 0)
    if backend_latency_ms is None and isinstance(raw_response, dict):
        backend_latency_ms = _extract_backend_latency_ms(raw_response)

    return AttemptResult(
        attempt_index=attempt_index,
        run_index=run_index,
        mode=mode,
        stream=True,
        query_id=case.query_id,
        query_set=case.query_set,
        query=case.query,
        ok=True,
        status_code=status_code,
        client_latency_ms=total_stream_ms,
        backend_latency_ms=backend_latency_ms,
        time_to_first_event_ms=first_event_ms,
        time_to_first_token_ms=first_token_ms,
        total_stream_ms=total_stream_ms,
    )


def _run_attempt(
    *,
    case: QueryCase,
    api_base_url: str,
    mode: str,
    timeout_s: float,
    run_index: int,
    attempt_index: int,
    stream: bool,
    model: str | None,
) -> AttemptResult:
    if stream:
        return _run_stream_attempt(
            case=case,
            api_base_url=api_base_url,
            mode=mode,
            timeout_s=timeout_s,
            run_index=run_index,
            attempt_index=attempt_index,
            model=model,
        )
    return _run_non_stream_attempt(
        case=case,
        api_base_url=api_base_url,
        mode=mode,
        timeout_s=timeout_s,
        run_index=run_index,
        attempt_index=attempt_index,
        model=model,
    )


def execute_attempts(
    *,
    cases: list[QueryCase],
    api_base_url: str,
    mode: str,
    runs: int,
    concurrency: int,
    stream: bool,
    timeout_s: float,
    model: str | None,
) -> list[AttemptResult]:
    """Execute measured attempts and return ordered results."""
    jobs: list[tuple[int, int, QueryCase]] = []
    attempt_index = 0
    for run_index in range(1, runs + 1):
        for case in cases:
            attempt_index += 1
            jobs.append((attempt_index, run_index, case))

    if not jobs:
        return []

    if concurrency <= 1:
        results = [
            _run_attempt(
                case=case,
                api_base_url=api_base_url,
                mode=mode,
                timeout_s=timeout_s,
                run_index=run_index,
                attempt_index=attempt_idx,
                stream=stream,
                model=model,
            )
            for attempt_idx, run_index, case in jobs
        ]
        return results

    results_by_attempt: dict[int, AttemptResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_attempt: dict[concurrent.futures.Future[AttemptResult], int] = {}
        for attempt_idx, run_index, case in jobs:
            future = executor.submit(
                _run_attempt,
                case=case,
                api_base_url=api_base_url,
                mode=mode,
                timeout_s=timeout_s,
                run_index=run_index,
                attempt_index=attempt_idx,
                stream=stream,
                model=model,
            )
            future_to_attempt[future] = attempt_idx

        for future in concurrent.futures.as_completed(future_to_attempt):
            attempt_idx = future_to_attempt[future]
            try:
                result = future.result()
            except Exception as exc:
                result = AttemptResult(
                    attempt_index=attempt_idx,
                    run_index=0,
                    mode=mode,
                    stream=stream,
                    query_id="unknown",
                    query_set="unknown",
                    query="",
                    ok=False,
                    error=f"Unexpected benchmark worker error: {_safe_error_message(str(exc))}",
                )
            results_by_attempt[attempt_idx] = result

    return [results_by_attempt[idx] for idx, _, _ in jobs]


def warmup(
    *,
    cases: list[QueryCase],
    api_base_url: str,
    mode: str,
    warmup_runs: int,
    concurrency: int,
    stream: bool,
    timeout_s: float,
    model: str | None,
) -> None:
    """Run warmup requests (best effort, results ignored for metrics)."""
    if warmup_runs <= 0 or not cases:
        return
    print(f"Warmup: {warmup_runs} run(s) x {len(cases)} query(ies)")
    _ = execute_attempts(
        cases=cases,
        api_base_url=api_base_url,
        mode=mode,
        runs=warmup_runs,
        concurrency=concurrency,
        stream=stream,
        timeout_s=timeout_s,
        model=model,
    )


def summarize_results(results: list[AttemptResult]) -> dict[str, Any]:
    """Build benchmark summary payload."""
    successes = [item for item in results if item.ok]
    failures = [item for item in results if not item.ok]

    client_values = [
        float(item.client_latency_ms)
        for item in successes
        if item.client_latency_ms is not None
    ]
    backend_values = [
        float(item.backend_latency_ms)
        for item in successes
        if item.backend_latency_ms is not None
    ]
    first_event_values = [
        float(item.time_to_first_event_ms)
        for item in successes
        if item.time_to_first_event_ms is not None
    ]
    first_token_values = [
        float(item.time_to_first_token_ms)
        for item in successes
        if item.time_to_first_token_ms is not None
    ]
    stream_total_values = [
        float(item.total_stream_ms)
        for item in successes
        if item.total_stream_ms is not None
    ]

    by_query: dict[str, list[AttemptResult]] = {}
    for item in results:
        by_query.setdefault(item.query_id, []).append(item)

    per_query_summary: dict[str, Any] = {}
    for query_id, items in by_query.items():
        item_successes = [entry for entry in items if entry.ok]
        query_latencies = [
            float(entry.client_latency_ms)
            for entry in item_successes
            if entry.client_latency_ms is not None
        ]
        per_query_summary[query_id] = {
            "query_set": items[0].query_set if items else "unknown",
            "query": items[0].query if items else "",
            "attempts": len(items),
            "successes": len(item_successes),
            "failures": len(items) - len(item_successes),
            "client_latency_ms": summarize_values(query_latencies),
        }

    return {
        "attempts": len(results),
        "successes": len(successes),
        "failures": len(failures),
        "client_latency_ms": summarize_values(client_values),
        "backend_latency_ms": summarize_values(backend_values),
        "time_to_first_event_ms": summarize_values(first_event_values),
        "time_to_first_token_ms": summarize_values(first_token_values),
        "total_stream_ms": summarize_values(stream_total_values),
        "per_query": per_query_summary,
    }


def _print_attempts(results: list[AttemptResult]) -> None:
    print("\nMeasured results:")
    for item in results:
        prefix = "[ok]" if item.ok else "[err]"
        message = (
            f"{prefix} #{item.attempt_index:03d} "
            f"query={item.query_id} run={item.run_index} "
            f"client_ms={item.client_latency_ms} backend_ms={item.backend_latency_ms}"
        )
        if item.stream:
            message += (
                f" ttf_event_ms={item.time_to_first_event_ms}"
                f" ttf_token_ms={item.time_to_first_token_ms}"
                f" stream_total_ms={item.total_stream_ms}"
            )
        if not item.ok:
            message += f" error={item.error}"
        print(message)


def _print_summary(summary: dict[str, Any], *, stream: bool) -> None:
    def _fmt(metric: dict[str, Any]) -> str:
        def _value(field: str) -> str:
            value = metric.get(field)
            return "n/a" if value is None else f"{value}ms"

        return (
            f"count={metric['count']} "
            f"p50={_value('p50_ms')} "
            f"p90={_value('p90_ms')} "
            f"p95={_value('p95_ms')} "
            f"max={_value('max_ms')}"
        )

    print("\nSummary:")
    print(
        f"attempts={summary['attempts']} successes={summary['successes']} failures={summary['failures']}"
    )
    print(f"client_latency_ms: {_fmt(summary['client_latency_ms'])}")
    print(f"backend_latency_ms: {_fmt(summary['backend_latency_ms'])}")
    if stream:
        print(f"time_to_first_event_ms: {_fmt(summary['time_to_first_event_ms'])}")
        print(f"time_to_first_token_ms: {_fmt(summary['time_to_first_token_ms'])}")
        print(f"total_stream_ms: {_fmt(summary['total_stream_ms'])}")

    print("\nPer-query summary:")
    per_query: dict[str, Any] = summary.get("per_query", {})
    for query_id in sorted(per_query.keys()):
        item = per_query[query_id]
        metric = item["client_latency_ms"]
        print(
            f"- {query_id} ({item['query_set']}): successes={item['successes']}/{item['attempts']} "
            f"p50={'n/a' if metric['p50_ms'] is None else str(metric['p50_ms']) + 'ms'} "
            f"p95={'n/a' if metric['p95_ms'] is None else str(metric['p95_ms']) + 'ms'} "
            f"max={'n/a' if metric['max_ms'] is None else str(metric['max_ms']) + 'ms'}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latency benchmark for /query and /query/stream."
    )
    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="Base API URL (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "advanced", "compare"],
        default="standard",
        help="Workflow mode to benchmark.",
    )
    parser.add_argument(
        "--query-file",
        default=None,
        help="Path to query JSON/JSONL file.",
    )
    parser.add_argument(
        "--query-set",
        action="append",
        choices=["simple_extractive", "normal", "complex"],
        help="Built-in sample query set (repeat flag to include multiple sets).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Benchmark streaming endpoint /query/stream instead of /query.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests for measured runs.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup run count (not included in summary).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured run count.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Read timeout seconds per request.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override to pass in query payload.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output file to store raw results and summary as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    concurrency = max(1, int(args.concurrency))
    runs = max(1, int(args.runs))
    warmup_runs = max(0, int(args.warmup))
    timeout_s = max(5.0, float(args.timeout))

    try:
        cases = load_query_cases(args.query_file, args.query_set or [])
    except Exception as exc:
        print(f"Failed to load queries: {exc}")
        return 0

    if not cases:
        print("No queries available to benchmark.")
        return 0

    print("Latency benchmark configuration:")
    print(f"- api_base_url: {args.api_base_url}")
    print(f"- mode: {args.mode}")
    print(f"- stream: {args.stream}")
    print(f"- queries: {len(cases)}")
    print(f"- warmup runs: {warmup_runs}")
    print(f"- measured runs: {runs}")
    print(f"- concurrency: {concurrency}")
    print(f"- timeout_s: {timeout_s}")
    if args.model:
        print(f"- model override: {args.model}")

    warmup(
        cases=cases,
        api_base_url=args.api_base_url,
        mode=args.mode,
        warmup_runs=warmup_runs,
        concurrency=concurrency,
        stream=args.stream,
        timeout_s=timeout_s,
        model=args.model,
    )

    started = time.perf_counter()
    results = execute_attempts(
        cases=cases,
        api_base_url=args.api_base_url,
        mode=args.mode,
        runs=runs,
        concurrency=concurrency,
        stream=args.stream,
        timeout_s=timeout_s,
        model=args.model,
    )
    benchmark_total_ms = elapsed_ms(started)

    _print_attempts(results)
    summary = summarize_results(results)
    _print_summary(summary, stream=args.stream)
    print(f"\nBenchmark total wall-clock: {benchmark_total_ms}ms")

    if summary["successes"] == 0:
        print(
            "\nNo successful requests. Check backend availability and API base URL, "
            "then retry. For stable numbers, run backend without --reload."
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "api_base_url": args.api_base_url,
                "mode": args.mode,
                "stream": bool(args.stream),
                "query_file": args.query_file,
                "query_sets": args.query_set or [],
                "runs": runs,
                "warmup": warmup_runs,
                "concurrency": concurrency,
                "timeout_s": timeout_s,
                "model": args.model,
            },
            "samples": [asdict(case) for case in cases],
            "results": [asdict(item) for item in results],
            "summary": summary,
            "benchmark_total_ms": benchmark_total_ms,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved benchmark output: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
