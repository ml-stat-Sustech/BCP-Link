from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Sequence
from urllib.parse import urldefrag

import openai
from tqdm import tqdm

from searcherkit.agent import SingleTurnAgent
from searcherkit.common.log import get_logger, setup_logger
from searcherkit.common.retry import RetryPolicy, retry_async
from searcherkit.llm.openai_client import OpenAIClient
from searcherkit.llm.parsers import UpstreamParser

logger = get_logger(__name__)

_ANSWER_PATTERN = re.compile(r"\\boxed\{(?P<answer>[^}]*)\}", re.DOTALL)
_MARKDOWN_LINK_PATTERN = re.compile(
    r"\[[^\]]*\]\((?P<url>https?://[^)\s]+)(?:\s+[^)]*)?\)"
)
_JUDGE_FIELDS = {
    "extracted_final_answer",
    "correct_answer",
    "reasoning",
    "correct",
    "confidence",
}
_SEARCH_TOOL_NAMES = {"search"}
_VISIT_TOOL_NAMES = {"visit", "browse"}


@dataclass(slots=True)
class JudgeConfig:
    model: str = "qwen3-32b"
    api_key: str | None = None
    base_url: str | None = None
    default_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 4096,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
                "enable_thinking": False,
            },
        }
    )


def _attempt_extract_answer_content(
    content: str,
    pattern: re.Pattern[str] = _ANSWER_PATTERN,
) -> str:
    matched = pattern.search(content)
    if matched is None:
        return content

    answer = matched.group("answer").strip()
    return answer or content


def build_agent(config: JudgeConfig | None = None) -> SingleTurnAgent:
    judge = config or JudgeConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    system_prompt = """Output exactly one valid JSON object and nothing else.

Schema:
{
  "extracted_final_answer": "string",
  "correct_answer": "string",
  "reasoning": "string",
  "correct": bool,
  "confidence": int from 0 to 100
}
"""
    return SingleTurnAgent(
        llm_client=OpenAIClient(
            model=judge.model,
            api_key=judge.api_key,
            base_url=judge.base_url,
            default_kwargs=judge.default_kwargs,
            max_retries=0,
            retry_policy=RetryPolicy(exceptions=(openai.RateLimitError,)),
        ),
        parser=UpstreamParser(),
        system_prompt=system_prompt,
    )


def _build_judge_prompt(question: str, response: str, correct_answer: Any) -> str:
    return """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Output true if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Output false otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip().format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )


def validate_judge_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("judge output must be a JSON object")
    if set(value) != _JUDGE_FIELDS:
        raise ValueError(f"judge output fields must be exactly {_JUDGE_FIELDS}")
    for field_name in ("extracted_final_answer", "correct_answer", "reasoning"):
        if not isinstance(value[field_name], str):
            raise ValueError(f"judge output field {field_name!r} must be a string")
    if not isinstance(value["correct"], bool):
        raise ValueError("judge output field 'correct' must be a boolean")
    confidence = value["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int)
        or not 0 <= confidence <= 100
    ):
        raise ValueError(
            "judge output field 'confidence' must be an integer from 0 to 100"
        )
    return value


def _normalize_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = urldefrag(value.strip())[0].rstrip("/")
    return normalized or None


def _document_aliases(document: Any) -> set[str]:
    if not isinstance(document, Mapping):
        return set()
    aliases: set[str] = set()
    for key in ("id", "docid"):
        value = document.get(key)
        if value is not None and str(value):
            aliases.add(f"id:{value}")
    url = _normalize_url(document.get("url"))
    if url is not None:
        aliases.add(f"url:{url}")
    return aliases


def _evidence_targets(extra: Any) -> tuple[list[set[str]], str]:
    if not isinstance(extra, Mapping) or "evidence_docs" not in extra:
        return [], "missing_evidence_docs"
    raw_documents = extra.get("evidence_docs")
    if not isinstance(raw_documents, list):
        return [], "invalid_evidence_docs"
    if not raw_documents:
        return [], "empty_evidence_docs"

    targets: list[set[str]] = []
    seen: set[tuple[str, ...]] = set()
    invalid = 0
    for document in raw_documents:
        aliases = _document_aliases(document)
        if not aliases:
            invalid += 1
            continue
        identity = tuple(sorted(aliases))
        if identity in seen:
            continue
        seen.add(identity)
        targets.append(aliases)
    if not targets:
        return [], "invalid_evidence_docs"
    return targets, "partial_evidence_docs" if invalid else "available"


def _extension_items(
    message: Mapping[str, Any],
    key: str,
    call_id: str,
) -> list[Any]:
    extensions = message.get("extensions")
    if not isinstance(extensions, Mapping):
        return []
    by_call = extensions.get(key)
    if isinstance(by_call, Mapping):
        value = by_call.get(call_id)
    else:
        value = by_call
    return value if isinstance(value, list) else []


def _markdown_urls(content: Any) -> set[str]:
    if not isinstance(content, str):
        return set()
    urls: set[str] = set()
    for matched in _MARKDOWN_LINK_PATTERN.finditer(content):
        url = _normalize_url(matched.group("url"))
        if url is not None:
            urls.add(url)
    return urls


def _visit_targets(tool_call: Mapping[str, Any]) -> set[str]:
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, Mapping):
        return set()
    targets: set[str] = set()
    for key in ("url", "document_id"):
        raw_value = arguments.get(key)
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            normalized = _normalize_url(value)
            if normalized is not None:
                targets.add(normalized)
    return targets


def _trajectory_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("trajectory history must be a list")

    returned_aliases: set[str] = set()
    visible_links: set[str] = set()
    pending_calls: dict[str, dict[str, Any]] = {}
    search_calls = 0
    visit_calls = 0
    link_following_visit_calls = 0
    turns = 0

    for message in history:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "assistant":
            turns += 1
            raw_calls = message.get("tool_calls")
            if not isinstance(raw_calls, list):
                continue
            for raw_call in raw_calls:
                if not isinstance(raw_call, Mapping):
                    continue
                name = str(raw_call.get("name") or "")
                call_id = str(raw_call.get("id") or "")
                counted_link = False
                visible_snapshot: set[str] = set()
                if name in _SEARCH_TOOL_NAMES:
                    search_calls += 1
                elif name in _VISIT_TOOL_NAMES:
                    visit_calls += 1
                    visible_snapshot = set(visible_links)
                    if _visit_targets(raw_call) & visible_snapshot:
                        link_following_visit_calls += 1
                        counted_link = True
                if call_id:
                    pending_calls[call_id] = {
                        "name": name,
                        "visible_snapshot": visible_snapshot,
                        "counted_link": counted_link,
                    }
            continue

        if role != "tool":
            continue
        responses = message.get("tool_responses")
        if not isinstance(responses, Mapping):
            continue
        for raw_call_id, content in responses.items():
            call_id = str(raw_call_id)
            pending = pending_calls.pop(call_id, {})
            name = str(pending.get("name") or "")
            documents = _extension_items(message, "documents", call_id)
            if name in _SEARCH_TOOL_NAMES or name in _VISIT_TOOL_NAMES:
                for document in documents:
                    returned_aliases.update(_document_aliases(document))

            document_urls = {
                url
                for document in documents
                if isinstance(document, Mapping)
                for url in [_normalize_url(document.get("url"))]
                if url is not None
            }
            if (
                name in _VISIT_TOOL_NAMES
                and not pending.get("counted_link")
                and document_urls & set(pending.get("visible_snapshot") or ())
            ):
                link_following_visit_calls += 1

            if name not in _SEARCH_TOOL_NAMES and name not in _VISIT_TOOL_NAMES:
                continue

            new_links = _markdown_urls(content)
            # Search and visit formatters add title links for returned documents.
            # Only links found in the result content count as followable links.
            new_links.difference_update(document_urls)
            visible_links.update(new_links)

    evidence_targets, recall_status = _evidence_targets(payload.get("extra"))
    recall = (
        sum(bool(target & returned_aliases) for target in evidence_targets)
        / len(evidence_targets)
        if evidence_targets
        else None
    )
    extra = payload.get("extra")
    query_id = extra.get("query_id") if isinstance(extra, Mapping) else None
    return {
        "query_id": str(query_id) if query_id is not None else None,
        "recall": recall,
        "recall_status": recall_status,
        "search_calls": search_calls,
        "visit_calls": visit_calls,
        "link_following_visit_calls": link_following_visit_calls,
        "turns": turns,
    }


async def _evaluate_one(
    *,
    record_path: Path,
    output_path: Path,
    concurrency_lock: asyncio.Semaphore | None,
    judge_config: JudgeConfig,
    answer_pattern: re.Pattern[str],
) -> Path:
    lock = concurrency_lock or nullcontext()
    async with lock:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"record {record_path} must contain a JSON object")
        metrics = _trajectory_metrics(payload)
        history = payload.get("history", [])
        if not isinstance(history, list) or not history:
            raise ValueError(f"record {record_path} missing non-empty history")

        last_message = history[-1]
        if not isinstance(last_message, dict):
            raise ValueError(f"record {record_path} last history item is not a dict")

        question = payload.get("input", "")
        correct_answer = payload.get("answer")
        if last_message.get("role") != "assistant":
            response = "No valid response"
            extracted_final_answer = "No valid response"
            reasoning = ""
            correct = False
            confidence = 100
        else:
            response_content = last_message.get("content")
            response = _attempt_extract_answer_content(
                response_content if isinstance(response_content, str) else "",
                answer_pattern,
            )
            prompt = _build_judge_prompt(question, response, correct_answer)
            agent = build_agent(judge_config)

            async def call_and_parse(judge_prompt: str) -> dict[str, Any]:
                judge_history = await agent.run(judge_prompt)
                content = judge_history[-1].content
                if not isinstance(content, str):
                    raise ValueError("judge returned no text content")
                return validate_judge_result(json.loads(content))

            try:
                result = await retry_async(
                    call_and_parse,
                    prompt,
                    policy=RetryPolicy(
                        exceptions=(ValueError, json.JSONDecodeError),
                    ),
                )
            finally:
                await agent.client.close()

            extracted_final_answer = result["extracted_final_answer"]
            reasoning = result["reasoning"]
            correct = result["correct"]
            confidence = result["confidence"]

        output_path.write_text(
            json.dumps(
                {
                    **metrics,
                    "question": question,
                    "response": response,
                    "correct_answer": correct_answer,
                    "extracted_final_answer": extracted_final_answer,
                    "judge_reasoning": reasoning,
                    "correct": correct,
                    "confidence": confidence,
                    "judge_model": judge_config.model,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_path


def _average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
    ]
    return mean(values) if values else None


def _collect_stats(output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    total = 0
    invalid = 0
    rows: list[dict[str, Any]] = []

    for record_path in sorted(Path(output_dir).glob("*.json")):
        if record_path.name == "summary.json":
            continue
        total += 1
        try:
            judge_result = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            invalid += 1
            logger.warning(
                "Invalid evaluation result path=%s error=%s",
                record_path,
                exc,
            )
            continue
        if not isinstance(judge_result, dict) or not isinstance(
            judge_result.get("correct"), bool
        ):
            invalid += 1
            logger.warning("Invalid evaluation result schema path=%s", record_path)
            continue
        rows.append(judge_result)

    correct = sum(1 for row in rows if row["correct"])
    recall_valid = sum(
        1
        for row in rows
        if isinstance(row.get("recall"), (int, float))
        and not isinstance(row.get("recall"), bool)
    )
    stats = {
        "evaluated": total,
        "valid": len(rows),
        "invalid": invalid,
        "correct": correct,
        "judged_incorrect": len(rows) - correct,
        "avg_confidence": _average(rows, "confidence") or 0.0,
        "avg_recall": _average(rows, "recall"),
        "recall_valid_samples": recall_valid,
        "recall_missing_samples": len(rows) - recall_valid,
        "avg_search_calls": _average(rows, "search_calls") or 0.0,
        "avg_visit_calls": _average(rows, "visit_calls") or 0.0,
        "avg_link_following_visit_calls": (
            _average(rows, "link_following_visit_calls") or 0.0
        ),
        "avg_turns": _average(rows, "turns") or 0.0,
    }
    for key in (
        "search_calls",
        "visit_calls",
        "link_following_visit_calls",
        "turns",
    ):
        stats[f"total_{key}"] = sum(
            int(row[key])
            for row in rows
            if isinstance(row.get(key), int) and not isinstance(row.get(key), bool)
        )
    return stats


def _resolve_benchmark_total(
    *,
    input_dir: Path,
    source_total: int,
    benchmark_total: int | None,
) -> int:
    """Resolve the official denominator, including generation failures."""
    if benchmark_total is not None:
        resolved = benchmark_total
    else:
        generation_summary_path = input_dir.parent / "summary.json"
        if not generation_summary_path.is_file():
            resolved = source_total
        else:
            try:
                generation_summary = json.loads(
                    generation_summary_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid generation summary: {generation_summary_path}"
                ) from exc
            if not isinstance(generation_summary, Mapping):
                raise ValueError(
                    f"generation summary must be an object: {generation_summary_path}"
                )
            resolved = generation_summary.get("total")

    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 1:
        raise ValueError("benchmark_total must be a positive integer")
    if resolved < source_total:
        raise ValueError(
            f"benchmark_total ({resolved}) cannot be smaller than "
            f"the generated source total ({source_total})"
        )
    return resolved


async def run_evaluate(
    *,
    input_dir: str,
    output_dir: str,
    max_concurrency: int | None,
    judge_config: JudgeConfig | None = None,
    answer_pattern: str | re.Pattern[str] = _ANSWER_PATTERN,
    benchmark_total: int | None = None,
) -> dict[str, Any]:
    resolved_judge = judge_config or JudgeConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    resolved_pattern = (
        re.compile(answer_pattern, re.DOTALL)
        if isinstance(answer_pattern, str)
        else answer_pattern
    )
    if "answer" not in resolved_pattern.groupindex:
        raise ValueError("answer_pattern must define a named 'answer' capture group")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1 or None")

    src_dir = Path(input_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    record_paths = [
        path for path in sorted(src_dir.glob("*.json")) if path.name != "summary.json"
    ]
    resolved_benchmark_total = _resolve_benchmark_total(
        input_dir=src_dir,
        source_total=len(record_paths),
        benchmark_total=benchmark_total,
    )

    tasks: list[asyncio.Task[Path]] = []
    pbar = tqdm(total=0)
    for record_path in record_paths:
        output_path = dst_dir / record_path.name
        if output_path.exists():
            continue
        task = asyncio.create_task(
            _evaluate_one(
                record_path=record_path,
                output_path=output_path,
                concurrency_lock=semaphore,
                judge_config=resolved_judge,
                answer_pattern=resolved_pattern,
            )
        )

        def _on_done(_done_task: asyncio.Future[Path]) -> None:
            pbar.update(1)

        task.add_done_callback(_on_done)
        tasks.append(task)

    pbar.reset(total=len(tasks))
    results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
    failed_requests = 0
    for result in results:
        if isinstance(result, Exception):
            failed_requests += 1
            logger.error(
                "Evaluation task failed error_type=%s error=%s",
                type(result).__name__,
                result,
            )

    stats = _collect_stats(dst_dir)
    stats["source_total"] = len(record_paths)
    stats["benchmark_total"] = resolved_benchmark_total
    stats["generation_failed"] = resolved_benchmark_total - len(record_paths)
    stats["judge_missing"] = len(record_paths) - stats["valid"]
    stats["incorrect"] = resolved_benchmark_total - stats["correct"]
    stats["accuracy"] = stats["correct"] / resolved_benchmark_total
    stats["submitted"] = len(tasks)
    stats["skipped_existing"] = len(record_paths) - len(tasks)
    stats["failed_requests"] = failed_requests
    (dst_dir / "summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    pbar.close()
    return stats


def evaluate_main(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Evaluate saved agent run records and trajectory metrics.",
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing agent run record JSON files.",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write evaluation result JSON files.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent judge requests.",
    )
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "qwen3-32b"))
    parser.add_argument("--judge-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--judge-base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument(
        "--answer-pattern",
        default=_ANSWER_PATTERN.pattern,
        help=(
            "Regex used to extract the final answer before judging. It must "
            "define a named capture group '(?P<answer>...)'."
        ),
    )
    parser.add_argument(
        "--benchmark-total",
        type=int,
        default=None,
        help=(
            "Official benchmark denominator. By default, read 'total' from "
            "the generation summary next to the history directory; otherwise "
            "fall back to the number of generated history files."
        ),
    )
    args = parser.parse_args(argv)
    try:
        answer_pattern = re.compile(args.answer_pattern, re.DOTALL)
    except re.error as exc:
        parser.error(str(exc))

    setup_logger()
    asyncio.run(
        run_evaluate(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_concurrency=args.max_concurrency,
            judge_config=JudgeConfig(
                model=args.judge_model,
                api_key=args.judge_api_key,
                base_url=args.judge_base_url,
            ),
            answer_pattern=answer_pattern,
            benchmark_total=args.benchmark_total,
        )
    )
