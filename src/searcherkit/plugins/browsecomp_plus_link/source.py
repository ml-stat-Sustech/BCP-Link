"""BrowseComp Plus Link corpus reading and preprocessing."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from searcherkit.plugins.indexing import IndexDocument

logger = logging.getLogger(__name__)


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    value = str(value).strip()
    return value or None


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text.strip()

    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        return {}, text.strip()

    front_matter: dict[str, str] = {}
    for line in lines[1:closing_index]:
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if key:
            front_matter[key] = value

    body = "\n".join(lines[closing_index + 1 :]).strip()
    return front_matter, body


def _normalize_link(link: Mapping[str, Any]) -> dict[str, str]:
    text = _coerce_text(link.get("text")) or _coerce_text(link.get("url_text"))
    target = _coerce_text(link.get("target")) or _coerce_text(link.get("to_url"))
    url = _coerce_text(link.get("url")) or target

    normalized: dict[str, str] = {}
    if text:
        normalized["text"] = text
    if target:
        normalized["target"] = target
    if url:
        normalized["url"] = url
    return normalized


def _normalize_links(links: Any) -> list[dict[str, str]]:
    if not isinstance(links, list):
        return []

    normalized: list[dict[str, str]] = []
    for link in links:
        if isinstance(link, Mapping):
            normalized_link = _normalize_link(link)
            if normalized_link:
                normalized.append(normalized_link)
    return normalized


def _fallback_title(clean_text: str, *, docid: str, url: str) -> str:
    for line in clean_text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate[:120]
    if url:
        return url[:120]
    return docid[:120]


def preprocess_browsecomp_plus_link_record(
    record: Mapping[str, Any],
) -> IndexDocument | None:
    text_value = record.get("text")
    text_raw_value = record.get("text_raw")
    if not isinstance(text_value, str) or not text_value.strip():
        text_value = text_raw_value
    if not isinstance(text_raw_value, str) or not text_raw_value.strip():
        text_raw_value = text_value
    if not isinstance(text_value, str) or not text_value.strip():
        return None
    if not isinstance(text_raw_value, str) or not text_raw_value.strip():
        return None

    text_front_matter, clean_text = _split_front_matter(text_value)
    raw_front_matter, clean_text_raw = _split_front_matter(text_raw_value)
    front_matter = {**text_front_matter, **raw_front_matter}
    title = front_matter.pop("title", "")
    extra_metadata = front_matter

    docid = _coerce_text(record.get("docid"))
    if docid is None:
        raise ValueError("docid missing or empty in record")

    url = _coerce_text(record.get("url")) or _coerce_text(record.get("url_raw"))
    if url is None:
        raise ValueError("url missing or empty in record")

    if not title:
        title = _fallback_title(clean_text_raw, docid=docid, url=url)
        logger.debug("Title not found in frontmatter, using text fallback for docid=%s", docid)

    metadata: dict[str, Any] = dict(extra_metadata)
    metadata["source"] = "browsecomp_plus_link"

    raw_source = _coerce_text(record.get("source"))
    if raw_source is not None:
        metadata["raw_source"] = raw_source

    url_raw = _coerce_text(record.get("url_raw"))
    if url_raw is not None and url_raw != url:
        metadata["url_raw"] = url_raw

    return IndexDocument(
        id=docid,
        title=title,
        text=clean_text,
        url=url,
        text_raw=clean_text_raw,
        links=_normalize_links(record.get("links")),
        metadata=metadata,
    )


class BrowseCompPlusLinkSource:
    """Iterate normalized documents from a local file or Hugging Face dataset."""

    def __init__(self, dataset_path: str | Path, *, split: str = "train") -> None:
        self.dataset_path = str(dataset_path)
        self.split = split

    def iter_documents(self, *, limit: int | None = None) -> Iterator[IndexDocument]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")

        count = 0
        for record in self._iter_records():
            document = preprocess_browsecomp_plus_link_record(record)
            if document is None:
                continue
            yield document
            count += 1
            if limit is not None and count >= limit:
                return

    def _iter_records(self) -> Iterator[Mapping[str, Any]]:
        path = Path(self.dataset_path)
        if path.exists():
            if path.suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if isinstance(row, Mapping):
                            yield row
                return
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, Mapping):
                    yield data
                elif isinstance(data, list):
                    for row in data:
                        if isinstance(row, Mapping):
                            yield row
                return

        from datasets import load_dataset

        if path.exists():
            dataset = load_dataset(
                "parquet",
                data_files=str(path),
                split=self.split,
                streaming=True,
            )
        else:
            dataset = load_dataset(self.dataset_path, split=self.split, streaming=True)

        for row in dataset:
            if isinstance(row, Mapping):
                yield row
