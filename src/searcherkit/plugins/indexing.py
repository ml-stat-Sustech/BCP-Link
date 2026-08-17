"""Shared Elasticsearch indexing helpers for corpus plugins."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IndexDocument:
    """Normalized document shape used by plugin preprocessors."""

    id: str
    title: str
    text: str
    url: str
    text_raw: str | None = None
    links: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_source(self) -> dict[str, Any]:
        source = {
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "links": list(self.links),
        }
        if self.text_raw is not None:
            source["text_raw"] = self.text_raw
        if self.metadata:
            source["metadata"] = dict(self.metadata)
        return source


@dataclass(slots=True)
class SentenceEmbeddingModel:
    """SentenceTransformer plus optional multi-GPU process pool."""

    model: Any
    pool: Any | None = None
    devices: list[str] | None = None

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Any:
        encode_kwargs = {}
        if self.pool is not None:
            encode_kwargs["pool"] = self.pool
        if self.devices is not None:
            encode_kwargs["device"] = self.devices
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            **encode_kwargs,
        )

    def close(self) -> None:
        if self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None


@dataclass(slots=True)
class OpenAIEmbeddingModel:
    """Batched client for an OpenAI-compatible embedding endpoint."""

    client: Any
    model_name: str
    max_input_tokens: int = 8192

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            encoding_format="float",
            extra_body={"truncate_prompt_tokens": self.max_input_tokens},
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise ValueError(
                f"embedding endpoint returned {len(ordered)} vectors for {len(texts)} inputs"
            )

        vectors: list[list[float]] = []
        for item in ordered:
            vector = [float(value) for value in item.embedding]
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                raise ValueError("embedding endpoint returned a zero vector")
            vectors.append([value / norm for value in vector])
        return vectors

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        del normalize_embeddings, show_progress_bar
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._encode_batch(texts[start : start + batch_size]))
        return vectors

    def close(self) -> None:
        self.client.close()


def apply_embedding_prompt(text: str, strategy: str = "none") -> str:
    if strategy == "none":
        return text
    if strategy == "e5":
        return f"passage: {text}"
    if strategy == "qwen3":
        return text
    raise ValueError(f"unknown prompt strategy: {strategy!r}")


def iter_batches(items: Iterable[IndexDocument], batch_size: int) -> Iterator[list[IndexDocument]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batch: list[IndexDocument] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_elasticsearch_client(hosts: str, *, username: str | None = None, password: str | None = None, request_timeout: float = 100.0) -> Any:
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise ImportError(
            "Elasticsearch deployment requires the 'elasticsearch' package. "
            "Install with `pip install searcherkit` or `pip install 'searcherkit[indexing]'`."
        ) from exc
    if (username is None) != (password is None):
        raise ValueError("username and password must be provided together")
    kwargs: dict[str, Any] = {"request_timeout": request_timeout}
    if username is not None:
        kwargs["basic_auth"] = (username, password)
    return Elasticsearch(hosts, **kwargs)


def create_elasticsearch_index(
    client: Any,
    *,
    index_name: str,
    embedding_dim: int | None = None,
    vector_field: str = "text_vector",
    shards: int = 1,
    replicas: int = 0,
    overwrite: bool = False,
) -> None:
    if not index_name:
        raise ValueError("index_name must be non-empty")
    if embedding_dim is not None and embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")

    if client.indices.exists(index=index_name):
        if not overwrite:
            return
        client.indices.delete(index=index_name)

    mappings: dict[str, Any] = {
        "properties": {
            "title": {"type": "text", "analyzer": "standard"},
            "text": {"type": "text", "analyzer": "standard"},
            "text_raw": {"type": "text", "analyzer": "standard"},
            "url": {"type": "keyword"},
            "links": {
                "type": "nested",
                "dynamic": False,
                "properties": {
                    "text": {"type": "text", "index": False},
                    "target": {"type": "keyword", "index": False},
                    "url": {"type": "keyword", "index": False},
                },
            },
            "metadata": {"type": "object", "enabled": True, "dynamic": False},
        }
    }
    if embedding_dim is not None:
        mappings["properties"][vector_field] = {
            "type": "dense_vector",
            "dims": embedding_dim,
            "index": True,
            "similarity": "cosine",
        }

    client.indices.create(
        index=index_name,
        settings={"index": {"number_of_shards": shards, "number_of_replicas": replicas}},
        mappings=mappings,
    )


def load_sentence_transformer(model_name: str) -> Any:
    if not model_name:
        raise ValueError("model_name must be non-empty when vector indexing is enabled")
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as exc:
        raise ImportError(
            "Vector indexing requires the 'sentence-transformers' package. "
            "Install with `pip install 'searcherkit[indexing]'`."
        ) from exc

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cuda_count > 1:
        model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        devices = [f"cuda:{idx}" for idx in range(cuda_count)]
        if hasattr(model, "start_multi_process_pool"):
            pool = model.start_multi_process_pool(target_devices=devices)
            return SentenceEmbeddingModel(model=model, pool=pool)
        return SentenceEmbeddingModel(model=model, devices=devices)
    return SentenceEmbeddingModel(
        model=SentenceTransformer(model_name, trust_remote_code=True)
    )


def load_openai_embedding_model(
    endpoint: str,
    *,
    model_name: str,
    api_key: str = "unused",
    max_input_tokens: int = 8192,
) -> OpenAIEmbeddingModel:
    if not endpoint:
        raise ValueError("embedding endpoint must be non-empty")
    if not model_name:
        raise ValueError("model_name must be non-empty")
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be >= 1")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Remote vector indexing requires the 'openai' package. "
            "Install with `pip install searcherkit`."
        ) from exc

    base_url = endpoint.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "unused",
        max_retries=5,
        timeout=120.0,
    )
    return OpenAIEmbeddingModel(
        client=client,
        model_name=model_name,
        max_input_tokens=max_input_tokens,
    )


def encode_documents(
    model: Any,
    documents: list[IndexDocument],
    *,
    prompt_strategy: str,
    max_text_chars: int,
    batch_size: int,
) -> list[list[float]]:
    if max_text_chars < 1:
        raise ValueError("max_text_chars must be >= 1")
    texts = []
    for document in documents:
        embedding_text = document.text_raw if document.text_raw is not None else document.text
        texts.append(apply_embedding_prompt(embedding_text[:max_text_chars], prompt_strategy))
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [
        vector.tolist() if hasattr(vector, "tolist") else list(vector)
        for vector in vectors
    ]


def index_documents(
    client: Any,
    *,
    index_name: str,
    documents: Iterable[IndexDocument],
    embedding_model: Any | None = None,
    vector_field: str = "text_vector",
    prompt_strategy: str = "none",
    batch_size: int = 200,
    embedding_batch_size: int = 16,
    max_text_chars: int = 32768,
) -> int:
    try:
        from elasticsearch.helpers import bulk
    except ImportError as exc:
        raise ImportError(
            "Bulk indexing requires the 'elasticsearch' package. "
            "Install with `pip install searcherkit` or `pip install 'searcherkit[indexing]'`."
        ) from exc

    total = 0
    for batch in iter_batches(documents, batch_size):
        vectors = None
        if embedding_model is not None:
            vectors = encode_documents(
                embedding_model,
                batch,
                prompt_strategy=prompt_strategy,
                max_text_chars=max_text_chars,
                batch_size=embedding_batch_size,
            )

        actions = []
        for idx, document in enumerate(batch):
            source = document.to_source()
            if vectors is not None:
                source[vector_field] = vectors[idx]
            actions.append(
                {
                    "_index": index_name,
                    "_id": document.id,
                    "_source": source,
                }
            )
        bulk(client.options(request_timeout=100), actions, raise_on_error=True)
        total += len(actions)
    return total


def deploy_to_elasticsearch(
    *,
    documents: Iterable[IndexDocument],
    es_host: str,
    index_name: str,
    es_username: str | None = None,
    es_password: str | None = None,
    embedding_model_name: str | None = None,
    embedding_endpoint: str | None = None,
    embedding_api_key: str = "unused",
    embedding_max_input_tokens: int = 8192,
    embedding_dim: int | None = None,
    prompt_strategy: str = "none",
    overwrite: bool = False,
    batch_size: int = 200,
    embedding_batch_size: int = 16,
    max_text_chars: int = 32768,
    shards: int = 1,
    replicas: int = 0,
) -> int:
    client = build_elasticsearch_client(es_host, username=es_username, password=es_password)
    if embedding_endpoint is not None and embedding_model_name is None:
        raise ValueError("embedding_model_name is required with embedding_endpoint")
    if embedding_endpoint is not None:
        model = load_openai_embedding_model(
            embedding_endpoint,
            model_name=embedding_model_name or "",
            api_key=embedding_api_key,
            max_input_tokens=embedding_max_input_tokens,
        )
    else:
        model = load_sentence_transformer(embedding_model_name) if embedding_model_name else None
    create_elasticsearch_index(
        client,
        index_name=index_name,
        embedding_dim=embedding_dim if model is not None else None,
        overwrite=overwrite,
        shards=shards,
        replicas=replicas,
    )
    try:
        return index_documents(
            client,
            index_name=index_name,
            documents=documents,
            embedding_model=model,
            prompt_strategy=prompt_strategy,
            batch_size=batch_size,
            embedding_batch_size=embedding_batch_size,
            max_text_chars=max_text_chars,
        )
    finally:
        if model is not None:
            model.close()
