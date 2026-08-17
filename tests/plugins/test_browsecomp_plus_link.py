from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from searcherkit.cli import plugins as plugins_cli
from searcherkit.plugins.browsecomp_plus_link import (
    BrowseCompPlusLinkSource,
    preprocess_browsecomp_plus_link_record,
)
from searcherkit.plugins.browsecomp_plus_link import deploy_elasticsearch
from searcherkit.plugins.indexing import create_elasticsearch_index
from searcherkit.plugins.indexing import encode_documents
from searcherkit.plugins.indexing import IndexDocument
from searcherkit.plugins.indexing import OpenAIEmbeddingModel


FIXTURE_DIR = Path("tests/fixtures/plugin_sources")
BCP_LINK_PATH = FIXTURE_DIR / "bcp_link.jsonl"


def test_preprocess_browsecomp_plus_link_record_normalizes_fields() -> None:
    text = (
        "---\ntitle: Link Canonical Title\ndate: 2022-11-03\n---\n"
        "Canonical [body paragraph](https://example.test/body).\n\nSecond line."
    )
    text_raw = (
        "---\ntitle: Link Canonical Title\ndate: 2022-11-03\n---\n"
        "Canonical body paragraph.\n\nSecond line."
    )
    document = preprocess_browsecomp_plus_link_record(
        {
            "docid": 42,
            "source": "live_raw",
            "text": text,
            "text_raw": text_raw,
            "url": "https://example.test/canonical/42",
            "url_raw": "https://example.test/raw/42",
            "links": [
                {
                    "to_url": "https://en.wikipedia.org/wiki/Book_of_Genesis",
                    "type": "in_corpus",
                    "url_text": "Book of Genesis",
                },
                {
                    "text": "Existing text",
                    "target": "https://example.test/target",
                    "url": "https://example.test/url",
                },
            ],
        }
    )

    assert document is not None
    assert document.id == "42"
    assert document.title == "Link Canonical Title"
    assert document.text == (
        "Canonical [body paragraph](https://example.test/body).\n\nSecond line."
    )
    assert document.text_raw == "Canonical body paragraph.\n\nSecond line."
    assert document.to_source()["text_raw"] == document.text_raw
    assert document.url == "https://example.test/canonical/42"
    assert document.links == [
        {
            "text": "Book of Genesis",
            "target": "https://en.wikipedia.org/wiki/Book_of_Genesis",
            "url": "https://en.wikipedia.org/wiki/Book_of_Genesis",
        },
        {
            "text": "Existing text",
            "target": "https://example.test/target",
            "url": "https://example.test/url",
        },
    ]
    assert document.metadata == {
        "date": "2022-11-03",
        "source": "browsecomp_plus_link",
        "raw_source": "live_raw",
        "url_raw": "https://example.test/raw/42",
    }


def test_browsecomp_plus_link_source_reads_jsonl(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))

    documents = list(BrowseCompPlusLinkSource(BCP_LINK_PATH).iter_documents())

    assert len(documents) == 2
    assert documents[0].id == "42"
    assert documents[0].title == "Link Canonical Title"
    assert documents[0].text_raw == "Canonical body paragraph.\n\nSecond line."
    assert documents[0].metadata["raw_source"] == "live_raw"
    assert documents[1].id == "43"
    assert documents[1].title == "BCP Style Title"
    assert documents[1].text == "BCP style body."
    assert documents[1].text_raw == documents[1].text


def test_encode_documents_prefers_plain_text_raw() -> None:
    class FakeEmbeddingModel:
        texts: list[str] = []

        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            del kwargs
            self.texts = texts
            return [[1.0, 0.0] for _ in texts]

    model = FakeEmbeddingModel()
    document = IndexDocument(
        id="doc-1",
        title="Example",
        text="Read [the source](https://example.test/source).",
        text_raw="Read the source.",
        url="https://example.test/doc-1",
    )

    vectors = encode_documents(
        model,
        [document],
        prompt_strategy="qwen3",
        max_text_chars=32768,
        batch_size=16,
    )

    assert model.texts == ["Read the source."]
    assert vectors == [[1.0, 0.0]]


def test_elasticsearch_mapping_contains_canonical_bcp_link_fields() -> None:
    class FakeIndices:
        created: dict[str, Any] = {}

        def exists(self, *, index: str) -> bool:
            assert index == "bcp-link-test"
            return False

        def create(self, **kwargs: Any) -> None:
            self.created = kwargs

    class FakeClient:
        indices = FakeIndices()

    client = FakeClient()
    create_elasticsearch_index(
        client,
        index_name="bcp-link-test",
        embedding_dim=4096,
        shards=2,
        replicas=0,
    )

    properties = client.indices.created["mappings"]["properties"]
    assert set(properties) == {
        "title",
        "text",
        "text_raw",
        "url",
        "links",
        "metadata",
        "text_vector",
    }
    assert properties["text_vector"] == {
        "type": "dense_vector",
        "dims": 4096,
        "index": True,
        "similarity": "cosine",
    }
    assert client.indices.created["settings"] == {
        "index": {"number_of_shards": 2, "number_of_replicas": 0}
    }


def test_searcher_cli_deploys_browsecomp_plus_link(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_iter_records(self: BrowseCompPlusLinkSource) -> Iterator[Mapping[str, Any]]:
        assert self.dataset_path == str(BCP_LINK_PATH)
        assert self.split == "validation"
        yield {
            "docid": "doc-1",
            "text_raw": "---\ntitle: BrowseComp Plus Link\n---\nA benchmark corpus.",
            "url": "https://example.test/doc-1",
            "links": [],
        }

    def fake_deploy_to_elasticsearch(
        *,
        documents: Iterable[IndexDocument],
        es_host: str,
        index_name: str,
        embedding_model_name: str | None,
        embedding_endpoint: str | None,
        embedding_api_key: str,
        embedding_max_input_tokens: int,
        embedding_dim: int | None,
        prompt_strategy: str,
        overwrite: bool,
        batch_size: int,
        embedding_batch_size: int,
        max_text_chars: int,
        shards: int,
        replicas: int,
    ) -> int:
        captured.update(
            {
                "documents": list(documents),
                "es_host": es_host,
                "index_name": index_name,
                "embedding_model_name": embedding_model_name,
                "embedding_endpoint": embedding_endpoint,
                "embedding_api_key": embedding_api_key,
                "embedding_max_input_tokens": embedding_max_input_tokens,
                "embedding_dim": embedding_dim,
                "prompt_strategy": prompt_strategy,
                "overwrite": overwrite,
                "batch_size": batch_size,
                "embedding_batch_size": embedding_batch_size,
                "max_text_chars": max_text_chars,
                "shards": shards,
                "replicas": replicas,
            }
        )
        return len(captured["documents"])

    monkeypatch.setattr(BrowseCompPlusLinkSource, "_iter_records", fake_iter_records)
    monkeypatch.setattr(
        deploy_elasticsearch,
        "deploy_to_elasticsearch",
        fake_deploy_to_elasticsearch,
    )

    result = plugins_cli.main(
        [
            "deploy",
            "browsecomp-plus-link",
            "--dataset_path",
            str(BCP_LINK_PATH),
            "--split",
            "validation",
            "--es_host",
            "http://localhost:9200",
            "--index_name",
            "bcp-link-test",
            "--limit",
            "1",
            "--overwrite",
        ]
    )

    assert result == 0
    assert [document.id for document in captured["documents"]] == ["doc-1"]
    assert captured["es_host"] == "http://localhost:9200"
    assert captured["index_name"] == "bcp-link-test"
    assert captured["embedding_model_name"] is None
    assert captured["embedding_endpoint"] is None
    assert captured["embedding_api_key"] == "unused"
    assert captured["embedding_max_input_tokens"] == 8192
    assert captured["embedding_dim"] is None
    assert captured["prompt_strategy"] == "none"
    assert captured["overwrite"] is True
    assert "Indexed 1 BrowseComp Plus Link documents" in capsys.readouterr().out


def test_openai_embedding_model_preserves_order_and_normalizes() -> None:
    class FakeEmbeddings:
        calls: list[list[str]] = []

        def create(self, **kwargs: Any) -> Any:
            texts = kwargs["input"]
            self.calls.append(texts)
            assert kwargs["model"] == "Qwen3-Embedding-8B"
            assert kwargs["extra_body"] == {"truncate_prompt_tokens": 8192}
            vectors = {
                "first": [3.0, 4.0],
                "second": [0.0, 2.0],
                "third": [12.0, 5.0],
            }
            return type(
                "Response",
                (),
                {
                    "data": [
                        type(
                            "Embedding",
                            (),
                            {"index": index, "embedding": vectors[text]},
                        )()
                        for index, text in reversed(list(enumerate(texts)))
                    ]
                },
            )()

    class FakeClient:
        embeddings = FakeEmbeddings()
        closed = False

        def close(self) -> None:
            self.closed = True

    client = FakeClient()
    model = OpenAIEmbeddingModel(client=client, model_name="Qwen3-Embedding-8B")

    vectors = model.encode(
        ["first", "second", "third"],
        normalize_embeddings=True,
        batch_size=2,
        show_progress_bar=False,
    )
    model.close()

    assert vectors == [[0.6, 0.8], [0.0, 1.0], [12 / 13, 5 / 13]]
    assert client.embeddings.calls == [["first", "second"], ["third"]]
    assert client.closed is True
