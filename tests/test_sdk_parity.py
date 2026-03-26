"""SDK parity tests for gateway-exposed contracts."""

from __future__ import annotations

from latence._base import APIResponse, ResponseMetadata
import base64
import struct

from latence.resources.document_intelligence import DocumentIntelligence
from latence.resources.embedding import Embedding


class _FakeSyncClient:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.last_path: str | None = None
        self.last_json: dict | None = None

    def post(self, path: str, json: dict) -> APIResponse:
        self.last_path = path
        self.last_json = json
        return APIResponse(
            data=self._data,
            metadata=ResponseMetadata(request_id="req_test"),
            status_code=200,
        )


def test_embedding_accepts_batch_text() -> None:
    client = _FakeSyncClient(
        {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "dimension": 2,
            "shape": [2, 2],
            "encoding_format": "float",
            "model": "nomic-embed-text-v1.5",
        }
    )
    resource = Embedding(client)

    result = resource.embed(text=["first", "second"], dimension=256)

    assert result.dimension == 2
    assert client.last_path == "/api/v1/embedding/embed"
    assert client.last_json is not None
    assert client.last_json["text"] == ["first", "second"]
    assert client.last_json["dimension"] == 256


def test_embedding_decodes_batch_base64_rows() -> None:
    row1 = base64.b64encode(struct.pack("<2f", 0.1, 0.2)).decode()
    row2 = base64.b64encode(struct.pack("<2f", 0.3, 0.4)).decode()
    client = _FakeSyncClient(
        {
            "embeddings": [row1, row2],
            "dimension": 2,
            "shape": [2, 2],
            "encoding_format": "base64",
            "model": "nomic-embed-text-v1.5",
        }
    )
    resource = Embedding(client)
    result = resource.embed(text=["first", "second"], dimension=256)
    assert len(result.embeddings) == 2
    assert abs(result.embeddings[0][0] - 0.1) < 1e-5
    assert abs(result.embeddings[1][1] - 0.4) < 1e-5


def test_document_intelligence_process_exposes_top_level_flags() -> None:
    client = _FakeSyncClient(
        {
            "content": "ok",
            "content_type": "markdown",
            "pages_processed": 1,
            "success": True,
        }
    )
    resource = DocumentIntelligence(client)

    resource.process(
        file_base64="ZmFrZV9wZGY=",
        filename="doc.pdf",
        use_layout_detection=False,
        use_chart_recognition=True,
    )

    assert client.last_path == "/api/v1/document_intelligence/process"
    assert client.last_json is not None
    assert client.last_json["use_layout_detection"] is False
    assert client.last_json["use_chart_recognition"] is True


def test_document_intelligence_refine_contract() -> None:
    client = _FakeSyncClient(
        {
            "content": "refined",
            "content_type": "markdown",
            "pages": [],
            "refinement_stats": {"tables_merged": True},
            "success": True,
        }
    )
    resource = DocumentIntelligence(client)

    resource.refine(
        pages_result=[{"page_num": 1, "markdown": "# page"}],
        refine_options={"merge_tables": True},
        output_options={"indent": 2, "ensure_ascii": False},
    )

    assert client.last_path == "/api/v1/document_intelligence/refine"
    assert client.last_json is not None
    assert "pages_result" in client.last_json
    assert client.last_json["refine_options"]["merge_tables"] is True
    assert client.last_json["output_options"]["indent"] == 2
