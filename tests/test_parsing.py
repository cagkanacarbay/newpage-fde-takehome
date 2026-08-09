import json
from pathlib import Path

import pytest
from docling_core.types.doc.document import DoclingDocument
from llama_index.core import Document

from live_long_rnd.parsing import EmptyParseResultError, chunk_documents, parse_pdf


class EmptyReader:
    def __init__(self) -> None:
        self.document = Document(text=json.dumps(DoclingDocument(name="empty").export_to_dict()))

    def load_data(self, _file_path: Path) -> list[Document]:
        return [self.document]


def test_empty_docling_output_is_a_failure_not_an_empty_document(tmp_path: Path) -> None:
    source = tmp_path / "empty.pdf"
    source.touch()

    with pytest.raises(EmptyParseResultError, match="produced no content"):
        parse_pdf(source, reader=EmptyReader())


@pytest.mark.integration
def test_parse_pdf_preserves_structure_and_page_provenance() -> None:
    source = Path(
        "data/corpus/longevity/009-vetter-2026-comparing-fourteen-biomarkers-of-aging.pdf"
    )

    [document] = parse_pdf(source)
    payload = json.loads(document.get_content())
    headings = [item for item in payload["texts"] if item["label"] == "section_header"]
    content_items = [item for collection in ("texts", "tables") for item in payload[collection]]

    assert document.metadata["source_document_id"] == source.stem
    assert max(item["level"] for item in headings) >= 3
    assert content_items
    assert all(item["prov"] and item["prov"][0]["page_no"] >= 1 for item in content_items)


@pytest.mark.integration
def test_chunked_nodes_keep_exact_spot_citation_provenance() -> None:
    source = Path(
        "data/corpus/longevity/009-vetter-2026-comparing-fourteen-biomarkers-of-aging.pdf"
    )
    [document] = parse_pdf(source)

    nodes = chunk_documents([document])

    assert nodes
    for node in nodes:
        assert node.metadata["source_document_id"] == source.stem
        doc_items = node.metadata["doc_items"]
        provenance = [prov for item in doc_items for prov in item.get("prov", [])]
        assert provenance
        assert all(prov["page_no"] >= 1 for prov in provenance)
        assert all({"l", "t", "r", "b"} <= prov["bbox"].keys() for prov in provenance)
