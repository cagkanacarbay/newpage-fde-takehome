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
@pytest.mark.e2e
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
@pytest.mark.e2e
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


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.xfail(
    strict=True,
    reason="Docling 2.115.0 does not bind every paper 009 table caption",
)
def test_docling_binds_every_paper_009_table_caption() -> None:
    source = Path(
        "data/corpus/longevity/009-vetter-2026-comparing-fourteen-biomarkers-of-aging.pdf"
    )

    [document] = parse_pdf(source)
    tables = json.loads(document.get_content())["tables"]

    assert len(tables) == 3
    assert all(table["captions"] for table in tables)


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.xfail(
    strict=True,
    reason="Docling 2.115.0 scrambles paper 011 two-column reading order",
)
def test_docling_preserves_paper_011_two_column_reading_order() -> None:
    source = Path("data/corpus/longevity/011-maleszka-2025-no-epigenetic-clock-in-insect.pdf")

    [document] = parse_pdf(source)
    text = document.get_content()
    positions = [text.index(ordinal) for ordinal in ("First", "Second", "Third", "Last")]

    assert positions == sorted(positions)
