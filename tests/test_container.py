"""Container packaging end-to-end test."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, ClassVar

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from live_long_rnd.ingest import LanceDBNodeStore


class OpenAIHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible planning and embedding endpoint reachable from Docker."""

    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        body_size = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(body_size))
        self.requests.append(payload)
        if self.path == "/v1/responses":
            self._send_json(_query_plan_response(payload))
            return

        inputs = payload["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        value = 1.0 / math.sqrt(3_072)
        self._send_json(
            {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": [value] * 3_072,
                    }
                    for index, _text in enumerate(inputs)
                ],
                "model": "text-embedding-3-large",
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            }
        )

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        del args


def _query_plan_response(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload["input"]
    message = messages[-1]["content"]
    plan = json.dumps(
        {
            "action": "retrieve",
            "search_intents": [
                {
                    "dense_query": f"Evidence needed to answer: {message}",
                    "sparse_query": message,
                    "filters": {"document_id": None, "author": None},
                }
            ],
        }
    )
    return {
        "id": "resp_container_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": payload["model"],
        "output": [
            {
                "id": "msg_container_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": plan,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _create_fixture_index(index_dir: Path) -> None:
    value = 1.0 / math.sqrt(3_072)
    node = TextNode(
        text="Senolytics selectively remove senescent cells.",
        metadata={
            "document_id": "docker-paper",
            "source_path": "docker-paper.pdf",
            "page_numbers": json.dumps([2]),
            "bboxes": json.dumps([{"page": 2, "l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}]),
            "heading_path": json.dumps(["Results"]),
            "element_types": json.dumps(["text"]),
            "captions": json.dumps([]),
            "doc_item_refs": json.dumps(["#/texts/1"]),
            "original_text": "Senolytics selectively remove senescent cells.",
        },
        embedding=[value] * 3_072,
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="docker-paper")
    store = LanceDBNodeStore(index_dir)
    store.add([node])
    store.finalize()


def _create_fixture_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir()
    (corpus_dir / "MANIFEST.md").write_text(
        "# Corpus\n\n"
        "| # | Filename | Title |\n"
        "|---|---|---|\n"
        "| 001 | `docker-paper.pdf` | Senolytics in a container |\n",
        encoding="utf-8",
    )
    (corpus_dir / "docker-paper.pdf").write_bytes(b"%PDF-1.4\n%fixture\n%%EOF\n")


def _wait_for_api(port: int) -> None:
    deadline = time.monotonic() + 60
    url = f"http://127.0.0.1:{port}/docs"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (
            ConnectionResetError,
            urllib.error.URLError,
            RemoteDisconnected,
            TimeoutError,
        ):
            time.sleep(0.2)
    raise AssertionError("Container API did not become ready within 60 seconds")


def _published_port(container_id: str) -> int:
    result = subprocess.run(
        ["docker", "port", container_id, "8000/tcp"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.rsplit(":", maxsplit=1)[1])


def _run_api_container(
    image_tag: str,
    state_dir: Path,
    embedding_server_port: int,
) -> tuple[str, int]:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--add-host",
            "host.docker.internal:host-gateway",
            "--publish",
            "127.0.0.1::8000",
            "--volume",
            f"{state_dir}:/app/state",
            "--env",
            "OPENAI_API_KEY=test-key",
            "--env",
            f"OPENAI_BASE_URL=http://host.docker.internal:{embedding_server_port}/v1",
            "--env",
            "LIVE_LONG_LLM=stub",
            image_tag,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    container_id = result.stdout.strip()
    port = _published_port(container_id)
    _wait_for_api(port)
    return container_id, port


@pytest.mark.e2e
def test_api_image_serves_citations_from_its_baked_index(tmp_path: Path) -> None:
    if os.environ.get("RUN_DOCKER_E2E") != "1":
        pytest.skip("set RUN_DOCKER_E2E=1 to build and run the API image")

    index_dir = tmp_path / "index"
    _create_fixture_index(index_dir)
    corpus_dir = tmp_path / "corpus"
    _create_fixture_corpus(corpus_dir)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_dir.chmod(0o777)
    image_tag = f"live-long-rnd-e2e:{os.getpid()}"
    OpenAIHandler.requests = []
    server = ThreadingHTTPServer(("0.0.0.0", 0), OpenAIHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    container_id = ""

    try:
        subprocess.run(
            [
                "docker",
                "build",
                "--build-context",
                f"index={index_dir}",
                "--build-context",
                f"corpus={corpus_dir}",
                "--tag",
                image_tag,
                ".",
            ],
            check=True,
        )
        container_id, port = _run_api_container(
            image_tag,
            state_dir,
            server.server_port,
        )
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
            home_page = response.read().decode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/chat",
            data=json.dumps({"message": "What do senolytics do?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode()

        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        citation_event = next(event for event in events if event["type"] == "citations")
        assert "<title>Live Long R&amp;D</title>" in home_page
        assert response.status == 200
        assert citation_event["citations"][0]["document_id"] == "docker-paper"
        assert events[-1] == {"type": "done"}
        assert OpenAIHandler.requests
        assert (state_dir / "conversations.db").is_file()

        conversation_id = events[0]["id"]
        subprocess.run(
            ["docker", "rm", "--force", container_id],
            check=True,
            capture_output=True,
        )
        container_id, port = _run_api_container(
            image_tag,
            state_dir,
            server.server_port,
        )

        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/conversations/{conversation_id}",
            timeout=10,
        ) as conversation_response:
            conversation = json.loads(conversation_response.read().decode())
        messages = conversation["messages"]
        assert messages[0]["text"] == "What do senolytics do?"
        assert messages[1]["text"].startswith("Senolytics are drugs")

        restarted_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/chat",
            data=json.dumps({"message": "What do senolytics do after restart?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(restarted_request, timeout=30) as restarted_response:
            restarted_body = restarted_response.read().decode()
        restarted_events = [
            json.loads(line.removeprefix("data: "))
            for line in restarted_body.splitlines()
            if line.startswith("data: ")
        ]
        restarted_citations = next(
            event for event in restarted_events if event["type"] == "citations"
        )
        assert restarted_citations["citations"][0]["document_id"] == "docker-paper"

        metadata_url = f"http://127.0.0.1:{port}/api/documents/docker-paper"
        with urllib.request.urlopen(metadata_url, timeout=10) as metadata_response:
            metadata = json.loads(metadata_response.read().decode())
        assert metadata == {
            "document_id": "docker-paper",
            "title": "Senolytics in a container",
        }
        pdf_url = f"http://127.0.0.1:{port}/api/documents/docker-paper/pdf"
        with urllib.request.urlopen(pdf_url, timeout=10) as pdf_response:
            assert pdf_response.headers["Content-Type"] == "application/pdf"
            assert pdf_response.read().startswith(b"%PDF-")
        assert len(OpenAIHandler.requests) == 2
    finally:
        if container_id:
            subprocess.run(
                ["docker", "rm", "--force", container_id],
                check=False,
                capture_output=True,
            )
        subprocess.run(
            ["docker", "image", "rm", "--force", image_tag],
            check=False,
            capture_output=True,
        )
        server.shutdown()
        server.server_close()
        thread.join()
