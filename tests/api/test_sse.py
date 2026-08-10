from live_long_rnd.api.sse import encode_sse


def test_encode_sse_frames_a_json_event() -> None:
    assert encode_sse({"type": "done"}) == 'data: {"type":"done"}\n\n'
