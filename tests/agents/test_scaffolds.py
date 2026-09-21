"""Offline tests for agents/taubench_tool_calling/scaffolds.py.

No GPU and no LLM: the tau-bench user simulator is replaced by a scripted one,
and the upstream LLM server by a local stub that answers in SSE.
"""

import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agents" / "taubench_tool_calling"))
import scaffolds  # noqa: E402
from scaffolds import Bridge, Episode, install_tool_faults, parse_usage  # noqa: E402
from tau_bench.envs import get_env  # noqa: E402
from tau_bench.types import RESPOND_ACTION_NAME, Action  # noqa: E402

AIRLINE_TASK = 0


class ScriptedUser:
    """Stands in for the LLM user simulator: fixed opener, stops on request."""

    def __init__(self):
        self.said = []

    def reset(self, instruction=None):
        return "Hi, I need help with my reservation."

    def step(self, content):
        self.said.append(content)
        return "###STOP###" if "goodbye" in content.lower() else "Please go ahead."

    def get_total_cost(self):
        return 0.0


def airline_env():
    # the Env constructor already resets the user, so swap the simulator in first
    import tau_bench.envs.base as tb_base

    original = tb_base.load_user
    tb_base.load_user = lambda **kwargs: ScriptedUser()
    try:
        return get_env("airline", "llm", "stub", "test", "openai", AIRLINE_TASK)
    finally:
        tb_base.load_user = original


def rpc(port, method, params=None, mid=1):
    body = json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/mcp",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def test_mcp_bridge_matches_direct_env_reward():
    """Ground-truth actions through the MCP bridge give the same reward and
    action log as the same actions sent straight to env.step."""
    direct = airline_env()
    direct.reset(task_index=AIRLINE_TASK)
    gt = [a for a in direct.task.actions if a.name != RESPOND_ACTION_NAME]
    for a in gt:
        direct.step(Action(name=a.name, kwargs=dict(a.kwargs)))
    direct_res = direct.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Goodbye"}))
    assert direct_res.done

    env = airline_env()
    ep = Episode(env, env.wiki, AIRLINE_TASK)
    bridge = Bridge(ep, env.tools_info, "served/model", "http://127.0.0.1:9/v1", "0")
    port = bridge.start()
    try:
        init = rpc(port, "initialize", {"protocolVersion": "2025-06-18"})
        assert init["result"]["capabilities"]["tools"] == {"listChanged": False}
        names = {t["name"] for t in rpc(port, "tools/list")["result"]["tools"]}
        assert {a.name for a in gt} <= names
        for a in gt:
            out = rpc(port, "tools/call", {"name": a.name, "arguments": dict(a.kwargs)})
            assert out["result"]["isError"] is False
        assert ep.respond("Goodbye") == "###STOP###"
    finally:
        bridge.stop()

    assert ep.done and ep.end_reason == "customer_stop"
    assert ep.reward == direct_res.reward == env.reward
    assert [a.model_dump() for a in env.actions] == [a.model_dump() for a in direct.actions]
    # transcript has the ToolCallingAgent shape: tool call + tool result per action
    roles = [m["role"] for m in ep.messages]
    assert roles[:2] == ["system", "user"]
    assert roles.count("tool") == len(gt)
    assert ep.messages[-2] == {"role": "assistant", "content": "Goodbye"}


def test_step_cap_and_notifications():
    env = airline_env()
    ep = Episode(env, env.wiki, AIRLINE_TASK)
    bridge = Bridge(ep, env.tools_info, "m", "http://127.0.0.1:9/v1", "0")
    assert bridge.rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert "error" in bridge.rpc({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
    for _ in range(scaffolds.MAX_ENV_STEPS):
        ep.call_tool("think", {"thought": "x"})
    text, is_error = ep.call_tool("think", {"thought": "x"})
    assert is_error and ep.end_reason == "step_cap"
    assert ep.respond("anything") is None


def test_tool_faults_fail_only_tools_and_track_recovery():
    from hal.utils.fault_injection import FaultInjector

    env = airline_env()
    env.reset(task_index=AIRLINE_TASK)
    inj = FaultInjector(fault_rate=1.0)
    finalize = install_tool_faults(env, inj)
    res = env.step(Action(name="think", kwargs={"thought": "x"}))
    assert res.observation.startswith("Error:") and env.actions == []
    # customer messages are never faulted
    env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "hello"}))
    assert len(env.actions) == 1
    inj.fault_rate = 0.0
    env.step(Action(name="think", kwargs={"thought": "retry"}))
    finalize()
    assert inj.state["faults_injected"] == 1
    assert inj.state["recoveries_successful"] == 1
    assert inj.fault_events[0].recovered


def test_tool_faults_never_touch_reward_replay():
    """Env.calculate_reward replays the ground truth through env.step; with
    every tool call faulted the reward must still match a fault-free episode."""
    from hal.utils.fault_injection import FaultInjector

    env = airline_env()
    env.reset(task_index=AIRLINE_TASK)
    inj = FaultInjector(fault_rate=0.0)  # the agent's own calls go through...
    install_tool_faults(env, inj)
    for a in env.task.actions:
        if a.name != RESPOND_ACTION_NAME:
            env.step(Action(name=a.name, kwargs=dict(a.kwargs)))
    inj.fault_rate = 1.0  # ...then every call would fail, including the replay
    res = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Goodbye"}))
    assert res.done and res.reward == 1.0
    assert inj.state["faults_injected"] == 0 and inj.state["recoveries_successful"] == 0


class _StubLLM(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _StubLLM.seen.append((self.path, body))
        sse = (
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3}}\n\n'
            b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)


def test_forwarder_rewrites_model_pins_temperature_and_traces(tmp_path, monkeypatch):
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _StubLLM)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    monkeypatch.setenv("HAL_LOCAL_TRACE_DIR", str(tmp_path))
    env = airline_env()
    ep = Episode(env, env.wiki, AIRLINE_TASK)
    bridge = Bridge(ep, env.tools_info, "Qwen/Served", f"http://127.0.0.1:{stub.server_address[1]}/v1", "7")
    port = bridge.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {"model": "agent", "temperature": 0.9, "stream": True, "messages": [],
                 "tools": [{"type": "function", "function": {"name": "tau_think"}}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
    finally:
        bridge.stop()
        stub.shutdown()
    assert b"[DONE]" in raw
    path, sent = _StubLLM.seen[-1]
    assert path == "/v1/chat/completions"
    assert sent["model"] == "Qwen/Served" and sent["temperature"] == 0.0
    assert sent["stream_options"] == {"include_usage": True}
    assert bridge.first_request_tools == ["tau_think"]
    rec = json.loads((tmp_path / "7.jsonl").read_text().splitlines()[0])
    assert rec["usage"] == {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
    assert rec["model"] == "Qwen/Served"


class _FlakyLLM(BaseHTTPRequestHandler):
    """400 'Already borrowed' on the first call, then a normal JSON answer."""

    protocol_version = "HTTP/1.1"
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        _FlakyLLM.calls += 1
        if _FlakyLLM.calls == 1:
            code, body = 400, b'{"error":{"message":"Already borrowed","code":400}}'
        else:
            code, body = 200, b'{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_forwarder_retries_tokenizer_race():
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _FlakyLLM)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    env = airline_env()
    ep = Episode(env, env.wiki, AIRLINE_TASK)
    bridge = Bridge(ep, env.tools_info, "m", f"http://127.0.0.1:{stub.server_address[1]}/v1", "9")
    port = bridge.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"model": "agent", "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            assert r.status == 200
    finally:
        bridge.stop()
        stub.shutdown()
    assert _FlakyLLM.calls == 2 and bridge.transient_retries == 1 and bridge.llm_errors == []


class _NoReasoningLLM(BaseHTTPRequestHandler):
    """vLLM 0.17 /v1/responses: rejects any echoed `reasoning` input item."""

    protocol_version = "HTTP/1.1"
    bodies = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _NoReasoningLLM.bodies.append(req)
        if any(i.get("type") == "reasoning" for i in req["input"]):
            code, body = 400, b"{\"error\":{\"message\":\"1 validation error [{'type': 'literal_error', 'loc': ('body', 'input'), 'msg': \\\"Input should be 'custom_tool_call'\\\", 'input': 'reasoning'}]\"}}"
        else:
            code, body = 200, b'{"output":[],"usage":{"input_tokens":1,"output_tokens":1}}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_codex_reasoning_items_dropped_and_output_capped():
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _NoReasoningLLM)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    env = airline_env()
    ep = Episode(env, env.wiki, AIRLINE_TASK)
    bridge = Bridge(ep, env.tools_info, "m", f"http://127.0.0.1:{stub.server_address[1]}/v1", "9")
    port = bridge.start()
    items = [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "reasoning", "id": "rs_1", "content": [{"type": "reasoning_text", "text": "t"}], "encrypted_content": None},
        {"type": "message", "role": "assistant", "content": "hello"},
    ]
    try:
        for _ in range(2):  # the second request is cleaned before it is sent
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/responses",
                data=json.dumps({"model": "agent", "input": items, "tools": [{"type": "function", "name": "f"}]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                assert r.status == 200 and json.loads(r.read())["output"] == []
    finally:
        bridge.stop()
        stub.shutdown()
    first, dropped, second = _NoReasoningLLM.bodies
    assert [i["type"] for i in first["input"]] == ["message", "reasoning", "message"]
    assert [i["type"] for i in dropped["input"]] == ["message", "message"]
    assert [i["type"] for i in second["input"]] == ["message", "message"]
    assert first["max_output_tokens"] == scaffolds.MAX_OUTPUT_TOKENS
    assert bridge.reasoning_rewrites == 2 and bridge.llm_errors == []


class _FlakyStream(BaseHTTPRequestHandler):
    """A 200 SSE stream whose error event carries 'Already borrowed' on the first call."""

    protocol_version = "HTTP/1.1"
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        _FlakyStream.calls += 1
        if _FlakyStream.calls == 1:
            body = b'data: {"error":{"message":"Already borrowed","code":400}}\n\ndata: [DONE]\n\n'
        else:
            body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_forwarder_retries_tokenizer_race_inside_stream():
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _FlakyStream)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    env = airline_env()
    ep = Episode(env, env.wiki, AIRLINE_TASK)
    bridge = Bridge(ep, env.tools_info, "m", f"http://127.0.0.1:{stub.server_address[1]}/v1", "8")
    port = bridge.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"model": "agent", "stream": True, "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            out = r.read()
    finally:
        bridge.stop()
        stub.shutdown()
    assert b"ok" in out and b"Already borrowed" not in out
    assert _FlakyStream.calls == 2 and bridge.transient_retries == 1


def test_invalid_tool_arguments_wrapped_in_history():
    items = [
        {"type": "function_call", "call_id": "a", "name": "mcp__tau__think", "arguments": '{"thought": "ok"}'},
        {"type": "function_call", "call_id": "b", "name": "mcp__tau__think", "arguments": '{"thought": "unterminated'},
    ]
    scaffolds._simplify_input_items(items)
    assert items[0]["arguments"] == '{"thought": "ok"}'
    assert json.loads(items[1]["arguments"]) == {"_invalid_json": '{"thought": "unterminated'}
    msgs = [{"role": "assistant", "tool_calls": [{"id": "c", "type": "function",
             "function": {"name": "think", "arguments": "{bad"}}]}]
    scaffolds._wrap_invalid_chat_arguments(msgs)
    assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"]) == {"_invalid_json": "{bad"}


def test_claude_system_turns_and_api_errors():
    body = json.dumps({"system": "policy", "messages": [
        {"role": "user", "content": "hi"}, {"role": "system", "content": "note"}]}).encode()
    fixed = json.loads(scaffolds._system_turns_to_user(body))
    assert [m["role"] for m in fixed["messages"]] == ["user", "user"] and fixed["system"] == "policy"
    assert scaffolds._system_turns_to_user(json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()) is None
    runner = scaffolds.ClaudeRunner.__new__(scaffolds.ClaudeRunner)
    ok = json.dumps({"result": "Hello!", "session_id": "s1", "is_error": False})
    assert runner.parse(ok) == ("Hello!", "s1")
    bad = json.dumps({"result": "API Error: 400 ...", "session_id": "s1", "is_error": False, "api_error_status": 400})
    with pytest.raises(RuntimeError):
        runner.parse(bad)


def test_codex_namespace_round_trip():
    tools = [
        {"type": "function", "name": "get_goal"},
        {"type": "namespace", "name": "mcp__tau", "tools": [{"type": "function", "name": "think"}]},
        {"type": "namespace", "name": "multi_agent_v1", "tools": [{"type": "function", "name": "spawn"}]},
    ]
    assert scaffolds._flatten_mcp_namespaces(tools) == [{"type": "function", "name": "mcp__tau__think"}]
    line = b'data: {"type":"response.output_item.done","item":{"type":"function_call","name":"mcp__tau__think","arguments":"{}"}}'
    item = json.loads(scaffolds._namespace_sse_line(line)[5:])["item"]
    assert (item["namespace"], item["name"]) == ("mcp__tau", "think")
    echoed = {"input": [{"type": "function_call", "namespace": "mcp__tau", "name": "think"}]}
    scaffolds._walk_function_calls(echoed, scaffolds._to_flat_call)
    assert echoed["input"][0] == {"type": "function_call", "name": "mcp__tau__think"}
    assert scaffolds._namespace_sse_line(b"event: response.created") == b"event: response.created"


def test_codex_input_simplified_for_old_vllm():
    items = [
        {"type": "message", "id": "m1", "role": "developer", "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]},
        {"type": "message", "id": "m2", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
        {"type": "function_call", "call_id": "c", "name": "mcp__tau__think", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": [{"type": "input_text", "text": "Wall time: 0.0s\nok"}]},
        {"role": "user", "content": [{"type": "input_image", "image_url": "x"}]},
    ]
    scaffolds._simplify_input_items(items)
    assert items[0] == {"type": "message", "role": "developer", "content": "ab"}
    assert items[1] == {"type": "message", "role": "assistant", "content": "hi"}
    assert items[2]["name"] == "mcp__tau__think"
    assert items[3]["output"] == "Wall time: 0.0s\nok"
    assert items[4]["content"] == [{"type": "input_image", "image_url": "x"}]  # non-text left alone


@pytest.mark.parametrize(
    "raw,expected",
    [
        (b'{"usage":{"prompt_tokens":5,"completion_tokens":2}}', (5, 2)),
        (
            b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":40,"output_tokens":1}}}\n\n'
            b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":17}}\n\n',
            (40, 17),
        ),
        (
            b'data: {"type":"response.completed","response":{"usage":{"input_tokens":9,"output_tokens":4}}}\n\n',
            (9, 4),
        ),
        (b"data: [DONE]\n\n", None),
    ],
)
def test_parse_usage_formats(raw, expected):
    assert parse_usage(raw) == expected
