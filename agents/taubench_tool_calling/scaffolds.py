"""Alternative agent scaffolds for the tau-bench tool-calling agent.

tool_calling.run keeps everything around the agent loop (env setup, prompt and
structural perturbations, confidence, compliance, result assembly) and only
swaps the loop itself:

  tc        tau_bench ToolCallingAgent (the published protocol)
  react     tau_bench ChatReActAgent (text Thought/Action, no native tool calls)
  opencode  opencode CLI  --+
  codex     Codex CLI       +- solve_cli(): the CLI runs as a subprocess and
  claude    Claude Code   --+  reaches the env through a per-task bridge

Bridge (one loopback HTTP server per task, both sides of the CLI):
  /mcp     MCP over streamable HTTP (JSON responses) exposing the env's tools;
           every call becomes env.step(Action) on the (perturbed) env
  /v1/...  pass-through to the agent vLLM proxy: pins temperature=0, maps the
           alias model name to the served id, and appends one local-trace
           record per LLM call to $HAL_LOCAL_TRACE_DIR/<task_id>.jsonl

The customer channel mirrors tc: the CLI's plain-text reply is sent to the
user simulator as a respond action and the customer's answer becomes the next
prompt of the same (resumed) CLI session. The bridge records the episode as an
OpenAI-format message list, so confidence, abstention, compliance and the
safety judge read a CLI run exactly like a tool-calling run.

install_tool_faults() is the scaffold-agnostic fault channel (fault_mode=tool):
a tool call fails before reaching the env, and the agent sees the error.
"""

import http.client
import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from tau_bench.types import (
    RESPOND_ACTION_FIELD_NAME,
    RESPOND_ACTION_NAME,
    Action,
    EnvInfo,
    EnvResponse,
    SolveResult,
)

CLI_SCAFFOLDS = ("opencode", "codex", "claude")
CLI_BIN = os.environ.get("HAL_SCAFFOLD_CLI_BIN", "/mfs1/u/viet/envs/scaffold_clis/bin")
NODE_BIN = os.environ.get(
    "HAL_SCAFFOLD_NODE_BIN", "/h/300/viet/.nvm/versions/node/v24.15.0/bin"
)
# Shared across tasks so opencode installs its provider package once, not per task.
OPENCODE_CACHE = os.environ.get(
    "HAL_SCAFFOLD_OPENCODE_CACHE", "/mfs1/u/viet/envs/scaffold_clis/cache"
)
MODEL_ALIAS = "agent"  # what the CLIs ask for; the bridge maps it to the served id
MAX_ENV_STEPS = 30  # tc's max_num_steps: one env action per agent step
EPISODE_TIMEOUT_S = 1080  # inside hal-eval's 1200 s task timeout
LLM_PATHS = ("/chat/completions", "/messages", "/responses")
# Per-call output cap for Responses requests (Codex sends none). The job proxy
# clamps max_tokens for chat/messages at PROXY_MAX_TOKENS but does not know
# max_output_tokens, so an unbounded runaway generation would run into the
# episode timeout instead of ending as the model's (truncated) turn.
MAX_OUTPUT_TOKENS = int(
    os.environ.get("HAL_SCAFFOLD_MAX_OUTPUT_TOKENS") or os.environ.get("PROXY_MAX_TOKENS") or 4096
)


# ------------------------------------------------------------------ tool faults

_FAULT_TEXT = {
    "timeout": "Error: request timed out",
    "error_response": "Error: API returned error: 500 Internal Server Error",
    "rate_limit": "Error: Rate limit exceeded: 429 Too Many Requests",
    "network_error": "Error: Network error: Connection refused",
    "partial_failure": 'Error: incomplete response {"status": "partial", "data": null}',
    "invalid_response": "Error: invalid response format",
    "empty_response": "Error: empty response",
}


def install_tool_faults(env, injector):
    """Wrap env.step so each tool call (never a customer message) fails with
    probability injector.fault_rate before reaching the env. A fault counts as
    recovered once the same tool later succeeds. Returns a finalize() that
    books the unrecovered faults; call it after the episode."""
    from hal.utils.fault_injection import FaultEvent

    inner = env.step
    inner_reward = env.calculate_reward
    pending: List[Any] = []
    scoring = [False]

    def calculate_reward():
        # Env.calculate_reward replays the ground-truth actions through
        # self.step; those must neither fail nor count as recoveries.
        scoring[0] = True
        try:
            return inner_reward()
        finally:
            scoring[0] = False

    def step(action):
        if scoring[0]:
            return inner(action)
        if (
            action.name != RESPOND_ACTION_NAME
            and action.name in env.tools_map
            and random.random() < injector.fault_rate
        ):
            fault_type = injector._select_fault_type()
            injector.state["faults_injected"] += 1
            event = FaultEvent(
                fault_type=fault_type,
                recovered=False,
                recovery_time=0.0,
                context={"mode": "tool", "function_name": action.name},
            )
            injector.fault_events.append(event)
            pending.append(event)
            print(f"⚡ Tool fault injected: {action.name} -> {fault_type.value}")
            return EnvResponse(
                observation=_FAULT_TEXT[fault_type.value],
                reward=0.0,
                done=False,
                info=EnvInfo(task=env.task, source=action.name),
            )
        res = inner(action)
        for event in pending:
            if event.context["function_name"] == action.name:
                event.recovered = True
                injector.state["recoveries_successful"] += 1
        pending[:] = [e for e in pending if not e.recovered]
        return res

    def finalize():
        injector.state["recoveries_failed"] += len(pending)
        pending.clear()

    env.step = step
    env.calculate_reward = calculate_reward
    return finalize


# ------------------------------------------------------------------ episode


class Episode:
    """The env side of a CLI run: executes actions and records the transcript
    in the message format ToolCallingAgent.solve produces."""

    def __init__(self, env, wiki: str, task_index: int):
        self.env = env
        self.lock = threading.Lock()
        reset = env.reset(task_index=task_index)
        self.first_message = reset.observation
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": wiki},
            {"role": "user", "content": reset.observation},
        ]
        self.info = reset.info.model_dump()
        self.reward = 0.0
        self.steps = 0
        self.done = False
        self.end_reason: Optional[str] = None
        self.tool_calls = 0
        self.customer_turns = 0
        self._ids = 0

    def _step(self, action: Action):
        res = self.env.step(action)
        self.steps += 1
        self.reward = res.reward
        self.info = {**self.info, **res.info.model_dump()}
        if res.done:
            self.done = True
            self.end_reason = (
                "customer_stop"
                if action.name == RESPOND_ACTION_NAME
                else f"terminal_tool:{action.name}"
            )
        return res

    def _capped(self) -> bool:
        if self.steps >= MAX_ENV_STEPS and not self.done:
            self.end_reason = self.end_reason or "step_cap"
            return True
        return False

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Tuple[str, bool]:
        """One MCP tools/call. Returns (text, is_error)."""
        with self.lock:
            if self.done:
                return "The conversation has ended. Do not call any more tools.", True
            if self._capped():
                return "Step limit reached. Stop now.", True
            call_id = f"call_{self._ids}"
            self._ids += 1
            self.tool_calls += 1
            self.messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            )
            res = self._step(Action(name=name, kwargs=dict(arguments)))
            self.messages.append(
                {"role": "tool", "tool_call_id": call_id, "name": name, "content": res.observation}
            )
            if res.done:
                return res.observation + "\n\n[The conversation has ended.]", False
            return res.observation, False

    def respond(self, text: str) -> Optional[str]:
        """Send the agent's reply to the customer; returns the customer's answer,
        or None when the episode is over or out of steps."""
        with self.lock:
            if self.done or self._capped():
                return None
            self.customer_turns += 1
            self.messages.append({"role": "assistant", "content": text})
            res = self._step(Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: text}))
            self.messages.append({"role": "user", "content": res.observation})
            return res.observation


# ------------------------------------------------------------------ usage sniffing


def _usage_of(obj: Any) -> Optional[Tuple[int, int]]:
    """(prompt, completion) tokens from a chat, messages or responses payload/event."""
    if not isinstance(obj, dict):
        return None
    for holder in (obj, obj.get("response") or {}, obj.get("message") or {}):
        u = holder.get("usage") if isinstance(holder, dict) else None
        if isinstance(u, dict):
            p = u.get("prompt_tokens", u.get("input_tokens"))
            c = u.get("completion_tokens", u.get("output_tokens"))
            if p is not None or c is not None:
                return int(p or 0), int(c or 0)
    return None


def parse_usage(raw: bytes) -> Optional[Tuple[int, int]]:
    """Token usage of one upstream response, JSON or SSE. For Anthropic SSE the
    prompt count arrives in message_start and the completion count in
    message_delta, so the maximum of each field over all events is kept."""
    text = raw.decode("utf-8", "replace").strip()
    try:
        return _usage_of(json.loads(text))
    except ValueError:
        pass
    prompt = completion = None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            u = _usage_of(json.loads(payload))
        except ValueError:
            continue
        if u:
            prompt = max(prompt or 0, u[0])
            completion = max(completion or 0, u[1])
    if prompt is None:
        return None
    return prompt, completion


def _flatten_mcp_namespaces(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Codex sends MCP tools as a Responses `namespace` tool, which vLLM does
    not understand. Keep only the MCP tools, as plain functions named
    mcp__<server>__<tool> (Codex's flat MCP naming); drop everything else, so
    every scaffold offers the model exactly the env's tools."""
    flat = []
    for t in tools:
        if t.get("type") == "namespace" and str(t.get("name", "")).startswith("mcp__"):
            for fn in t.get("tools") or []:
                flat.append({**fn, "name": f"{t['name']}__{fn['name']}"})
    return flat


def _walk_function_calls(obj: Any, fix) -> None:
    if isinstance(obj, dict):
        if obj.get("type") == "function_call":
            fix(obj)
        for v in obj.values():
            _walk_function_calls(v, fix)
    elif isinstance(obj, list):
        for v in obj:
            _walk_function_calls(v, fix)


def _to_flat_call(item: Dict[str, Any]) -> None:
    """Codex echoes past calls as {name, namespace}; the model saw flat names."""
    ns = item.pop("namespace", None)
    if ns:
        item["name"] = f"{ns}__{item['name']}"


_TEXT_PARTS = ("input_text", "output_text", "text")


def _json_or_wrapped(arguments: Any) -> Any:
    """Tool-call arguments echoed back in a request's history. vLLM parses them
    and rejects the whole request (400) if a model once emitted invalid JSON,
    which the OpenAI APIs the CLIs target would accept. Keep valid JSON as is;
    wrap invalid text so the model still sees what it sent."""
    if not isinstance(arguments, str):
        return arguments
    try:
        json.loads(arguments)
        return arguments
    except ValueError:
        return json.dumps({"_invalid_json": arguments})


def _wrap_invalid_chat_arguments(messages: Any) -> None:
    """Chat-completions history: the same fix for assistant tool_calls."""
    for m in messages if isinstance(messages, list) else []:
        for tc in (m.get("tool_calls") or []) if isinstance(m, dict) else []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict):
                fn["arguments"] = _json_or_wrapped(fn.get("arguments"))


def _simplify_input_items(items: Any) -> None:
    """Codex echoes history in shapes vLLM 0.17's Responses schema rejects
    (0.23 accepts them): tool results as lists of text parts, and past replies
    as full output-message objects (id, status, output_text parts). Rewrite
    both to the plain {role, content: str} / string-output forms every version
    accepts. vLLM joins text parts into the prompt either way, so the model
    sees the same text."""
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            item["arguments"] = _json_or_wrapped(item.get("arguments"))
        if item.get("type") == "function_call_output" and isinstance(item.get("output"), list):
            item["output"] = "".join(p.get("text", "") for p in item["output"] if isinstance(p, dict))
        elif item.get("type", "message") == "message" and "role" in item:
            content = item.get("content")
            if isinstance(content, list) and all(
                isinstance(p, dict) and p.get("type") in _TEXT_PARTS for p in content
            ):
                item["content"] = "".join(p.get("text", "") for p in content)
                item.pop("id", None)
                item.pop("status", None)


def _to_namespaced_call(item: Dict[str, Any]) -> None:
    """The model calls mcp__tau__x; Codex routes {namespace: mcp__tau, name: x}."""
    parts = str(item.get("name", "")).split("__", 2)
    if len(parts) == 3 and parts[0] == "mcp":
        item["namespace"], item["name"] = f"mcp__{parts[1]}", parts[2]


def _namespace_sse_line(line: bytes) -> bytes:
    if not line.startswith(b"data:"):
        return line
    payload = line[5:].strip()
    if not payload.startswith(b"{") or b"function_call" not in payload:
        return line
    try:
        obj = json.loads(payload)
    except ValueError:
        return line
    _walk_function_calls(obj, _to_namespaced_call)
    return b"data: " + json.dumps(obj).encode()


# ------------------------------------------------------------------ bridge server


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            out = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    return out
                out += self.rfile.read(size)
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _send(self, code: int, body: bytes = b"", headers: Optional[Dict[str, str]] = None):
        self.send_response(code)
        if body:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/mcp"):
            return self._send(405)
        self.server.bridge.forward(self, b"")

    def do_DELETE(self):
        return self._send(405 if self.path.startswith("/mcp") else 404)

    def do_POST(self):
        body = self._body()
        if self.path.startswith("/mcp"):
            return self._mcp(body)
        self.server.bridge.forward(self, body)

    def _mcp(self, body: bytes):
        try:
            msg = json.loads(body)
        except ValueError:
            return self._send(400)
        bridge = self.server.bridge
        if isinstance(msg, list):
            out = [r for r in map(bridge.rpc, msg) if r is not None] or None
        else:
            out = bridge.rpc(msg)
        if out is None:
            return self._send(202)
        return self._send(200, json.dumps(out).encode(), {"Mcp-Session-Id": bridge.session_id})


def _drop_reasoning_items(items: Any) -> bool:
    """Remove echoed `reasoning` items from a Responses input list in place.
    vLLM 0.17 rejects them (Qwen3's come back with content but no summary); the
    Qwen3 chat template discards past reasoning anyway, so the model's input is
    unchanged. Returns True if anything was removed."""
    if not isinstance(items, list):
        return False
    kept = [i for i in items if not (isinstance(i, dict) and i.get("type") == "reasoning")]
    if len(kept) == len(items):
        return False
    items[:] = kept
    return True


def _system_turns_to_user(body: bytes) -> Optional[bytes]:
    """Anthropic-format request with any role=system turns inside `messages`
    re-labelled as user turns; None if there are none."""
    try:
        data = json.loads(body)
    except ValueError:
        return None
    msgs = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(msgs, list) or not any(isinstance(m, dict) and m.get("role") == "system" for m in msgs):
        return None
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            m["role"] = "user"
    return json.dumps(data).encode()


class _Replay:
    """An upstream response whose body was already read (to inspect an error)."""

    def __init__(self, resp, body: bytes):
        self.status = resp.status
        self._headers = resp.getheaders()
        self._body = body

    def getheaders(self):
        return self._headers

    def read1(self, n: int = -1) -> bytes:
        out, self._body = self._body, b""
        return out


class Bridge:
    def __init__(self, episode: Episode, tools_info, served_model: str, api_base: str, task_id: str):
        self.episode = episode
        self.served_model = served_model
        up = urlsplit(api_base)
        self.up_host, self.up_port = up.hostname, up.port or 80
        self.task_id = task_id
        self.session_id = f"tau-{task_id}-{os.getpid()}"
        self.tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "inputSchema": t["function"].get("parameters")
                or {"type": "object", "properties": {}},
            }
            for t in tools_info
        ]
        trace_dir = os.environ.get("HAL_LOCAL_TRACE_DIR")
        self.trace_path = os.path.join(trace_dir, f"{task_id}.jsonl") if trace_dir else None
        self.trace_lock = threading.Lock()
        self.llm_calls = 0
        self.transient_retries = 0
        self.system_role_rewrites = 0
        self.reasoning_rewrites = 0
        self.drop_reasoning = False  # set once upstream has rejected a reasoning item
        self.llm_errors: List[str] = []
        self.first_request_tools: Optional[List[str]] = None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.server.bridge = self
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> int:
        self.thread.start()
        return self.port

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    # -- MCP -------------------------------------------------------------------
    def rpc(self, m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        mid, method = m.get("id"), m.get("method")
        if mid is None:  # notification
            return None
        params = m.get("params") or {}
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion") or "2025-06-18",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "tau", "version": "1.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": self.tools}
        elif method == "tools/call":
            args = params.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            text, is_error = self.episode.call_tool(params.get("name", ""), args)
            result = {"content": [{"type": "text", "text": text}], "isError": is_error}
        else:
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    # -- LLM pass-through --------------------------------------------------------
    def forward(self, handler: _Handler, body: bytes):
        path = handler.path
        is_llm = path.split("?")[0].endswith(LLM_PATHS)
        if body:
            try:
                data = json.loads(body)
            except ValueError:
                data = None
                if is_llm:
                    enc = handler.headers.get("Content-Encoding", "none")
                    self.llm_errors.append(f"{path}: unparsed request body (Content-Encoding={enc}, {len(body)} bytes)")
            if isinstance(data, dict):
                if "model" in data:
                    data["model"] = self.served_model
                if is_llm:
                    data["temperature"] = 0.0
                    if path.split("?")[0].endswith("/responses") and data.get("tools"):
                        data["tools"] = _flatten_mcp_namespaces(data["tools"])
                        _walk_function_calls(data.get("input"), _to_flat_call)
                        _simplify_input_items(data.get("input"))
                        data.pop("include", None)  # reasoning.encrypted_content: OpenAI-only
                        if self.drop_reasoning and _drop_reasoning_items(data.get("input")):
                            self.reasoning_rewrites += 1
                    if path.split("?")[0].endswith("/responses"):
                        cap = data.get("max_output_tokens")
                        if not isinstance(cap, int) or cap > MAX_OUTPUT_TOKENS:
                            data["max_output_tokens"] = MAX_OUTPUT_TOKENS
                    if path.split("?")[0].endswith("/chat/completions"):
                        _wrap_invalid_chat_arguments(data.get("messages"))
                    if data.get("stream") and path.split("?")[0].endswith("/chat/completions"):
                        data.setdefault("stream_options", {})["include_usage"] = True
                    if self.first_request_tools is None and data.get("tools"):
                        self.first_request_tools = [
                            (t.get("function") or t).get("name", "?") for t in data["tools"]
                        ]
                body = json.dumps(data).encode()
                dump_dir = os.environ.get("HAL_SCAFFOLD_DUMP_DIR")  # debugging aid
                if dump_dir and is_llm:
                    os.makedirs(dump_dir, exist_ok=True)
                    name = f"{self.task_id}_{self.llm_calls:03d}_{os.getpid()}.json"
                    with open(os.path.join(dump_dir, name), "wb") as f:
                        f.write(body)
        headers = {
            k: v
            for k, v in handler.headers.items()
            if k.lower()
            not in ("host", "content-length", "transfer-encoding", "connection", "accept-encoding",
                    "authorization", "x-api-key")  # the job proxy injects the real key
        }
        headers["Content-Length"] = str(len(body))
        started = datetime.now()
        conn = None
        try:
            # vLLM's fast tokenizer can race under concurrent requests and fail
            # with "Already borrowed": as a 400, or, when streaming, as an error
            # event inside a 200 stream. It is a server-side transient, so an LLM
            # reply is read in full first and the request resent if it carries
            # that error; the CLI (which retries neither) never sees it. The CLI
            # still receives the same SSE bytes, only all at once.
            for attempt in range(7):
                conn = http.client.HTTPConnection(self.up_host, self.up_port, timeout=EPISODE_TIMEOUT_S)
                conn.request(handler.command, path, body=body, headers=headers)
                resp = conn.getresponse()
                if not is_llm:
                    break
                resp = _Replay(resp, resp.read())
                if (
                    resp.status == 400
                    and b"Input should be 'user' or 'assistant'" in resp._body
                    and _system_turns_to_user(body)
                    and attempt < 6
                ):
                    # vLLM 0.17's /v1/messages rejects the mid-conversation
                    # system-role turns Claude Code sends (0.23 accepts them).
                    body = _system_turns_to_user(body)
                    headers["Content-Length"] = str(len(body))
                    self.system_role_rewrites += 1
                    conn.close()
                    continue
                if (
                    resp.status == 400
                    and b"'input': 'reasoning'" in resp._body
                    and not self.drop_reasoning
                    and attempt < 6
                ):
                    # vLLM 0.17's /v1/responses rejects echoed reasoning items;
                    # drop them now and from every later request of this task.
                    self.drop_reasoning = True
                    data = json.loads(body)
                    if _drop_reasoning_items(data.get("input")):
                        body = json.dumps(data).encode()
                        headers["Content-Length"] = str(len(body))
                        self.reasoning_rewrites += 1
                        conn.close()
                        continue
                if b"Already borrowed" not in resp._body or attempt == 6:
                    break
                conn.close()
                self.transient_retries += 1
                time.sleep(0.5 * (attempt + 1))
            handler.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "content-length", "connection", "content-encoding"):
                    handler.send_header(k, v)
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()

            def emit(data: bytes):
                if data:
                    handler.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                    handler.wfile.flush()

            # Responses API (Codex): rewrite flat MCP calls back to namespaced
            # ones, line by line so SSE events stream through unchanged otherwise.
            rewrite = path.split("?")[0].endswith("/responses")
            raw, carry = b"", b""
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                raw += chunk
                if not rewrite:
                    emit(chunk)
                    continue
                carry += chunk
                *lines, carry = carry.split(b"\n")
                emit(b"".join(_namespace_sse_line(ln) + b"\n" for ln in lines))
            if carry:  # a non-streamed JSON body arrives as one unterminated line
                try:
                    obj = json.loads(carry)
                    _walk_function_calls(obj, _to_namespaced_call)
                    carry = json.dumps(obj).encode()
                except ValueError:
                    carry = _namespace_sse_line(carry)
                emit(carry)
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
        except (OSError, http.client.HTTPException) as e:
            self.llm_errors.append(f"{path}: {type(e).__name__}: {e}")
            return
        finally:
            if conn is not None:
                conn.close()
        if not is_llm:
            return
        self.llm_calls += 1
        if resp.status >= 400:
            text = raw.decode("utf-8", "replace")
            if len(text) > 2800:  # validation errors echo the whole request first
                text = text[:300] + " ... " + text[-2500:]
            self.llm_errors.append(f"{path}: HTTP {resp.status}: {text}")
            return
        usage = parse_usage(raw)
        if usage and self.trace_path:
            rec = {
                "task_id": self.task_id,
                "model": self.served_model,
                "started_at": started.isoformat(),
                "ended_at": datetime.now().isoformat(),
                "usage": {
                    "prompt_tokens": usage[0],
                    "completion_tokens": usage[1],
                    "total_tokens": usage[0] + usage[1],
                },
            }
            with self.trace_lock, open(self.trace_path, "a") as f:
                f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ CLI runners


class _Runner:
    """One CLI invocation per customer turn; the session carries the history."""

    def __init__(self, work: str, port: int, policy: str, context_window: int):
        self.work = work
        self.port = port
        self.context_window = context_window
        self.home = os.path.join(work, "home")
        self.cwd = os.path.join(work, "cwd")
        for d in (self.home, self.cwd):
            os.makedirs(d, exist_ok=True)
        self.policy_path = os.path.join(work, "policy.md")
        with open(self.policy_path, "w") as f:
            f.write(policy)
        self.env = {
            "PATH": f"{CLI_BIN}:{NODE_BIN}:/usr/local/bin:/usr/bin:/bin",
            "HOME": self.home,
            "XDG_CONFIG_HOME": os.path.join(self.home, ".config"),
            "XDG_DATA_HOME": os.path.join(self.home, ".local", "share"),
            "XDG_STATE_HOME": os.path.join(self.home, ".local", "state"),
            "XDG_CACHE_HOME": os.path.join(self.home, ".cache"),
            "TMPDIR": work,
            "LANG": "C.UTF-8",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
        self.setup()

    def setup(self):
        pass

    def command(self, prompt: str, session: Optional[str]) -> List[str]:
        raise NotImplementedError

    def parse(self, stdout: str) -> Tuple[str, Optional[str]]:
        raise NotImplementedError

    def run(self, prompt: str, session: Optional[str], timeout: float) -> Tuple[str, Optional[str], Dict[str, Any]]:
        t0 = time.time()
        try:
            p = subprocess.run(
                self.command(prompt, session),
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            rc = "timeout"
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        try:
            text, new_session = self.parse(out)
        except Exception as e:  # a CLI error result, or a changed output format
            text, new_session = "", None
            err = f"{err}\nparse error: {type(e).__name__}: {e}"
            if rc == 0:
                rc = "parse_error"  # an infrastructure failure, not a silent model
        diag = {
            "rc": rc,
            "seconds": round(time.time() - t0, 1),
            "reply_chars": len(text),
            "stdout_tail": out[-1500:],
            "stderr_tail": err[-1500:],
        }
        return text, new_session or session, diag


class OpencodeRunner(_Runner):
    def setup(self):
        tools_off = {
            t: False
            for t in (
                "bash", "edit", "write", "read", "grep", "glob", "list", "patch",
                "apply_patch", "todowrite", "todoread", "webfetch", "websearch",
                "task", "skill", "question", "lsp", "codesearch",
            )
        }
        config = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "provider": {
                "hal": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "hal",
                    "options": {
                        "baseURL": f"http://127.0.0.1:{self.port}/v1",
                        "apiKey": "hal-local",
                        "includeUsage": True,
                    },
                    "models": {
                        MODEL_ALIAS: {
                            "name": MODEL_ALIAS,
                            "tool_call": True,
                            "limit": {"context": self.context_window, "output": 4096},
                        }
                    },
                }
            },
            "model": f"hal/{MODEL_ALIAS}",
            "mcp": {
                "tau": {"type": "remote", "url": f"http://127.0.0.1:{self.port}/mcp", "enabled": True}
            },
            "agent": {
                "tau": {
                    "mode": "primary",
                    "description": "tau-bench airline agent",
                    "prompt": "{file:" + self.policy_path + "}",
                    "temperature": 0,
                    "tools": tools_off,
                    "permission": {"edit": "deny", "bash": "deny", "webfetch": "deny"},
                }
            },
        }
        self.config_path = os.path.join(self.work, "opencode.json")
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=1)
        self.env.update(
            {
                "OPENCODE_CONFIG": self.config_path,
                "XDG_CACHE_HOME": OPENCODE_CACHE,
                "OPENCODE_DISABLE_AUTOUPDATE": "1",
                "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
                "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
                "OPENCODE_DISABLE_MODELS_FETCH": "1",
                "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            }
        )

    def command(self, prompt, session):
        cmd = ["opencode", "run", "--pure", "--format", "json", "--agent", "tau", "-m", f"hal/{MODEL_ALIAS}"]
        if session:
            cmd += ["--session", session]
        return cmd + [prompt]

    def parse(self, stdout):
        session, texts = None, []
        for line in stdout.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            session = ev.get("sessionID") or (ev.get("part") or {}).get("sessionID") or session
            part = ev.get("part") or {}
            if ev.get("type") == "text" and part.get("text"):
                texts.append(part["text"])
            elif ev.get("type") in ("tool_use", "tool") or part.get("type") == "tool":
                texts = []  # only the text after the last tool call is the reply
        return "\n".join(texts).strip(), session


class CodexRunner(_Runner):
    def setup(self):
        self.codex_home = os.path.join(self.home, ".codex")
        os.makedirs(self.codex_home, exist_ok=True)
        self.last_path = os.path.join(self.work, "last_message.txt")
        config = f"""model = "{MODEL_ALIAS}"
model_provider = "hal"
model_instructions_file = "{self.policy_path}"
approval_policy = "never"
sandbox_mode = "read-only"
web_search = "disabled"
check_for_update_on_startup = false
model_context_window = {self.context_window}

[model_providers.hal]
name = "hal"
base_url = "http://127.0.0.1:{self.port}/v1"
env_key = "HAL_LOCAL_KEY"
wire_api = "responses"

[features]
shell_tool = false
unified_exec = false
multi_agent = false
goals = false
view_image = false
image_generation = false
apps = false
browser_use = false
computer_use = false
plugins = false
hooks = false
tool_suggest = false
sleep_tool = false
skill_search = false

[mcp_servers.tau]
url = "http://127.0.0.1:{self.port}/mcp"
tool_timeout_sec = 600
startup_timeout_sec = 60
default_tools_approval_mode = "approve"
"""
        with open(os.path.join(self.codex_home, "config.toml"), "w") as f:
            f.write(config)
        self.env.update({"CODEX_HOME": self.codex_home, "HAL_LOCAL_KEY": "hal-local"})

    def command(self, prompt, session):
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "-o", self.last_path]
        if session:
            cmd += ["resume", session]
        return cmd + [prompt]

    def parse(self, stdout):
        session, text = None, ""
        for line in stdout.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get("type") == "thread.started":
                session = ev.get("thread_id") or session
            item = ev.get("item") or {}
            if ev.get("type") == "item.completed" and item.get("type") == "agent_message":
                text = item.get("text") or text
        if os.path.exists(self.last_path):
            with open(self.last_path) as f:
                text = f.read().strip() or text
            os.remove(self.last_path)
        return text.strip(), session


class ClaudeRunner(_Runner):
    def setup(self):
        self.mcp_path = os.path.join(self.work, "mcp.json")
        with open(self.mcp_path, "w") as f:
            json.dump(
                {"mcpServers": {"tau": {"type": "http", "url": f"http://127.0.0.1:{self.port}/mcp"}}}, f
            )
        self.env.update(
            {
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{self.port}",
                "ANTHROPIC_API_KEY": "hal-local",
                "ANTHROPIC_MODEL": MODEL_ALIAS,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": MODEL_ALIAS,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": MODEL_ALIAS,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL_ALIAS,
                "ANTHROPIC_SMALL_FAST_MODEL": MODEL_ALIAS,
                "CLAUDE_CODE_SUBAGENT_MODEL": MODEL_ALIAS,
                "CLAUDE_CONFIG_DIR": os.path.join(self.home, ".claude"),
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096",
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                "DISABLE_TELEMETRY": "1",
                "DISABLE_ERROR_REPORTING": "1",
                "DISABLE_PROMPT_CACHING": "1",
                "MCP_TIMEOUT": "60000",
                "MCP_TOOL_TIMEOUT": "600000",
            }
        )

    def command(self, prompt, session):
        cmd = [
            "claude", "--bare", "-p", "--output-format", "json", "--model", MODEL_ALIAS,
            "--mcp-config", self.mcp_path, "--strict-mcp-config",
            "--tools", "", "--allowedTools", "mcp__tau__*",
            "--system-prompt-file", self.policy_path, "--permission-mode", "dontAsk",
        ]
        if session:
            cmd += ["--resume", session]
        return cmd + [prompt]

    def parse(self, stdout):
        obj = json.loads(stdout.strip().splitlines()[-1])
        # On an upstream API error `claude -p` still exits 0 and puts
        # "API Error: ..." in `result`; that is not a reply to the customer.
        if obj.get("is_error") or obj.get("api_error_status"):
            raise RuntimeError(f"claude api error {obj.get('api_error_status')}: {str(obj.get('result'))[:200]}")
        return (obj.get("result") or "").strip(), obj.get("session_id")


RUNNERS = {"opencode": OpencodeRunner, "codex": CodexRunner, "claude": ClaudeRunner}


def solve_cli(
    scaffold: str,
    env,
    tools_info,
    wiki: str,
    task_index: int,
    served_model: str,
    api_base: str,
    task_id: str,
    context_window: int = 65536,
) -> SolveResult:
    """Run one tau-bench episode with a CLI agent. Mirrors ToolCallingAgent.solve:
    resets the env, returns the reward and an OpenAI-format transcript."""
    episode = Episode(env, wiki, task_index)
    bridge = Bridge(episode, tools_info, served_model, api_base, task_id)
    port = bridge.start()
    work = tempfile.mkdtemp(prefix=f"hal_{scaffold}_")
    turns: List[Dict[str, Any]] = []
    try:
        runner = RUNNERS[scaffold](work, port, wiki, context_window)
        deadline = time.time() + EPISODE_TIMEOUT_S
        prompt, session = episode.first_message, None
        while not episode.done:
            remaining = deadline - time.time()
            if remaining < 10:
                episode.end_reason = episode.end_reason or "timeout"
                break
            text, session, diag = runner.run(prompt, session, remaining)
            turns.append(diag)
            if episode.done:
                break
            if diag["rc"] == "timeout":
                episode.end_reason = "timeout"
                break
            if not text:
                episode.end_reason = episode.end_reason or ("cli_error" if diag["rc"] else "cli_no_reply")
                break
            if not session:
                episode.end_reason = "cli_no_session"
                break
            reply = episode.respond(text)
            if reply is None or episode.done:
                break
            prompt = reply
    finally:
        bridge.stop()
        shutil.rmtree(work, ignore_errors=True)
    if not episode.done:
        episode.end_reason = episode.end_reason or "step_cap"
    # Infrastructure failures must not score as a model's 0: tc surfaces the
    # same cases (an LLM call that errors, the task timeout) as task errors,
    # which the unit check then retries. A silent model (cli_no_reply) or the
    # step cap are the model's outcome and keep their reward.
    infra = episode.end_reason in ("cli_error", "cli_no_session", "timeout")
    if infra or (bridge.llm_errors and not episode.done):
        last = turns[-1] if turns else {}
        raise RuntimeError(
            f"{scaffold} scaffold failure (end={episode.end_reason}, steps={episode.steps}): "
            f"llm_errors={bridge.llm_errors[:3]} rc={last.get('rc')} stderr={last.get('stderr_tail', '')[-400:]!r} "
            f"stdout={last.get('stdout_tail', '')[-800:]!r} secs={last.get('seconds')}"
        )
    info = dict(episode.info)
    info["scaffold"] = {
        "name": scaffold,
        "end_reason": episode.end_reason,
        "env_steps": episode.steps,
        "tool_calls": episode.tool_calls,
        "customer_turns": episode.customer_turns,
        "cli_invocations": len(turns),
        "llm_calls": bridge.llm_calls,
        "transient_retries": bridge.transient_retries,
        "system_role_rewrites": bridge.system_role_rewrites,
        "reasoning_rewrites": bridge.reasoning_rewrites,
        "llm_errors": bridge.llm_errors[:20],
        "model_tools": bridge.first_request_tools,
        "turns": turns[-3:],
    }
    print(
        f"🧩 {scaffold}: end={episode.end_reason} reward={episode.reward} steps={episode.steps} "
        f"tools={episode.tool_calls} turns={episode.customer_turns} llm_calls={bridge.llm_calls}"
    )
    return SolveResult(reward=episode.reward, messages=episode.messages, info=info, total_cost=0.0)
