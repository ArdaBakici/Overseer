"""A scripted stand-in for the OpenAI Responses API (codex) and Chat Completions API (opencode).

Turn 1 (no tool output yet): call the tool whose name ends with `target_tool` with `target_args`.
Turn 2 (tool output present): answer "DONE: <first 80 chars of the tool output>".
Requests without tools (titles, one-shot reports) get a plain "OK" answer.
Every request body is recorded in `requests`.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeModel:
    def __init__(self, target_tool="read_file", target_args=None):
        self.target_tool = target_tool
        self.target_args = target_args or {}
        self.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()

    def tool_names(self, body):
        """Flat tool names; tools inside a Responses-API namespace are reported as 'namespace/tool'."""
        names = []
        for t in body.get("tools") or []:
            if t.get("type") == "namespace":
                names += [f"{t['name']}/{sub.get('name')}" for sub in t.get("tools") or []]
            else:
                names.append(t.get("name") or (t.get("function") or {}).get("name"))
        return [n for n in names if n]

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"object": "list", "data": [{"id": "fake-model", "object": "model"}]}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                fake.requests.append({"path": self.path, "body": body})
                if self.path.endswith("/responses"):
                    events = fake.responses(body)
                elif self.path.endswith("/chat/completions"):
                    events = fake.chat(body)
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                for name, data in events:
                    chunk = (f"event: {name}\n" if name else "") + f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()

        return H

    def _decide(self, tools, outputs):
        """Return ('call', name, args) or ('text', text)."""
        if not tools:
            return ("text", "OK")
        if not outputs:
            match = [t for t in tools if t.endswith(self.target_tool)]
            if match:
                return ("call", match[0], self.target_args)
            return ("text", "OK")
        out = outputs[-1].split("Output:", 1)[-1].lstrip()  # codex prefixes "Wall time: ...\nOutput:"
        return ("text", "DONE: " + out[:80])

    # ---- OpenAI Responses API (codex)
    def responses(self, body):
        outputs = []
        for item in body.get("input") or []:
            if isinstance(item, dict) and item.get("type") in ("function_call_output", "custom_tool_call_output"):
                out = item.get("output")
                if isinstance(out, list):
                    out = "".join(p.get("text", "") for p in out if isinstance(p, dict))
                elif isinstance(out, dict):
                    out = out.get("content") or json.dumps(out)
                outputs.append(str(out))
        decision = self._decide(self.tool_names(body), outputs)
        n = len(self.requests)
        if decision[0] == "call":
            namespace, _, name = decision[1].rpartition("/")
            item = {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "name": name,
                    "arguments": json.dumps(decision[2]), "status": "completed"}
            if namespace:
                item["namespace"] = namespace
        else:
            item = {"type": "message", "role": "assistant", "id": f"msg_{n}", "status": "completed",
                    "content": [{"type": "output_text", "text": decision[1], "annotations": []}]}
        usage = {"input_tokens": 1, "input_tokens_details": None, "output_tokens": 1,
                 "output_tokens_details": None, "total_tokens": 2}
        return [
            ("response.created", {"type": "response.created", "response": {"id": f"resp_{n}"}}),
            ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0, "item": item}),
            ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": item}),
            ("response.completed", {"type": "response.completed", "response": {"id": f"resp_{n}", "usage": usage}}),
        ]

    # ---- Chat Completions API (opencode via @ai-sdk/openai-compatible)
    def chat(self, body):
        outputs = [str(m.get("content")) for m in body.get("messages") or [] if m.get("role") == "tool"]
        decision = self._decide(self.tool_names(body), outputs)
        n = len(self.requests)
        base = {"id": f"chat_{n}", "object": "chat.completion.chunk", "created": 0, "model": "fake-model"}
        if decision[0] == "call":
            delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": f"call_{n}", "type": "function",
                                                          "function": {"name": decision[1], "arguments": json.dumps(decision[2])}}]}
            finish = "tool_calls"
        else:
            delta = {"role": "assistant", "content": decision[1]}
            finish = "stop"
        return [
            (None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}),
            (None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
            (None, "[DONE]"),
        ]
