"""A scripted stand-in for the model APIs the agent CLIs talk to.

* Anthropic Messages API (Claude Code):      POST /v1/messages        (SSE)
* OpenAI Responses API (Codex):              POST /v1/responses       (SSE)

`FakeModel(script=[cmd1, cmd2, ...])`: on each request that offers tools, the model calls
the CLI's shell tool with the next command; once all commands ran it answers
"DONE" plus the outputs. Requests without tools (titles, one-shot reports) get "OK".
With `think=True` each tool call is preceded by a reasoning summary ("THOUGHT <n>") and a
short message ("NOTE <n>"), like real models' thinking and progress updates.
All tool outputs the CLI sends back are recorded in `.outputs` (in order).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeModel:
    def __init__(self, script, think: bool = False):
        self.script = list(script)
        self.think = think
        self.requests = []
        self.outputs: list[str] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._json({"object": "list", "data": [{"id": "fake-model", "object": "model", "type": "model",
                                                        "display_name": "Fake"}], "has_more": False})

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("content-length", "0")
                self.end_headers()

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("content-length", 0)) or 0)
                body = json.loads(raw or b"{}")
                path = self.path.split("?")[0]
                fake.requests.append({"path": path, "body": body})
                if path.endswith("/count_tokens"):
                    return self._json({"input_tokens": 10})
                if path.endswith("/messages"):
                    events = fake.anthropic(body)
                elif path.endswith("/responses"):
                    events = fake.responses(body)
                else:
                    return self._json({"error": "not found"}, 404)
                if not body.get("stream"):
                    return self._json(events)  # non-streaming fallback: a full message object
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "close")
                self.end_headers()
                for name, data in events:
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
                self.close_connection = True

        return H

    def _next(self, n_outputs: int, has_tools: bool):
        if not has_tools:
            return ("text", "OK")
        if n_outputs < len(self.script):
            return ("call", self.script[n_outputs])
        return ("text", "DONE\n" + "\n---\n".join(self.outputs[-len(self.script):]))

    # ---------------------------------------------------------------- Anthropic Messages (Claude Code)
    def anthropic(self, body):
        outputs = []
        for m in body.get("messages") or []:
            if m.get("role") != "user" or not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                    outputs.append(str(c))
        tools = [t.get("name") for t in body.get("tools") or []]
        if outputs and len(outputs) > len(self.outputs):
            self.outputs = outputs
        decision = self._next(len(outputs), "Bash" in tools)
        n = len(self.requests)
        usage = {"input_tokens": 1, "output_tokens": 1}
        msg = {"id": f"msg_{n}", "type": "message", "role": "assistant", "model": body.get("model", "fake"),
               "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage}
        if decision[0] == "call":
            block = {"type": "tool_use", "id": f"toolu_{n}", "name": "Bash", "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps({"command": decision[1], "description": "step"})}
            stop = "tool_use"
        else:
            block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": decision[1]}
            stop = "end_turn"
        blocks = [(block, [delta])]
        if self.think and decision[0] == "call":
            k = len(outputs) + 1
            blocks = [({"type": "thinking", "thinking": "", "signature": ""},
                       [{"type": "thinking_delta", "thinking": f"THOUGHT {k}: checking step {k}"},
                        {"type": "signature_delta", "signature": "c2ln"}]),
                      ({"type": "text", "text": ""}, [{"type": "text_delta", "text": f"NOTE {k}"}])] + blocks
        if not body.get("stream"):
            content = []
            for b, deltas in blocks:
                full = dict(b)
                for d in deltas:
                    if d["type"] == "input_json_delta":
                        full["input"] = json.loads(d["partial_json"])
                    elif d["type"] == "text_delta":
                        full["text"] = d["text"]
                    elif d["type"] == "thinking_delta":
                        full["thinking"] = d["thinking"]
                    elif d["type"] == "signature_delta":
                        full["signature"] = d["signature"]
                content.append(full)
            return {**msg, "content": content, "stop_reason": stop}
        events = [("message_start", {"type": "message_start", "message": msg})]
        for i, (b, deltas) in enumerate(blocks):
            events.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": b}))
            events += [("content_block_delta", {"type": "content_block_delta", "index": i, "delta": d}) for d in deltas]
            events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
        return events + [
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                               "usage": {"output_tokens": 1}}),
            ("message_stop", {"type": "message_stop"}),
        ]

    # ---------------------------------------------------------------- OpenAI Responses (Codex)
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
        if outputs and len(outputs) > len(self.outputs):
            self.outputs = outputs
        tools = {}
        for t in body.get("tools") or []:
            if t.get("name"):
                tools[t["name"]] = t
        shell = next((n for n in ("exec_command", "shell_command", "shell") if n in tools), None)
        decision = self._next(len(outputs), shell is not None)
        n = len(self.requests)
        if decision[0] == "call":
            if shell == "exec_command":
                args = {"cmd": decision[1]}
            elif shell == "shell_command":
                args = {"command": decision[1]}
            else:
                args = {"command": ["bash", "-lc", decision[1]]}
            item = {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "name": shell,
                    "arguments": json.dumps(args), "status": "completed"}
        else:
            item = {"type": "message", "role": "assistant", "id": f"msg_{n}", "status": "completed",
                    "content": [{"type": "output_text", "text": decision[1], "annotations": []}]}
        items = [item]
        if self.think and decision[0] == "call":
            k = len(outputs) + 1
            items = [{"type": "reasoning", "id": f"rs_{n}", "encrypted_content": None,
                      "summary": [{"type": "summary_text", "text": f"THOUGHT {k}: checking step {k}"}]},
                     {"type": "message", "role": "assistant", "id": f"msg_{n}_note", "status": "completed",
                      "content": [{"type": "output_text", "text": f"NOTE {k}", "annotations": []}]}] + items
        usage = {"input_tokens": 1, "input_tokens_details": None, "output_tokens": 1,
                 "output_tokens_details": None, "total_tokens": 2}
        events = [("response.created", {"type": "response.created", "response": {"id": f"resp_{n}"}})]
        for i, it in enumerate(items):
            events.append(("response.output_item.added", {"type": "response.output_item.added", "output_index": i, "item": it}))
            events.append(("response.output_item.done", {"type": "response.output_item.done", "output_index": i, "item": it}))
        return events + [
            ("response.completed", {"type": "response.completed", "response": {"id": f"resp_{n}", "usage": usage}}),
        ]
