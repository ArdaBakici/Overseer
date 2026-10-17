"""The dashboard: a Telegram Mini App for browsing what chat messages are poor at. The platform serves the page and
checks who asks; the domain gives its tabs (dashboard_js) and their data (dashboard_api) - for the HPC: jobs,
results, findings and a timeline (hpc/dashboard.py, hpc/dashboard.js).

Private by construction:
- Overseer serves it on this machine only (dashboard.host, 127.0.0.1 by default). `tailscale serve` makes it
  reachable over HTTPS on your tailnet - not the internet (that would be `tailscale funnel`). The Mini App opens in
  Telegram on your own device, so it only needs to reach your tailnet; Telegram's servers never fetch it.
- Every data request carries Telegram's signed launch data (initData, HMAC-signed with the bot's token): Overseer
  checks the signature, its age and that the user is one of telegram.allowed_chat_ids. A browser outside Telegram,
  or another device on the tailnet, gets nothing.
- It only reads: no button in it changes anything (that stays in the chat, where every action is announced).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import time
from importlib import resources
from urllib.parse import parse_qs, parse_qsl, urlsplit

from .config import Config
from .store import Store

log = logging.getLogger(__name__)

MAX_REQUEST = 16_384
JOB_ID = re.compile(r"^[\w.\[\]-]{1,64}$")
STATUS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 405: "Method Not Allowed",
          500: "Internal Server Error", 502: "Bad Gateway"}
CSP = ("default-src 'none'; script-src 'self' https://telegram.org; style-src 'unsafe-inline'; "
       "img-src 'self' blob: data:; connect-src 'self'; base-uri 'none'; form-action 'none'")


def verify_init_data(init_data: str, bot_token: str, allowed: list[int], max_age: float = 86400,
                     now: float | None = None) -> int | None:
    """The Telegram user id in a Mini App's initData if its signature is the bot's, it is recent and the user is
    allowed; else None. (https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app)"""
    if not init_data or not bot_token:
        return None
    try:
        pairs = parse_qsl(init_data, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        return None
    fields = dict(pairs)
    got = fields.pop("hash", "")
    if not got or len(fields) + 1 != len(pairs):  # (a field given twice)
        return None
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    want = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(want, got):
        return None
    try:
        age = (now or time.time()) - int(fields.get("auth_date", "0"))
        user = json.loads(fields.get("user", "{}"))
        uid = int(user.get("id"))
    except (ValueError, TypeError, AttributeError):
        return None
    if age > max_age or age < -300 or uid not in allowed:
        return None
    return uid


def json_answer(data) -> tuple[int, str, bytes, dict]:
    return 200, "application/json", json.dumps(data).encode(), {}


class Dashboard:
    def __init__(self, cfg: Config, store: Store, bot_token: str, domain):
        self.cfg = cfg
        self.store = store
        self.token = bot_token
        self.domain = domain  # its tabs (dashboard_js) and their data (dashboard_api)
        self.server: asyncio.AbstractServer | None = None

    @property
    def url(self) -> str:
        return self.cfg.dashboard.url.strip()

    async def start(self) -> None:
        c = self.cfg.dashboard
        self.server = await asyncio.start_server(self._client, host=c.host, port=c.port, limit=MAX_REQUEST)
        self.port = self.server.sockets[0].getsockname()[1]  # (port 0: any free one - tests)
        log.info("dashboard on http://%s:%s (%s)", c.host, c.port, self.url or "dashboard.url not set")

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    # ------------------------------------------------------------------ HTTP (just enough: GET, one per connection)
    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            try:
                head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError):
                return
            lines = head.decode("latin-1").split("\r\n")
            try:
                method, target, _ = lines[0].split(" ", 2)
            except ValueError:
                await self._send(writer, 400, b"bad request")
                return
            headers = {k.strip().lower(): v.strip() for k, _, v in (x.partition(":") for x in lines[1:] if x)}
            if method not in ("GET", "HEAD"):
                await self._send(writer, 405, b"read-only")
                return
            status, ctype, body, extra = await self.route(target, headers)
            await self._send(writer, status, body if method == "GET" else b"", ctype, extra,
                             length=len(body))
        except Exception:
            log.exception("dashboard request failed")
            try:
                await self._send(writer, 500, b"error")
            except Exception:
                pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _send(self, writer, status: int, body: bytes, ctype: str = "text/plain; charset=utf-8",
                    extra: dict | None = None, length: int | None = None) -> None:
        head = [f"HTTP/1.1 {status} {STATUS.get(status, '')}", f"Content-Type: {ctype}",
                f"Content-Length: {len(body) if length is None else length}", "Cache-Control: no-store",
                "X-Content-Type-Options: nosniff", "Referrer-Policy: no-referrer", "Connection: close",
                *[f"{k}: {v}" for k, v in (extra or {}).items()]]
        writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body)
        await writer.drain()

    # ------------------------------------------------------------------ routes
    async def route(self, target: str, headers: dict[str, str]) -> tuple[int, str, bytes, dict]:
        url = urlsplit(target)
        path, query = url.path, {k: v[-1] for k, v in parse_qs(url.query).items()}
        if path in ("/", "/index.html"):
            page = resources.files("overseer.core").joinpath("dashboard.html").read_bytes()
            return 200, "text/html; charset=utf-8", page, {"Content-Security-Policy": CSP}
        if path == "/app.js":  # the domain's tabs
            return 200, "text/javascript; charset=utf-8", self.domain.dashboard_js(), {}
        if not path.startswith("/api/"):
            return 404, "text/plain", b"not found", {}
        if verify_init_data(headers.get("x-telegram-init-data", ""), self.token,
                            self.cfg.telegram.allowed_chat_ids) is None:
            return 401, "application/json", b'{"error": "open the dashboard from Telegram"}', {}
        parts = path[5:].strip("/").split("/")
        try:
            answer = await self.domain.dashboard_api(parts, query)  # the domain's data (e.g. jobs, results)
        except ValueError:
            return 400, "text/plain", b"bad request", {}
        return answer or (404, "text/plain", b"not found", {})

