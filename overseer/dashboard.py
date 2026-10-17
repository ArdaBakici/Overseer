"""The dashboard: a Telegram Mini App with your jobs, results, findings and a timeline - for browsing what chat
messages are poor at (sorting and filtering results, paging through graphs, seeing a week of jobs at once).

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
import html
import json
import logging
import re
import time
from importlib import resources
from urllib.parse import parse_qs, parse_qsl, urlsplit

from .config import Config
from .slurm import gpus_from_tres
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


def plain(card_html: str) -> str:
    """A card's Telegram HTML as plain text (the page escapes it again)."""
    return html.unescape(re.sub(r"<[^>]+>", "", card_html or "")).strip()


class Dashboard:
    def __init__(self, cfg: Config, store: Store, monitor, hpc, bot_token: str):
        self.cfg = cfg
        self.store = store
        self.monitor = monitor
        self.hpc = hpc
        self.token = bot_token
        self.server: asyncio.AbstractServer | None = None
        self._timeline: tuple[float, int, dict] | None = None

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
            page = resources.files("overseer").joinpath("dashboard.html").read_bytes()
            return 200, "text/html; charset=utf-8", page, {"Content-Security-Policy": CSP}
        if path == "/app.js":
            js = resources.files("overseer").joinpath("dashboard.js").read_bytes()
            return 200, "text/javascript; charset=utf-8", js, {}
        if not path.startswith("/api/"):
            return 404, "text/plain", b"not found", {}
        if verify_init_data(headers.get("x-telegram-init-data", ""), self.token,
                            self.cfg.telegram.allowed_chat_ids) is None:
            return 401, "application/json", b'{"error": "open the dashboard from Telegram"}', {}
        parts = path[5:].strip("/").split("/")
        try:
            if parts == ["overview"]:
                return self._json(self.overview())
            if parts == ["results"]:
                return self._json(self.results(query.get("project", "")))
            if parts == ["findings"]:
                return self._json(self.findings())
            if parts == ["timeline"]:
                return self._json(await self.timeline(int(query.get("days", "7") or 7)))
            if len(parts) == 3 and parts[0] == "plot" and JOB_ID.match(parts[1]) and parts[2].isdigit():
                plots = self.monitor.cached_plots(parts[1]) if self.monitor else []
                i = int(parts[2])
                if i < len(plots):
                    return 200, "image/png", plots[i][1], {}
                return 404, "text/plain", b"no such graph", {}
            if len(parts) == 2 and parts[0] == "log" and JOB_ID.match(parts[1]):
                return self._json(await self.log(parts[1], int(query.get("lines", "80") or 80)))
        except ValueError:
            return 400, "text/plain", b"bad request", {}
        return 404, "text/plain", b"not found", {}

    @staticmethod
    def _json(data) -> tuple[int, str, bytes, dict]:
        return 200, "application/json", json.dumps(data).encode(), {}

    # ------------------------------------------------------------------ data
    def overview(self) -> dict:
        active = [r for r in self.store.active_jobs() if r.name != "overseer-worker"]
        seen = {r.job_id for r in active}
        recent = [r for r in self.store.recent_jobs(80) if r.job_id not in seen and r.name != "overseer-worker"]
        jobs = []
        for r in active + recent:
            try:
                card = json.loads(self.store.get(f"cardview:{r.job_id}") or "null")
            except ValueError:
                card = None
            idle = self.monitor.idle_state(r.job_id) if self.monitor and not r.ended_at else {}
            jobs.append({
                "id": r.job_id, "name": r.name, "project": r.project or "", "state": r.state,
                "agent": bool(r.by_agent), "seen": r.first_seen, "ended": r.ended_at or 0,
                "card": plain(card[0]) if card else "", "report": r.report or "",
                "plots": len(self.monitor.cached_plots(r.job_id)) if self.monitor else 0,
                "idle": bool(idle.get("alerted")), "start": self.store.get(f"eststart:{r.job_id}") or "",
            })
        return {"jobs": jobs, "projects": [p.name for p in self.store.list_projects()], "now": time.time()}

    def results(self, project: str = "") -> dict:
        scope: str | None | bool = False if project in ("", "all") else (None if project == "main" else project)
        rows, counts = [], {}
        for r in self.store.results(scope):
            try:
                metrics = json.loads(r.metrics or "{}") or {}
            except ValueError:
                metrics = {}
            for k in metrics:
                counts[k] = counts.get(k, 0) + 1
            rows.append({"id": r.id, "name": r.name, "project": r.project or "", "verdict": r.verdict,
                         "state": r.state, "success": bool(r.success), "elapsed": r.elapsed,
                         "ended": r.ended or r.created, "metrics": metrics, "hypothesis": r.hypothesis,
                         "conclusion": r.conclusion, "tags": r.tags})
        keys = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:6]
        return {"rows": rows, "metric_keys": keys}

    def findings(self) -> dict:
        return {"rows": [{"id": f"F{f.id}", "project": f.project or "", "kind": f.kind, "text": f.text,
                          "job": f.job_id, "created": f.created} for f in reversed(self.store.findings())]}

    async def timeline(self, days: int) -> dict:
        """Your jobs that ran in the last `days` (SLURM accounting, the cluster's clock): kept for a minute."""
        days = max(1, min(days, 30))
        if self._timeline and self._timeline[1] == days and time.time() - self._timeline[0] < 60:
            return self._timeline[2]
        try:
            clock = await self.hpc.clock()
            rows = await self.hpc.sacct_recent(days * 1440)
        except Exception as e:
            return {"error": f"couldn't read SLURM's accounting: {e}", "jobs": [], "clock": ""}
        tracked = {r.job_id: r for r in self.store.recent_jobs(2000)}
        jobs = []
        for j in rows:
            if j.name == "overseer-worker" or not j.start_time[:1].isdigit():
                continue  # (never started: nothing to draw)
            rec = tracked.get(j.job_id)
            jobs.append({"id": j.job_id, "name": j.name, "state": j.state, "start": j.start_time,
                         "end": j.end_time if j.end_time[:1].isdigit() else "", "gpus": int(gpus_from_tres(
                             j.alloc_tres) or 0), "project": (rec.project or "") if rec else "",
                         "tasks": j.extra.get("tasks", ""), "elapsed": j.elapsed})
        data = {"jobs": jobs, "clock": clock, "days": days}
        self._timeline = (time.time(), days, data)
        return data

    async def log(self, job_id: str, lines: int) -> dict:
        """The end of a tracked job's log (the path Overseer knows, never one from the request)."""
        rec = self.store.get_job(job_id)
        if not rec or rec.ignored:
            return {"error": "Overseer doesn't track this job", "text": ""}
        path = rec.info_dict.get("progress_log") or rec.stdout
        if not path:
            return {"error": "no log known for this job", "text": ""}
        text = await self.hpc.try_tail(path, max(10, min(lines, 400)))
        return {"path": path, "text": text[-60_000:]}
