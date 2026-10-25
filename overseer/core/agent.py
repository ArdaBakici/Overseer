"""Agent conversations: Telegram message -> Claude Code / Codex run -> Telegram reply.

The agent is sandboxed on this machine (see runners.py); what it can reach beyond that is its domain's (e.g. the HPC
over `ssh hpc`, see hpc/domain.py). The platform's job here is bookkeeping: conversations and their projects,
resuming the CLI's session, reporting the commands it ran, the files it sends, and the domain's hooks around each run
(its instructions, context, and what it watches in the commands).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config import BACKENDS, Config
from .ctl import TOKEN_ENV
from .live import Progress, Transcript
from .richtext import fold, fold_long, md_to_html, rich_code, rich_details, rich_markdown
from .notify import SILENT, Notifier, Outgoing, code, esc, pre
from .runners import Activity, AgentRunError, Runner, make_runner
from .store import SessionRecord, Store

if TYPE_CHECKING:
    from .domain import Domain

log = logging.getLogger(__name__)

LIVE_FLUSH_SECONDS = 5
NAMES = {"claude_code": "Claude Code", "codex": "Codex"}
# The CLI is too old (e.g. for a new model): "version 2.1.280 or newer is required. Run 'claude update'"
OUTDATED_RE = re.compile(r"or newer is required|(?:claude|codex) update\b|upgrade (?:claude|codex)", re.I)

# Normal mode: the agent may propose planner mode (subagents.py); the researcher approves or declines.
PROPOSE_SECTION = """

## Proposing planner mode (subagents)
For work with several independent lines (ablations, competing ideas, a sweep plus its analysis) or long experiments, \
you may propose planner mode: you would plan and review on {planner}, and subagents on cheaper models \
({workers}) would do the hands-on work, each in its own thread. Write the plan as JSON and run \
`ov propose plan.json` - {{"goal": ..., "why": "what splitting it gains", "subagents": [{{"name": "cosine", "task": \
"a self-contained brief", "model": ..., "backend": ...}}]}} (model and backend are optional). The researcher sees \
it and approves or declines it; if they write to you instead, that declines it and you get their message - revise \
and propose again if it still makes sense. After proposing, end your reply and set no wake-ups for it. Don't propose \
it for a single job, a quick fix or a question, and at most once per task unless asked."""

PLANNER_SECTION = """

## Planner mode: you plan, subagents do the work
You run on a strong model; subagents on cheaper ones ({workers}). Plan, delegate the hands-on work (running and \
watching experiments, debugging a job, writing analysis or plotting scripts, reading logs), review the results and \
decide the next step. Start one with `ov spawn <name> "<brief>" [--model M] [--backend claude_code|codex]` (at most \
{max_running} run at once across all conversations; more wait their turn). A subagent doesn't see this \
conversation: make each brief self-contained - goal, context, exact paths, constraints, its own output folder \
(so subagents never overwrite each other's results), and what to report back. Pick the cheapest model that can do \
the task. You're woken with their reports once all the subagents you're waiting for are done (or as soon as one \
fails or is stopped); subagents watch their own jobs, so don't watch theirs. `ov agents`, `ov agent show|stop \
<name>`, `ov agent send <name> <text>`. In a project, you alone write the research log: record each subagent's \
results (link its NOTES.md). When the goal is reached, say so and run `ov planner off`."""

REVIEWER_SECTION = """

## You are Overseer's feedback reviewer
Agents report what limits them in Overseer (`ov feedback list --all` shows every item). You turn that into designs the
researcher can act on: group and rank the items, read the relevant code in {src} (you can read it, not change it),
and send designs - and patches when a change is clear - as files in ./outbox/. The researcher talks to you in the 🛠
feedback topic: answer questions about the designs and revise them there. Never propose weakening the sandbox,
exposing secrets, or agents changing Overseer itself."""

WORKER_SECTION = """

## You are a subagent: "{name}"
A planner (another conversation, on a stronger model) gave you a task; the researcher can follow you in your own \
Telegram thread and write to you there. Do the task on your own - submit jobs, watch them, wake yourself - and \
keep your detailed notes in `{notes}`. Give each of your jobs its verdict and conclusion (`ov result`). Use your \
own output folder; never overwrite other runs' results. Don't \
start subagents or propose planner mode. When the task is done, your final reply is your report to the planner: \
what you did, the key numbers, where the results are (at most ~15 lines). If you're blocked or the task turns out \
wrong, say so in that report rather than guessing."""


IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
ANIMATION_TYPES = {".gif"}  # sent with sendAnimation (a GIF as a picture would show only its first frame)
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".webm"}
MAX_PHOTO = 10 * 1024 * 1024  # Telegram's limits for pictures and documents sent by bots
MAX_FILE = 50 * 1024 * 1024
MAX_ATTACHMENTS = 10


INBOX = "inbox"  # where the researcher's files go, in the agent's working dir

# Runs in progress or queued (kv), and those a restart cut off, waiting for the researcher's ▶️ Continue / ✖.
RUN_KEY = "run:"
CUT_OFF_KEY = "cutoff:"
RUN_TEXT_MAX = 30_000
CONTINUE_PROMPT = (
    "[Overseer was restarted while you were working on the request below, and your run was cut off partway. "
    "Some of it may already be done: check before redoing anything (files you changed, jobs you submitted - "
    "`ov job list`, `squeue`), then finish it.]\n\n")


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{max(1, round(n / 1e3))} KB"


def _upload_name(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", Path(name or "").name).strip("._-") or "file"
    return name[-100:]


def deliver_uploads(workdir: Path, files: list[tuple[str, Path]]) -> list[tuple[str, int]]:
    """Move the researcher's files (name, staged path) into the agent's ./inbox: [(path for the agent, size)].
    The agent can write its working dir, so ./inbox is opened without following links (Overseer isn't
    sandboxed: a symlinked inbox must not make it write elsewhere), and existing files are never replaced."""
    inbox = workdir / INBOX
    try:
        if not stat.S_ISDIR(os.lstat(inbox).st_mode):
            inbox.unlink()
            raise FileNotFoundError
    except FileNotFoundError:
        inbox.mkdir(mode=0o700)
    fd = os.open(inbox, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    out = []
    try:
        taken = set(os.listdir(fd))
        for name, src in files:
            base = _upload_name(name)
            stem, dot, ext = base.rpartition(".") if "." in base else (base, "", "")
            candidate, i = base, 1
            while candidate in taken:
                candidate, i = f"{stem}-{i}{dot}{ext}", i + 1
            size = src.stat().st_size
            os.rename(src, candidate, dst_dir_fd=fd)
            taken.add(candidate)
            out.append((f"./{INBOX}/{candidate}", size))
    finally:
        os.close(fd)
    return out


def uploads_note(items: list[tuple[str, int]], backend: str) -> str:
    look = "your view_image tool" if backend == "codex" else "your Read tool (it shows images and PDFs)"
    listing = ", ".join(f"{p} ({_size(n)})" for p, n in items)
    return (f"[The researcher sent {'a file' if len(items) == 1 else f'{len(items)} files'} with this message, saved "
            f"in your working directory: {listing}. Look at images with {look}. To use a file on the HPC, copy it "
            "there with scp.]")


def collect_outbox(folder: Path) -> tuple[list[tuple[str, bytes]], list[tuple[str, bytes]], list[str],
                                         list[tuple[str, str, bytes]]]:
    """Take the files the agent left in its outbox: (pictures, documents, notes about skipped files, GIFs and
    videos as (animation | video, name, content)).
    Overseer isn't sandboxed, so only regular files are read - never symlinks or hard links, which
    could point at something the agent itself can't read."""
    photos: list[tuple[str, bytes]] = []
    docs: list[tuple[str, bytes]] = []
    notes: list[str] = []
    media: list[tuple[str, str, bytes]] = []
    try:
        entries = sorted((e for e in os.scandir(folder) if not e.name.startswith(".")),
                         key=lambda e: e.stat(follow_symlinks=False).st_mtime)
    except OSError:
        return photos, docs, notes, media
    for e in entries:
        if len(photos) + len(docs) + len(media) >= MAX_ATTACHMENTS:
            notes.append(f"more than {MAX_ATTACHMENTS} files: the rest are sent with the next reply")
            break
        path = Path(e.path)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            notes.append(f"{e.name}: not sent (not a regular file)")
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "rb") as fh:
            st = os.fstat(fh.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                notes.append(f"{e.name}: not sent (not a regular file)")
            elif st.st_size > MAX_FILE:
                notes.append(f"{e.name}: not sent ({st.st_size / 2**20:.0f} MB, the limit is 50 MB)")
            elif path.suffix.lower() in IMAGE_TYPES and st.st_size <= MAX_PHOTO:
                photos.append((e.name, fh.read()))
            elif path.suffix.lower() in ANIMATION_TYPES | VIDEO_TYPES:
                kind = "animation" if path.suffix.lower() in ANIMATION_TYPES else "video"
                media.append((kind, e.name, fh.read()))
            else:
                docs.append((e.name, fh.read()))
        path.unlink(missing_ok=True)
    return photos, docs, notes, media


def _command_fold(commands: list[str]) -> str:
    """Commands in a collapsible section: the first few lines show, the rest on tap."""
    def clip(c: str) -> str:
        c = c.strip()
        return c if len(c) <= 300 else c[:299] + "…"
    return fold(esc("\n".join(f"$ {clip(c)}" for c in commands)))


def strip_tags(html_text: str) -> str:
    return re.sub(r"<[^>]+>", "", html_text)


def _human(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def compaction_line(compactions: list[dict]) -> str:
    """🗜 The conversation was compacted during the run (HTML), or "" if it wasn't."""
    if not compactions:
        return ""
    if len(compactions) > 1:
        return f"🗜 <i>The conversation was compacted {len(compactions)} times during this run: older turns are now a summary.</i>"
    c = compactions[0]
    how = "on request" if c.get("trigger") == "manual" else "automatically, as the context was full"
    tokens = f" ({_human(c['pre_tokens'])} → {_human(c['post_tokens'])} tokens)" if c.get("pre_tokens") and c.get("post_tokens") else ""
    return f"🗜 <i>The conversation was compacted {how}{tokens}: older turns are now a summary.</i>"


@dataclass
class Session:
    rec: SessionRecord
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task | None = None
    origin: str = "user"  # what started the current turn: user | wakeup | checkin | subagent
    last_reply: str = ""  # the agent's last answer (a subagent's report)
    last_status: str = ""  # how its last run ended: done | error | stopped

    @property
    def id(self) -> str:
        return self.rec.id

    @property
    def job_id(self) -> str | None:
        return self.rec.job_id

    @property
    def project(self) -> str | None:
        return self.rec.project


class Agent:
    def __init__(self, cfg: Config, store: Store, notifier: Notifier, domain: Domain):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.domain = domain  # what makes this bot what it is (hpc/, ...): its instructions, context and hooks
        domain.attach(self)
        self.autopilot = None  # set by Autopilot(...): wakes conversations when their jobs finish
        self.subagents = None  # set by Subagents(...): planner mode
        self.feedback = None  # set by Feedback(...): agents' feedback about Overseer, and its review
        self.addons = None  # set by Addons(...): skills agents proposed and the researcher approved
        self.ctl = None  # set by ControlServer(...): the agent's `ov` command
        self.updating = False  # the CLIs are being updated (update_clis)
        self.shutting_down = False  # Overseer is stopping: runs cut off now are offered again at the next start
        self.sessions: dict[str, Session] = {}
        self._runners: dict[str, Runner] = {}

    # ------------------------------------------------------------------ backend, model, effort
    # Stored per scope: the main chat ("backend", "model:<backend>", ...) and each project
    # ("backend@dreamer", ...). A project copies the main chat's settings when it is created;
    # an empty value there means "the config / CLI default".
    def _setting(self, key: str, project: str | None = None) -> str | None:
        if project:
            v = self.store.get(f"{key}@{project}")
            if v is not None:
                return v or None
        return self.store.get(key)

    def _set_setting(self, key: str, value: str | None, project: str | None = None) -> None:
        if project:
            self.store.set(f"{key}@{project}", value or "")
        elif value:
            self.store.set(key, value)
        else:
            self.store.unset(key)

    def init_project_settings(self, project: str, backend: str | None = None, model: str | None = None,
                              effort: str | None = None) -> None:
        """A new project starts with the main chat's backend, models and efforts, then the given ones."""
        for key in ["backend"] + [f"{k}:{b}" for b in BACKENDS for k in ("model", "effort")]:
            if self.store.get(f"{key}@{project}") is None:
                self.store.set(f"{key}@{project}", self.store.get(key) or "")
        if backend:
            self.set_backend(backend, project)
        if model:
            self.set_model(model, project=project)
        if effort:
            self.set_effort(effort, project=project)

    @property
    def backend(self) -> str:
        """The main chat's backend."""
        return self.backend_of(None)

    def backend_of(self, project: str | None) -> str:
        b = self._setting("backend", project) or self.cfg.agent.backend
        return b if b in BACKENDS else self.cfg.agent.backend

    def set_backend(self, name: str, project: str | None = None) -> None:
        if name not in BACKENDS:
            raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
        self._set_setting("backend", name, project)

    def runner(self, backend: str | None = None, project: str | None = None) -> Runner:
        b = backend or self.backend_of(project)
        # One runner per backend and project: concurrent runs in different projects can't see
        # each other's model / effort.
        key = f"{b}@{project or ''}"
        if key not in self._runners:
            self._runners[key] = make_runner(self.cfg, b, self.domain.sandbox())
        r = self._runners[key]
        # Overrides set from Telegram (/model, /effort) win over config.yaml.
        r.model_override = self._setting(f"model:{b}", project)
        r.effort_override = self._setting(f"effort:{b}", project)
        return r

    # ------------------------------------------------------------------ a conversation's own backend and model
    # A subagent runs on the backend and model it was given; a planner (planner mode) on the planner model of its
    # backend; any other conversation on its project's settings (/backend, /model, /effort).
    def planner_on(self, s: Session) -> bool:
        if s.rec.is_worker:
            return False
        return bool(s.rec.planner) or self._setting("subagents", s.project) == "on"

    def planner_model(self, backend: str, project: str | None) -> str | None:
        return self._setting(f"sub_planner:{backend}", project) or self.cfg.agent.subagents.planner.get(backend)

    def worker_models(self, backend: str, project: str | None) -> list[str]:
        v = self._setting(f"sub_workers:{backend}", project)
        return [m for m in (v or "").split(",") if m] or list(self.cfg.agent.subagents.workers.get(backend, []))

    def session_backend(self, s: Session) -> str:
        b = s.rec.agent_backend
        return b if b in BACKENDS else self.backend_of(s.project)

    def session_runner(self, s: Session) -> Runner:
        b = self.session_backend(s)
        if s.rec.is_worker:
            model, effort = s.rec.model or None, self.cfg.agent.subagents.worker_effort
        elif s.rec.role == "reviewer" or self.planner_on(s):  # the feedback reviewer plans too
            model, effort = self.planner_model(b, s.project), self._setting(f"effort:{b}", s.project)
        else:
            return self.runner(b, s.project)
        key = f"{b}@{s.project or ''}#{model}#{effort}"  # its own runner: the model never changes under a run
        if key not in self._runners:
            self._runners[key] = make_runner(self.cfg, b, self.domain.sandbox())
        r = self._runners[key]
        r.model_override, r.effort_override = model, effort
        return r

    def set_model(self, name: str, backend: str | None = None, project: str | None = None) -> None:
        """Set the model for a backend; "default" goes back to config.yaml."""
        b = backend or self.backend_of(project)
        name = name.strip()
        if not name or any(c.isspace() for c in name):
            raise ValueError("model name must be a single word, e.g. opus or gpt-5.5-codex")
        self._set_setting(f"model:{b}", None if name == "default" else name, project)

    def set_effort(self, level: str, backend: str | None = None, project: str | None = None) -> None:
        b = backend or self.backend_of(project)
        level = level.strip().lower()
        if level != "default" and level not in self.runner(b, project).efforts:
            raise ValueError(f"{b} effort must be one of: {', '.join(self.runner(b, project).efforts)} (or default)")
        self._set_setting(f"effort:{b}", None if level == "default" else level, project)

    def describe(self, backend: str, project: str | None = None) -> str:
        """e.g. 'codex: model default (last run: gpt-5.5) · effort default (last run: medium (model default))'."""
        r = self.runner(backend, project)
        last_model = self.store.get(f"last_model:{backend}")
        last_effort = self.store.get(f"last_effort:{backend}")
        model = r.model or "CLI default"
        if r.model_override:
            model += " (set via /model)"
        if last_model and last_model != r.model:
            model += f" (last run: {last_model})"
        effort = r.effort or "CLI default"
        if r.effort_override:
            effort += " (set via /effort)"
        if last_effort and last_effort != r.effort:
            effort += f" (last run: {last_effort})"
        return f"{backend}: model {model} · effort {effort}"

    def _usage_delta(self, backend: str, res, resumed: bool) -> None:
        """Turn a running total (codex) into this request's usage, remembering the total."""
        if not res.usage_is_total or not res.session_id:
            return
        key = f"usage:{backend}:{res.session_id}"
        total = (res.input_tokens, res.cached_tokens, res.output_tokens)
        prev = self.store.get(key)
        self.store.set(key, ",".join(map(str, total)))
        if prev:
            p = [int(x) for x in prev.split(",")]
            if all(t >= q for t, q in zip(total, p)):
                res.input_tokens, res.cached_tokens, res.output_tokens = (t - q for t, q in zip(total, p))
                res.usage_is_total = False
        elif not resumed:
            res.usage_is_total = False  # a new conversation: its total is this request's usage
        # else: resumed from before Overseer kept totals - the footer says it's the conversation's total

    def _record_usage(self, s: Session, backend: str, res) -> None:
        """Tokens and cost of this run, for /usage (a codex total Overseer can't split yet isn't counted)."""
        if res.usage_is_total:
            return
        self.store.add_usage(session_id=s.id, project=s.project, backend=backend, model=res.model, origin=s.origin,
                             input=res.input_tokens, cached=res.cached_tokens, output=res.output_tokens,
                             cost=res.cost_usd)

    def footer(self, backend: str, res, project: str | None = None, runner: Runner | None = None) -> str:
        r = runner or self.runner(backend, project)
        parts = [backend, res.model or r.model or "default model", f"effort {res.effort or r.effort or 'default'}"]
        tokens = res.input_tokens + res.output_tokens
        if tokens:
            cached = f" ({_human(res.cached_tokens)} cached)" if res.cached_tokens else ""
            parts.append(f"{_human(tokens)} tokens{cached}" + (" in this conversation so far" if res.usage_is_total else ""))
        if res.cost_usd is not None:
            parts.append(f"${res.cost_usd:.2f}")
        return " · ".join(parts)

    # ------------------------------------------------------------------ sessions
    def _load(self, rec: SessionRecord) -> Session:
        if rec.id not in self.sessions:
            self.sessions[rec.id] = Session(rec)
        return self.sessions[rec.id]

    def _new(self, job_id: str | None, project: str | None = None) -> Session:
        prefix = f"j{job_id}-" if job_id else f"{project}-" if project else "chat-"
        rec = SessionRecord(id=prefix + uuid.uuid4().hex[:6], job_id=job_id, created=time.time(), project=project)
        self.store.upsert_session(rec)
        return self._load(rec)

    def get(self, sid: str | None) -> Session | None:
        if not sid:
            return None
        if sid in self.sessions:
            return self.sessions[sid]
        rec = self.store.get_session(sid)
        return self._load(rec) if rec else None

    def session_for_job(self, job_id: str) -> Session:
        job = self.store.get_job(job_id)
        # A job the agent submitted continues the conversation that submitted it.
        if job and job.session_id and (s := self.get(job.session_id)):
            return s
        rec = self.store.session_for_job(job_id)
        return self._load(rec) if rec else self._new(job_id, job.project if job else None)

    def new_conversation(self, project: str | None = None) -> Session:
        """A fresh conversation (e.g. for a new Telegram thread)."""
        return self._new(None, project)

    def general_session(self, reset: bool = False, project: str | None = None) -> Session:
        """The ongoing conversation of the main chat, or of a project's topic."""
        key = f"general_session@{project}" if project else "general_session"
        sid = None if reset else self.store.get(key)
        s = self.get(sid)
        if s is None:
            s = self._new(None, project)
            self.store.set(key, s.id)
        return s

    # Notes for the agent's next turn in a conversation, about things that happened while it wasn't running.
    def add_note(self, sid: str, text: str) -> None:
        notes = json.loads(self.store.get(f"notes:{sid}") or "[]")
        self.store.set(f"notes:{sid}", json.dumps((notes + [text])[-10:]))

    def take_notes(self, sid: str) -> list[str]:
        raw = self.store.get(f"notes:{sid}")
        if not raw or raw == "[]":
            return []
        self.store.set(f"notes:{sid}", "[]")
        return json.loads(raw)

    async def compact(self, s: Session, focus: str = "") -> bool:
        """Compact the conversation now (the researcher's /compact): the CLI summarises its older turns."""
        backend = self.session_backend(s)
        if not s.rec.cli_session_id or s.rec.backend != backend:
            await self.notifier.send(Outgoing(f"🗜 Nothing to compact: this conversation has no history with "
                                              f"{esc(backend)} yet.", session_id=s.id, job_id=s.job_id))
            return False
        if self.is_running(s):
            await self.notifier.send(Outgoing("The agent is working in this conversation: send /compact when it's "
                                              "done (or /stop it first).", session_id=s.id, job_id=s.job_id))
            return False
        runner = self.session_runner(s)
        async with s.lock:
            s.task = asyncio.current_task()
            transcript = Transcript(self.cfg, s.id)
            transcript.write({"kind": "turn", "backend": backend, "model": runner.model, "effort": runner.effort,
                              "text": ("/compact " + focus).strip()})
            activity = Activity()
            activity.on_progress = transcript.write
            await self.notifier.send(Outgoing(f"🗜 Compacting the conversation ({esc(backend)})…", session_id=s.id,
                                              job_id=s.job_id, silent=True))
            try:
                res = await runner.compact(session_id=s.rec.cli_session_id, key=s.id, system=self.system_prompt(s.id),
                                           activity=activity, focus=focus)
            except asyncio.CancelledError:
                transcript.write({"kind": "end", "status": "stopped"})
                await self.notifier.send(Outgoing("⏹ Stopped.", session_id=s.id, job_id=s.job_id))
                return False
            except AgentRunError as e:
                transcript.write({"kind": "end", "status": "error", "text": str(e)[:2000]})
                await self.notifier.send(Outgoing(f"⚠️ Couldn't compact: {esc(e)}", session_id=s.id, job_id=s.job_id))
                return False
            finally:
                s.task = None
            if res.session_id and res.session_id != s.rec.cli_session_id:
                s.rec.cli_session_id = res.session_id
                self.store.upsert_session(s.rec)
            s.origin = "compact"
            self._record_usage(s, backend, res)
            s.origin = "user"
            c = res.compactions[-1]
            tokens = (f": {_human(c['pre_tokens'])} → {_human(c['post_tokens'])} tokens"
                      if c.get("pre_tokens") and c.get("post_tokens") else "")
            note = " (Codex doesn't take a focus: it summarised everything)" if focus and backend == "codex" else ""
            text = f"🗜 <b>Compacted the conversation</b>{tokens}. Older turns are now a summary{note}."
            transcript.write({"kind": "end", "status": "done", "text": strip_tags(text)})
            await self.notifier.send(Outgoing(text, session_id=s.id, job_id=s.job_id))
            return True

    def is_running(self, s: Session) -> bool:
        return s.task is not None and not s.task.done()

    async def stop(self, sid: str) -> bool:
        s = self.sessions.get(sid)
        if not s or not self.is_running(s):
            return False
        s.task.cancel()
        return True

    def running_sessions(self) -> list[Session]:
        return [s for s in self.sessions.values() if self.is_running(s)]

    # ------------------------------------------------------------------ keeping the CLIs up to date
    async def update_clis(self) -> list[str] | None:
        """Run each installed CLI's own updater. None if agent runs are in progress (a CLI isn't replaced under a
        running agent); otherwise one line (HTML) per CLI. New runs wait while this runs."""
        if self.running_sessions() or self.updating:
            return None
        self.updating = True
        lines = []
        try:
            for backend in BACKENDS:
                r = self.runner(backend)
                if not r.installed():
                    continue
                before = await r.version()
                ok, out = await r.update()
                after = await r.version()
                name = NAMES[backend]
                if not ok:
                    lines.append(f"⚠️ {name}: the update failed ({esc(before or 'version unknown')}):"
                                 + pre(out or "(no output)", 600))
                elif after and after != before:
                    lines.append(f"⬆️ {name} updated: {esc(before or '?')} → {esc(after)}")
                else:
                    lines.append(f"✓ {name} {esc(after or before or '')} is up to date")
        finally:
            self.updating = False
            self.store.set("cli_update_at", str(time.time()))
        return lines

    async def update_loop(self) -> None:
        """Once a day: update the CLIs when nothing is running (agent.auto_update). Tells you only about changes."""
        while True:
            await asyncio.sleep(600)
            if not self.cfg.agent.auto_update:
                continue
            if time.time() - float(self.store.get("cli_update_at") or 0) < 86400:
                continue
            try:
                lines = await self.update_clis()
            except Exception:
                log.exception("updating the agent CLIs failed")
                self.store.set("cli_update_at", str(time.time()))
                continue
            changed = [x for x in lines or [] if not x.startswith("✓")]
            if changed:
                await self.notifier.send(Outgoing("\n".join(changed), silent=True))

    # ------------------------------------------------------------------ prompts
    def web_line(self, project: str | None) -> str:
        if self.backend_of(project) == "codex":
            if self.cfg.agent.codex.web_search == "disabled":
                return ""
            tools = "your web search tool"
        else:
            tools = "your WebSearch and WebFetch tools"
        return (f" You can still search the web and read pages with {tools}: use them for papers, library docs, "
                "error messages and cluster documentation.")

    def system_prompt(self, session_id: str | None = None) -> str:
        sess = self.get(session_id)
        prompt = self.domain.system_prompt(sess)
        if self.ctl and self.subagents and sess and sess.rec.role != "reviewer" and self.domain.planner:
            prompt += self._subagent_section(sess)
        if sess and sess.rec.role == "reviewer":
            from .feedback import SOURCE
            prompt += REVIEWER_SECTION.format(src=SOURCE)
        if self.cfg.agent.i_have_adhd:  # (last: it shapes the replies, and says the instructions above win)
            from . import adhd
            prompt += adhd.section()
        return prompt

    def _subagent_section(self, s: Session) -> str:
        """Planner mode: how to propose it (normal mode), how to plan (planner), or how to work (subagent)."""
        if s.rec.is_worker:
            return WORKER_SECTION.format(name=s.rec.name, notes=self.subagents.notes_path(s.rec))
        b = self.session_backend(s)
        workers = ", ".join(m for bb in BACKENDS for m in self.worker_models(bb, s.project))
        if self.planner_on(s):
            return PLANNER_SECTION.format(workers=workers, max_running=self.cfg.agent.subagents.max_running)
        return PROPOSE_SECTION.format(planner=self.planner_model(b, s.project) or "a stronger model", workers=workers)

    # ------------------------------------------------------------------ running a request
    async def handle(self, s: Session, user_text: str, quiet: bool = False, origin: str = "user",
                     files: list[tuple[str, Path]] | None = None) -> bool:
        """Run one request in a conversation. True if the agent answered. `quiet`: send the answer
        without a notification sound (check-ins). `origin`: who asked - the researcher (user), or a
        wake-up the agent set up (wakeup, checkin). `files`: what the researcher sent with it (name, staged
        path), moved into the agent's ./inbox."""
        if not self.cfg.agent.enabled:
            await self.notifier.send(Outgoing("The agent is disabled in the config."))
            return False
        ok = False
        run = self._run_begins(s, user_text, origin, files)
        while self.updating:  # the CLIs are being updated: runs wait until that's done
            await asyncio.sleep(1)
        async with s.lock:
            if self.shutting_down:  # it was waiting its turn: left for the next start
                return False
            s.task = asyncio.current_task()
            s.origin = origin
            if origin == "user" and self.autopilot:
                self.autopilot.user_turn(s)
            try:
                if files:  # (moved while no run of this conversation can touch its inbox)
                    items = deliver_uploads(self.session_runner(s).workdir(s.id), files)
                    user_text = f"{user_text}\n\n{uploads_note(items, self.session_backend(s))}".strip()
                self._run_update(run, text=user_text, files=[], started=time.time())
                s.last_status = "running"
                await self._turn(s, user_text, quiet)
                ok, s.last_status = True, "done"
            except asyncio.CancelledError:
                if self.shutting_down:  # not stopped by the researcher: offered again at the next start
                    raise
                s.last_status = "stopped"
                await self.notifier.send(Outgoing("⏹ Stopped.", session_id=s.id, job_id=s.job_id))
            except Exception as e:
                if self.shutting_down:  # its CLI was stopped with Overseer: offered again at the next start
                    raise
                s.last_status, s.last_reply = "error", str(e)[:1500]
                if isinstance(e, AgentRunError):
                    await self._run_failed(s, e)
                else:
                    log.exception("agent run failed")
                    await self.notifier.send(Outgoing(f"⚠️ Agent error: {esc(type(e).__name__)}: {esc(str(e)[:800])}",
                                                      session_id=s.id, job_id=s.job_id))
            finally:
                s.task = None
                s.origin = "user"
        self.store.unset(RUN_KEY + run)  # (not reached when the run is cut off by Overseer stopping)
        if self.autopilot:
            self.autopilot.turn_ended(s)
        if self.subagents and s.rec.is_worker:  # done, waiting for its jobs, failed or stopped: its planner hears
            self.subagents.turn_ended(s)
        return ok

    async def _run_failed(self, s: Session, e: AgentRunError) -> None:
        hint = ("\nOverseer can update it: /update (it also updates the CLIs once a day on its own)."
                if OUTDATED_RE.search(str(e)) else "")
        await self.notifier.send(Outgoing(f"⚠️ {esc(e)}{hint}", session_id=s.id, job_id=s.job_id))

    # ------------------------------------------------------------------ runs cut off by a restart
    def _run_begins(self, s: Session, text: str, origin: str, files: list[tuple[str, Path]] | None) -> str:
        """Note a run (queued or running) until it ends: a note left at the next start means Overseer stopped
        or crashed meanwhile, and the researcher is offered to continue it."""
        rid = uuid.uuid4().hex[:10]
        self.store.set(RUN_KEY + rid, json.dumps({
            "session": s.id, "text": text[:RUN_TEXT_MAX], "origin": origin, "started": 0, "at": time.time(),
            "files": [[name, str(path)] for name, path in files or []]}))
        return rid

    def _run_update(self, rid: str, **changes) -> None:
        raw = self.store.get(RUN_KEY + rid)
        if raw:
            rec = json.loads(raw)
            if "text" in changes:
                changes["text"] = changes["text"][:RUN_TEXT_MAX]
            self.store.set(RUN_KEY + rid, json.dumps({**rec, **changes}))

    def take_cut_off_runs(self) -> list[dict]:
        """At startup: the runs the last Overseer didn't finish (oldest first), kept until the researcher
        continues or dismisses them."""
        out = []
        for key in self.store.keys(RUN_KEY):
            raw = self.store.get(key)
            self.store.unset(key)
            try:
                rec = json.loads(raw or "")
            except ValueError:
                continue
            rid = key[len(RUN_KEY):]
            self.store.set(CUT_OFF_KEY + rid, json.dumps(rec))
            out.append({**rec, "id": rid})
        return sorted(out, key=lambda r: r.get("at", 0))

    def pop_cut_off_run(self, rid: str) -> dict | None:
        raw = self.store.get(CUT_OFF_KEY + rid)
        self.store.unset(CUT_OFF_KEY + rid)
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return None

    @staticmethod
    def continue_prompt(rec: dict) -> tuple[str, list[tuple[str, Path]]]:
        """What to send to continue a cut-off run: (text, files still waiting to be delivered)."""
        files = [(name, Path(p)) for name, p in rec.get("files") or [] if Path(p).is_file()]
        if not rec.get("started"):  # it was waiting its turn: simply run it now
            return rec.get("text", ""), files
        return (CONTINUE_PROMPT + rec.get("text", "")), files

    async def _turn(self, s: Session, user_text: str, quiet: bool = False) -> None:
        backend = self.session_backend(s)
        runner = self.session_runner(s)
        resume = s.rec.cli_session_id if s.rec.backend == backend else None
        parts = await self.domain.context(s, fresh=resume is None)  # (a new CLI session knows nothing yet)
        notes = self.take_notes(s.id)
        if notes:  # things that happened since the agent last ran (e.g. the researcher declined a request)
            parts.append("<overseer_notes>\n" + "\n".join(f"- {n}" for n in notes) + "\n</overseer_notes>")
        parts.append(user_text)
        prompt = "\n\n".join(parts)

        outbox = runner.workdir(s.id) / "outbox"  # files the agent leaves here are sent with its reply
        outbox.mkdir(exist_ok=True)
        if self.addons:  # the skills the researcher approved for this project (or for every agent)
            try:
                self.addons.install(s, runner.workdir(s.id), backend)
            except OSError:
                log.warning("could not install the approved skills for %s", s.id, exc_info=True)
        transcript = Transcript(self.cfg, s.id)
        transcript.write({"kind": "turn", "backend": backend, "model": runner.model, "effort": runner.effort,
                          "text": user_text, **({"project": s.project} if s.project else {})})
        progress = Progress(self.notifier, backend, s.id, s.job_id) if self.cfg.agent.live_progress else None
        activity, live, buffer = self._activity(s)

        def on_progress(ev: dict) -> None:
            transcript.write(ev)
            if progress:
                progress.on_event(ev)
            if ev["kind"] == "session" and (s.rec.backend, s.rec.cli_session_id) != (backend, ev["id"]):
                # Saved right away, so `overseer open` finds a conversation that is still running.
                s.rec.backend, s.rec.cli_session_id = backend, ev["id"]
                self.store.upsert_session(s.rec)

        activity.on_progress = on_progress
        token = self.ctl.open(s.id) if self.ctl else None
        if token:
            activity.env = {TOKEN_ENV: token}
        if progress:
            progress.start()
        try:
            state = await self.domain.before_run(s)
            res = await runner.run(prompt, system=self.system_prompt(s.id), session_id=resume, key=s.id,
                                   activity=activity)
        except asyncio.CancelledError:
            transcript.write({"kind": "end", "status": "stopped"})
            raise
        except Exception as e:
            transcript.write({"kind": "end", "status": "error", "text": str(e)[:2000]})
            raise
        finally:
            if token:
                self.ctl.close(token)
            if live:
                live.cancel()
            if progress:
                await progress.close()
            await self._report_activity(s, activity, buffer)
        note = await self.domain.after_run(s, activity, state, res.text)  # e.g. the notes it changed, ↩️ Undo
        if res.session_id:
            s.rec.backend, s.rec.cli_session_id = backend, res.session_id
            self.store.upsert_session(s.rec)
        if res.model:
            self.store.set(f"last_model:{backend}", res.model)
        if res.effort:
            self.store.set(f"last_effort:{backend}", res.effort)
        self._usage_delta(backend, res, resumed=resume is not None)
        footer = self.footer(backend, res, s.project, runner)
        s.last_reply = res.text or ""
        self._record_usage(s, backend, res)
        text, autopilot_line = res.text, ""
        if self.autopilot:
            autopilot_line = self.autopilot.after_turn(s, activity)
        photos, docs, notes, media = collect_outbox(outbox)
        if photos or docs or media:
            transcript.write({"kind": "tool", "name": "attachments",
                              "detail": ", ".join([*(n for n, _ in photos + docs), *(n for _, n, _ in media)])})
        transcript.write({"kind": "end", "status": "done", "text": text, "footer": footer})
        skipped = "\n".join(f"📎 {esc(n)}" for n in notes)
        compacted = compaction_line(res.compactions)
        extra = [x for x in (*(note.lines if note else []), compacted, autopilot_line, skipped) if x]
        if s.origin != "user" and (text or "").strip() == SILENT and not (photos or docs or media):
            # A wake-up with nothing worth saying: no message (what it changed, if anything, quietly).
            if extra:
                await self.notifier.send(Outgoing("\n\n".join(extra), session_id=s.id, job_id=s.job_id, silent=True,
                                                  buttons=note.buttons if note else []))
            return
        # (HTML, which Rich Markdown accepts as well)
        await self.notifier.send(Outgoing(
            f"🤖 {fold_long(md_to_html(text)) or '(no answer)'}" + "".join(f"\n\n{x}" for x in extra)
            + f"\n\n<i>{esc(footer)}</i>",
            markdown="\n\n".join([rich_markdown(text) or "(no answer)", *extra, f"<i>{esc(footer)}</i>"]),
            session_id=s.id, job_id=s.job_id, silent=quiet, buttons=note.buttons if note else [],
            photos=photos, files=docs, media=media))

    def _activity(self, s: Session) -> tuple[Activity, asyncio.Task | None, list[str]]:
        activity = self.domain.new_activity(s)  # (the domain's: it watches the commands, e.g. for submitted jobs)
        buffer: list[str] = []  # live mode: commands not yet sent
        live = None
        if self.cfg.agent.report_commands == "live":
            activity.on_command = buffer.append

            async def flush_loop():
                while True:
                    await asyncio.sleep(LIVE_FLUSH_SECONDS)
                    if buffer:
                        batch = buffer[:]
                        buffer.clear()
                        await self.notifier.send(Outgoing(
                            "💻 " + _command_fold(batch), session_id=s.id, job_id=s.job_id, silent=True))

            live = asyncio.get_running_loop().create_task(flush_loop())
        return activity, live, buffer

    async def _report_activity(self, s: Session, activity: Activity, buffer: list[str]) -> None:
        mode = self.cfg.agent.report_commands
        if mode == "off":
            return
        if mode == "live":
            if buffer:
                await self.notifier.send(Outgoing("💻 " + _command_fold(buffer),
                                                  session_id=s.id, job_id=s.job_id, silent=True))
            return
        if activity.blocked:
            lines = [f"⚠️ <b>{len(activity.blocked)} command(s) were refused by {esc(self.session_backend(s))} itself</b> "
                     "(not by Overseer - there is no approval step):"]
            for cmd, reason in activity.blocked[:10]:
                lines.append(pre(cmd, 300) + f"→ {esc(reason)}")
            await self.notifier.send(Outgoing("\n".join(lines), session_id=s.id, job_id=s.job_id))
        if not (activity.commands or activity.files):
            return
        n = len(activity.commands)
        lines = [f"🛠 <b>{n} command(s)</b>" + (" (the last 40 here, all in commands.txt)" if n > 40 else "")]
        if activity.commands:
            lines.append(_command_fold(activity.commands[-40:]))
        lines += activity.report_lines()  # the domain's remarks (e.g. heavy work on the HPC's login node)
        if activity.files:
            lines.append("✏️ local files: " + ", ".join(code(f) for f in activity.files[:20]))
        files = [("commands.txt", "\n\n".join(activity.commands).encode())] if len(activity.commands) > 15 else []
        # As a rich message: the commands in a collapsible block, as a real code block.
        rich = [lines[0]]
        if activity.commands:
            rich.append(rich_details("Show", rich_code("\n".join(f"$ {c.strip()}" for c in activity.commands[-40:]), "bash")))
        rich += lines[2:] if activity.commands else lines[1:]
        await self.notifier.send(Outgoing("\n".join(lines), markdown="\n\n".join(rich), session_id=s.id,
                                          job_id=s.job_id, silent=True, files=files))
