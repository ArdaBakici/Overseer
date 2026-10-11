"""Agent conversations: Telegram message -> Claude Code / Codex run -> Telegram reply.

The agent has full access to the HPC (`ssh hpc ...`) and is sandboxed on this machine
(see runners.py). Overseer's job here is bookkeeping: which conversation belongs to which
job, resuming the CLI's session, reporting the commands it ran, and noticing the jobs it
submits so the monitor reports on them and replies continue the same conversation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .config import BACKENDS, Config
from .hpc import HPC
from .autopilot import parse_status
from .live import Progress, Transcript
from .richtext import fold, fold_long, md_to_html, rich_code, rich_details, rich_markdown
from .notify import Notifier, Outgoing, code, esc, pre
from .runners import Activity, AgentRunError, Runner, make_runner
from .sshlink import ensure_master
from .store import JobRecord, SessionRecord, Store

log = logging.getLogger(__name__)

LIVE_FLUSH_SECONDS = 5

SYSTEM_PROMPT = """You are Overseer's agent. You help an ML researcher run, debug and fix SLURM jobs on their \
university HPC cluster. They talk to you through Telegram.

## Environment
- You run on the researcher's homelab, NOT on the HPC. Your local working directory is a scratch folder; \
you can't write anywhere else on this machine, and the local shell has no internet access{domains}.
- The HPC is reachable as `{host}`: `ssh {host} '<command>'`, and `rsync`/`scp` with `{host}:<path>`. This goes \
through a connection Overseer keeps open. On the HPC you have the researcher's full permissions.
- `ssh {host} '<command>'` runs a non-login shell that doesn't read the login profile, so SLURM commands \
and modules are often not on PATH there. Run remote commands through a login shell: \
`ssh {host} 'bash -lc "{setup}squeue -u $USER"'`. Load modules / activate environments explicitly inside it.

## How to work
- Investigate with real commands: squeue, `sacct -j <id> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem`, \
scontrol show job, the job's log files and batch script, the code, disk quota, module/env availability.
- The login node is only for quick commands: looking at files and logs, git, editing, squeue/sacct, submitting \
jobs, installing packages. Anything that computes - running Python scripts, plotting, processing data, \
compiling, tests, evaluation - runs on a compute node: `hpc-run --dir <project dir> -- <command>` runs it in a \
compute node's login shell and returns its output (it reuses a node Overseer keeps for a while, so it usually \
starts at once; `hpc-run --cpus N --mem 64G --time 120 -- ...` or `--gpu` gets a node of its own; \
`hpc-run --help` for more). Long runs and experiments are sbatch jobs as usual.{login_limit}
- For non-trivial changes to the researcher's code, prefer a copy or a git branch, and say where it is.
- When you submit a job, leave sbatch's normal output ("Submitted batch job <id>") visible: Overseer uses it to \
track the job and report on it.
- Don't cancel or modify jobs you weren't asked about.
- Nobody can approve commands for you: there is no approval step. If a command is refused (sandbox or \
permission error), it will not be approved later - do it another way, or tell the researcher exactly what failed \
and quote the error. Never ask them to approve something.
- Your final message is sent to Telegram, rendered from Markdown. Keep it short: what you found, what you changed \
(paths), the job ids you submitted, and what happens next. **Bold**, lists, `code`, small code blocks and small \
tables are fine; avoid headings and wide tables (it's read on a phone).

## Sending plots and files
Anything you put in `./outbox/` (in your local working directory) is sent to the researcher with your reply: \
images (.png, .jpg) as pictures, other files as documents - up to 10 per reply, images up to 10 MB, files up to \
50 MB. Plots are made on the HPC, where the data is: write a small plotting script there (matplotlib in the \
project's environment), run it with `hpc-run --dir <dir> -- python plot.py` (not on the login node), save a PNG \
(e.g. dpi=150, readable on a phone), copy it back with `scp {host}:/path/plot.png ./outbox/`, and say in your \
reply what it shows."""


PROJECT_SECTION = """

## Project: {name}
This conversation belongs to the project in `{dir}` on the HPC. Work there unless asked otherwise.{context}
Project notes: `{notes}` holds durable notes for this project - how to run things, environments, data \
locations, conventions, known issues. It is shown to you at the start of each conversation. When you learn \
something future conversations here should know, add it there (keep it short and current; create it if missing)."""

NOTES_FILE = "OVERSEER.md"


IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_PHOTO = 10 * 1024 * 1024  # Telegram's limits for pictures and documents sent by bots
MAX_FILE = 50 * 1024 * 1024
MAX_ATTACHMENTS = 10


def collect_outbox(folder: Path) -> tuple[list[tuple[str, bytes]], list[tuple[str, bytes]], list[str]]:
    """Take the files the agent left in its outbox: (pictures, documents, notes about skipped files).
    Overseer isn't sandboxed, so only regular files are read - never symlinks or hard links, which
    could point at something the agent itself can't read."""
    photos: list[tuple[str, bytes]] = []
    docs: list[tuple[str, bytes]] = []
    notes: list[str] = []
    try:
        entries = sorted((e for e in os.scandir(folder) if not e.name.startswith(".")),
                         key=lambda e: e.stat(follow_symlinks=False).st_mtime)
    except OSError:
        return photos, docs, notes
    for e in entries:
        if len(photos) + len(docs) >= MAX_ATTACHMENTS:
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
            else:
                docs.append((e.name, fh.read()))
        path.unlink(missing_ok=True)
    return photos, docs, notes


def _command_fold(commands: list[str]) -> str:
    """Commands in a collapsible section: the first few lines show, the rest on tap."""
    def clip(c: str) -> str:
        c = c.strip()
        return c if len(c) <= 300 else c[:299] + "…"
    return fold(esc("\n".join(f"$ {clip(c)}" for c in commands)))


def _duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def _human(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


@dataclass
class Session:
    rec: SessionRecord
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task | None = None

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
    def __init__(self, cfg: Config, store: Store, notifier: Notifier, hpc: HPC):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.hpc = hpc
        self.monitor = None  # set by the app; poked when the agent submits a job
        self.autopilot = None  # set by Autopilot(...): wakes conversations when their jobs finish
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
            self._runners[key] = make_runner(self.cfg, b)
        r = self._runners[key]
        # Overrides set from Telegram (/model, /effort) win over config.yaml.
        r.model_override = self._setting(f"model:{b}", project)
        r.effort_override = self._setting(f"effort:{b}", project)
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

    def footer(self, backend: str, res, project: str | None = None) -> str:
        r = self.runner(backend, project)
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

    def move_job(self, job_id: str, project: str | None) -> JobRecord | None:
        """Reassign a job (and the conversation about it) to another project, or the main chat."""
        rec = self.store.get_job(job_id)
        if not rec:
            return None
        rec.project = project
        self.store.upsert_job(rec)
        for sess in [self.store.session_for_job(job_id)]:
            if sess and sess.job_id == job_id:
                sess.project = project
                self.store.upsert_session(sess)
                if sess.id in self.sessions:
                    self.sessions[sess.id].rec.project = project
        return rec

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

    # ------------------------------------------------------------------ prompts
    def system_prompt(self, session_id: str | None = None) -> str:
        domains = self.cfg.agent.allowed_domains
        setup = self.cfg.slurm.setup.strip()
        cpu = self.cfg.slurm.compute.login_cpu_seconds
        prompt = SYSTEM_PROMPT.format(
            login_limit=(f" Commands run directly on the login node are stopped after {cpu} CPU seconds." if cpu else ""),
            host=self.cfg.ssh.host,
            setup=f"{setup}; " if setup else "",
            domains=f" (except {', '.join(domains)})" if domains else "",
        )
        sess = self.get(session_id)
        proj = self.store.get_project(sess.project) if sess else None
        if proj:
            context = f"\n\nAbout this project, from the researcher:\n{proj.context.strip()}" if proj.context.strip() else ""
            prompt += PROJECT_SECTION.format(name=proj.name, dir=proj.dir, context=context,
                                             notes=f"{proj.dir.rstrip('/')}/{NOTES_FILE}")
        if self.autopilot and session_id:
            prompt += self.autopilot.system_section(session_id)
        return prompt

    def _remote_path(self, path: str) -> str:
        """'~/x' -> '/home/me/x' (paths are passed quoted, so the remote shell won't expand ~)."""
        if path.startswith("~/") and self.hpc.home:
            return self.hpc.home.rstrip("/") + path[1:]
        return path

    async def _project_notes(self, project: str) -> str:
        proj = self.store.get_project(project)
        if not proj or not proj.dir:
            return ""
        path = f"{proj.dir.rstrip('/')}/{NOTES_FILE}"
        notes = await self.hpc.try_head(self._remote_path(path), 400)
        if not notes.strip():
            return (f"<project_notes path='{path}'>(no notes yet - create the file when there is something "
                    "worth keeping)</project_notes>")
        return f"<project_notes path='{path}'>\n{notes[:12000].strip()}\n</project_notes>"

    async def _job_context(self, job_id: str) -> str:
        rec = self.store.get_job(job_id)
        if not rec:
            return f"(Job {job_id} is not known to Overseer; look it up with sacct.)"
        tail = await self.hpc.try_tail(rec.stdout, 60) if rec.stdout else ""
        return "\n".join(x for x in [
            f"<job id='{rec.job_id}' name='{rec.name}' state='{rec.state}' workdir='{rec.workdir}' "
            f"stdout='{rec.stdout}' stderr='{rec.stderr}'>",
            f"<overseer_analysis>\n{rec.report}\n</overseer_analysis>" if rec.report else "",
            f"<batch_script>\n{rec.batch_script[:8000]}\n</batch_script>" if rec.batch_script else "",
            f"<stdout_tail>\n{tail[-6000:]}\n</stdout_tail>" if tail else "",
            "</job>",
        ] if x)

    # ------------------------------------------------------------------ running a request
    async def handle(self, s: Session, user_text: str, quiet: bool = False) -> bool:
        """Run one request in a conversation. True if the agent answered. `quiet`: send the answer
        without a notification sound (autopilot check-ins) unless the autopilot needs attention."""
        if not self.cfg.agent.enabled:
            await self.notifier.send(Outgoing("The agent is disabled in the config."))
            return False
        ok = False
        async with s.lock:
            s.task = asyncio.current_task()
            try:
                await self._turn(s, user_text, quiet)
                ok = True
            except asyncio.CancelledError:
                await self.notifier.send(Outgoing("⏹ Stopped.", session_id=s.id, job_id=s.job_id))
            except AgentRunError as e:
                await self.notifier.send(Outgoing(f"⚠️ {esc(e)}", session_id=s.id, job_id=s.job_id))
            except Exception as e:
                log.exception("agent run failed")
                await self.notifier.send(Outgoing(f"⚠️ Agent error: {esc(type(e).__name__)}: {esc(str(e)[:800])}",
                                                  session_id=s.id, job_id=s.job_id))
            finally:
                s.task = None
        if self.autopilot:
            self.autopilot.turn_ended(s)
        return ok

    async def _turn(self, s: Session, user_text: str, quiet: bool = False) -> None:
        backend = self.backend_of(s.project)
        runner = self.runner(backend, s.project)
        resume = s.rec.cli_session_id if s.rec.backend == backend else None
        parts = []
        if resume is None and s.project:  # a new CLI session: give it the project's notes
            parts.append(await self._project_notes(s.project))
        if resume is None and s.job_id:
            parts.append(await self._job_context(s.job_id))
        parts.append(user_text)
        prompt = "\n\n".join(parts)

        outbox = runner.workdir(s.id) / "outbox"  # files the agent leaves here are sent with its reply
        outbox.mkdir(exist_ok=True)
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
        if progress:
            progress.start()
        try:
            await ensure_master(self.cfg)
            res = await runner.run(prompt, system=self.system_prompt(s.id), session_id=resume, key=s.id,
                                   activity=activity)
        except asyncio.CancelledError:
            transcript.write({"kind": "end", "status": "stopped"})
            raise
        except Exception as e:
            transcript.write({"kind": "end", "status": "error", "text": str(e)[:2000]})
            raise
        finally:
            if live:
                live.cancel()
            if progress:
                await progress.close()
            await self._report_activity(s, activity, buffer)
        if res.session_id:
            s.rec.backend, s.rec.cli_session_id = backend, res.session_id
            self.store.upsert_session(s.rec)
        if res.model:
            self.store.set(f"last_model:{backend}", res.model)
        if res.effort:
            self.store.set(f"last_effort:{backend}", res.effort)
        self._usage_delta(backend, res, resumed=resume is not None)
        footer = self.footer(backend, res, s.project)
        text, autopilot_line, status = res.text, "", None
        if self.autopilot and self.autopilot.active(s.id):
            text, status, detail = parse_status(res.text)
            autopilot_line = self.autopilot.after_turn(s, status, detail, activity.cancelled)
        photos, docs, notes = collect_outbox(outbox)
        if photos or docs:
            transcript.write({"kind": "tool", "name": "attachments", "detail": ", ".join(n for n, _ in photos + docs)})
        transcript.write({"kind": "end", "status": "done", "text": text, "footer": footer,
                          **({"autopilot": status} if status else {})})
        skipped = "\n".join(f"📎 {esc(n)}" for n in notes)
        extra = [x for x in (autopilot_line, skipped) if x]  # HTML, which Rich Markdown accepts as well
        await self.notifier.send(Outgoing(
            f"🤖 {fold_long(md_to_html(text)) or '(no answer)'}" + "".join(f"\n\n{x}" for x in extra)
            + f"\n\n<i>{esc(footer)}</i>",
            markdown="\n\n".join([rich_markdown(text) or "(no answer)", *extra, f"<i>{esc(footer)}</i>"]),
            session_id=s.id, job_id=s.job_id, silent=quiet and status not in ("DONE", "NEED_HELP"),
            photos=photos, files=docs))

    def _activity(self, s: Session) -> tuple[Activity, asyncio.Task | None, list[str]]:
        activity = Activity(login_warn_seconds=self.cfg.slurm.compute.login_warn_seconds)
        buffer: list[str] = []  # live mode: commands not yet sent

        def on_job(job_id: str, command: str) -> None:
            self._record_job(s, job_id)
            asyncio.get_running_loop().create_task(self.notifier.send(Outgoing(
                f"🚀 <b>Agent submitted job {esc(job_id)}</b>\n{pre(command, 600)}", job_id=job_id, session_id=s.id)))

        activity.on_job = on_job
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
            lines = [f"⚠️ <b>{len(activity.blocked)} command(s) were refused by {esc(self.backend_of(s.project))} itself</b> "
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
        for cmd, took in activity.login_node[:5]:  # heavy work that should have gone to a compute node
            lines.append(f"⚠️ ran {_duration(took)} on the login node: {code(cmd[:200])}")
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

    def _record_job(self, s: Session, job_id: str) -> None:
        rec = self.store.get_job(job_id) or JobRecord(job_id=job_id, state="PENDING", first_seen=time.time(),
                                                      last_status_at=time.time())
        rec.by_agent = 1
        rec.session_id = s.id
        rec.project = s.project
        self.store.upsert_job(rec)
        if self.monitor:
            self.monitor.poke()
