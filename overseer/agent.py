"""Agent conversations: Telegram message -> Claude Code / Codex run -> Telegram reply.

The agent has full access to the HPC (`ssh hpc ...`) and is sandboxed on this machine
(see runners.py). Overseer's job here is bookkeeping: which conversation belongs to which
job, resuming the CLI's session, reporting the commands it ran, and noticing the jobs it
submits so the monitor reports on them and replies continue the same conversation.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from .config import BACKENDS, Config
from .hpc import HPC
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
- Never run heavy computation on the login node; use sbatch (or srun inside an allocation).
- For non-trivial changes to the researcher's code, prefer a copy or a git branch, and say where it is.
- When you submit a job, leave sbatch's normal output ("Submitted batch job <id>") visible: Overseer uses it to \
track the job and report on it.
- Don't cancel or modify jobs you weren't asked about.
- Nobody can approve commands for you: there is no approval step. If a command is refused (sandbox or \
permission error), it will not be approved later - do it another way, or tell the researcher exactly what failed \
and quote the error. Never ask them to approve something.
- Your final message is sent to Telegram: short plain text (no tables or headings) - what you found, what you \
changed (paths), the job ids you submitted, and what happens next."""


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


class Agent:
    def __init__(self, cfg: Config, store: Store, notifier: Notifier, hpc: HPC):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.hpc = hpc
        self.monitor = None  # set by the app; poked when the agent submits a job
        self.sessions: dict[str, Session] = {}
        self._runners: dict[str, Runner] = {}

    # ------------------------------------------------------------------ backend
    @property
    def backend(self) -> str:
        b = self.store.get("backend") or self.cfg.agent.backend
        return b if b in BACKENDS else self.cfg.agent.backend

    def set_backend(self, name: str) -> None:
        if name not in BACKENDS:
            raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
        self.store.set("backend", name)

    def runner(self, backend: str | None = None) -> Runner:
        b = backend or self.backend
        if b not in self._runners:
            self._runners[b] = make_runner(self.cfg, b)
        r = self._runners[b]
        # Overrides set from Telegram (/model, /effort) win over config.yaml.
        r.model_override = self.store.get(f"model:{b}")
        r.effort_override = self.store.get(f"effort:{b}")
        return r

    def set_model(self, name: str, backend: str | None = None) -> None:
        """Set the model for a backend; "default" goes back to config.yaml."""
        b = backend or self.backend
        name = name.strip()
        if not name or any(c.isspace() for c in name):
            raise ValueError("model name must be a single word, e.g. opus or gpt-5.5-codex")
        if name == "default":
            self.store.unset(f"model:{b}")
        else:
            self.store.set(f"model:{b}", name)

    def set_effort(self, level: str, backend: str | None = None) -> None:
        b = backend or self.backend
        level = level.strip().lower()
        if level == "default":
            self.store.unset(f"effort:{b}")
            return
        allowed = self.runner(b).efforts
        if level not in allowed:
            raise ValueError(f"{b} effort must be one of: {', '.join(allowed)} (or default)")
        self.store.set(f"effort:{b}", level)

    def describe(self, backend: str) -> str:
        """e.g. 'claude_code: model opus · effort high (last run: claude-opus-5-5)'."""
        r = self.runner(backend)
        last = self.store.get(f"last_model:{backend}")
        model = r.model or "CLI default"
        if r.model_override:
            model += " (set via /model)"
        effort = r.effort or "CLI default"
        if r.effort_override:
            effort += " (set via /effort)"
        out = f"{backend}: model {model} · effort {effort}"
        if last and last != r.model:
            out += f" · last run used {last}"
        return out

    def footer(self, backend: str, res) -> str:
        r = self.runner(backend)
        parts = [backend, res.model or r.model or "default model", f"effort {r.effort or 'default'}"]
        tokens = res.input_tokens + res.output_tokens
        if tokens:
            cached = f" ({_human(res.cached_tokens)} cached)" if res.cached_tokens else ""
            parts.append(f"{_human(tokens)} tokens{cached}")
        if res.cost_usd is not None:
            parts.append(f"${res.cost_usd:.2f}")
        return " · ".join(parts)

    # ------------------------------------------------------------------ sessions
    def _load(self, rec: SessionRecord) -> Session:
        if rec.id not in self.sessions:
            self.sessions[rec.id] = Session(rec)
        return self.sessions[rec.id]

    def _new(self, job_id: str | None) -> Session:
        rec = SessionRecord(id=(f"j{job_id}-" if job_id else "chat-") + uuid.uuid4().hex[:6], job_id=job_id,
                            created=time.time())
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
        return self._load(rec) if rec else self._new(job_id)

    def general_session(self, reset: bool = False) -> Session:
        sid = None if reset else self.store.get("general_session")
        s = self.get(sid)
        if s is None:
            s = self._new(None)
            self.store.set("general_session", s.id)
        return s

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
    def system_prompt(self) -> str:
        domains = self.cfg.agent.allowed_domains
        setup = self.cfg.slurm.setup.strip()
        return SYSTEM_PROMPT.format(
            host=self.cfg.ssh.host,
            setup=f"{setup}; " if setup else "",
            domains=f" (except {', '.join(domains)})" if domains else "",
        )

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
    async def handle(self, s: Session, user_text: str) -> None:
        if not self.cfg.agent.enabled:
            await self.notifier.send(Outgoing("The agent is disabled in the config."))
            return
        async with s.lock:
            s.task = asyncio.current_task()
            try:
                await self._turn(s, user_text)
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

    async def _turn(self, s: Session, user_text: str) -> None:
        backend = self.backend
        runner = self.runner(backend)
        resume = s.rec.cli_session_id if s.rec.backend == backend else None
        parts = []
        if resume is None and s.job_id:
            parts.append(await self._job_context(s.job_id))
        parts.append(user_text)
        prompt = "\n\n".join(parts)

        await ensure_master(self.cfg)
        activity, live, buffer = self._activity(s)
        try:
            res = await runner.run(prompt, system=self.system_prompt(), session_id=resume, key=s.id, activity=activity)
        finally:
            if live:
                live.cancel()
            await self._report_activity(s, activity, buffer)
        if res.session_id:
            s.rec.backend, s.rec.cli_session_id = backend, res.session_id
            self.store.upsert_session(s.rec)
        if res.model:
            self.store.set(f"last_model:{backend}", res.model)
        await self.notifier.send(Outgoing(
            f"🤖 {esc(res.text or '(no answer)')}\n\n<i>{esc(self.footer(backend, res))}</i>",
            session_id=s.id, job_id=s.job_id))

    def _activity(self, s: Session) -> tuple[Activity, asyncio.Task | None, list[str]]:
        activity = Activity()
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
                            "💻 " + "\n".join(pre(c, 400) for c in batch), session_id=s.id, job_id=s.job_id, silent=True))

            live = asyncio.get_running_loop().create_task(flush_loop())
        return activity, live, buffer

    async def _report_activity(self, s: Session, activity: Activity, buffer: list[str]) -> None:
        mode = self.cfg.agent.report_commands
        if mode == "off":
            return
        if mode == "live":
            if buffer:
                await self.notifier.send(Outgoing("💻 " + "\n".join(pre(c, 400) for c in buffer),
                                                  session_id=s.id, job_id=s.job_id, silent=True))
            return
        if activity.blocked:
            lines = [f"⚠️ <b>{len(activity.blocked)} command(s) were refused by {esc(self.backend)} itself</b> "
                     "(not by Overseer - there is no approval step):"]
            for cmd, reason in activity.blocked[:10]:
                lines.append(pre(cmd, 300) + f"→ {esc(reason)}")
            await self.notifier.send(Outgoing("\n".join(lines), session_id=s.id, job_id=s.job_id))
        if not (activity.commands or activity.files):
            return
        lines = [f"🛠 <b>{len(activity.commands)} command(s)</b>"]
        for c in activity.commands[-40:]:
            lines.append(pre(c, 300))
        if len(activity.commands) > 40:
            lines.insert(1, f"(showing the last 40 of {len(activity.commands)})")
        if activity.files:
            lines.append("✏️ local files: " + ", ".join(code(f) for f in activity.files[:20]))
        files = [("commands.txt", "\n\n".join(activity.commands).encode())] if len(activity.commands) > 15 else []
        await self.notifier.send(Outgoing("\n".join(lines), session_id=s.id, job_id=s.job_id, silent=True, files=files))

    def _record_job(self, s: Session, job_id: str) -> None:
        rec = self.store.get_job(job_id) or JobRecord(job_id=job_id, state="PENDING", first_seen=time.time(),
                                                      last_status_at=time.time())
        rec.by_agent = 1
        rec.session_id = s.id
        self.store.upsert_job(rec)
        if self.monitor:
            self.monitor.poke()
