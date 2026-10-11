"""`ov`: the agent talks to Overseer.

The agent's shell has an `ov` command (in local.tmp_dir/agent-bin). It sends requests over a unix
socket (local.tmp_dir/ctl.sock) to the running Overseer, which answers right away:

  ov job add 18746955          "I submitted this": Overseer checks it with SLURM, then tracks it here
  ov job list | show <id>      jobs of this conversation (--all: every tracked job)
  ov job move <id> <project>   reassign a job to a project (or "main")
  ov job delete <id>           stop tracking it (the job itself keeps running; scancel cancels it)
  ov project list | new | set | archive
  ov conversation move <project>   this conversation belongs to a project (or "main")
  ov notify <text>             a message to the researcher right now
  ov status                    this conversation: project, autopilot, jobs

Each agent run gets a one-time token (OVERSEER_CTL in its environment) that tells Overseer which
conversation is asking; it stops working when the run ends. Every change is announced in Telegram.
Claims are verified: a job is only tracked if SLURM knows it as one of your jobs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Config
from .notify import Outgoing, code, esc
from .projects import ProjectError, Projects

if TYPE_CHECKING:
    from .agent import Agent

log = logging.getLogger(__name__)

TOKEN_ENV = "OVERSEER_CTL"  # (codex hides variables named *TOKEN*/*KEY*/*SECRET* from commands)


class CtlError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # an answer to the agent, not an exit
        raise CtlError(f"{message}\n\n{self.format_usage()}")


def _parser() -> _Parser:
    p = _Parser(prog="ov", description="talk to Overseer", add_help=False)
    sub = p.add_subparsers(dest="area")
    job = sub.add_parser("job", add_help=False).add_subparsers(dest="action")
    a = job.add_parser("add", add_help=False)
    a.add_argument("ids", nargs="+")
    a = job.add_parser("list", add_help=False)
    a.add_argument("--all", action="store_true")
    job.add_parser("show", add_help=False).add_argument("id")
    a = job.add_parser("move", add_help=False)
    a.add_argument("id")
    a.add_argument("project")
    job.add_parser("delete", add_help=False).add_argument("ids", nargs="+")
    proj = sub.add_parser("project", add_help=False).add_subparsers(dest="action")
    proj.add_parser("list", add_help=False)
    a = proj.add_parser("new", add_help=False)
    a.add_argument("name")
    a.add_argument("--dir", required=True)
    a.add_argument("--about", default="")
    a.add_argument("--move-here", action="store_true", help="also move this conversation into it")
    a = proj.add_parser("set", add_help=False)
    a.add_argument("name")
    a.add_argument("--dir")
    a.add_argument("--about")
    proj.add_parser("archive", add_help=False).add_argument("name")
    conv = sub.add_parser("conversation", add_help=False).add_subparsers(dest="action")
    conv.add_parser("move", add_help=False).add_argument("project")
    sub.add_parser("notify", add_help=False).add_argument("text", nargs="+")
    sub.add_parser("status", add_help=False)
    sub.add_parser("help", add_help=False)
    return p


HELP = """ov - talk to Overseer (the system that runs you and reports to the researcher on Telegram)

  ov job add <id> [<id>...]        register jobs you submitted (any way: sbatch, a script, a CLI). Overseer
                                   checks them with SLURM, reports on them, and autopilot waits for them.
  ov job list [--all]              jobs of this conversation (--all: every job Overseer tracks)
  ov job show <id>
  ov job move <id> <project|main>  reassign a job to a project
  ov job delete <id> [<id>...]     stop tracking jobs (they keep running; use scancel to cancel them)
  ov project list
  ov project new <name> --dir <folder on the HPC> [--about "what it is"] [--move-here]
  ov project set <name> [--dir ...] [--about ...]
  ov project archive <name>
  ov conversation move <project|main>
  ov notify <text>                 send the researcher a message now
  ov status                        this conversation: project, autopilot, its jobs
"""


class ControlServer:
    def __init__(self, cfg: Config, agent: Agent, projects: Projects, notifier, hpc):
        self.cfg = cfg
        self.agent = agent
        self.store = agent.store
        self.projects = projects
        self.notifier = notifier
        self.hpc = hpc
        self.tokens: dict[str, str] = {}  # token -> conversation
        self.server: asyncio.AbstractServer | None = None
        agent.ctl = self

    # ------------------------------------------------------------------ tokens
    def open(self, session_id: str) -> str:
        token = secrets.token_urlsafe(24)
        self.tokens[token] = session_id
        return token

    def close(self, token: str) -> None:
        self.tokens.pop(token, None)

    # ------------------------------------------------------------------ socket
    async def start(self) -> None:
        path = self.cfg.ctl_socket
        path.unlink(missing_ok=True)
        self.server = await asyncio.start_unix_server(self._client, path=str(path))
        os.chmod(path, 0o600)

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.cfg.ctl_socket.unlink(missing_ok=True)

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), 30)
            req = json.loads(line or b"{}")
            ok, text = await self.handle(str(req.get("token", "")), [str(a) for a in req.get("argv", [])])
        except Exception as e:  # never let a bad request hurt Overseer
            log.exception("ov request failed")
            ok, text = False, f"Overseer error: {type(e).__name__}: {e}"
        writer.write((json.dumps({"ok": ok, "text": text}) + "\n").encode())
        try:
            await writer.drain()
        finally:
            writer.close()

    # ------------------------------------------------------------------ requests
    async def handle(self, token: str, argv: list[str]) -> tuple[bool, str]:
        sid = self.tokens.get(token)
        if not sid:
            return False, ("Not connected to Overseer: `ov` only works inside a run Overseer started "
                           "(the token in $OVERSEER_CTL expired or is missing).")
        session = self.agent.get(sid)
        if session is None:
            return False, "This conversation no longer exists."
        if not argv or argv[0] in ("help", "-h", "--help"):
            return True, HELP
        try:
            args = _parser().parse_args(argv)
            area = args.area
            if area == "job" and args.action:
                return True, await getattr(self, f"job_{args.action}")(session, args)
            if area == "project" and args.action:
                return True, await getattr(self, f"project_{args.action}")(session, args)
            if area == "conversation" and args.action == "move":
                return True, await self.conversation_move(session, args)
            if area == "notify":
                return True, await self.notify(session, " ".join(args.text))
            if area == "status":
                return True, self.status(session)
            return False, HELP
        except (CtlError, ProjectError, ValueError) as e:
            return False, str(e)

    async def _announce(self, session, text: str, **kw) -> None:
        await self.notifier.send(Outgoing(text, session_id=session.id if session else None, silent=True, **kw))

    # ---- jobs
    async def _verify(self, job_id: str) -> tuple[str, str, str]:
        """(name, state, workdir) of one of your jobs, according to SLURM; raises if it isn't one."""
        if not job_id.isdigit():
            raise CtlError(f"{job_id!r} isn't a job id")
        await self.hpc.ensure_connected()
        info = await self.hpc.scontrol_job(job_id)
        if info:
            owner = info.get("UserId", "").split("(")[0]
            if owner and self.hpc.user and owner != self.hpc.user:
                raise CtlError(f"job {job_id} belongs to {owner}, not to {self.hpc.user}")
            return info.get("JobName", ""), info.get("JobState", ""), info.get("WorkDir", "")
        job = await self.hpc.sacct_job(job_id)  # finished a while ago: only accounting still knows it
        if job:
            return job.name, job.state, job.workdir
        raise CtlError(f"SLURM doesn't know a job {job_id} of yours")

    async def job_add(self, session, args) -> str:
        out = []
        for jid in args.ids:
            try:
                name, state, workdir = await self._verify(jid)
            except CtlError as e:
                out.append(f"✗ {e}")
                continue
            rec = self.store.get_job(jid)
            already = bool(rec and rec.by_agent and rec.session_id == session.id and not rec.ignored)
            if rec and rec.ignored:
                rec.ignored = 0
                self.store.upsert_job(rec)
            if not already:
                self.agent._record_job(session, jid)
                rec = self.store.get_job(jid)
                if not rec.name:  # what SLURM said, until the monitor's next poll
                    rec.name, rec.state, rec.workdir = name, state or rec.state, workdir
                    self.store.upsert_job(rec)
                await self._announce(session, f"🚀 <b>Agent submitted job {esc(jid)}</b> {esc(name)} ({esc(state)})",
                                     job_id=jid)
            out.append(f"✓ job {jid} ({name or '?'}, {state or '?'}) is tracked in this conversation"
                       + (" (already was)" if already else "") + ": Overseer reports on it"
                       + (", and autopilot waits for it" if self.agent.autopilot and self.agent.autopilot.active(session.id)
                          else ""))
        return "\n".join(out)

    async def job_list(self, session, args) -> str:
        rows = self.store.recent_jobs(40) if args.all else self.store.session_jobs(session.id)
        if not rows:
            return "No jobs." if args.all else "No jobs in this conversation (register yours with: ov job add <id>)."
        return "\n".join(f"{r.job_id}  {r.state or '?':<10} {r.name or ''}"
                         + (f"  [project {r.project}]" if r.project else "")
                         + ("  (this conversation)" if r.session_id == session.id else "") for r in rows)

    async def job_show(self, session, args) -> str:
        rec = self.store.get_job(args.id)
        if not rec:
            raise CtlError(f"Overseer doesn't track job {args.id} (ov job add {args.id} to track it)")
        return "\n".join(x for x in [
            f"job {rec.job_id}: {rec.name} - {rec.state}" + (" (ended)" if rec.ended_at else ""),
            f"project: {rec.project or 'main'}" + (f"; conversation: {rec.session_id}" if rec.session_id else ""),
            f"workdir: {rec.workdir}" if rec.workdir else "",
            f"stdout: {rec.stdout}" if rec.stdout else "",
            f"report: {rec.report[:1500]}" if rec.report else "",
        ] if x)

    async def job_move(self, session, args) -> str:
        project = None if args.project.lower() in ("main", "none", "general") else args.project.lower()
        if project and not self.store.get_project(project):
            raise CtlError(f"no project {project!r} (ov project list)")
        rec = self.store.get_job(args.id) or None
        if not rec:
            raise CtlError(f"Overseer doesn't track job {args.id}")
        old = rec.project
        self.agent.move_job(args.id, project)
        where = f"project {code(project)}" if project else "the main chat"
        await self._announce(None, f"🔀 Job {code(args.id)} {esc(rec.name)} moved to {where} by the agent "
                                   f"(was: {esc(old or 'main')}).", job_id=args.id)
        return f"✓ job {args.id} moved to {project or 'main'}"

    async def job_delete(self, session, args) -> str:
        out = []
        for jid in args.ids:
            rec = self.store.get_job(jid)
            if not rec or rec.ignored:
                out.append(f"✗ Overseer doesn't track job {jid}")
                continue
            rec.ignored = 1
            self.store.upsert_job(rec)
            await self._announce(session, f"🗑 Overseer no longer tracks job {code(jid)} {esc(rec.name)} "
                                          "(the job itself wasn't touched).")
            out.append(f"✓ job {jid} is no longer tracked (it keeps running; scancel {jid} would cancel it)")
        return "\n".join(out)

    # ---- projects
    async def project_list(self, session, args) -> str:
        rows = self.store.list_projects(include_archived=True)
        if not rows:
            return "No projects yet (ov project new <name> --dir <folder>)."
        here = session.project if session else None
        return "\n".join(f"{p.name}{' (archived)' if p.archived else ''}  {p.dir}"
                         + (f"  - {p.context[:120]}" if p.context else "")
                         + ("  <- this conversation" if p.name == here else "") for p in rows)

    async def project_new(self, session, args) -> str:
        p = self.projects.create(args.name, args.dir, args.about, "agent")
        if hasattr(self.notifier, "ensure_topic"):
            await self.notifier.ensure_topic(p.name)
        await self._announce(None, f"📁 The agent created project {code(p.name)}: {code(p.dir)}"
                                   + (f"\n{esc(p.context)}" if p.context else ""), project=p.name)
        msg = f"✓ project {p.name} created ({p.dir})"
        if args.move_here:
            msg += "\n" + await self.conversation_move(session, argparse.Namespace(project=p.name))
        return msg

    async def project_set(self, session, args) -> str:
        if args.dir is None and args.about is None:
            raise CtlError("nothing to change: give --dir and/or --about")
        p = self.projects.update(args.name.lower(), dir=args.dir, context=args.about)
        await self._announce(None, f"📁 The agent updated project {code(p.name)}: {code(p.dir)}"
                                   + (f"\n{esc(p.context)}" if args.about is not None else ""), project=p.name)
        return f"✓ project {p.name} updated"

    async def project_archive(self, session, args) -> str:
        p = self.projects.archive(args.name.lower())
        await self._announce(None, f"🗄 The agent archived project {code(p.name)} (/project restore {esc(p.name)} "
                                   "brings it back).")
        return f"✓ project {p.name} archived"

    async def conversation_move(self, session, args) -> str:
        project = None if args.project.lower() in ("main", "none", "general") else args.project.lower()
        if project and not self.store.get_project(project):
            raise CtlError(f"no project {project!r} (ov project list)")
        session.rec.project = project
        self.store.upsert_session(session.rec)
        await self._announce(session, "📁 This conversation now belongs to "
                                      + (f"project {code(project)}." if project else "no project."))
        return (f"✓ this conversation is now in {project or 'no project'}"
                + (" (the project's notes and context apply from your next run)" if project else ""))

    # ---- other
    async def notify(self, session, text: str) -> str:
        await self.notifier.send(Outgoing(f"📣 {esc(text[:3500])}", session_id=session.id))
        return "✓ sent"

    def status(self, session) -> str:
        ap = self.agent.autopilot.active(session.id) if self.agent.autopilot else None
        jobs = [j for j in self.store.session_jobs(session.id) if not j.ended_at]
        lines = [f"conversation {session.id}, project: {session.project or 'none'}",
                 (f"autopilot: on, round {ap.rounds}/{ap.max_rounds}, {ap.status}" if ap else "autopilot: off"),
                 "jobs Overseer waits for: " + (", ".join(f"{j.job_id} ({j.state})" for j in jobs) or "none")]
        return "\n".join(lines)


# ============================================================================ the agent's `ov` command

OV_CLIENT = r'''#!@PYTHON@ -I
# ov - talk to Overseer (see `ov help`).
import json, os, socket, sys

SOCK = @SOCK@
token = os.environ.get("OVERSEER_CTL", "")
# Folders are on the HPC: an unquoted ~/x was expanded by the local shell to this machine's home - undo that.
home = os.environ.get("HOME", "").rstrip("/")
argv = [("~" + a[len(home):]) if home and (a == home or a.startswith(home + "/")) else a for a in sys.argv[1:]]
try:
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(120)
    s.connect(SOCK)
    s.sendall((json.dumps({"token": token, "argv": argv}) + "\n").encode())
    data = b""
    while not data.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    res = json.loads(data or b"{}")
except (OSError, ValueError) as e:
    sys.stderr.write("ov: can't reach Overseer (%s)\n" % e)
    sys.exit(2)
(sys.stdout if res.get("ok") else sys.stderr).write(res.get("text", "") + "\n")
sys.exit(0 if res.get("ok") else 1)
'''


def write_ov_tool(cfg: Config, bin_dir: Path, write) -> None:
    write(bin_dir / "ov", OV_CLIENT.replace("@PYTHON@", sys.executable).replace("@SOCK@", repr(str(cfg.ctl_socket))),
          0o700)

