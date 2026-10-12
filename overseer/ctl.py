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
  ov watch <ids> | wake in 2h  what wakes the agent (autopilot.py)
  ov auto request|done|help    ask for autopilot (an approval card in Telegram), finish it, ask for help

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

from .compute import WORKER_NAME
from .config import Config
from .notify import Outgoing, code, esc
from .autopilot import WatchError, human_interval, parse_duration
from .projects import ProjectError, Projects
from .slurm import TERMINAL_STATES, array_parent

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
    a = sub.add_parser("watch", add_help=False)
    a.add_argument("ids", nargs="+")
    a.add_argument("--every")
    a.add_argument("--note", default="")
    sub.add_parser("unwatch", add_help=False).add_argument("ref")
    sub.add_parser("watches", add_help=False)
    a = sub.add_parser("wake", add_help=False)
    a.add_argument("mode", choices=["in", "every"])
    a.add_argument("duration")
    a.add_argument("note", nargs="*")
    auto = sub.add_parser("auto", add_help=False).add_subparsers(dest="action")
    a = auto.add_parser("request", add_help=False)
    a.add_argument("--goal", required=True)
    a.add_argument("--wakeups", type=int)
    a.add_argument("--hours", type=float)
    a.add_argument("--checkins")
    auto.add_parser("status", add_help=False)
    auto.add_parser("done", add_help=False).add_argument("text", nargs="*")
    auto.add_parser("help", add_help=False).add_argument("text", nargs="+")
    sub.add_parser("notify", add_help=False).add_argument("text", nargs="+")
    sub.add_parser("status", add_help=False)
    sub.add_parser("help", add_help=False)
    return p


HELP = """ov - talk to Overseer (the system that runs you and reports to the researcher on Telegram)

  ov job add <id> [<id>...]        register jobs you submitted (any way: sbatch, a script, a CLI). Overseer
                                   checks them with SLURM and reports on them to the researcher.
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
  ov status                        this conversation: project, autopilot, its jobs, what will wake you

Being woken (you only run when someone writes to you - or when something you set up wakes you):
  ov watch <id> [<id>...] [--every 2h] [--note "what to do then"]
                                   wake me when these jobs have ended (all of them, or as soon as one fails);
                                   --every: check-ins while they run (autopilot only)
  ov wake in 90m ["what to do then"]      wake me once, later
  ov wake every 2h ["..."]         wake me repeatedly (autopilot only)
  ov watches                       what will wake me (ids like w12)
  ov unwatch <w12 | job id | all>
  Without autopilot each of these wakes you once ("one-off"); a few can be pending, and after a few
  one-off wake-ups in a row without the researcher writing, ask for autopilot instead.

Autopilot (working on your own over many runs; the researcher approves it in Telegram):
  ov auto request --goal "..." [--wakeups 20] [--hours 48] [--checkins 1h|off]
                                   ask; then end your turn. If they approve (they may change the limits),
                                   you're woken to start.
  ov auto status
  ov auto done ["conclusion"]      the goal is reached / nothing sensible left: ends autopilot
  ov auto help "question"          you need the researcher's decision (they get a notification)
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
            if area in ("watch", "unwatch", "watches", "wake"):
                return True, await getattr(self, area)(session, args)
            if area == "auto" and args.action:
                return True, await getattr(self, f"auto_{args.action}")(session, args)
            if area == "notify":
                return True, await self.notify(session, " ".join(args.text))
            if area == "status":
                return True, self.status(session)
            return False, HELP
        except (CtlError, ProjectError, WatchError, ValueError) as e:
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
        for jid in dict.fromkeys(array_parent(x) for x in args.ids):  # a job array is tracked by its id
            try:
                name, state, workdir = await self._verify(jid)
            except CtlError as e:
                out.append(f"✗ {e}")
                continue
            if name == WORKER_NAME:
                out.append(f"✗ job {jid} is hpc-run's shared compute worker, not one of your jobs: don't register or "
                           "watch it (hpc-run starts, reuses and ends it)")
                continue
            rec = self.store.get_job(jid)
            alive = bool(state) and state.split()[0] not in TERMINAL_STATES
            already = bool(rec and rec.by_agent and rec.session_id == session.id and not rec.ignored
                           and not (rec.ended_at and alive))
            if rec and (rec.ignored or (rec.ended_at and alive)):  # deleted, or wrongly taken as ended: track again
                rec.ignored, rec.missing_polls = 0, 0
                if alive:
                    rec.ended_at, rec.state = None, state.split()[0]
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
                       + (f" (ov watch {jid} to be woken when it ends)" if not self._watched(session, jid) else ""))
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
            if self.agent.autopilot:
                self.agent.autopilot.forget_job(jid)
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
        jobs = [j for j in self.store.session_jobs(session.id) if not j.ended_at]
        lines = [f"conversation {session.id}, project: {session.project or 'none'}", self._auto_line(session),
                 "running jobs of this conversation: " + (", ".join(f"{j.job_id} ({j.state})" for j in jobs) or "none")]
        if self.agent.autopilot:
            lines.append("what will wake you:\n" + self.agent.autopilot.watch_lines(session.id))
        return "\n".join(lines)

    # ---- being woken
    def _ap(self):
        if not self.agent.autopilot:
            raise CtlError("wake-ups aren't available in this Overseer")
        return self.agent.autopilot

    def _watched(self, session, job_id: str) -> bool:
        return bool(self.agent.autopilot) and any(
            w.session_id == session.id for w in self.store.job_watches(job_id))

    def _auto_line(self, session) -> str:
        ap = self.agent.autopilot
        a = ap.active(session.id) if ap else None
        if a:
            return (f"autopilot: ON - wake-up {a.rounds} of {a.max_rounds} used, {ap._left(a)} left, check-ins "
                    f"{ap._checkin_text(a)}; goal: {a.goal}")
        rec = self.store.get_autopilot(session.id) if ap else None
        if rec and rec.status == "requested":
            return "autopilot: requested - waiting for the researcher's answer"
        return "autopilot: off" + (f" (last: {rec.status})" if rec and rec.status else "")

    def _duration(self, text: str, what: str) -> float:
        d = parse_duration(text)
        if d is None or d <= 0:
            raise CtlError(f"{what}: {text!r} isn't a duration (e.g. 30m, 2h, 1d)")
        return d

    async def watch(self, session, args) -> str:
        ap = self._ap()
        every = self._duration(args.every, "--every") if args.every else 0.0
        ids = list(dict.fromkeys(array_parent(x) for x in args.ids))  # a job array is watched by its id
        for jid in ids:
            rec = self.store.get_job(jid)
            if rec and rec.name == WORKER_NAME:
                raise CtlError(f"job {jid} is hpc-run's shared compute worker, not one of your jobs: there's nothing "
                               "to watch")
            if rec and rec.ended_at and not rec.ignored and rec.state != "UNKNOWN":
                raise CtlError(f"job {jid} has already ended ({rec.state}): look at it now (ov job show {jid})")
            if not rec or rec.ignored or rec.ended_at:  # not tracked (yet), or its end wasn't confirmed: ask SLURM
                name, state, _ = await self._verify(jid)
                if name == WORKER_NAME:
                    raise CtlError(f"job {jid} is hpc-run's shared compute worker, not one of your jobs: there's "
                                   "nothing to watch")
                if state and state.split()[0] in TERMINAL_STATES:
                    raise CtlError(f"job {jid} has already ended ({state}): look at it now")
                await self.job_add(session, argparse.Namespace(ids=[jid]))
        w, remark = ap.add_watch(session, ids, every, args.note)
        what = f"job{'s' if len(ids) > 1 else ''} {', '.join(ids)}"
        if w.oneoff:
            await self._oneoff_announce(session, w, f"look at {what} when {'they end' if len(ids) > 1 else 'it ends'}")
            return (f"✓ w{w.id}: you'll be woken once, when {what} {'have' if len(ids) > 1 else 'has'} ended "
                    "(a one-off: autopilot isn't on). The researcher was told and can cancel it.")
        return (f"✓ w{w.id}: you'll be woken when {what} {'have' if len(ids) > 1 else 'has'} ended"
                + (" (or as soon as one fails)" if len(ids) > 1 else "")
                + (f", with check-ins every {human_interval(w.every)} while running" if w.every else "") + remark)

    async def _oneoff_announce(self, session, w, what: str) -> None:
        note = f"\n{esc(w.note)}" if w.note else ""
        await self.notifier.send(Outgoing(f"⏰ The agent will {esc(what)} (a one-off wake-up).{note}",
                                          session_id=session.id, silent=True,
                                          buttons=[[("✖ Cancel", f"uw:{w.id}")]]))

    async def wake(self, session, args) -> str:
        ap = self._ap()
        d = self._duration(args.duration, "wake")
        note = " ".join(args.note)
        w = ap.add_timer(session, d, d if args.mode == "every" else 0.0, note)
        if w.oneoff:
            await self._oneoff_announce(session, w, f"look again in {human_interval(d)}")
            return (f"✓ w{w.id}: you'll be woken once, in {human_interval(d)} (a one-off: autopilot isn't on). The "
                    "researcher was told and can cancel it.")
        return f"✓ w{w.id}: you'll be woken in {human_interval(d)}" + (
            f", then every {human_interval(d)} (each one is a wake-up)" if w.every else " (one wake-up)")

    async def watches(self, session, args) -> str:
        return self._ap().watch_lines(session.id)

    async def unwatch(self, session, args) -> str:
        changed = self._ap().unwatch(session.id, args.ref.strip())
        if not changed:
            raise CtlError(f"nothing matches {args.ref!r} (ov watches lists them)")
        return "✓ updated: " + ", ".join(f"w{w.id} ({'removed' if w.state == 'cancelled' else 'now ' + w.jobs})"
                                          for w in changed)

    # ---- autopilot
    async def auto_request(self, session, args) -> str:
        ap = self._ap()
        checkin = None
        if args.checkins is not None:
            checkin = parse_duration(args.checkins)
            if checkin is None:
                raise CtlError(f"--checkins: {args.checkins!r} isn't a duration (e.g. 1h, or off)")
        a = ap.request(session, args.goal, args.wakeups, args.hours, checkin)
        text, buttons = ap.card(a)
        await self.notifier.send(Outgoing(text, session_id=session.id, buttons=buttons))
        return ("✓ asked: the researcher sees an approval card in Telegram (they may change the limits). End your "
                "turn now. If they approve, you're woken to start; if not, you'll hear it next time they write.")

    async def auto_status(self, session, args) -> str:
        return self._auto_line(session) + "\nwhat will wake you:\n" + self._ap().watch_lines(session.id)

    async def auto_done(self, session, args) -> str:
        self._ap().finish(session.id, " ".join(args.text))
        return "✓ autopilot finished (announced with your reply); your autopilot watches and timers are removed"

    async def auto_help(self, session, args) -> str:
        self._ap().need_help(session.id, " ".join(args.text))
        return ("✓ the researcher will be asked (with a notification) when your reply is sent. End your reply now; "
                "your watches stay, and their answer comes as a message.")


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


SKILL = Path(__file__).parent / "skills" / "overseer" / "SKILL.md"


def install_skill(skills_dir: Path) -> None:
    """Put the `overseer` skill (how to use ov) into an agent CLI's skills folder."""
    text = SKILL.read_text()
    d = skills_dir / "overseer"
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not (d / "SKILL.md").exists() or (d / "SKILL.md").read_text() != text:
        (d / "SKILL.md").write_text(text)


def write_skills(cfg: Config) -> None:
    """Claude Code's copy of the skill, in CLAUDE_CONFIG_DIR/skills (Claude Code reads skills itself).
    Codex has the model read a skill's file with a command, and its sandbox can't read Overseer's state
    dir: its copy goes into each scratch dir, as .agents/skills (CodexRunner)."""
    install_skill(cfg.local.resolve(cfg.agent.claude_code.home) / "skills")
