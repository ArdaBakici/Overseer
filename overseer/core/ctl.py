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
  ov status                    this conversation: project, jobs, what will wake the agent
  ov watch <ids> | wake in 2h  what wakes the agent (autopilot.py); no approval needed

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
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Config
from .notify import Outgoing, code, esc
from .autopilot import WatchError, human_interval, parse_duration
from .projects import ProjectError, Projects
from .subagents import ENDED, STATE_LABEL, SubagentError, model_label
from .feedback import FeedbackError
from .addons import SkillError

if TYPE_CHECKING:
    from .agent import Agent

log = logging.getLogger(__name__)

TOKEN_ENV = "OVERSEER_CTL"  # (codex hides variables named *TOKEN*/*KEY*/*SECRET* from commands)


class CtlError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # an answer to the agent, not an exit
        raise CtlError(f"{message}\n\n{self.format_usage()}")


def _parser(ext=None) -> _Parser:
    p = _Parser(prog="ov", description="talk to Overseer", add_help=False)
    sub = p.add_subparsers(dest="area")
    if ext is not None:  # the domain's commands (e.g. the HPC's job, metrics, results, ...)
        ext.register(sub)
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
    sub.add_parser("unwatch", add_help=False).add_argument("ref")
    sub.add_parser("watches", add_help=False)
    a = sub.add_parser("wake", add_help=False)
    a.add_argument("mode", choices=["in", "every"])
    a.add_argument("duration")
    a.add_argument("--for", dest="for_", help="every: how long to keep waking (default: agent.autopilot.repeat_hours)")
    a.add_argument("note", nargs="*")
    sub.add_parser("auto", add_help=False).add_argument("rest", nargs=argparse.REMAINDER)  # (no longer used)
    a = sub.add_parser("feedback", add_help=False)
    a.add_argument("words", nargs="*")
    a.add_argument("--kind", default="limitation")
    a.add_argument("--context", default="")
    a.add_argument("--all", action="store_true")
    sk = sub.add_parser("skill", add_help=False).add_subparsers(dest="action")
    a = sk.add_parser("propose", add_help=False)
    a.add_argument("text")
    a.add_argument("--all", action="store_true")
    sk.add_parser("list", add_help=False)
    sk.add_parser("show", add_help=False).add_argument("name")
    sub.add_parser("propose", add_help=False).add_argument("plan")
    a = sub.add_parser("spawn", add_help=False)
    a.add_argument("name")
    a.add_argument("task", nargs="+")
    a.add_argument("--model")
    a.add_argument("--backend")
    sub.add_parser("agents", add_help=False)
    ag = sub.add_parser("agent", add_help=False).add_subparsers(dest="action")
    ag.add_parser("show", add_help=False).add_argument("name")
    ag.add_parser("stop", add_help=False).add_argument("name")
    a = ag.add_parser("send", add_help=False)
    a.add_argument("name")
    a.add_argument("text", nargs="+")
    sub.add_parser("planner", add_help=False).add_argument("mode", choices=["off"])
    sub.add_parser("notify", add_help=False).add_argument("text", nargs="+")
    hist = sub.add_parser("history", add_help=False).add_subparsers(dest="action")
    a = hist.add_parser("search", add_help=False)
    a.add_argument("text", nargs="+")
    a.add_argument("--project")
    a.add_argument("--kind", action="append")
    a.add_argument("--since")
    a.add_argument("--limit", type=int, default=12)
    a = hist.add_parser("show", add_help=False)
    a.add_argument("conversation")
    a.add_argument("--last", type=int, default=5)
    a.add_argument("--grep", default="")
    sub.add_parser("status", add_help=False)
    sub.add_parser("help", add_help=False)
    return p


HELP = """ov - talk to Overseer (the system that runs you and reports to the researcher on Telegram)

  ov project list | new <name> --dir <folder> [--about "..."] [--move-here] | set <name> ... | archive <name>
  ov conversation move <project|main>
  ov notify <text>                 send the researcher a message now
  ov status                        this conversation: project, what will wake you
  ov history search <text> [--project p|main] [--kind conversation|...] [--since 30d] [--limit N]
  ov history show <conversation> [--last 5] [--grep text]   read a past conversation's turns

Feedback about Overseer itself (what limited you; shown in the researcher's 🛠 feedback topic):
  ov feedback "<what limited you>" [--kind limitation|bug|idea] [--context "what you were doing"]
  ov feedback list [--all]         ov feedback +1 FB7
Skills (know-how for agents, installed once the researcher approves):
  ov skill propose skill.md [--all]   ov skill list | show <name>

Subagents (planner mode, off unless the researcher turned it on or approved your plan):
  ov propose plan.json   ov spawn <name> <task...> [--model M] [--backend B]   ov agents
  ov agent show|stop <name>   ov agent send <name> <text...>   ov planner off

Being woken (no permission needed; the researcher sees each one and can cancel it):
  ov wake in 90m ["what to do then"]           wake me once, later
  ov wake every 2h [--for 12h] ["..."]         wake me repeatedly (at least 15 minutes apart; 24h unless --for)
  ov watches                       what will wake me (ids like w12)
  ov unwatch <w12 | all>
"""


class ControlServer:
    def __init__(self, cfg: Config, agent: Agent, projects: Projects, notifier):
        self.cfg = cfg
        self.agent = agent
        self.store = agent.store
        self.projects = projects
        self.notifier = notifier
        self.tokens: dict[str, str] = {}  # token -> conversation
        self._bg: set[asyncio.Task] = set()  # card refreshes in flight
        self.server: asyncio.AbstractServer | None = None
        self._searcher = None  # search.Search, made on first use
        agent.ctl = self
        self.ext = agent.domain.ctl_commands(self)  # the domain's commands (help, register, dispatch, errors)
        self.help = getattr(self.ext, "help", None) or HELP
        self.errors = (CtlError, ProjectError, WatchError, SubagentError, FeedbackError, SkillError, ValueError,
                       *getattr(self.ext, "errors", ()))

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
            return True, self.help
        if self.ext is not None and hasattr(self.ext, "preprocess"):
            argv = self.ext.preprocess(argv)
        try:
            args = _parser(self.ext).parse_args(argv)
            area = args.area
            if self.ext is not None:
                answer = await self.ext.dispatch(session, args)
                if answer is not None:
                    return True, answer
            if area == "project" and args.action:
                return True, await getattr(self, f"project_{args.action}")(session, args)
            if area == "conversation" and args.action == "move":
                return True, await self.conversation_move(session, args)
            if area in ("unwatch", "watches", "wake"):
                return True, await getattr(self, area)(session, args)
            if area == "auto":
                return False, ("Autopilot no longer needs approval and has no separate mode: watch your jobs "
                               "(ov watch) and set wake-ups (ov wake) yourself. To finish, set no more wake-ups (ov "
                               "unwatch all) and say so; to ask the researcher, ask in your reply or ov notify.")
            if area == "feedback":
                return True, await self.feedback_cmd(session, args)
            if area == "skill" and args.action:
                return True, await getattr(self, f"skill_{args.action}")(session, args)
            if area in ("propose", "spawn", "agents", "planner"):
                return True, await getattr(self, area)(session, args)
            if area == "agent" and args.action:
                return True, await getattr(self, f"agent_{args.action}")(session, args)
            if area == "notify":
                return True, await self.notify(session, " ".join(args.text))
            if area == "status":
                return True, self.status(session)
            if area == "history" and args.action:
                return True, await getattr(self, f"history_{args.action}")(session, args)
            return False, self.help
        except self.errors as e:  # (SearchError is a ValueError)
            return False, str(e)

    # ---- past work (search.py)
    def _search(self):
        if self._searcher is None:
            from .search import Search
            self._searcher = Search(self.cfg, self.store)
        return self._searcher

    async def history_search(self, session, args) -> str:
        from .search import parse_since, render_hits
        text = " ".join(args.text)
        project = False
        if args.project:
            project = None if args.project.lower() in ("main", "none") else args.project.lower()
        hits, total = await self._search().asearch(text, project=project, kinds=tuple(args.kind or ()),
                                                   since=parse_since(args.since), limit=max(1, min(args.limit, 50)))
        return render_hits(hits, total, text)

    async def history_show(self, session, args) -> str:
        from .search import render_turns
        turns = await self._search().aconversation(args.conversation, args.last, args.grep)
        return render_turns(args.conversation, turns)

    # ---- the cluster (cluster.py)

    # ---- feedback about Overseer (feedback.py) and skills as addons (addons.py)
    async def feedback_cmd(self, session, args) -> str:
        fb = getattr(self.agent, "feedback", None)
        if not fb:
            raise CtlError("feedback isn't available in this Overseer")
        words = args.words
        if not words or words[0] == "list":
            items = fb.listing(None if args.all else ("open", "noted"))
            return "\n".join(fb.lines(items)) or "no open feedback yet"
        if words[0] in ("+1", "vote", "me-too"):
            ids = [w.upper().lstrip("FB") for w in words[1:]]
            if not ids or not all(i.isdigit() for i in ids):
                raise CtlError("which one? ov feedback +1 FB7")
            for i in ids:
                await fb.vote(session, int(i))
            return f"✓ +1 on {', '.join('FB' + i for i in ids)}"
        f = await fb.report(session, " ".join(words), args.kind, args.context)
        return (f"✓ FB{f.id} sent to the researcher's 🛠 feedback topic. You're told if they fix it or decide not "
                "to. Carry on with your task meanwhile: work around the limitation if you can.")

    def _addons(self):
        addons = getattr(self.agent, "addons", None)
        if not addons:
            raise CtlError("skills aren't available in this Overseer")
        return addons

    async def skill_propose(self, session, args) -> str:
        k = await self._addons().propose(session, args.text, everyone=args.all)
        return (f"✓ skill {k.name} proposed for {'every agent' if not k.project else 'project ' + k.project}: the "
                "researcher approves or declines it in Telegram. Agents get it from their next run once installed.")

    async def skill_list(self, session, args) -> str:
        installed = self._addons().for_project(session.project)
        pending = [k for k in self.store.skills(("proposed",)) if k.project in ("", session.project or "")]
        lines = [f"{k.name} ({'every agent' if not k.project else 'project ' + k.project}): {k.description}"
                 for k in installed] or ["no skills installed (besides overseer)"]
        lines += [f"proposed, waiting for the researcher: {k.name}" for k in pending]
        return "\n".join(lines)

    async def skill_show(self, session, args) -> str:
        for k in self._addons().for_project(session.project):
            if k.name == args.name:
                return k.content
        raise CtlError(f"no installed skill {args.name!r} (ov skill list)")

    # ---- results and findings (results.py)








    # ---- planner mode and subagents (subagents.py)
    def _subs(self):
        if not self.agent.subagents:
            raise CtlError("subagents aren't available in this Overseer")
        return self.agent.subagents

    async def propose(self, session, args) -> str:
        try:
            plan = json.loads(args.plan)
        except ValueError as e:
            raise CtlError(f"the plan isn't valid JSON ({e}). Format: {{\"goal\": ..., \"why\": ..., \"subagents\": "
                           "[{\"name\": ..., \"task\": ..., \"model\": ..., \"backend\": ...}]}")
        p = await self._subs().propose(session, plan)
        return (f"✓ plan {p['id']} sent to the researcher: {len(p['subagents'])} subagent(s), planner "
                f"{p['planner']}. They approve it with a button (the subagents then start as listed and you're woken "
                "with their reports) or decline it, possibly with a message for you. End your reply now and wait: "
                "set no wake-ups for this.")

    async def spawn(self, session, args) -> str:
        rec = await self._subs().spawn(session, args.name, " ".join(args.task), args.backend, args.model)
        where = "started" if rec.state == "running" else "queued (it starts when one of the running ones is done)"
        return (f"✓ subagent {rec.name} ({rec.model}, {rec.agent_backend}) {where}. It has its own thread the "
                "researcher can follow. You're woken with its report when it's done (with the others, if several "
                "are running).")

    async def agents(self, session, args) -> str:
        subs = self._subs()
        if session.rec.is_worker:
            return (f"You are subagent {session.rec.name} ({session.rec.model}); your planner reads your final reply. "
                    "Subagents can't start subagents.")
        mode = "on" if self.agent.planner_on(session) else "off"
        recs = self.store.subagents(parent=session.id)
        lines = [f"planner mode: {mode}" + (f" ({self.agent.session_runner(session).model})" if mode == "on" else ""),
                 f"running: {subs.running()} of {subs.c.max_running} at once (all conversations)"]
        lines += [f"  {x}" for x in subs.describe(recs)] or ["  (no subagents in this conversation)"]
        return "\n".join(lines)

    async def agent_show(self, session, args) -> str:
        w = self._subs().find(session, args.name)
        r = w.rec
        report = r.report if r.state in ENDED else "(not done yet)"
        return (f"{r.name}: {STATE_LABEL.get(r.state, r.state)} · {model_label(r.model)} ({r.agent_backend})\n"
                f"notes: {self._subs().notes_path(r)}\ntask:\n{r.task}\n\nreport:\n{report}")

    async def agent_stop(self, session, args) -> str:
        w = self._subs().find(session, args.name)
        if w.rec.state in ENDED:
            return f"{args.name} has already ended ({w.rec.state})"
        await self._subs().stop(w, by="planner")
        return f"✓ stopped {args.name} (its run, its wake-ups or its place in the queue)"

    async def agent_send(self, session, args) -> str:
        w = self._subs().find(session, args.name)
        await self._subs().send(w, " ".join(args.text))
        return (f"✓ sent to {args.name}" + (" (it reads it when it starts)" if w.rec.state == "queued" else
                                             ": it works on it and reports again when done"))

    async def planner(self, session, args) -> str:
        return "✓ " + self._subs().planner_off(session)

    async def _announce(self, session, text: str, **kw) -> None:
        await self.notifier.send(Outgoing(text, session_id=session.id if session else None, silent=True, **kw))

    # ---- jobs






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
            if hasattr(self.notifier, "ensure_jobs_topics"):
                await self.notifier.ensure_jobs_topics([p.name])
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

    # ---- what matters in a job: its card's metrics, and alerts






    # ---- graphs (plots.py)






    # ---- other
    async def notify(self, session, text: str) -> str:
        await self.notifier.send(Outgoing(f"📣 {esc(text[:3500])}", session_id=session.id))
        return "✓ sent"

    def status(self, session) -> str:
        lines = [f"conversation {session.id}, project: {session.project or 'none'}"]
        if self.ext is not None and hasattr(self.ext, "status_lines"):
            lines += self.ext.status_lines(session)  # e.g. the conversation's running jobs
        if self.agent.autopilot:
            lines.append("what will wake you:\n" + self.agent.autopilot.watch_lines(session.id))
        if self.agent.subagents:
            if session.rec.is_worker:
                lines.append(f"you are subagent {session.rec.name} ({session.rec.model}); your final reply is your "
                             "report to your planner")
            else:
                on = self.agent.planner_on(session)
                lines.append(f"planner mode: {'on' if on else 'off'}")
                recs = self.store.subagents(parent=session.id)
                if recs:
                    lines += ["subagents:"] + [f"  {x}" for x in self.agent.subagents.describe(recs)]
        return "\n".join(lines)

    # ---- being woken
    def _ap(self):
        if not self.agent.autopilot:
            raise CtlError("wake-ups aren't available in this Overseer")
        return self.agent.autopilot


    def _duration(self, text: str, what: str) -> float:
        d = parse_duration(text)
        if d is None or d <= 0:
            raise CtlError(f"{what}: {text!r} isn't a duration (e.g. 30m, 2h, 1d)")
        return d



    async def _announce_watch(self, session, w, what: str) -> None:
        note = f"\n{esc(w.note)}" if w.note else ""
        await self.notifier.send(Outgoing(f"⏰ The agent will {esc(what)}.{note}",
                                          session_id=session.id, silent=True,
                                          buttons=[[("✖ Cancel", f"uw:{w.id}")]]))

    async def wake(self, session, args) -> str:
        ap = self._ap()
        d = self._duration(args.duration, "wake")
        hours = self._duration(args.for_, "--for") / 3600 if args.for_ else None
        note = " ".join(args.note)
        w, remark = ap.add_timer(session, d, d if args.mode == "every" else 0.0, note, hours)
        if w.every:
            span = human_interval(w.until - time.time())
            what = f"look again every {human_interval(w.every)} for {span}"
            reply = f"✓ w{w.id}: you'll be woken every {human_interval(w.every)} for {span}{remark}"
        else:
            what = f"look again in {human_interval(d)}"
            reply = f"✓ w{w.id}: you'll be woken in {human_interval(d)}"
        await self._announce_watch(session, w, what)
        return reply + ". The researcher was told and can cancel it."

    async def watches(self, session, args) -> str:
        return self._ap().watch_lines(session.id)

    async def unwatch(self, session, args) -> str:
        changed = self._ap().unwatch(session.id, args.ref.strip())
        if not changed:
            raise CtlError(f"nothing matches {args.ref!r} (ov watches lists them)")
        return "✓ updated: " + ", ".join(f"w{w.id} ({'removed' if w.state == 'cancelled' else 'now ' + w.jobs})"
                                          for w in changed)


# ============================================================================ the agent's `ov` command

OV_CLIENT = r'''#!@PYTHON@ -I
# ov - talk to Overseer (see `ov help`).
import json, os, socket, sys

SOCK = @SOCK@
token = os.environ.get("OVERSEER_CTL", "")
# Folders are on the HPC: an unquoted ~/x was expanded by the local shell to this machine's home - undo that.
home = os.environ.get("HOME", "").rstrip("/")
argv = [("~" + a[len(home):]) if home and (a == home or a.startswith(home + "/")) else a for a in sys.argv[1:]]
# A plan (ov propose plan.json) and long texts ("-": read from stdin) are read here, with the agent's own access.
if argv[:1] == ["propose"] and len(argv) == 2 and not argv[1].lstrip().startswith("{"):
    try:
        argv[1] = sys.stdin.read() if argv[1] == "-" else open(os.path.expanduser(sys.argv[2])).read()
    except OSError as e:
        sys.stderr.write("ov: can't read the plan: %s\n" % e)
        sys.exit(1)
elif argv[:2] == ["skill", "propose"] and len(argv) >= 3 and not argv[2].lstrip().startswith("---"):
    try:
        argv[2] = sys.stdin.read() if argv[2] == "-" else open(os.path.expanduser(sys.argv[3])).read()
    except OSError as e:
        sys.stderr.write("ov: can't read the skill: %s\n" % e)
        sys.exit(1)
elif "-" in argv[1:] and argv[:1] in (["spawn"], ["agent"]):
    argv[argv.index("-")] = sys.stdin.read()
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


def install_skill(skills_dir: Path, name: str, text: str) -> None:
    """Put a skill (e.g. the domain's `overseer`: how to use ov) into an agent CLI's skills folder."""
    d = skills_dir / name
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not (d / "SKILL.md").exists() or (d / "SKILL.md").read_text() != text:
        (d / "SKILL.md").write_text(text)


def write_skills(cfg: Config, skills: dict[str, str]) -> None:
    """Claude Code's copy of the domain's skills, in CLAUDE_CONFIG_DIR/skills (Claude Code reads skills itself).
    Codex has the model read a skill's file with a command, and its sandbox can't read Overseer's state
    dir: its copy goes into each scratch dir, as .agents/skills (CodexRunner)."""
    for name, text in skills.items():
        install_skill(cfg.local.resolve(cfg.agent.claude_code.home) / "skills", name, text)
