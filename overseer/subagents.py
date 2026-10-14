"""Planner mode: a conversation plans and delegates to subagents, each a conversation of its own.

Off unless the researcher turns it on (/subagents on, for a project) or approves an agent's proposal (`ov propose`,
for that conversation). When on, the conversation (the planner) runs on the planner model of its backend and starts
subagents with `ov spawn`: each runs on a cheaper worker model (either backend), in its own Telegram thread, with the
same sandbox and `ov`. A subagent can submit and watch jobs and wake itself; once it's done (it answered and has no
wake-ups left), its final reply is its report. The planner is woken when all the subagents it's waiting for are done
(or as soon as one fails or is stopped), with their reports. At most `max_running` subagents run at once; more queue.

Proposals: the agent sends a plan (goal, planner model, subagents and their tasks). The researcher approves it with a
button (then planner mode goes on for that conversation and the listed subagents start, exactly as approved), or
declines it - with the button, or by writing to the agent, which declines the plan and passes on what they wrote.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
import uuid

from .autopilot import human_interval
from .config import BACKENDS, Config
from .notify import Notifier, Outgoing, code, esc
from .results import context as results_context
from .store import SessionRecord, Store

log = logging.getLogger(__name__)

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
ACTIVE = ("queued", "running", "waiting")
ENDED = ("done", "failed", "stopped")
MAX_PLAN = 10  # subagents in one proposal
PROPOSAL_KEY = "proposal:"  # kv, by conversation: its latest plan
PROPOSAL_ID_KEY = "proposal_id:"  # kv: plan id -> conversation

STATE_LABEL = {"queued": "⏳ queued", "running": "▶️ running", "waiting": "⏰ waiting for its jobs",
               "done": "✅ done", "failed": "⚠️ failed", "stopped": "⏹ stopped"}

WORKER_BRIEF = """[Task from your planner - you are subagent "{name}"]
{task}

Work on it on your own: you can submit and watch jobs and wake yourself up, like any conversation. Keep your detailed \
notes (commands, intermediate results, dead ends) in {notes}. When the task is done, reply with a short report for \
your planner: what you did, the key numbers, where the results are (at most ~15 lines). That reply ends your task."""

REPORT_HEADER = "[Subagents you started have finished]"


class SubagentError(ValueError):
    pass


def model_label(model: str) -> str:
    """claude-sonnet-5-5 -> Sonnet 5.5; gpt-6.1-sol -> gpt-6.1-sol."""
    m = re.match(r"^claude-([a-z]+)-(\d+)(?:-(\d+))?$", model or "")
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)}" + (f".{m.group(3)}" if m.group(3) else "")
    return model or "default model"


class Subagents:
    def __init__(self, cfg: Config, store: Store, agent, notifier: Notifier):
        self.cfg = cfg
        self.store = store
        self.agent = agent
        self.notifier = notifier
        self._tasks: set[asyncio.Task] = set()
        agent.subagents = self

    @property
    def c(self):
        return self.cfg.agent.subagents

    def _spawn_task(self, coro) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self) -> None:
        """Wait for subagent runs and planner wake-ups in flight (tests)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------------ checks
    def _check(self, planner, name: str, task: str, backend: str | None, model: str | None,
               taken: set[str]) -> tuple[str, str]:
        """(backend, model) for a subagent, or SubagentError saying what's wrong."""
        if not NAME_RE.match(name or ""):
            raise SubagentError(f"subagent name {name!r}: use lowercase letters, digits, - and _ (e.g. cosine-lr)")
        if name in taken:
            raise SubagentError(f"there's already a subagent named {name!r} in this conversation: pick another name")
        if not (task or "").strip():
            raise SubagentError(f"subagent {name!r} needs a task")
        backend = backend or self.agent.session_backend(planner)
        if backend not in BACKENDS:
            raise SubagentError(f"backend must be one of {', '.join(BACKENDS)}")
        models = self.agent.worker_models(backend, planner.project)
        if not models:
            raise SubagentError(f"no worker models are set for {backend}")
        model = model or models[0]
        if model not in models:
            raise SubagentError(f"{model!r} isn't a subagent model for {backend}: use {', '.join(models)}")
        return backend, model

    def _names(self, planner) -> set[str]:
        return {r.name for r in self.store.subagents(parent=planner.id)}

    # ------------------------------------------------------------------ subagents
    async def spawn(self, planner, name: str, task: str, backend: str | None = None, model: str | None = None,
                    check_mode: bool = True) -> SessionRecord:
        if planner.rec.is_worker:
            raise SubagentError("subagents can't start subagents of their own: ask your planner in your report")
        if check_mode and not self.agent.planner_on(planner):
            raise SubagentError("planner mode is off in this conversation. Propose it to the researcher with "
                                "`ov propose` (they approve or decline it), or they turn it on with /subagents on.")
        backend, model = self._check(planner, name, task, backend, model, self._names(planner))
        rec = SessionRecord(id=f"{planner.project or 'chat'}-{name}-{uuid.uuid4().hex[:4]}", created=time.time(),
                            project=planner.project, parent=planner.id, role="worker", name=name,
                            agent_backend=backend, model=model, state="queued", task=task.strip())
        self.store.upsert_session(rec)
        w = self.agent.get(rec.id)
        if hasattr(self.notifier, "open_thread"):  # its own thread: you can read along, steer it, stop it
            await self.notifier.open_thread(rec.id, self.thread_name(rec))
        await self.notifier.send(Outgoing(
            f"🤖 <b>Subagent {esc(name)}</b> · {esc(model_label(model))} ({esc(backend)}) · started by "
            f"{self._planner_label(planner)}<blockquote expandable>{esc(rec.task[:3000])}</blockquote>",
            session_id=rec.id, silent=True))
        self.pump()
        return w.rec

    def thread_name(self, rec: SessionRecord) -> str:
        return (f"🤖 {rec.project} · {rec.name}" if rec.project else f"🤖 {rec.name}")[:128]

    def _planner_label(self, planner) -> str:
        return f"the {esc(planner.project)} planner" if planner.project else "the planner"

    def running(self) -> int:
        return len(self.store.subagents(states=("running",)))

    def pump(self) -> None:
        """Start queued subagents while fewer than max_running are running."""
        free = self.c.max_running - self.running()
        for rec in self.store.subagents(states=("queued",)):
            if free <= 0:
                break
            w = self.agent.get(rec.id)
            if not w or w.rec.state != "queued":
                continue
            w.rec.state, w.rec.started = "running", time.time()
            self.store.upsert_session(w.rec)
            free -= 1
            self._spawn_task(self._run_brief(w))

    async def _run_brief(self, w) -> None:
        notes = self.notes_path(w.rec)
        await self.agent.handle(w, WORKER_BRIEF.format(name=w.rec.name, task=w.rec.task, notes=notes),
                                origin="subagent")

    def notes_path(self, rec: SessionRecord) -> str:
        proj = self.store.get_project(rec.project) if rec.project else None
        base = (proj.dir.rstrip("/") if proj and proj.dir else "~/overseer-subagents")
        return f"{base}/subagents/{rec.name}/NOTES.md"

    def turn_ended(self, w) -> None:
        """After any run of a subagent: still waiting for its jobs, or done (its planner hears)."""
        prev = w.rec.state
        if prev == "stopped" and w.last_status in ("stopped", "error"):
            return  # stopped by its planner or the researcher: already handled
        if w.last_status == "stopped":
            state = "stopped"
        elif w.last_status == "error":
            state = "failed"
        elif self.store.watches(w.id):
            state = "waiting"
        else:
            state = "done"
        if prev in ENDED and state in ENDED:  # the researcher chatted with a finished subagent: nothing changes
            return
        w.rec.state = state
        if state in ENDED:
            w.rec.report = (w.last_reply or "")[:20000]
            w.rec.finished, w.rec.delivered = time.time(), 0
        self.store.upsert_session(w.rec)
        if state in ENDED:
            self._spawn_task(self._ended(w))
        self.pump()

    async def _ended(self, w) -> None:
        rec = w.rec
        took = human_interval(max(60.0, rec.finished - (rec.started or rec.created)))
        cost = self.store.session_cost(rec.id)
        word = {"done": "finished", "failed": "failed", "stopped": "was stopped"}[rec.state]
        await self.notifier.send(Outgoing(
            f"{STATE_LABEL[rec.state].split()[0]} <b>{esc(rec.name)}</b> {word} · {took} · "
            f"{esc(model_label(rec.model))}" + (f" · ${cost:.2f}" if cost else ""),
            session_id=rec.id, silent=rec.state == "done",
            buttons=[[("🗑 Delete thread", f"subdel:{rec.id}"), ("Keep", f"subkeep:{rec.id}")]]))
        await self.wake_planner(rec.parent)

    async def wake_planner(self, planner_id: str | None) -> None:
        """Give the planner the reports it hasn't seen, once all its subagents are done (or one failed/stopped)."""
        planner = self.agent.get(planner_id)
        if planner is None:
            return
        subs = self.store.subagents(parent=planner_id)
        new = [r for r in subs if r.state in ENDED and not r.delivered]
        active = [r for r in subs if r.state in ACTIVE]
        if not new or (active and not any(r.state != "done" for r in new)):
            return
        for r in new:
            w = self.agent.get(r.id)
            w.rec.delivered = 1
            self.store.upsert_session(w.rec)
        prompt = self.report_prompt(new, active)
        ap = self.agent.autopilot
        if ap and ap.paused(planner_id):  # its wake-ups are paused: it reads them when the researcher writes
            self.agent.add_note(planner_id, prompt)
            return
        await self.agent.handle(planner, prompt, origin="wakeup")

    def report_prompt(self, ended: list[SessionRecord], active: list[SessionRecord]) -> str:
        parts = [REPORT_HEADER]
        for r in ended:
            took = human_interval(max(60.0, r.finished - (r.started or r.created)))
            parts.append(f"<subagent name='{r.name}' model='{r.model}' state='{r.state}' took='{took}' "
                         f"notes='{self.notes_path(r)}'>\n{(r.report or '(no report)').strip()}\n</subagent>")
        if active:
            parts.append("Still working: " + ", ".join(f"{r.name} ({r.state})" for r in active)
                         + ". You'll be woken when they're done.")
        if ended:
            ctx = results_context(self.store, ended[0].project)
            if ctx:
                parts.append(ctx)
        parts.append("Record the results in the research log (their rows: `ov results`, and give each a verdict if its "
                     "subagent didn't), then decide: start more subagents, end planner mode (`ov planner off`) when the "
                     "goal is reached, or ask the researcher.")
        return "\n\n".join(parts)

    async def stop(self, w, by: str = "planner") -> None:
        """Stop a subagent (its run, its wake-ups, or its place in the queue)."""
        prev = w.rec.state
        w.rec.state, w.rec.finished = "stopped", time.time()
        w.rec.report = w.rec.report or f"(stopped by the {by} before it reported)"
        w.rec.delivered = 1 if by == "planner" else 0  # the planner knows why it stopped it
        self.store.upsert_session(w.rec)
        if self.agent.autopilot:
            self.agent.autopilot.cancel_all([w.id])
        await self.agent.stop(w.id)
        if prev in ACTIVE:
            await self._ended(w)
        self.pump()

    async def send(self, w, text: str) -> None:
        """A message from the planner: the subagent works on it (and reports again when done)."""
        if w.rec.state == "queued":
            w.rec.task += f"\n\nAlso, from your planner: {text}"
            self.store.upsert_session(w.rec)
            return
        if w.rec.state in ENDED:
            w.rec.state, w.rec.started = "running", time.time()
            self.store.upsert_session(w.rec)
        self._spawn_task(self.agent.handle(w, f"[Message from your planner]\n{text}", origin="subagent"))

    def find(self, planner, name: str):
        for r in self.store.subagents(parent=planner.id):
            if r.name == name:
                return self.agent.get(r.id)
        raise SubagentError(f"no subagent named {name!r} in this conversation (ov agents lists them)")

    def describe(self, recs: list[SessionRecord]) -> list[str]:
        out = []
        for r in recs:
            cost = self.store.session_cost(r.id)
            out.append(f"{r.name}: {STATE_LABEL.get(r.state, r.state)} · {model_label(r.model)} ({r.agent_backend})"
                       + (f" · ${cost:.2f}" if cost else ""))
        return out

    def planner_off(self, s) -> str:
        if self._project_on(s):
            return ("planner mode is on for the whole " + (f"project {s.project}" if s.project else "main chat")
                    + " (/subagents on): the researcher turns it off with /subagents off")
        s.rec.planner = 0
        self.store.upsert_session(s.rec)
        return "planner mode is off: your next runs use the conversation's usual model. Running subagents finish."

    def _project_on(self, s) -> bool:
        return self.agent._setting("subagents", s.project) == "on"

    # ------------------------------------------------------------------ proposals
    def validate_plan(self, s, plan: dict) -> dict:
        if s.rec.is_worker:
            raise SubagentError("subagents can't propose planner mode")
        if self.agent.planner_on(s):
            raise SubagentError("planner mode is already on here: start subagents with `ov spawn`")
        if not isinstance(plan, dict):
            raise SubagentError("the plan must be a JSON object: {goal, planner?, subagents: [{name, task, model?, "
                                "backend?}]}")
        goal = str(plan.get("goal") or "").strip()
        if not goal:
            raise SubagentError("the plan needs a goal")
        backend = self.agent.session_backend(s)
        planner = self.agent.planner_model(backend, s.project)
        if plan.get("planner") and plan["planner"] != planner:
            raise SubagentError(f"the planner here is {planner} (this conversation runs on {backend}, which it keeps)")
        items = plan.get("subagents")
        if not isinstance(items, list) or not items:
            raise SubagentError("the plan needs at least one subagent")
        if len(items) > MAX_PLAN:
            raise SubagentError(f"at most {MAX_PLAN} subagents in a plan")
        taken = self._names(s)
        subs = []
        for it in items:
            if not isinstance(it, dict):
                raise SubagentError("each subagent is an object: {name, task, model?, backend?}")
            b, m = self._check(s, str(it.get("name") or ""), str(it.get("task") or ""), it.get("backend"),
                               it.get("model"), taken)
            taken.add(it["name"])
            subs.append({"name": it["name"], "task": str(it["task"]).strip(), "backend": b, "model": m})
        return {"goal": goal, "planner": planner, "backend": backend, "subagents": subs,
                "why": str(plan.get("why") or "").strip()}

    def card_text(self, p: dict) -> str:
        lines = ["🧭 <b>Plan: switch this conversation to planner mode?</b>", "",
                 f"<b>Goal:</b> {esc(p['goal'])}"]
        if p.get("why"):
            lines.append(f"<b>Why subagents:</b> {esc(p['why'])}")
        lines.append(f"<b>Planner:</b> {esc(model_label(p['planner']))} (this conversation)")
        lines.append(f"<b>Subagents ({len(p['subagents'])}):</b>")
        for it in p["subagents"]:
            lines.append(f" • <b>{esc(it['name'])}</b> · {esc(model_label(it['model']))}"
                         + (f" ({esc(it['backend'])})" if it["backend"] != p["backend"] else "")
                         + f" · {esc(it['task'][:300])}")
        if len(p["subagents"]) > self.c.max_running:
            lines.append(f"({self.c.max_running} run at a time; the rest wait their turn)")
        lines += ["", "Reply to change the plan."]
        return "\n".join(lines)

    async def propose(self, s, plan: dict) -> dict:
        p = self.validate_plan(s, plan)
        old = self.pending(s.id)
        if old:  # only the latest plan can be approved
            old["state"] = "superseded"
            self._save(s.id, old)
            await self._edit_card(old, "⏭ <i>Replaced by a newer plan.</i>")
        p.update(id=secrets.token_hex(4), session=s.id, state="pending", created=time.time())
        self.store.set(PROPOSAL_ID_KEY + p["id"], s.id)
        msg = Outgoing(self.card_text(p), session_id=s.id,
                       buttons=[[("✅ Approve", f"plan:{p['id']}"), ("✖ No", f"planx:{p['id']}")]])
        p["refs"] = (await self.notifier.post(msg)) if hasattr(self.notifier, "post") else {}
        if not hasattr(self.notifier, "post"):
            await self.notifier.send(msg)
        self._save(s.id, p)
        return p

    def _save(self, sid: str, p: dict) -> None:
        self.store.set(PROPOSAL_KEY + sid, json.dumps(p))

    def pending(self, sid: str) -> dict | None:
        try:
            p = json.loads(self.store.get(PROPOSAL_KEY + sid) or "null")
        except ValueError:
            return None
        return p if p and p.get("state") == "pending" else None

    def proposal(self, pid: str) -> dict | None:
        sid = self.store.get(PROPOSAL_ID_KEY + pid)
        try:
            p = json.loads(self.store.get(PROPOSAL_KEY + sid) or "null") if sid else None
        except ValueError:
            return None
        return p if p and p.get("id") == pid else None

    async def _edit_card(self, p: dict, footer: str) -> None:
        if p.get("refs") and hasattr(self.notifier, "edit"):
            body = self.card_text(p).rsplit("\n\n", 1)[0]  # without "Reply to change the plan."
            await self.notifier.edit(p["refs"], f"{body}\n\n{footer}")

    async def approve(self, pid: str) -> str:
        """✅ on a plan: planner mode on for that conversation, and the subagents start as listed."""
        p = self.proposal(pid)
        if not p or p.get("state") != "pending":
            return "This plan was already answered or replaced."
        s = self.agent.get(p["session"])
        if s is None:
            return "This conversation no longer exists."
        p["state"] = "approved"
        self._save(s.id, p)
        s.rec.planner = 1
        self.store.upsert_session(s.rec)
        started, failed = [], []
        for it in p["subagents"]:
            try:
                await self.spawn(s, it["name"], it["task"], it["backend"], it["model"])
                started.append(it["name"])
            except SubagentError as e:
                failed.append(f"{it['name']}: {e}")
        self.agent.add_note(s.id, (
            f"The researcher approved your plan (goal: {p['goal']}). Planner mode is on: you run on "
            f"{p['planner']} now. Started subagents: {', '.join(started) or 'none'}"
            + (f"; couldn't start: {'; '.join(failed)}" if failed else "")
            + ". You're woken with their reports when they're done."))
        queued = [r.name for r in self.store.subagents(parent=s.id, states=("queued",))]
        summary = (f"✅ <b>Approved</b>: planner mode is on ({esc(model_label(p['planner']))}). Started: "
                   + ", ".join(esc(n) for n in started if n not in queued)
                   + (f"; queued: {', '.join(esc(n) for n in queued)}" if queued else "")
                   + (f"\n⚠️ Not started: {esc('; '.join(failed))}" if failed else ""))
        await self._edit_card(p, summary)
        return summary

    async def decline(self, pid: str, reply: str | None = None) -> str:
        """✖ on a plan, or a message written while it's open (`reply`): the agent is told, nothing starts."""
        p = self.proposal(pid)
        if not p or p.get("state") != "pending":
            return "This plan was already answered or replaced."
        p["state"] = "declined"
        self._save(p["session"], p)
        if reply is None:
            self.agent.add_note(p["session"], f"The researcher declined your plan (goal: {p['goal']}) without "
                                              "comment. Carry on in normal mode.")
            text = "✖ <b>Declined.</b>"
        else:
            short = " ".join(reply.split())
            text = f"✖ <b>Declined</b>: you replied “{esc(short[:200] + ('…' if len(short) > 200 else ''))}”"
        await self._edit_card(p, text)
        return text

    async def reject_by_reply(self, s, text: str) -> str:
        """A message in a conversation with an open plan declines it: what the agent is given instead."""
        p = self.pending(s.id)
        if not p:
            return text
        await self.decline(p["id"], reply=text)
        return f"[The researcher declined your proposed plan (goal: {p['goal']}) and replied:]\n{text}"

    # ------------------------------------------------------------------ startup
    def resume(self) -> None:
        """At startup: queued subagents get their turn."""
        self.pump()
