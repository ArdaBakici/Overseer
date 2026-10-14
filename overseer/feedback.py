"""Agents' feedback about Overseer itself, and a review that turns it into designs.

An agent that runs into a limitation of Overseer (a missing `ov` command, information it should have had, the sandbox
blocking something legitimate, confusing behaviour), a bug, or has an idea, reports it: `ov feedback "..."`. Each item
is a card in the 🛠 feedback topic; other agents that hit the same thing add a +1 instead of a duplicate. The
researcher marks items noted, fixed or won't fix (the agents that reported them are told).

`/feedback review` starts the reviewer: a conversation on the planner model, bound to the feedback topic (the
researcher talks to it there). It reads the open feedback and Overseer's source (it can't change it: its sandbox only
writes its scratch folder), groups and ranks the items, and writes a design - and a patch when one is clear - that it
sends as files. Applying them is the researcher's call: Overseer holds the SSH key and starts the sandboxes, so its own
code is never changed by an agent. Know-how gaps become skill proposals instead (addons.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import Config
from .notify import Notifier, Outgoing, esc
from .store import FeedbackRecord, SessionRecord, Store

KINDS = {"limitation": "🧱", "bug": "🐞", "idea": "💡"}
STATUS = {"open": "", "noted": "👀 noted", "fixed": "✅ fixed", "wontfix": "✖ won't fix"}
PER_DAY = 3  # new items per conversation per day (+1s are unlimited)
SOURCE = Path(__file__).resolve().parent.parent  # Overseer's source (README, overseer/, tests/)
REVIEWER_KEY = "feedback_reviewer"

REVIEW_PROMPT = """[Feedback review]{focus}
Agents reported these about Overseer (open ones, most +1s first):

{items}

1. Group the items that are the same problem, and rank the groups by how much they'd help (how often agents hit them,
   how much work they block, how hard a fix is).
2. For the top groups, read the relevant parts of Overseer's source in {src} (read-only: you can't change it) and write
   a design: the problem, the proposed change (files, functions, new `ov` commands or settings), how to test it, and
   the risks. Put it in ./outbox/DESIGN-{date}.md - it's sent to the researcher with your reply.
3. When a change is clear and contained, also write it as a patch: copy the source - only the code, not its state or
   secrets - into your scratch folder (`mkdir -p a b && cp -r {src}/overseer {src}/tests {src}/README.md
   {src}/pyproject.toml a/ && cp -r a/. b/`), edit the copy in b/, and save `diff -ruN a b` as
   ./outbox/<short-name>.patch (ready for `git apply -p1`). Run its tests if you can (`cd b && python -m pytest -q
   tests/test_integration.py`).
4. When an item is really missing know-how (how to do something on this cluster or with these tools) rather than a
   missing feature, propose a skill instead (`ov skill propose`).
Never propose weakening the sandbox, exposing the SSH key or tokens, or letting agents change Overseer itself; say
when a change touches security. Reply with a short summary: the groups, what you'd do first, and what you sent."""


class FeedbackError(ValueError):
    pass


class Feedback:
    def __init__(self, cfg: Config, store: Store, agent, notifier: Notifier):
        self.cfg = cfg
        self.store = store
        self.agent = agent
        self.notifier = notifier
        agent.feedback = self

    # ------------------------------------------------------------------ reporting
    async def report(self, s, text: str, kind: str = "limitation", context: str = "") -> FeedbackRecord:
        if kind not in KINDS:
            raise FeedbackError(f"--kind must be one of: {', '.join(KINDS)}")
        text = " ".join((text or "").split())
        if len(text) < 10:
            raise FeedbackError("say what limited you, in a sentence or two")
        since = time.time() - 86400
        mine = [f for f in self.store.feedback() if f.session_id == s.id and f.created >= since]
        if len(mine) >= PER_DAY:
            raise FeedbackError(f"this conversation already reported {PER_DAY} items today: add a +1 to an existing "
                                "one (`ov feedback list`) or wait")
        runner = self.agent.session_runner(s)
        f = self.store.add_feedback(FeedbackRecord(
            kind=kind, text=text[:2000], context=" ".join((context or "").split())[:1000], session_id=s.id,
            project=s.project, backend=self.agent.session_backend(s), model=runner.model or ""))
        await self._post(f)
        return f

    async def vote(self, s, fid: int) -> FeedbackRecord:
        f = self.store.get_feedback(fid)
        if not f:
            raise FeedbackError(f"no feedback FB{fid} (ov feedback list)")
        votes = f.votes
        if s.id != f.session_id and s.id not in votes:
            f.voters = json.dumps(votes + [s.id])
            self.store.save_feedback(f)
            await self._update_card(f)
        return f

    def card_text(self, f: FeedbackRecord) -> str:
        who = " · ".join(x for x in [f.project or "main", f"{f.backend} {f.model}".strip()] if x)
        status = STATUS.get(f.status, f.status)
        lines = [f"{KINDS.get(f.kind, '•')} <b>FB{f.id}</b> · {esc(f.kind)} · {esc(who)}"
                 + (f" · <b>{esc(status)}</b>" if status else ""),
                 esc(f.text)]
        if f.context:
            lines.append(f"<i>while: {esc(f.context)}</i>")
        if f.votes:
            lines.append(f"➕ {len(f.votes)} other conversation(s) ran into this too")
        return "\n".join(lines)

    def buttons(self, f: FeedbackRecord) -> list[list[tuple[str, str]]]:
        row = [(label, f"fb:{f.id}:{st}") for st, label in (("noted", "👀 Noted"), ("fixed", "✅ Fixed"),
                                                             ("wontfix", "✖ Won't fix")) if st != f.status]
        return [row]

    async def _post(self, f: FeedbackRecord) -> None:
        if hasattr(self.notifier, "ensure_feedback_topic"):
            await self.notifier.ensure_feedback_topic()
        msg = Outgoing(self.card_text(f), topic="feedback", silent=True, buttons=self.buttons(f))
        if hasattr(self.notifier, "post"):
            f.refs = json.dumps(await self.notifier.post(msg))
            self.store.save_feedback(f)
        else:
            await self.notifier.send(msg)

    async def _update_card(self, f: FeedbackRecord) -> None:
        refs = json.loads(f.refs or "{}")
        if refs and hasattr(self.notifier, "edit"):
            await self.notifier.edit(refs, self.card_text(f), self.buttons(f))

    async def set_status(self, fid: int, status: str) -> FeedbackRecord | None:
        """The researcher's answer: noted, fixed or won't fix. The agents that reported it are told."""
        f = self.store.get_feedback(fid)
        if not f or status not in STATUS:
            return None
        f.status = status
        self.store.save_feedback(f)
        await self._update_card(f)
        if status in ("fixed", "wontfix"):
            word = "fixed it" if status == "fixed" else "decided not to change this"
            for sid in dict.fromkeys([f.session_id, *f.votes]):
                if sid and self.store.get_session(sid):
                    self.agent.add_note(sid, f"About your feedback FB{f.id} ({f.text[:120]}): the researcher {word}.")
        return f

    def listing(self, statuses: tuple[str, ...] = ("open", "noted")) -> list[FeedbackRecord]:
        return sorted(self.store.feedback(statuses), key=lambda f: (-len(f.votes), -(f.id or 0)))

    def lines(self, items: list[FeedbackRecord], html: bool = False) -> list[str]:
        e = esc if html else (lambda x: x)
        out = []
        for f in items:
            extra = [f"+{len(f.votes)}" if f.votes else "", STATUS.get(f.status, "")]
            out.append(f"{KINDS.get(f.kind, '•')} FB{f.id} {f.kind}" + "".join(f" · {x}" for x in extra if x)
                       + f": {e(f.text)}" + (f" (while: {e(f.context)})" if f.context else ""))
        return out

    # ------------------------------------------------------------------ the review
    def reviewer(self):
        """The reviewer's conversation (on the planner model), in the feedback topic: made once, kept."""
        s = self.agent.get(self.store.get(REVIEWER_KEY))
        if s is None:
            rec = SessionRecord(id=f"feedback-review-{int(time.time())}", created=time.time(), role="reviewer")
            self.store.upsert_session(rec)
            self.store.set(REVIEWER_KEY, rec.id)
            s = self.agent.get(rec.id)
        if hasattr(self.notifier, "feedback_thread"):
            for chat_id in getattr(self.notifier, "chat_ids", []):
                thread = self.notifier.feedback_thread(chat_id)
                if thread and self.store.session_thread(s.id, chat_id) != thread:
                    self.store.bind_thread(chat_id, thread, s.id)
        return s

    def review_prompt(self, focus: str = "") -> str:
        items = self.listing()
        body = "\n".join(self.lines(items)) or "(no open feedback)"
        return REVIEW_PROMPT.format(focus=f"\nFocus on: {focus}" if focus else "", items=body, src=SOURCE,
                                    date=time.strftime("%Y-%m-%d"))
