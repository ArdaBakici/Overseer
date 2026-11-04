"""/pending: everything the bot's agents asked of you and you haven't answered yet, in one place.

Each kind of request registers how to list, approve and decline its own (a `Kind`): handbook changes, skills, planner
plans, questions (`ov ask`) - and the domain's, e.g. the assistant's calendar changes. /pending lists them all with ✅ /
✖ (a question: its choices), per kind and for everything at once (after a confirmation that says what's in it); an
answer here is the same as on the request's own card, which is updated too. Agents can list them (`ov pending`) and send
you the list (`ov pending --send`).

What waits EXPIRE_DAYS days unanswered is declined (a question: dropped), and its agent is told.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from .notify import Outgoing, esc

log = logging.getLogger(__name__)

EXPIRE_DAYS = 14
SHOWN = 15  # requests with buttons in the list (Telegram allows ~100 buttons; more is unreadable on a phone)


@dataclass
class Item:
    kind: str
    id: str
    title: str  # what's asked, in a line (plain text)
    session: str | None = None  # the conversation that asked
    at: float = 0.0  # when
    choices: list[str] = field(default_factory=list)  # a question's answers (no ✅ / ✖, never in bulk)


class Kind:
    key = ""  # in the buttons' data: short
    emoji = ""
    title = ""  # e.g. "Handbook changes"
    noun = ""  # e.g. "handbook change"

    def items(self) -> list[Item]:
        return []

    async def answer(self, item_id: str, ok: bool) -> str:
        """✅ / ✖: what happened (HTML). Updates the request's own card, tells the agent."""
        return ""

    async def choose(self, item_id: str, choice: int) -> str:
        """A question's answer."""
        return ""

    async def expire(self, item: Item) -> None:
        await self.answer(item.id, False)


def age(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}m" if s < 3600 else f"{s // 3600}h" if s < 86400 else f"{s // 86400}d"


class Pending:
    def __init__(self, store, agent, notifier):
        self.store = store
        self.agent = agent
        self.notifier = notifier
        self.kinds: dict[str, Kind] = {}
        agent.pending = self

    def register(self, kind: Kind) -> None:
        self.kinds[kind.key] = kind

    # ------------------------------------------------------------------ what waits
    def items(self, kind: str | None = None) -> list[Item]:
        out = []
        for k in self.kinds.values():
            if kind in (None, k.key):
                try:
                    out += sorted(k.items(), key=lambda i: i.at)
                except Exception:
                    log.exception("pending: listing %s failed", k.key)
        return out

    def count(self) -> int:
        return len(self.items())

    def _where(self, item: Item) -> str:
        s = self.agent.get(item.session) if item.session else None
        if s is None:
            return ""
        return s.rec.name or s.project or "main"

    def text(self, plain: bool = False) -> str:
        """The list, grouped by kind (HTML, or plain for `ov pending`)."""
        items = self.items()
        if not items:
            return "Nothing waits for an answer." if plain else "✅ <b>Nothing waits for your answer.</b>"
        now, lines = time.time(), []
        for k in self.kinds.values():
            mine = [i for i in items if i.kind == k.key]
            if not mine:
                continue
            lines.append(f"{k.emoji} {k.title} ({len(mine)})" if plain else f"\n{k.emoji} <b>{esc(k.title)}</b> "
                                                                             f"({len(mine)})")
            for i in mine:
                where = self._where(i)
                bits = [i.title[:160]] + ([where] if where else []) + [age(now - i.at) if i.at else ""]
                line = " · ".join(b for b in bits if b)
                lines.append(f"  {k.key}{i.id} {line}" if plain else f"• {esc(line)}")
        head = [] if plain else [f"⏳ <b>{len(items)} waiting for your answer</b> (declined after {EXPIRE_DAYS} days)"]
        return "\n".join(head + lines).strip()

    def buttons(self, confirm: str = "") -> list[list[tuple[str, str]]]:
        """✅ / ✖ per request (a question: its choices), then per kind and for all. `confirm`: "y" / "n" - the step
        that asks before answering everything."""
        items = self.items()
        bulk = [i for i in items if not i.choices]
        if confirm:
            n = len(bulk)
            return [[(f"{'✅ Yes, approve' if confirm == 'y' else '✖ Yes, decline'} all {n}", f"pdall:{confirm}"),
                     ("Cancel", "pdlist:")]]
        rows: list[list[tuple[str, str]]] = []
        for i in items[:SHOWN]:
            short = i.title[:24] + ("…" if len(i.title) > 24 else "")
            if i.choices:
                rows.append([(c[:24], f"pd:{i.kind}:{i.id}:{n}") for n, c in enumerate(i.choices)][:4])
            else:
                rows.append([(f"✅ {short}", f"pd:{i.kind}:{i.id}:y"), ("✖", f"pd:{i.kind}:{i.id}:n")])
        for k in self.kinds.values():
            n = len([i for i in bulk if i.kind == k.key])
            if n > 1:
                rows.append([(f"✅ All {k.noun}s ({n})", f"pdkind:{k.key}:y"), (f"✖ All ({n})", f"pdkind:{k.key}:n")])
        if len(bulk) > 1:
            rows.append([(f"✅ Approve all ({len(bulk)})", "pdask:y"), ("✖ Decline all", "pdask:n")])
        return rows

    def summary(self, ok: bool) -> str:
        """What "approve / decline all" covers, said before doing it."""
        counts = []
        for k in self.kinds.values():
            n = len([i for i in self.items(k.key) if not i.choices])
            if n:
                counts.append(f"{n} {k.noun}{'s' if n > 1 else ''}")
        return (f"{'Approve' if ok else 'Decline'} all of them: {', '.join(counts)}? (Questions aren't included: they "
                "need your choice.)")

    # ------------------------------------------------------------------ answering
    async def answer(self, kind: str, item_id: str, what: str) -> str:
        k = self.kinds.get(kind)
        if not k:
            return "That kind of request no longer exists."
        try:
            if what in ("y", "n"):
                return await k.answer(item_id, what == "y")
            if what.isdigit():
                return await k.choose(item_id, int(what))
        except Exception as e:
            log.exception("pending: answering %s %s failed", kind, item_id)
            return f"⚠️ {esc(str(e)[:300])}"
        return ""

    async def answer_all(self, ok: bool, kind: str | None = None) -> str:
        done = 0
        for i in self.items(kind):
            if not i.choices:
                await self.answer(i.kind, i.id, "y" if ok else "n")
                done += 1
        return f"{'✅ Approved' if ok else '✖ Declined'} {done} request{'s' if done != 1 else ''}."

    async def expire(self) -> int:
        """What has waited EXPIRE_DAYS days: declined (a question: dropped); its agent is told."""
        cutoff, n = time.time() - EXPIRE_DAYS * 86400, 0
        for i in self.items():
            if i.at and i.at < cutoff:
                try:
                    await self.kinds[i.kind].expire(i)
                except Exception:
                    log.exception("pending: expiring %s %s failed", i.kind, i.id)
                    continue
                if i.session:
                    self.agent.add_note(i.session, f"Your request \"{i.title[:200]}\" waited {EXPIRE_DAYS} days "
                                                   "without an answer, so it was dropped.")
                n += 1
        return n

    async def loop(self) -> None:
        while True:
            try:
                n = await self.expire()
                if n:
                    log.info("pending: %d request(s) expired after %d days", n, EXPIRE_DAYS)
            except Exception:
                log.exception("pending: expiry failed")
            await asyncio.sleep(3600)

    async def send(self, session_id: str | None = None, note: str = "") -> None:
        """The list, with its buttons, to the researcher (in a conversation's topic: an agent sent it)."""
        text = (f"{esc(note)}\n\n" if note else "") + self.text()
        await self.notifier.send(Outgoing(text, session_id=session_id, buttons=self.buttons()))


# ============================================================================ the platform's kinds
class HandbookKind(Kind):
    key, emoji, title, noun = "hb", "📖", "Handbook changes", "handbook change"

    def __init__(self, handbook):
        self.hb = handbook

    def items(self) -> list[Item]:
        return [Item(self.key, str(p.id), self.hb._describe(p), p.session_id, p.created) for p in self.hb.pending()]

    async def answer(self, item_id: str, ok: bool) -> str:
        return await (self.hb.approve if ok else self.hb.decline)(int(item_id))


class SkillKind(Kind):
    key, emoji, title, noun = "sk", "🧩", "Skills", "skill"

    def __init__(self, store, addons):
        self.store, self.addons = store, addons

    def items(self) -> list[Item]:
        return [Item(self.key, str(k.id), f"{k.name} ({self.addons._scope(k)}): {k.description}", k.session_id,
                     k.created) for k in self.store.skills(("proposed",))]

    async def answer(self, item_id: str, ok: bool) -> str:
        return await (self.addons.approve if ok else self.addons.decline)(int(item_id))


class PlanKind(Kind):
    key, emoji, title, noun = "pl", "🗺", "Plans (planner mode)", "plan"

    def __init__(self, store, subagents):
        self.store, self.subagents = store, subagents

    def items(self) -> list[Item]:
        from .subagents import PROPOSAL_KEY
        out = []
        for key in self.store.keys(PROPOSAL_KEY):
            try:
                p = json.loads(self.store.get(key) or "null")
            except ValueError:
                continue
            if p and p.get("state") == "pending":
                out.append(Item(self.key, p["id"], f"{p.get('goal', '')} ({len(p.get('subagents') or [])} subagents)",
                                p.get("session"), float(p.get("created") or 0)))
        return out

    async def answer(self, item_id: str, ok: bool) -> str:
        return await (self.subagents.approve if ok else self.subagents.decline)(item_id)


class QuestionKind(Kind):
    """`ov ask`: answered with one of its choices (the bot runs the agent with it)."""
    key, emoji, title, noun = "q", "❓", "Questions", "question"

    def __init__(self, store, answer):
        self.store, self._answer = store, answer  # answer(qid, choice) -> str: the bot's

    def items(self) -> list[Item]:
        out = []
        for key in self.store.keys("ask:"):
            try:
                rec = json.loads(self.store.get(key) or "null")
            except ValueError:
                continue
            if rec and rec.get("options"):
                out.append(Item(self.key, key[4:], rec["question"], rec.get("session"), float(rec.get("at") or 0),
                                choices=list(rec["options"])))
        return out

    async def choose(self, item_id: str, choice: int) -> str:
        return await self._answer(item_id, choice)

    async def expire(self, item: Item) -> None:
        self.store.unset(f"ask:{item.id}")
