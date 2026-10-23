"""Runs Routines.md: each routine at its times (a `do:` one wakes the assistant, a `say:` one is a follow-up
message), the first-run setup conversation while the note doesn't exist, and a notice when the note can't be read.
Overseer decides the times; the model never does."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

from ..core.notify import Outgoing, code, esc
from . import prompts as P
from . import routines as R

log = logging.getLogger(__name__)

CHECKED_KEY = "assistant:routines_checked"
ONBOARDED_KEY = "assistant:onboarding_started"
TICK = 20


class Scheduler:
    def __init__(self, domain):
        self.domain = domain
        self.store = domain.store
        self._seen: tuple[float, int] | None = None  # (mtime, size) of the note last read
        self._parsed: R.Routines | None = None
        self._reported: tuple | None = None  # the problems last reported, for that version of the note

    @property
    def rel(self) -> str:
        return self.domain._rel("Routines.md")

    @property
    def path(self):
        return self.domain.vault.root / self.rel

    def load(self) -> R.Routines | None:
        """The note, parsed (again only when it changed); None while it doesn't exist."""
        try:
            st = self.path.stat()
        except OSError:
            self._seen, self._parsed = None, None
            return None
        if self.path.is_symlink():
            return R.Routines(errors=["it's a link: Overseer doesn't follow links out of the vault"])
        key = (st.st_mtime, st.st_size)
        if key != self._seen or self._parsed is None:
            self._parsed = R.parse(self.domain.vault.read_note(self.rel, 200_000))
            self._seen = key
        return self._parsed

    def state_line(self) -> str:
        """For the agent's instructions: the note's state."""
        rs = self.load()
        if rs is None:
            return P.ROUTINES_MISSING.format(routines_path=self.rel)
        parts = []
        if rs.routines:
            parts.append(P.ROUTINES_OK.format(summary="; ".join(
                f"{r.name} ({r.describe_when()}{', paused' if not r.active(datetime.now().date()) else ''})"
                for r in rs.routines)))
        if rs.errors:
            parts.append(P.ROUTINES_ERRORS.format(errors="; ".join(rs.errors)))
        return "\n".join(parts)

    def check_text(self) -> str:
        """`ov routine check`: problems, and when each routine runs next."""
        rs = self.load()
        if rs is None:
            return f"{self.rel} doesn't exist yet"
        now = datetime.now()
        lines = [f"⚠️ {e}" for e in rs.errors]
        for r in rs.routines:
            nxt = r.next_time(now)
            kind = "do" if r.do else "say"
            fu = f" · follow-up: {R.describe_policy(rs.policy(r.follow_up))}" if r.say else ""
            lines.append(f"✓ {r.name} ({kind}): {r.describe_when()} · next "
                         + (f"{nxt:%a %d %b %H:%M}" if nxt else "never (paused or off)") + fu)
        for name, p in rs.policies.items():
            lines.append(f"follow-up {name}: {R.describe_policy(p)}")
        return "\n".join(lines) or "no routines in it"

    # ------------------------------------------------------------------ the clock
    async def tick(self, now: datetime | None = None) -> list[str]:
        """Run what's due since the last tick (a missed one only if it's recent). The names of those run."""
        now = now or datetime.now()
        rs = self.load()
        if rs is None:
            return []
        await self._report(rs)
        last = self.store.get(CHECKED_KEY)
        self.store.set(CHECKED_KEY, str(now.timestamp()))
        if not last:  # the first time: from now on (nothing from before is "missed")
            return []
        since = datetime.fromtimestamp(float(last))
        fired = []
        for r in rs.routines:
            due = r.occurrences(since, now)
            if due and (now - due[-1]).total_seconds() <= R.CATCH_UP:
                await self.run(r, due[-1], rs)
                fired.append(r.name)
        return fired

    async def run(self, r: R.Routine, when: datetime, rs: R.Routines) -> None:
        d = self.domain
        if r.say:
            d.reminders.add(time.time(), r.say, kind="say", policy=rs.policy(r.follow_up), routine=r.name)
            return
        agent = d.agent
        if not agent or not agent.cfg.agent.enabled or not agent.autopilot:
            return
        s = agent.general_session()
        prompt = "\n\n".join([
            P.ROUTINE_PROMPT.format(name=r.name, routines_path=self.rel, when=f"{when:%A %H:%M}", do=r.do),
            await d.now_context()])
        agent.autopilot.wake(s, prompt, kind="routine", by_user=True)  # (yours: runs even if wake-ups are paused)

    async def _report(self, rs: R.Routines) -> None:
        """Problems in the note (e.g. after an edit in Obsidian): told once per version of it."""
        key = (self._seen, tuple(rs.errors))
        if not rs.errors or key == self._reported:
            self._reported = key if rs.errors else None
            return
        self._reported = key
        await self.domain.notifier.send(Outgoing(
            f"⚠️ {code(self.rel)} has problems, so these routines don't run until they're fixed:\n"
            + "\n".join(f"• {esc(e)}" for e in rs.errors[:10]) + "\nTell me how it should be, or fix it in Obsidian.",
            silent=True))

    async def onboard(self) -> bool:
        """The first run: no routines note yet, and never set up - the assistant starts the setup conversation."""
        d = self.domain
        agent = d.agent
        if self.path.exists() or self.store.get(ONBOARDED_KEY) or not agent or not agent.cfg.agent.enabled \
                or not agent.autopilot:
            return False
        self.store.set(ONBOARDED_KEY, str(time.time()))
        s = agent.general_session()
        prompt = "\n\n".join([P.ONBOARDING_PROMPT.format(routines_path=self.rel, example=R.EXAMPLE),
                              await d.now_context()])
        agent.autopilot.wake(s, prompt, kind="setup", by_user=True)
        return True

    async def loop(self) -> None:
        await asyncio.sleep(3)  # (Telegram's topics are set up first)
        try:
            await self.onboard()
        except Exception:
            log.exception("first-run setup")
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("routines")
            await asyncio.sleep(TICK)
