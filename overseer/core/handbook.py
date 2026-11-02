"""The agents' handbook: what one agent learns, every agent of this bot knows.

Short entries, in the system prompt of every conversation of the bot (each bot - the HPC bot, the assistant - has its
own, in its own database):

* notes - what agents learned about where they work (the HPC bot: how the cluster behaves, a partition's quirk, a limit
  that doesn't apply where you'd think);
* rules - the researcher's own (e.g. "always --account=pi_lab");
* tools - the researcher's scripts, with what they're for. The domain may make them commands of their own (the HPC
  bot: a script of the researcher's becomes a command, on the login node or a compute node).

Agents add, change and remove entries with `ov handbook`; the researcher approves each change on a card (✅ / ✖) - except
an agent's edit of its own entry, which applies at once (the researcher is told, with ↩️ Undo). The researcher changes
it directly with /handbook. A changed handbook reaches conversations already going (agent.py hands over changed
instructions).
"""

from __future__ import annotations

import json
import re

from .notify import Outgoing, esc
from .store import HandbookRecord, Store

SECTIONS = {"notes": "What agents learned", "rules": "The researcher's rules", "tools": "Tools"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,30}$")
MAX_TEXT = 600  # an entry: a few lines at most
MAX_CHARS = 8000  # the whole handbook (it's in every prompt): past this, agents are asked to condense it


class HandbookError(ValueError):
    pass


def place(h: HandbookRecord) -> dict:
    """Where a tool runs: {} for the default (the HPC bot: the login node)."""
    try:
        return json.loads(h.place) if h.place else {}
    except ValueError:
        return {}


def ref(h: HandbookRecord) -> str:
    return f"H{h.target if h.action else h.id}"


class Handbook:
    def __init__(self, store: Store, agent, notifier):
        self.store = store
        self.agent = agent
        self.notifier = notifier
        agent.handbook = self
        self._tools_changed()

    # ------------------------------------------------------------------ reading
    def entries(self, section: str | None = None) -> list[HandbookRecord]:
        return [h for h in self.store.handbook() if not h.action and (section is None or h.section == section)]

    def pending(self) -> list[HandbookRecord]:
        return self.store.handbook(("proposed",))

    def entry(self, text: str) -> HandbookRecord:
        hid = text.strip().upper().lstrip("H")
        h = self.store.get_handbook(int(hid)) if hid.isdigit() else None
        if not h or h.action or h.status != "active":
            raise HandbookError(f"no handbook entry {text} (ov handbook lists them)")
        return h

    def size(self) -> int:
        return sum(len(h.text) + len(h.command) for h in self.entries())

    def _line(self, h: HandbookRecord) -> str:
        if h.section == "tools":
            return f"- {self.agent.domain.handbook_tool(h.name, h.command, h.text, place(h))} ({ref(h)})"
        return f"- {h.text} ({ref(h)})"

    def _tool(self, h: HandbookRecord) -> str:
        where = self.agent.domain.handbook_place_text(place(h))
        return f"{h.name}: {h.command}" + (f" ({where})" if where else "") + f" - {h.text}"

    def section(self) -> str:
        """The system prompt's part: how to use the handbook, and what's in it."""
        parts = ["\n\n## Handbook\nShared by every agent of this bot: what one learns, the others know. Use what's here. "
                 "When you learn something the others should know - how this environment behaves, a quirk, a limit, a "
                 "faster way - add it: `ov handbook add notes \"<one fact, with the conditions it holds under>\" "
                 "--why \"<the evidence>\"` (check the entries below first: change one rather than repeat it). A "
                 "script of the researcher's that agents should use: `ov handbook tool <name> --command \"<command>\" "
                 "--about \"<what it's for, when to use it>\"`" + self.agent.domain.handbook_tool_options() + ". Fix or update an entry: `ov handbook edit H3 \"...\"`, "
                 "drop a wrong or stale one: `ov handbook remove H3 --why \"...\"`. A standing instruction the "
                 "researcher gives you (how they want you to work, e.g. what a message prefix means) is a rule: "
                 "`ov handbook add rules \"...\"`, so every agent follows it - you have no memory of your own between "
                 "conversations. The researcher approves additions, "
                 "removals and changes to others' entries; your edits of your own entries apply at once. Keep entries "
                 "short and factual - project matters belong in the project's notes, results in `ov result`."]
        for key, title in SECTIONS.items():
            items = self.entries(key)
            if items:
                parts.append(f"\n### {title}\n" + "\n".join(self._line(h) for h in items))
        if self.size() > MAX_CHARS:
            parts.append(f"\n(The handbook is past {MAX_CHARS // 1000} KB, and it's in every prompt: when you can, "
                         "propose merging or removing entries.)")
        return "\n".join(parts)

    def listing(self) -> str:
        """For `ov handbook` and /handbook (plain text)."""
        out = []
        for key, title in SECTIONS.items():
            items = self.entries(key)
            if items:
                out.append(f"{title}:")
                out += [f"  {ref(h)} " + (self._tool(h) if key == "tools" else h.text) for h in items]
        pending = self.pending()
        if pending:
            out.append("Waiting for the researcher:")
            out += [f"  #{p.id} {self._describe(p)}" for p in pending]
        return "\n".join(out) or "The handbook is empty."

    def _describe(self, p: HandbookRecord) -> str:
        if p.action == "remove":
            return f"remove {ref(p)}"
        what = self._tool(p) if p.section == "tools" else p.text
        return f"{'change ' + ref(p) if p.action == 'edit' else 'add to ' + p.section}: {what}"

    # ------------------------------------------------------------------ an agent's changes
    def _check(self, section: str, text: str, name: str = "", command: str = "") -> None:
        if section not in SECTIONS:
            raise HandbookError(f"which section? {' | '.join(SECTIONS)}")
        if not text.strip():
            raise HandbookError("say what the entry is")
        if len(text) > MAX_TEXT:
            raise HandbookError(f"keep an entry under {MAX_TEXT} characters (it's in every agent's prompt): one fact "
                                "or rule per entry")
        if section == "tools":
            if not NAME_RE.match(name or ""):
                raise HandbookError("a tool's name: lowercase letters, digits, - or _ (e.g. queue-report)")
            if not command.strip():
                raise HandbookError("--command: what the tool runs (e.g. ~/bin/<script>.sh)")
            if self.agent.domain.command_taken(name):
                raise HandbookError(f"{name!r} is already a command here: name the tool differently")
            if any(h.name == name for h in self.entries("tools")):
                raise HandbookError(f"there's a tool {name} already: ov handbook edit its entry")

    def _place(self, section: str, opts: dict | None, before: dict | None = None) -> str:
        """Where a tool runs (JSON), from the options given (an edit: over what it had); the domain checks them."""
        opts = {k: v for k, v in (opts or {}).items() if v not in (None, False, "")}
        if section != "tools":
            if opts:
                raise HandbookError("--on, --dir and the resources are for tools")
            return ""
        merged = dict(before or {})
        if opts.get("on") == "login":  # (back to the login node: its compute options go)
            merged = {k: v for k, v in merged.items() if k == "dir"}
        merged.update(opts)
        return json.dumps(self.agent.domain.handbook_place(merged)) if merged else ""

    async def propose(self, s, action: str, section: str = "", text: str = "", name: str = "", command: str = "",
                      target: str = "", why: str = "", where: dict | None = None) -> str:
        """An agent's addition, change or removal: a card for the researcher - or, its edit of its own entry, done.
        `where`: a tool's placement options (on, dir, partition, gpu, gres, cpus, mem, time, qos)."""
        text, why = text.strip(), why.strip()
        old = self.entry(target) if action in ("edit", "remove") else None
        if old:
            section, name = old.section, old.name
        if action == "add":
            self._check(section, text, name, command)
            spot = self._place(section, where)
        elif action == "edit":
            text, command = text or old.text, command or old.command
            self._check_text(text)
            spot = self._place(section, where, place(old)) if where and any(where.values()) else old.place
            if (text, command, spot) == (old.text, old.command, old.place):
                raise HandbookError("that's what it says already: give the new text, --command or placement")
        else:
            text, command, spot = old.text, old.command, old.place  # (what goes)
        p = self.store.add_handbook(HandbookRecord(section=section, text=text, name=name, command=command,
                                                   place=spot, status="proposed", action=action,
                                                   target=old.id if old else 0, why=why, session_id=s.id))
        if action == "edit" and old.session_id == s.id:  # (its own entry: done, and the researcher is told)
            before = self._tool(old) if section == "tools" else old.text
            self._apply(p)
            after = self._tool(p) if section == "tools" else text
            await self._post(p, f"✏️ <b>An agent updated its handbook entry {ref(p)}</b> ({esc(SECTIONS[section])})"
                                f"\n<s>{esc(before)}</s>\n{esc(after)}" + (f"\n<i>why: {esc(why)}</i>" if why else ""),
                             [[("↩️ Undo", f"hbu:{p.id}")]])
            return f"✓ {ref(p)} updated (your own entry: done; the researcher is told)."
        await self._post(p, self.card_text(p), [[("✅ Apply", f"hb:{p.id}"), ("✖ No", f"hbx:{p.id}")]])
        return (f"✓ proposed (#{p.id}): the researcher approves or declines it in Telegram; you're told which. "
                "Carry on meanwhile.")

    def _check_text(self, text: str) -> None:
        if not text:
            raise HandbookError("say what the entry is")
        if len(text) > MAX_TEXT:
            raise HandbookError(f"keep an entry under {MAX_TEXT} characters")

    def card_text(self, p: HandbookRecord) -> str:
        head = {"add": f"add to {SECTIONS[p.section].lower()}", "edit": f"change {ref(p)}",
                "remove": f"remove {ref(p)}"}[p.action]
        old = self.store.get_handbook(p.target) if p.target else None
        lines = [f"📖 <b>Handbook: {esc(head)}</b> · from an agent"]

        def shown(h: HandbookRecord) -> str:
            if h.section != "tools":
                return esc(h.text)
            where = self.agent.domain.handbook_place_text(place(h))
            return (f"<code>{esc(h.name)}</code> runs <code>{esc(h.command)}</code>"
                    + (f" <b>{esc(where)}</b>" if where else "") + f": {esc(h.text)}")
        if old and p.action in ("edit", "remove"):
            lines.append(f"<s>{shown(old)}</s>")
        if p.action != "remove":
            lines.append(shown(p))
        if p.why:
            lines.append(f"<i>why: {esc(p.why)}</i>")
        return "\n".join(lines)

    async def _post(self, p: HandbookRecord, text: str, buttons) -> None:
        msg = Outgoing(text, session_id=p.session_id, buttons=buttons)
        if hasattr(self.notifier, "post"):
            p.refs = json.dumps(await self.notifier.post(msg))
            self.store.save_handbook(p)
        else:
            await self.notifier.send(msg)

    async def _answer(self, p: HandbookRecord, text: str) -> None:
        refs = json.loads(p.refs or "{}")
        if refs and hasattr(self.notifier, "edit"):
            await self.notifier.edit(refs, f"{self.card_text(p)}\n\n{text}")

    def _apply(self, p: HandbookRecord) -> HandbookRecord:
        if p.action == "add":
            e = self.store.add_handbook(HandbookRecord(section=p.section, text=p.text, name=p.name,
                                                       command=p.command, place=p.place, session_id=p.session_id))
            p.target = e.id
        else:
            e = self.store.get_handbook(p.target)
            if p.action == "edit":
                p.old = json.dumps({"text": e.text, "command": e.command, "place": e.place})
                e.text, e.command, e.place = p.text, p.command or e.command, p.place
            else:
                e.status = "removed"
            self.store.save_handbook(e)
        p.status = "applied"
        self.store.save_handbook(p)
        if p.section == "tools":
            self._tools_changed()
        return e

    async def approve(self, pid: int) -> str:
        p = self.store.get_handbook(pid)
        if not p or p.status != "proposed":
            return "This handbook proposal was already answered."
        if p.target and p.action != "add":
            e = self.store.get_handbook(p.target)
            if not e or e.status != "active":
                p.status = "declined"
                self.store.save_handbook(p)
                await self._answer(p, "✖ Its entry is gone.")
                return "✖ Its entry is gone."
        self._apply(p)
        if p.session_id:
            self.agent.add_note(p.session_id, f"The researcher applied your handbook change (#{p.id}, {ref(p)}): every "
                                              "agent of this bot has it from its next request.")
        text = f"✅ <b>Applied</b> ({ref(p)}): every agent has it from its next request. (/handbook)"
        await self._answer(p, text)
        return text

    async def decline(self, pid: int) -> str:
        p = self.store.get_handbook(pid)
        if not p or p.status != "proposed":
            return "This handbook proposal was already answered."
        p.status = "declined"
        self.store.save_handbook(p)
        if p.session_id:
            self.agent.add_note(p.session_id, f"The researcher declined your handbook change #{p.id} "
                                              f"({self._describe(p)[:200]}).")
        await self._answer(p, "✖ <b>Declined.</b>")
        return "✖ Declined."

    async def undo(self, pid: int) -> str:
        """↩️ on an agent's edit of its own entry: the text before."""
        p = self.store.get_handbook(pid)
        e = self.store.get_handbook(p.target) if p and p.status == "applied" and p.action == "edit" else None
        if not e or e.status != "active":
            return "This change can't be undone any more."
        before = json.loads(p.old)
        e.text, e.command, e.place = before["text"], before["command"], before["place"]
        self.store.save_handbook(e)
        if e.section == "tools":
            self._tools_changed()
        p.status = "undone"
        self.store.save_handbook(p)
        if p.session_id:
            self.agent.add_note(p.session_id, f"The researcher undid your edit of {ref(p)}: it's back to what it was.")
        return f"↩️ {ref(p)} is back to: {esc(e.text)}"

    # ------------------------------------------------------------------ the researcher's changes (/handbook)
    def add(self, section: str, text: str, name: str = "", command: str = "",
            where: dict | None = None) -> HandbookRecord:
        self._check(section, text.strip(), name, command.strip())
        e = self.store.add_handbook(HandbookRecord(section=section, text=text.strip(), name=name,
                                                   command=command.strip(), place=self._place(section, where)))
        if section == "tools":
            self._tools_changed()
        return e

    def edit(self, target: str, text: str) -> HandbookRecord:
        e = self.entry(target)
        self._check_text(text.strip())
        e.text = text.strip()
        self.store.save_handbook(e)
        return e

    def remove(self, target: str) -> HandbookRecord:
        e = self.entry(target)
        e.status = "removed"
        self.store.save_handbook(e)
        if e.section == "tools":
            self._tools_changed()
        return e

    def _tools_changed(self) -> None:
        """The domain makes the tools commands of their own, if it can (the HPC bot: on the login node)."""
        self.agent.domain.handbook_tools([(h.name, h.command, place(h)) for h in self.entries("tools")])
