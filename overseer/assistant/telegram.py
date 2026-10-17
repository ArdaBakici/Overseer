"""The assistant's Telegram commands and buttons: /tasks, /task, /remind, /reminders, /undo, /vault, /persona, and the
buttons (✅ on tasks, ↩️ Undo under replies, reminders' 💤 and ✖). The platform's bot (core/bot.py) registers them."""

from __future__ import annotations

import asyncio
import logging
import time

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from ..core.notify import Outgoing, code, esc
from . import tasks as T
from .domain import files_line
from .reminders import when_text
from .when import parse_moment

log = logging.getLogger(__name__)

HELP = """<b>Your assistant</b> — talk to it like a person: just send a message (pictures and files too).
Its memory is your Obsidian vault: it reads and writes notes there, and every reply that changed notes has a ↩️ Undo button.

/tasks [today|week|overdue|upcoming|undated|all|#tag|words] — tasks from the whole vault, ✅ to tick them off
/task &lt;text&gt; — add a task (Obsidian Tasks syntax works: 📅 2026-10-20 ⏫ 🔁 every week #tag)
/remind &lt;when&gt; &lt;text&gt; — e.g. /remind tomorrow 9am call the bank · /remind 2h stretch · /reminders — pending ones
/undo — undo a recent change of the assistant's · /vault — the vault: recent changes, sync conflicts
/persona — how the assistant behaves (a note in your vault: edit it there, or ask the assistant)
/new — a fresh conversation · /stop — stop the current reply · /backend · /model · /effort
/auto &lt;goal&gt; — let it work on something on its own (it wakes itself) · /auto off
/digest · /search &lt;text&gt; · /dashboard · /usage · /health
/project — projects (each gets its own topic) · /feedback · /skills"""

MENU = [("tasks", "tasks due, ✅ to tick off"), ("task", "add a task"), ("remind", "remind me at/in ..."),
        ("reminders", "pending reminders"), ("undo", "undo a change of the assistant's"),
        ("vault", "the vault: changes, conflicts"), ("persona", "how the assistant behaves")]

WHENS = ("open", "today", "overdue", "week", "upcoming", "undated", "done", "all")


class AssistantBot:
    """Registered with the platform's bot by AssistantDomain."""

    help = HELP
    menu = MENU
    errors = (T.TaskError,)

    def __init__(self, bot, domain):
        self.bot = bot
        self.domain = domain

    def __getattr__(self, name):  # reply, _scope, _edit, store, agent, notifier, cfg, ...
        return getattr(self.bot, name)

    @property
    def vault(self):
        return self.domain.vault

    def commands(self) -> list[tuple[str | list[str], object]]:
        return [("tasks", self.cmd_tasks), ("task", self.cmd_task), ("remind", self.cmd_remind),
                ("reminders", self.cmd_reminders), ("undo", self.cmd_undo), ("vault", self.cmd_vault),
                ("persona", self.cmd_persona)]

    async def on_button(self, update: Update, q, chat_id: int, action: str, arg: str) -> bool:
        if action == "tdone":
            await self._done(update, q, arg)
        elif action == "vundo":
            await self._undo(update, q, arg)
        elif action == "rrm":
            r = self.domain.reminders.remove(arg)
            await self._edit(q, f"🗑 Reminder cancelled: {esc(r['text'])}" if r else "That reminder is gone already.")
        elif action == "rsnz":
            rid, _, secs = arg.partition(":")
            r = self.domain.reminders.snooze(rid, float(secs) if secs.isdigit() else 3600)
            if r:
                await self._edit(q, f"💤 {esc(r['text'])}\n<i>again {esc(when_text(r['at']))}</i>",
                                 [[("✖ Cancel", f"rrm:{r['id']}")]])
            else:
                await self._edit(q, "That reminder was snoozed already.")
        else:
            return False
        return True

    async def health_lines(self) -> list[str]:
        notes = await asyncio.to_thread(self.vault.notes)
        conflicts = await asyncio.to_thread(self.vault.conflicts)
        return [f"vault {code(str(self.vault.root))} · {len(notes)} notes · history "
                + ("on" if self.vault.history else "off (git not found)" if self.cfg.vault.history else "off"),
                f"sync conflicts: {len(conflicts)}" if conflicts else "no sync conflicts",
                f"pending reminders: {len(self.domain.reminders.pending())}"]

    # ------------------------------------------------------------------ tasks
    async def cmd_tasks(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        args = ctx.args or []
        when, tag, words = "week", "", []
        for a in args:
            if a.lower() in WHENS:
                when = a.lower()
            elif a.startswith("#"):
                tag = a
            else:
                words.append(a)
        if (tag or words) and not any(a.lower() in WHENS for a in args):
            when = "open"
        today = self.domain.today()
        items = T.select(await asyncio.to_thread(T.scan, self.vault), when, tag, " ".join(words), today)
        title = {"week": "Due this week (and overdue)", "today": "Today (and overdue)"}.get(when, f"Tasks: {when}")
        if not items:
            await self.reply(update, f"<b>{esc(title)}</b>\nNothing. 🎉" if when in ("week", "today")
                             else f"No tasks ({esc(when)}).")
            return
        lines = [f"<b>{esc(title)}</b>"] + [esc(T.describe(t, today)) for t in items[:30]]
        if len(items) > 30:
            lines.append(f"(+{len(items) - 30} more)")
        buttons = [[(f"✅ {t.text[:28]}", f"tdone:{t.id}")] for t in items[:8] if t.open]
        await self.notifier.send(self._out("\n".join(lines), update, buttons))

    def _out(self, text: str, update: Update, buttons=None) -> Outgoing:
        return Outgoing(text, buttons=buttons or [], project=self._scope(update))

    async def _done(self, update: Update, q, tid: str) -> None:
        try:
            await self.domain.record_before()
            t, remark = T.complete(self.vault, tid, self.domain.today())
        except T.TaskError as e:
            await self.reply(update, f"⚠️ {esc(e)}")
            return
        change = await self.domain.record(f"Done: {t.text[:60]}")
        buttons = [[("↩️ Undo", f"vundo:{change.commit[:12]}")]] if change else None
        await self.notifier.send(self._out(f"✅ {esc(t.text)}" + (f" · {esc(remark)}" if remark else ""),
                                           update, buttons))

    async def cmd_task(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        text = " ".join(ctx.args or [])
        if not text.strip():
            await self.reply(update, "Usage: /task &lt;text&gt; (📅 2026-10-20, ⏫, 🔁 every week and #tags work)")
            return
        await self.domain.record_before()
        line = text.strip() if text.lstrip().startswith("- [") else T.line_for(text)
        rel = T.add(self.vault, line)
        change = await self.domain.record(f"Task added: {text[:60]}")
        buttons = [[("↩️ Undo", f"vundo:{change.commit[:12]}")]] if change else None
        await self.notifier.send(self._out(f"➕ {esc(line[6:])}\n<i>in {esc(rel)}</i>", update, buttons))

    # ------------------------------------------------------------------ reminders
    async def cmd_remind(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        words = ctx.args or []
        when = None
        for n in (3, 2, 1):
            if len(words) > n:
                try:
                    when = parse_moment(" ".join(words[:n]))
                    break
                except T.TaskError:
                    continue
        if when is None:
            await self.reply(update, "Usage: /remind &lt;when&gt; &lt;text&gt; — when: 17:30, tomorrow 9am, "
                                     "friday 18:00, 2026-10-20 09:00, 2h, 45m")
            return
        r = self.domain.reminders.add(when.timestamp(), " ".join(words[n:]), project=self._scope(update))
        await self.notifier.send(self._out(f"⏰ {esc(when_text(r['at']))}: {esc(r['text'])}", update,
                                           [[("✖ Cancel", f"rrm:{r['id']}")]]))

    async def cmd_reminders(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        items = self.domain.reminders.pending()
        if not items:
            await self.reply(update, "No reminders. /remind &lt;when&gt; &lt;text&gt; sets one.")
            return
        lines = ["<b>Reminders</b>"] + [f"⏰ {esc(when_text(r['at']))}: {esc(r['text'])}" for r in items[:30]]
        buttons = [[(f"✖ {when_text(r['at'])} {r['text'][:20]}", f"rrm:{r['id']}")] for r in items[:10]]
        await self.notifier.send(self._out("\n".join(lines), update, buttons))

    # ------------------------------------------------------------------ the vault
    async def _undo(self, update: Update, q, commit: str) -> None:
        ok, what = await self.vault.undo(commit)
        if ok:
            try:  # the button has done its job
                await q.edit_message_reply_markup(reply_markup=None)
            except BadRequest:
                pass
            await self.reply(update, f"↩️ Undone: {esc(what)}")
        else:
            await self.reply(update, f"⚠️ Couldn't undo: {esc(what)}")

    async def cmd_undo(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        changes = [c for c in await self.vault.log(30) if c.message not in ("Your edits", "Vault history starts")
                   and not c.message.startswith("Revert ")][:6]
        if not changes:
            await self.reply(update, "No changes of the assistant's to undo." if self.vault.history
                             else "The vault's history is off (vault.history, and git must be installed).")
            return
        lines = ["<b>Recent changes</b> (undo one: your own edits since are kept)"]
        for c in changes:
            lines.append(f"• {esc(time.strftime('%d %b %H:%M', time.localtime(c.when)))} — {esc(c.message[:80])}\n"
                         f"  {files_line(c.files, 5)}")
        buttons = [[(f"↩️ {time.strftime('%H:%M', time.localtime(c.when))} {c.message.split(': ', 1)[-1][:24]}",
                     f"vundo:{c.commit[:12]}")] for c in changes]
        await self.notifier.send(self._out("\n".join(lines), update, buttons))

    async def cmd_vault(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        notes = await asyncio.to_thread(self.vault.notes)
        conflicts = await asyncio.to_thread(self.vault.conflicts)
        lines = [f"<b>Vault</b> {code(str(self.vault.root))} · {len(notes)} notes",
                 f"today's note: {code(self.vault.daily_note())} · tasks go to {code(self.vault.rel(self.vault.tasks_file))}"]
        if conflicts:
            lines.append("⚠️ <b>Sync conflicts</b> (two devices changed a note at once): "
                         + ", ".join(code(c) for c in conflicts[:10]))
        changes = await self.vault.log(8)
        if changes:
            lines.append("<b>Recent changes</b>")
            lines += [f"• {esc(time.strftime('%d %b %H:%M', time.localtime(c.when)))} {esc(c.message[:70])} · "
                      f"{files_line(c.files, 3)}" for c in changes]
        elif not self.vault.history:
            lines.append("History is off (vault.history, and git must be installed).")
        await self.reply(update, "\n".join(lines))

    async def cmd_persona(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        rel = self.domain._rel("Persona.md")
        text = self.vault.read_note(rel, 3500).strip() or "(empty)"
        await self.reply(update, f"<b>{esc(self.domain.persona_name())}</b> — from {code(rel)} in your vault:\n"
                                 f"<blockquote expandable>{esc(text)}</blockquote>\nEdit the note in Obsidian, or "
                                 "tell the assistant how it should change.")
