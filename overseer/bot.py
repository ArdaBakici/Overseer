"""Telegram front-end. Only chats listed in telegram.allowed_chat_ids can talk to the bot.

Note: the agent has full access to your HPC account, so whoever controls an allowed chat
effectively has a shell there. Keep allowed_chat_ids to your own chat and protect your
Telegram account (2FA).
"""

from __future__ import annotations

import asyncio
import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agent import Agent, Session
from .config import BACKENDS, Config
from .hpc import HPC
from .monitor import Monitor
from .notify import TG_LIMIT, Outgoing, code, esc, pre, split_message
from .slurm import STATE_EMOJI
from .ssh import SSHError
from .sshlink import master_alive
from .store import Store

log = logging.getLogger(__name__)

HELP = """<b>Overseer</b> — your SLURM jobs, from Telegram.

/jobs — active and recent jobs
/job &lt;id&gt; — status + log tail of a job
/log &lt;id&gt; [lines] — tail of a job's log
/fix &lt;id&gt; [instructions] — ask the agent to diagnose/fix a job
/ask &lt;text&gt; — talk to the agent (or just send a message)
/new — start a fresh agent conversation
/backend [claude_code|codex] — show or switch the agent
/model [name|default] — show or set the active agent's model
/effort [level|default] — show or set the active agent's effort
/stop — stop the agent's current run
/health — connection status

Reply to any job message to talk to the agent about that job.
The agent has full access to your HPC account; you'll see every command it runs and every job it submits."""


class TelegramNotifier:
    def __init__(self, app: Application, chat_ids: list[int], store: Store):
        self.app = app
        self.chat_ids = chat_ids
        self.store = store

    async def send(self, msg: Outgoing) -> None:
        """Deliver a message. Never raises: a Telegram hiccup must not break monitoring."""
        for attempt in range(3):
            try:
                await self._send(msg)
                return
            except BadRequest:
                log.exception("telegram rejected message")
                return
            except Exception:
                log.exception("telegram send failed (attempt %d)", attempt + 1)
                await asyncio.sleep(2 * (attempt + 1))

    async def _send(self, msg: Outgoing) -> None:
        markup = None
        if msg.buttons:
            markup = InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in row] for row in msg.buttons])
        for chat_id in self.chat_ids:
            chunks = split_message(msg.text)
            for i, chunk in enumerate(chunks):
                last = i == len(chunks) - 1
                try:
                    sent = await self.app.bot.send_message(
                        chat_id, chunk, parse_mode=ParseMode.HTML, disable_notification=msg.silent,
                        reply_markup=markup if last else None, link_preview_options=None,
                    )
                except BadRequest:
                    # Malformed HTML after splitting: fall back to plain text.
                    sent = await self.app.bot.send_message(chat_id, chunk, disable_notification=msg.silent,
                                                           reply_markup=markup if last else None)
                self.store.link_message(chat_id, sent.message_id, msg.job_id, msg.session_id)
            for name, data in msg.files:
                sent = await self.app.bot.send_document(chat_id, document=data, filename=name, disable_notification=True)
                self.store.link_message(chat_id, sent.message_id, msg.job_id, msg.session_id)


    async def post_live(self, msg: Outgoing) -> list[tuple[int, int]]:
        handle = []
        for chat_id in self.chat_ids:
            try:
                sent = await self.app.bot.send_message(chat_id, msg.text[:TG_LIMIT], parse_mode=ParseMode.HTML,
                                                       disable_notification=True)
            except Exception:
                log.debug("live message failed", exc_info=True)
                continue
            handle.append((chat_id, sent.message_id))
            self.store.link_message(chat_id, sent.message_id, msg.job_id, msg.session_id)
        return handle

    async def edit_live(self, handle: list[tuple[int, int]], text: str) -> None:
        for chat_id, message_id in handle:
            try:
                await self.app.bot.edit_message_text(text[:TG_LIMIT], chat_id=chat_id, message_id=message_id,
                                                     parse_mode=ParseMode.HTML)
            except Exception:  # "message is not modified", rate limits: the next update catches up
                log.debug("live message edit failed", exc_info=True)

    async def delete_live(self, handle: list[tuple[int, int]]) -> None:
        for chat_id, message_id in handle:
            try:
                await self.app.bot.delete_message(chat_id, message_id)
            except Exception:
                log.debug("live message delete failed", exc_info=True)


class Bot:
    def __init__(self, cfg: Config, app: Application, store: Store, hpc: HPC, monitor: Monitor,
                 agent: Agent, notifier: TelegramNotifier):
        self.cfg = cfg
        self.app = app
        self.store = store
        self.hpc = hpc
        self.monitor = monitor
        self.agent = agent
        self.notifier = notifier
        self._tasks: set[asyncio.Task] = set()

    def register(self) -> None:
        allowed = filters.Chat(chat_id=self.cfg.telegram.allowed_chat_ids)
        add = self.app.add_handler
        add(CommandHandler(["start", "help"], self.cmd_help, filters=allowed))
        add(CommandHandler("jobs", self.cmd_jobs, filters=allowed))
        add(CommandHandler("job", self.cmd_job, filters=allowed))
        add(CommandHandler("log", self.cmd_log, filters=allowed))
        add(CommandHandler("fix", self.cmd_fix, filters=allowed))
        add(CommandHandler("ask", self.cmd_ask, filters=allowed))
        add(CommandHandler("new", self.cmd_new, filters=allowed))
        add(CommandHandler("backend", self.cmd_backend, filters=allowed))
        add(CommandHandler("model", self.cmd_model, filters=allowed))
        add(CommandHandler("effort", self.cmd_effort, filters=allowed))
        add(CommandHandler("stop", self.cmd_stop, filters=allowed))
        add(CommandHandler("health", self.cmd_health, filters=allowed))
        add(CallbackQueryHandler(self.on_button))
        add(MessageHandler(allowed & filters.TEXT & ~filters.COMMAND, self.on_text))
        add(MessageHandler(~allowed, self.on_stranger))

    # ------------------------------------------------------------------ helpers
    async def reply(self, update: Update, text: str) -> None:
        for chunk in split_message(text):
            try:
                await update.effective_message.reply_text(chunk, parse_mode=ParseMode.HTML)
            except BadRequest:
                await update.effective_message.reply_text(chunk)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_agent(self, chat_id: int, session: Session, text: str) -> None:
        if self.agent.is_running(session):
            await self.notifier.send(Outgoing("⏳ The agent is still working on the previous message in this "
                                              "conversation; yours is queued. /stop cancels the current run.",
                                              session_id=session.id, silent=True))

        async def typing():
            while True:
                try:
                    await self.app.bot.send_chat_action(chat_id, ChatAction.TYPING)
                except Exception:
                    pass
                await asyncio.sleep(5)

        t = asyncio.create_task(typing())
        try:
            await self.agent.handle(session, text)
        finally:
            t.cancel()

    # ------------------------------------------------------------------ commands
    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await self.reply(update, HELP + f"\n\nCurrent agent: {code(self.agent.backend)}")

    async def cmd_jobs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        rows = self.store.recent_jobs(15)
        if not rows:
            await self.reply(update, "No jobs seen yet.")
            return
        lines = []
        for r in rows:
            tag = " 🤖" if r.by_agent else ""
            lines.append(f"{STATE_EMOJI.get(r.state, '•')} {code(r.job_id)} {esc(r.name)} — {esc(r.state)}{tag}")
        await self.reply(update, "<b>Recent jobs</b> (🤖 = submitted by the agent)\n" + "\n".join(lines))

    async def cmd_job(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.args:
            await self.reply(update, "Usage: /job &lt;id&gt;")
            return
        await self._send_status(ctx.args[0])

    async def _send_status(self, job_id: str) -> None:
        rec = self.store.get_job(job_id)
        if not rec:
            await self.notifier.send(Outgoing(f"Unknown job {code(job_id)}."))
            return
        live = {j.job_id: j for j in await self.hpc.squeue()}
        msg = await self.monitor.status_message(rec, live.get(job_id))
        msg.silent = False
        if rec.report:
            msg.text += f"\n\n🧠 {esc(rec.report[:1500])}"
        await self.notifier.send(msg)

    async def cmd_log(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.args:
            await self.reply(update, "Usage: /log &lt;id&gt; [lines]")
            return
        lines = int(ctx.args[1]) if len(ctx.args) > 1 and ctx.args[1].isdigit() else 100
        await self._send_log(ctx.args[0], lines)

    async def _send_log(self, job_id: str, lines: int = 100) -> None:
        rec = self.store.get_job(job_id)
        if not rec:
            await self.notifier.send(Outgoing(f"Unknown job {code(job_id)}."))
            return
        lines = max(1, min(lines, 5000))
        out = await self.hpc.try_tail(rec.stdout, lines)
        files = [(f"job-{job_id}-stdout.log", out.encode())]
        if rec.stderr and rec.stderr != rec.stdout:
            files.append((f"job-{job_id}-stderr.log", (await self.hpc.try_tail(rec.stderr, lines)).encode()))
        await self.notifier.send(Outgoing(f"📜 Job {code(job_id)} {code(rec.stdout)}" + pre(out, 3000),
                                          job_id=job_id, files=files if lines > 60 else []))

    async def cmd_fix(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.args:
            await self.reply(update, "Usage: /fix &lt;job id&gt; [instructions]")
            return
        await self._start_fix(update.effective_chat.id, ctx.args[0], " ".join(ctx.args[1:]))

    async def _start_fix(self, chat_id: int, job_id: str, instr: str = "") -> None:
        session = self.agent.session_for_job(job_id)
        text = instr or (f"Job {job_id} failed. Find out why, fix it, and resubmit it if the fix is clear. "
                         "If it needs a decision from me, explain the options instead.")
        await self.notifier.send(Outgoing(f"🔧 {esc(self.agent.backend)} is working on job {code(job_id)}…",
                                          job_id=job_id, session_id=session.id, silent=True))
        self._spawn(self._run_agent(chat_id, session, text))

    async def cmd_ask(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        text = " ".join(ctx.args or [])
        if not text:
            await self.reply(update, "Usage: /ask &lt;question&gt;")
            return
        self._spawn(self._run_agent(update.effective_chat.id, self.agent.general_session(), text))

    async def cmd_new(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        s = self.agent.general_session(reset=True)
        await self.reply(update, f"Started a fresh conversation ({code(s.id)}).")

    def _agents_summary(self) -> str:
        lines = []
        for b in BACKENDS:
            mark = "▶️" if b == self.agent.backend else "▫️"
            lines.append(f"{mark} {esc(self.agent.describe(b))}")
        return "\n".join(lines)

    async def cmd_backend(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.args:
            names = ", ".join(code(b) for b in BACKENDS)
            await self.reply(update, f"{self._agents_summary()}\n\nSwitch with /backend &lt;name&gt; ({names}); "
                                     "change the model / effort with /model and /effort.")
            return
        try:
            self.agent.set_backend(ctx.args[0])
        except ValueError as e:
            await self.reply(update, f"⛔ {esc(e)}")
            return
        await self.reply(update, f"Agent backend is now {code(self.agent.backend)}. Existing conversations continue "
                                 "with a fresh session on this backend (the job context is re-sent).\n\n"
                                 + self._agents_summary())

    async def cmd_model(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        b = self.agent.backend
        if not ctx.args:
            examples = "opus, sonnet, claude-opus-5-5" if b == "claude_code" else "gpt-5.5-codex"
            await self.reply(update, f"{esc(self.agent.describe(b))}\n\nSet with /model &lt;name&gt; (e.g. {examples}), "
                                     "or /model default to use config.yaml / the CLI default.")
            return
        try:
            self.agent.set_model(ctx.args[0])
        except ValueError as e:
            await self.reply(update, f"⛔ {esc(e)}")
            return
        await self.reply(update, f"✅ {esc(self.agent.describe(b))}\nApplies from the next message "
                                 "(an unknown model name shows up as an error from the CLI).")

    async def cmd_effort(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        b = self.agent.backend
        levels = " | ".join(self.agent.runner(b).efforts)
        if not ctx.args:
            await self.reply(update, f"{esc(self.agent.describe(b))}\n\nSet with /effort &lt;{levels}&gt; or /effort default.")
            return
        try:
            self.agent.set_effort(ctx.args[0])
        except ValueError as e:
            await self.reply(update, f"⛔ {esc(e)}")
            return
        await self.reply(update, f"✅ {esc(self.agent.describe(b))}\nApplies from the next message.")

    async def cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        running = self.agent.running_sessions()
        if not running:
            await self.reply(update, "Nothing is running.")
            return
        for s in running:
            await self.agent.stop(s.id)

    async def cmd_health(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        last = f"{int(time.time() - self.monitor.last_ok)}s ago" if self.monitor.last_ok else "never"
        err = f"\nlast error: {code(self.monitor.last_error[:300])}" if self.monitor.last_error else ""
        conn = "up" if await master_alive(self.cfg) else "down (reopens on the next poll)"
        running = len(self.agent.running_sessions())
        await self.reply(update, f"HPC user {code(self.hpc.user or '?')} · SSH connection {conn}\n"
                                 f"last successful poll: {last}{err}\n"
                                 f"agent runs in progress: {running}\n\n{self._agents_summary()}")

    # ------------------------------------------------------------------ free text & buttons
    async def on_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        chat_id = update.effective_chat.id
        session = None
        if msg.reply_to_message:
            job_id, sid = self.store.message_link(chat_id, msg.reply_to_message.message_id)
            session = self.agent.get(sid)
            if session is None and job_id:
                session = self.agent.session_for_job(job_id)
        if session is None:
            session = self.agent.general_session()
        self._spawn(self._run_agent(chat_id, session, msg.text))

    async def on_stranger(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat:
            log.warning("ignored message from unauthorised chat %s", update.effective_chat.id)

    async def on_button(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        chat = update.effective_chat
        if not chat or chat.id not in self.cfg.telegram.allowed_chat_ids:
            await q.answer("Not authorised.", show_alert=True)
            return
        action, _, arg = (q.data or "").partition(":")
        await q.answer()
        try:
            if action == "fix":
                await self._start_fix(chat.id, arg)
            elif action == "log":
                await self._send_log(arg, 200)
            elif action == "status":
                await self._send_status(arg)
        except SSHError as e:
            await self.notifier.send(Outgoing(f"⚠️ {esc(e)}"))
