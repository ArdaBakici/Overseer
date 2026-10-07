"""Telegram front-end. Only chats listed in telegram.allowed_chat_ids can talk to the bot."""

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
from .config import Config
from .hpc import HPC
from .monitor import Monitor
from .notify import Outgoing, code, esc, pre, split_message
from .policy import PolicyError
from .proposals import Submitter
from .slurm import STATE_EMOJI
from .ssh import SSHError
from .store import Store

log = logging.getLogger(__name__)

HELP = """<b>Overseer</b> — your SLURM jobs, from Telegram.

/jobs — active and recent jobs
/job &lt;id&gt; — status + log tail of a job
/log &lt;id&gt; [lines] — tail of a job's log
/fix &lt;id&gt; [instructions] — ask the agent to diagnose/fix a job
/ask &lt;text&gt; — talk to the agent (or just send a message)
/new — start a fresh agent conversation
/pending — proposals waiting for your approval
/cancel &lt;id&gt; — cancel a job Overseer submitted (asks to confirm)
/access — folders Overseer may read / write on the HPC
/grant &lt;path&gt; [write|read] — allow a folder (default: write)
/revoke &lt;path&gt; — remove a folder granted here
/health — connection status

Reply to any job message to talk to the agent about that job.
The agent only writes inside allowed folders (it asks you for new ones), backs up every file it changes,
and every sbatch/scancel needs your approval."""


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


class Bot:
    def __init__(self, cfg: Config, app: Application, store: Store, hpc: HPC, monitor: Monitor,
                 agent: Agent, submitter: Submitter, notifier: TelegramNotifier):
        self.cfg = cfg
        self.app = app
        self.store = store
        self.hpc = hpc
        self.monitor = monitor
        self.agent = agent
        self.submitter = submitter
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
        add(CommandHandler("pending", self.cmd_pending, filters=allowed))
        add(CommandHandler("cancel", self.cmd_cancel, filters=allowed))
        add(CommandHandler("health", self.cmd_health, filters=allowed))
        add(CommandHandler("access", self.cmd_access, filters=allowed))
        add(CommandHandler("grant", self.cmd_grant, filters=allowed))
        add(CommandHandler("revoke", self.cmd_revoke, filters=allowed))
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
        await self.reply(update, HELP)

    async def cmd_jobs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        rows = self.store.recent_jobs(15)
        if not rows:
            await self.reply(update, "No jobs seen yet.")
            return
        lines = []
        for r in rows:
            tag = " 🤖" if r.overseer else ""
            lines.append(f"{STATE_EMOJI.get(r.state, '•')} {code(r.job_id)} {esc(r.name)} — {esc(r.state)}{tag}")
        await self.reply(update, "<b>Recent jobs</b> (🤖 = submitted by Overseer)\n" + "\n".join(lines))

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
        job_id, instr = ctx.args[0], " ".join(ctx.args[1:])
        await self._start_fix(update.effective_chat.id, job_id, instr)

    async def _start_fix(self, chat_id: int, job_id: str, instr: str = "") -> None:
        if not self.store.get_job(job_id):
            await self.notifier.send(Outgoing(f"Unknown job {code(job_id)}."))
            return
        session = self.agent.session_for_job(job_id)
        text = instr or ("Diagnose why this job failed. Then fix it and propose a resubmission. "
                         "If the fix needs something outside your reach, explain what I should do instead.")
        await self.notifier.send(Outgoing(f"🔧 Working on job {code(job_id)}…", job_id=job_id, session_id=session.id, silent=True))
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

    async def cmd_pending(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        ps = self.store.pending_proposals()
        if not ps:
            await self.reply(update, "Nothing pending.")
            return
        for p in ps:
            what = f"submit {code(p.script_path)}" if p.kind == "submit" else f"cancel job {code(p.target_job)}"
            await self.notifier.send(Outgoing(
                f"⏳ {code(p.id)}: {what}\n{esc(p.summary)}",
                session_id=p.session_id,
                buttons=[[("✅ Approve", f"approve:{p.id}"), ("❌ Reject", f"reject:{p.id}")]]))

    async def cmd_cancel(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.args:
            await self.reply(update, "Usage: /cancel &lt;job id&gt;")
            return
        job_id = ctx.args[0]
        rec = self.store.get_job(job_id)
        sid = (rec.session_id if rec else None) or self.agent.general_session().id
        session = self.agent.get(sid) or self.agent.general_session()
        try:
            await self.submitter.propose_cancel(session, job_id, "Requested via /cancel")
        except PolicyError as e:
            await self.reply(update, f"⛔ {esc(e)}")

    async def cmd_health(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        last = f"{int(time.time() - self.monitor.last_ok)}s ago" if self.monitor.last_ok else "never"
        err = f"\nlast error: {code(self.monitor.last_error[:300])}" if self.monitor.last_error else ""
        await self.reply(update, f"HPC user {code(self.hpc.user or '?')} · local state {code(self.cfg.local.state())}\n"
                                 f"last successful poll: {last}{err}\n"
                                 f"active agent sessions: {len(self.agent.sessions)}")

    async def cmd_access(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        def fmt(paths: list[str]) -> str:
            return "\n".join(f"• {code(p)}" + (" (granted here)" if p in self.hpc.grants else "") for p in paths) or "• (none)"
        await self.reply(update, f"✏️ <b>Writable</b>\n{fmt(self.hpc.write_roots)}\n\n📖 <b>Readable</b>\n{fmt(self.hpc.read_roots)}\n\n"
                                 "Config-file folders can only be changed in config.yaml.")

    async def cmd_grant(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        args = ctx.args or []
        mode = args[1] if len(args) > 1 else "write"
        if not args or mode not in ("read", "write"):
            await self.reply(update, "Usage: /grant &lt;path&gt; [write|read]")
            return
        try:
            await self.hpc.ensure_connected()
            await self.reply(update, esc(await self.submitter.grant(args[0], mode, "granted via /grant")))
        except (PolicyError, SSHError) as e:
            await self.reply(update, f"⛔ {esc(e)}")

    async def cmd_revoke(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.args:
            await self.reply(update, "Usage: /revoke &lt;path&gt;")
            return
        try:
            await self.reply(update, esc(await self.submitter.revoke(ctx.args[0])))
        except (PolicyError, SSHError) as e:
            await self.reply(update, f"⛔ {esc(e)}")

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
            if action in ("approve", "reject"):
                await q.edit_message_reply_markup(None)
                p = self.store.get_proposal(arg)
                result = await (self.submitter.approve(arg) if action == "approve" else self.submitter.reject(arg))
                p2 = self.store.get_proposal(arg)
                await self.notifier.send(Outgoing(esc(result), session_id=p.session_id if p else None,
                                                  job_id=(p2.job_id or None) if p2 else None))
                if p:
                    if p2 and p2.job_id:
                        self.store.link_message(chat.id, q.message.message_id, p2.job_id, p.session_id)
                    verdict = "approved" if action == "approve" else "rejected"
                    session = self.agent.get(p.session_id)
                    if p.kind == "access" and session and p2 and p2.status in ("done", "rejected"):
                        # The agent ended its turn waiting for this decision: resume it.
                        self._spawn(self._run_agent(chat.id, session,
                                                    f"(automatic) The user {verdict} your access request: {result} Continue."))
                    else:
                        self.agent.note(p.session_id, f"User {verdict} proposal {arg}: {result}")
            elif action == "revert":
                s = self.agent.get(arg)
                lines = s.changes.summary_lines() if s else []
                await self.notifier.send(Outgoing(
                    "↩️ Revert these changes?\n" + "\n".join(f"• {esc(x)}" for x in lines[:30]),
                    session_id=arg, buttons=[[("↩️ Yes, revert", f"revertok:{arg}"), ("Keep", "noop:")]]))
            elif action == "revertok":
                await q.edit_message_reply_markup(None)
                await self.notifier.send(Outgoing(esc(await self.agent.revert(arg)), session_id=arg))
            elif action == "noop":
                await q.edit_message_reply_markup(None)
            elif action == "fix":
                await self._start_fix(chat.id, arg)
            elif action == "log":
                await self._send_log(arg, 200)
            elif action == "status":
                await self._send_status(arg)
        except (SSHError, PolicyError) as e:
            await self.notifier.send(Outgoing(f"⚠️ {esc(e)}"))
