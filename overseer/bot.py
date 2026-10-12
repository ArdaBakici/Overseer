"""Telegram front-end. Only chats listed in telegram.allowed_chat_ids can talk to the bot.

Note: the agent has full access to your HPC account, so whoever controls an allowed chat
effectively has a shell there. Keep allowed_chat_ids to your own chat and protect your
Telegram account (2FA).
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, InputMediaPhoto, Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, EndPointNotFound
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from .agent import Agent, Session
from .autopilot import parse_duration
from .projects import ProjectError, Projects
from .richtext import RICH_LIMIT, strip_html
from .config import BACKENDS, Config
from .hpc import HPC
from .monitor import Monitor, progress_log
from .notify import TG_LIMIT, Outgoing, code, esc, pre, split_message
from .slurm import STATE_EMOJI
from .ssh import SSHError
from .sshlink import master_alive
from .store import ProjectRecord, Store

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
/update — update Claude Code and Codex now (Overseer also does it once a day, while nothing runs)
/compact [focus] — compact the conversation now: older turns become a summary, which frees context (as a reply: that conversation). Claude Code takes a focus, e.g. /compact keep the lr sweep results
/auto [wake-ups] [every 2h] [for 48h] &lt;goal&gt; — autopilot: the agent keeps experimenting on its own, woken by the jobs it watches (as a reply: in that conversation). The agent can also ask for it: you get a card to approve
/auto — autopilot progress and pending wake-ups · /auto off — stop it
/health — connection status

<b>Projects</b> (each gets its own topic; commands in a topic apply to that project)
/project — list · /project new &lt;name&gt; &lt;dir on HPC&gt; [what it's about]
/project context|dir|notes|archive|restore …
/assign &lt;job&gt; &lt;project|main&gt; — move a job (or reply to its message: /assign &lt;project&gt;)

Reply to any job message to talk to the agent about that job.
The agent has full access to your HPC account; you'll see every command it runs and every job it submits."""


def _thread_error(e: Exception) -> bool:
    text = str(e).lower()
    return "thread" in text or "topic" in text


class TelegramNotifier:
    """Sends to the allowed chats. Messages of a project go to its topic (when the bot has topics
    in private chats enabled); otherwise they are labelled with the project's name."""

    def __init__(self, app: Application, chat_ids: list[int], store: Store, rich: bool = True,
                 streaming: bool = True):
        self.app = app
        self.chat_ids = chat_ids
        self.store = store
        self.rich = rich  # native rich messages; switched off if this Telegram server doesn't have them
        self.streaming = streaming
        self.topics = False  # set by detect_topics()
        self._drafts: dict[int, str | None] = {}  # live draft id -> conversation
        self._recreated: dict[str, float] = {}  # project -> when its topic was last recreated

    async def detect_topics(self) -> bool:
        try:
            me = await self.app.bot.get_me()
            self.topics = bool(getattr(me, "has_topics_enabled", False))
        except Exception:
            log.warning("could not check whether the bot has topics enabled", exc_info=True)
        return self.topics

    def _conversation(self, msg: Outgoing):
        """The conversation a message belongs to: its own, or that of the agent that submitted its job."""
        sess = self.store.get_session(msg.session_id) if msg.session_id else None
        job = self.store.get_job(msg.job_id) if msg.job_id else None
        if sess is None and job and job.by_agent and job.session_id:
            sess = self.store.get_session(job.session_id)
        return sess, job

    def project_of(self, msg: Outgoing) -> ProjectRecord | None:
        sess, job = self._conversation(msg)
        name = msg.project or (sess.project if sess else None) or (job.project if job else None)
        return self.store.get_project(name) if name else None

    def _target(self, msg: Outgoing, chat_id: int) -> tuple[int | None, str, bool]:
        """(thread, label, is the conversation's own thread): a conversation's thread first, then
        its project's topic, then the main chat (labelled with the project if there is one)."""
        if self.topics and not msg.project:
            sess, _ = self._conversation(msg)
            thread = self.store.session_thread(sess.id, chat_id) if sess else None
            if thread:
                return thread, "", True
        thread, label = self._route(self.project_of(msg), chat_id)
        return thread, label, False

    async def ensure_topic(self, project: str, intro: str | None = None) -> None:
        """Create the project's topic in each chat that doesn't have one yet."""
        if not self.topics:
            return
        for chat_id in self.chat_ids:
            p = self.store.get_project(project)
            if not p or str(chat_id) in p.thread_map:
                continue
            await self._create_topic(p, chat_id, intro)

    async def _create_topic(self, p: ProjectRecord, chat_id: int, intro: str | None = None) -> int | None:
        try:
            topic = await self.app.bot.create_forum_topic(chat_id, name=p.name)
        except Exception:
            log.warning("could not create a topic for project %s", p.name, exc_info=True)
            return None
        p = self.store.get_project(p.name) or p
        p.set_thread(chat_id, topic.message_thread_id)
        self.store.upsert_project(p)
        text = intro or (f"📁 <b>{esc(p.name)}</b> · {code(p.dir)}\nMessages here go to this project's agent; "
                         "its jobs are reported here.")
        try:
            await self.app.bot.send_message(chat_id, text, parse_mode=ParseMode.HTML,
                                            message_thread_id=topic.message_thread_id, disable_notification=True)
        except Exception:
            log.debug("topic intro failed", exc_info=True)
        return topic.message_thread_id

    async def rename_topic(self, chat_id: int, thread_id: int, name: str) -> None:
        try:
            await self.app.bot.edit_forum_topic(chat_id, message_thread_id=thread_id, name=name[:128])
        except Exception:
            log.debug("could not rename topic %s", thread_id, exc_info=True)

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

    def general_thread(self, chat_id: int) -> int | None:
        """The "general" topic: everything that belongs to no project."""
        v = self.store.get(f"general_topic:{chat_id}")
        return int(v) if v else None

    async def ensure_general_topic(self) -> None:
        if not self.topics:
            return
        for chat_id in self.chat_ids:
            if not self.general_thread(chat_id):
                await self._create_general(chat_id)

    async def _create_general(self, chat_id: int) -> int | None:
        try:
            topic = await self.app.bot.create_forum_topic(chat_id, name="general")
        except Exception:
            log.warning("could not create the general topic", exc_info=True)
            return None
        self.store.set(f"general_topic:{chat_id}", str(topic.message_thread_id))
        try:
            await self.app.bot.send_message(
                chat_id, "💬 <b>general</b> - for anything that isn't about a project: questions, quick runs, your "
                "jobs outside every project's folder. Projects have their own topics; a new thread is a new "
                "conversation.", parse_mode=ParseMode.HTML, message_thread_id=topic.message_thread_id,
                disable_notification=True)
        except Exception:
            log.debug("general topic intro failed", exc_info=True)
        return topic.message_thread_id

    def _route(self, proj: ProjectRecord | None, chat_id: int) -> tuple[int | None, str]:
        """(topic to post in, label to prefix): a project without a topic gets labelled; no project means
        the general topic (or the main chat without topics)."""
        if not proj:
            return (self.general_thread(chat_id) if self.topics else None), ""
        thread = proj.thread_map.get(str(chat_id)) if self.topics else None
        return thread, "" if thread else f"📁 <b>{esc(proj.name)}</b> · "

    async def _send(self, msg: Outgoing) -> None:
        proj = self.project_of(msg)
        for chat_id in self.chat_ids:
            thread, label, own = self._target(msg, chat_id)
            try:
                await self._deliver_any(chat_id, thread, label, msg)
            except BadRequest as e:
                if not (thread and _thread_error(e)):
                    raise
                if own:  # the conversation's thread was deleted: continue in its project's topic / main chat
                    log.warning("thread %s is gone (%s)", thread, e)
                    self.store.unbind_thread(chat_id, thread)
                    thread, label = self._route(proj, chat_id)
                    try:
                        await self._deliver_any(chat_id, thread, label, msg)
                        continue
                    except BadRequest as e2:
                        if not (thread and _thread_error(e2)):
                            raise
                # The topic was deleted (or Telegram lost it): make a new one - at most once an hour, so an
                # outage on Telegram's side can't spawn a topic per message - else use the main chat.
                key = proj.name if proj else "\0general"
                log.warning("topic %s unavailable (%s)", proj.name if proj else "general", e)
                if time.time() - self._recreated.get(key, 0) > 3600:
                    self._recreated[key] = time.time()
                    new = await (self._create_topic(proj, chat_id) if proj else self._create_general(chat_id))
                    proj = (self.store.get_project(proj.name) or proj) if proj else None
                    if new:
                        try:
                            await self._deliver_any(chat_id, new, "", msg)
                            continue
                        except BadRequest:
                            pass
                await self._deliver_any(chat_id, None, f"📁 <b>{esc(proj.name)}</b> · " if proj else "", msg)

    async def _deliver_any(self, chat_id: int, thread: int | None, label: str, msg: Outgoing) -> None:
        """A rich message when there is one and Telegram takes it, else the regular formatted message."""
        if msg.markdown and self.rich:
            try:
                await self._deliver_rich(chat_id, thread, label + msg.markdown, msg)
                return
            except BadRequest as e:
                if thread and _thread_error(e):
                    raise
                log.warning("rich message rejected (%s); sending a regular message", e)
            except EndPointNotFound:
                log.warning("this Telegram server has no rich messages; using regular messages from now on")
                self.rich = False
            except ValueError as e:  # too long for one rich message
                log.info("%s; sending a regular message", e)
        await self._deliver(chat_id, thread, label + msg.text, msg)

    async def _deliver_rich(self, chat_id: int, thread: int | None, markdown: str, msg: Outgoing) -> None:
        """sendRichMessage (Bot API 10.1; python-telegram-bot doesn't wrap it yet). Pictures and files
        are embedded as media blocks, uploaded as attach:// parts of the same request."""
        uploads: dict[str, InputFile] = {}
        media, blocks = [], []
        for kind, items in (("photo", msg.photos), ("document", msg.files)):
            for i, (name, data) in enumerate(items):
                key = f"{kind[0]}{i}"
                uploads[f"file_{key}"] = InputFile(data, filename=name)
                media.append({"id": key, "media": {"type": kind, "media": f"attach://file_{key}"}})
                caption = re.sub(r'["\\\]\[]', "", name)
                blocks.append(f'![](tg://{kind}?id={key} "{caption}")')
        if blocks:
            markdown += "\n\n" + "\n\n".join(blocks)
        if len(markdown) > RICH_LIMIT:
            raise ValueError(f"message too long for a rich message ({len(markdown)} characters)")
        rich: dict = {"markdown": markdown}
        if media:
            rich["media"] = media
        kwargs: dict = {"chat_id": chat_id, "rich_message": rich, "disable_notification": msg.silent, **uploads}
        if thread:
            kwargs["message_thread_id"] = thread
        if msg.buttons:
            kwargs["reply_markup"] = InlineKeyboardMarkup(
                [[InlineKeyboardButton(t, callback_data=d) for t, d in row] for row in msg.buttons])
        sent = await self.app.bot.do_api_request("sendRichMessage", api_kwargs=kwargs, return_type=Message)
        self.store.link_message(chat_id, sent.message_id, msg.job_id, msg.session_id)

    async def _deliver(self, chat_id: int, thread: int | None, text: str, msg: Outgoing) -> None:
        markup = None
        if msg.buttons:
            markup = InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in row] for row in msg.buttons])
        chunks = split_message(text)
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            kw = dict(disable_notification=msg.silent, reply_markup=markup if last else None, message_thread_id=thread)
            try:
                sent = await self.app.bot.send_message(chat_id, chunk, parse_mode=ParseMode.HTML,
                                                       link_preview_options=None, **kw)
            except BadRequest as e:
                if thread and _thread_error(e):
                    raise
                # Telegram rejected the formatting: send the same text without it.
                log.warning("telegram rejected the formatting (%s); sending plain text", e)
                sent = await self.app.bot.send_message(chat_id, strip_html(chunk), **kw)
            self.store.link_message(chat_id, sent.message_id, msg.job_id, msg.session_id)
        for i in range(0, len(msg.photos), 10):  # an album holds up to 10 pictures
            await self._send_photos(chat_id, thread, msg.photos[i:i + 10], msg)
        for name, data in msg.files:
            sent = await self.app.bot.send_document(chat_id, document=data, filename=name, disable_notification=True,
                                                    message_thread_id=thread)
            self.store.link_message(chat_id, sent.message_id, msg.job_id, msg.session_id)

    async def _send_photos(self, chat_id: int, thread: int | None, photos: list[tuple[str, bytes]],
                           msg: Outgoing) -> None:
        bot = self.app.bot
        try:
            if len(photos) == 1:
                name, data = photos[0]
                sent = [await bot.send_photo(chat_id, photo=data, filename=name, caption=name,
                                             disable_notification=True, message_thread_id=thread)]
            else:
                media = [InputMediaPhoto(media=data, filename=name, caption=name) for name, data in photos]
                sent = list(await bot.send_media_group(chat_id, media=media, disable_notification=True,
                                                       message_thread_id=thread))
        except BadRequest as e:
            if thread and _thread_error(e):
                raise
            # Telegram refuses some images as pictures (e.g. extreme dimensions): send them as files.
            log.warning("pictures rejected (%s); sending them as files", e)
            sent = [await bot.send_document(chat_id, document=data, filename=name, disable_notification=True,
                                            message_thread_id=thread) for name, data in photos]
        for m in sent:
            self.store.link_message(chat_id, m.message_id, msg.job_id, msg.session_id)

    # ------------------------------------------------------------------ live progress
    # A streaming draft with a Stop button (Bot API 9.3+/10.3) in private chats; otherwise a message
    # that is edited in place. Drafts disappear by themselves when the reply is sent.
    async def post_live(self, msg: Outgoing) -> dict:
        if self.streaming and all(c > 0 for c in self.chat_ids):  # drafts only exist in private chats
            draft_id = secrets.randbelow(2**31 - 1) + 1
            targets = [(c, self._target(msg, c)[0]) for c in self.chat_ids]
            try:
                for chat_id, thread in targets:
                    await self._draft(chat_id, thread, draft_id, msg.text)
                self._drafts[draft_id] = msg.session_id
                return {"draft": draft_id, "targets": targets, "session": msg.session_id}
            except EndPointNotFound:
                log.warning("this Telegram server has no streaming drafts; using edited messages from now on")
                self.streaming = False
            except Exception as e:
                log.warning("streaming draft failed (%s); using an edited message", e)
        handle = []
        for chat_id in self.chat_ids:
            thread, label, _ = self._target(msg, chat_id)
            try:
                sent = await self.app.bot.send_message(chat_id, (label + msg.text)[:TG_LIMIT], parse_mode=ParseMode.HTML,
                                                       disable_notification=True, message_thread_id=thread)
            except Exception:
                log.debug("live message failed", exc_info=True)
                continue
            handle.append((chat_id, sent.message_id, label))
            self.store.link_message(chat_id, sent.message_id, msg.job_id, msg.session_id)
        return {"messages": handle}

    async def _draft(self, chat_id: int, thread: int | None, draft_id: int, text: str) -> None:
        # can_stop (the Stop button, Bot API 10.3) is newer than python-telegram-bot: passed as an extra field.
        await self.app.bot.send_message_draft(chat_id, draft_id, text=text[:TG_LIMIT], message_thread_id=thread,
                                              parse_mode=ParseMode.HTML, api_kwargs={"can_stop": True})

    async def edit_live(self, handle: dict, text: str) -> None:
        if "draft" in handle:  # same draft id: Telegram animates the change; refreshing also keeps it alive
            for chat_id, thread in handle["targets"]:
                try:
                    await self._draft(chat_id, thread, handle["draft"], text)
                except Exception:
                    log.debug("draft update failed", exc_info=True)
            return
        for chat_id, message_id, label in handle["messages"]:
            try:
                await self.app.bot.edit_message_text((label + text)[:TG_LIMIT], chat_id=chat_id, message_id=message_id,
                                                     parse_mode=ParseMode.HTML)
            except Exception:  # "message is not modified", rate limits: the next update catches up
                log.debug("live message edit failed", exc_info=True)

    async def delete_live(self, handle: dict) -> None:
        if "draft" in handle:  # a draft vanishes when the reply arrives (or after 30 s)
            self._drafts.pop(handle["draft"], None)
            return
        for chat_id, message_id, _ in handle["messages"]:
            try:
                await self.app.bot.delete_message(chat_id, message_id)
            except Exception:
                log.debug("live message delete failed", exc_info=True)

    def draft_session(self, draft_id: int) -> str | None:
        """The conversation whose progress draft this is (for the Stop button)."""
        return self._drafts.get(draft_id)


class Bot:
    def __init__(self, cfg: Config, app: Application, store: Store, hpc: HPC, monitor: Monitor,
                 agent: Agent, notifier: TelegramNotifier, projects: Projects | None = None):
        self.projects = projects or Projects(cfg, store, agent)
        self.cfg = cfg
        self.app = app
        self.store = store
        self.hpc = hpc
        self.monitor = monitor
        self.agent = agent
        self.notifier = notifier
        self._tasks: set[asyncio.Task] = set()
        self._pending: dict[tuple[int, int | None], str] = {}  # new thread -> its first message, until a project is picked

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
        add(CommandHandler("compact", self.cmd_compact, filters=allowed))
        add(CommandHandler("update", self.cmd_update, filters=allowed))
        add(CommandHandler("stop", self.cmd_stop, filters=allowed))
        add(CommandHandler("health", self.cmd_health, filters=allowed))
        add(CommandHandler("auto", self.cmd_auto, filters=allowed))
        add(CommandHandler(["project", "projects"], self.cmd_project, filters=allowed))
        add(CommandHandler("assign", self.cmd_assign, filters=allowed))
        # Updates python-telegram-bot doesn't know yet (e.g. the Stop button of a streaming draft).
        add(TypeHandler(Update, self.on_raw_update), group=-1)
        add(CallbackQueryHandler(self.on_button))
        add(MessageHandler(allowed & filters.TEXT & ~filters.COMMAND, self.on_text))
        add(MessageHandler(~allowed, self.on_stranger))

    # ------------------------------------------------------------------ helpers
    async def reply(self, update: Update, text: str) -> None:
        for chunk in split_message(text):
            try:
                await update.effective_message.reply_text(chunk, parse_mode=ParseMode.HTML)
            except BadRequest:
                await update.effective_message.reply_text(strip_html(chunk))

    def _thread(self, update: Update | None) -> int | None:
        msg = update.effective_message if update else None
        return msg.message_thread_id if msg is not None and getattr(msg, "is_topic_message", False) else None

    def _scope(self, update: Update | None) -> str | None:
        """The project of the topic this was sent in (its home topic or one of its threads); None = no project."""
        thread = self._thread(update)
        if not thread:
            return None
        p = self.store.project_for_thread(update.effective_chat.id, thread)
        if p:
            return p.name
        sess = self.agent.get(self.store.thread_session(update.effective_chat.id, thread))
        return sess.project if sess else None

    def _is_general(self, update: Update | None) -> bool:
        thread = self._thread(update)
        gen = self.notifier.general_thread(update.effective_chat.id) if hasattr(self.notifier, "general_thread") else None
        return bool(thread) and thread == gen

    def _here(self, update: Update | None) -> Session | None:
        """The conversation of the place a message was sent: its thread's, a project topic's, or the main
        chat's (= the general topic's). None for a new thread nobody has assigned yet."""
        thread = self._thread(update)
        if not thread or self._is_general(update):
            return self.agent.general_session()
        chat_id = update.effective_chat.id
        p = self.store.project_for_thread(chat_id, thread)
        if p:
            return self.agent.general_session(project=p.name)
        return self.agent.get(self.store.thread_session(chat_id, thread))

    def _where(self, scope: str | None) -> str:
        return f"project {code(scope)}" if scope else "the main chat"

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
        scope = self._scope(update)
        await self.reply(update, HELP + f"\n\nHere: {self._where(scope)} · agent {code(self.agent.backend_of(scope))}")

    async def cmd_jobs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        scope = self._scope(update)
        rows = self.store.project_jobs(scope, 15) if scope else self.store.recent_jobs(20)
        if not rows:
            await self.reply(update, "No jobs seen yet.")
            return
        lines = []
        for r in rows:
            tag = " 🤖" if r.by_agent else ""
            proj = f" · 📁{esc(r.project)}" if r.project and not scope else ""
            lines.append(f"{STATE_EMOJI.get(r.state, '•')} {code(r.job_id)} {esc(r.name)} — {esc(r.state)}{tag}{proj}")
        title = f"Jobs of {esc(scope)}" if scope else "Recent jobs"
        await self.reply(update, f"<b>{title}</b> (🤖 = submitted by the agent)\n" + "\n".join(lines))

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
        msg = await self.monitor.status_message(rec, live.get(job_id), gpu=True)
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
        log_path = await progress_log(self.hpc, rec)
        self.store.upsert_job(rec)
        out = await self.hpc.try_tail(rec.stdout, lines)
        files = [(f"job-{job_id}-stdout.log", out.encode())]
        if rec.stderr and rec.stderr != rec.stdout:
            files.append((f"job-{job_id}-stderr.log", (await self.hpc.try_tail(rec.stderr, lines)).encode()))
        if log_path:  # the program writes its real output here; SLURM's stdout only has the preamble
            main = await self.hpc.try_tail(log_path, lines)
            files.insert(0, (log_path.rsplit("/", 1)[-1], main.encode()))
            text = f"📜 Job {code(job_id)} {code(log_path)}" + pre(main, 3000) + f"\nSLURM stdout: {code(rec.stdout)}"
            files = files if lines > 60 else []
            await self.notifier.send(Outgoing(text, job_id=job_id, files=files))
            return
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
        await self.notifier.send(Outgoing(f"🔧 {esc(self.agent.backend_of(session.project))} is working on job {code(job_id)}…",
                                          job_id=job_id, session_id=session.id, silent=True))
        self._spawn(self._run_agent(chat_id, session, text))

    async def _start_plot(self, chat_id: int, job_id: str) -> None:
        """📈: the agent plots the job's metrics on the HPC and sends the picture."""
        rec = self.store.get_job(job_id)
        if not rec:
            await self.notifier.send(Outgoing(f"Unknown job {code(job_id)}."))
            return
        log_path = (rec.info_dict.get("progress_log") or rec.stdout) if rec else ""
        text = (f"Plot how job {job_id} ({rec.name}, {rec.state}) is doing. Find its metrics - its log {log_path}, or "
                "metric files (metrics.jsonl, CSV, TensorBoard events) in its output folder - and make one clear PNG "
                "with a plotting script run on a compute node (hpc-run): the main metric(s) against step or time, a "
                "few panels if useful, the job id in the title. Copy it to ./outbox/ and describe the trend in two "
                "or three sentences.")
        session = self.agent.session_for_job(job_id)
        await self.notifier.send(Outgoing(f"📈 {esc(self.agent.backend_of(session.project))} is plotting job "
                                          f"{code(job_id)}…", job_id=job_id, session_id=session.id, silent=True))
        self._spawn(self._run_agent(chat_id, session, text))

    async def cmd_ask(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        text = " ".join(ctx.args or [])
        if not text:
            await self.reply(update, "Usage: /ask &lt;question&gt;")
            return
        session = self._here(update)
        if session is None:
            await self._new_thread(update, text)
            return
        self._spawn(self._run_agent(update.effective_chat.id, session, text))

    async def cmd_new(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        scope, thread = self._scope(update), self._thread(update)
        if thread and not self._is_general(update) and not self.store.project_for_thread(update.effective_chat.id, thread):
            s = self.agent.new_conversation(scope)  # a thread of its own: it gets a fresh conversation
            self.store.bind_thread(update.effective_chat.id, thread, s.id)
        else:
            s = self.agent.general_session(reset=True, project=scope)
        await self.reply(update, f"Started a fresh conversation ({code(s.id)}) in {self._where(scope)}. "
                                 "(Or start a new thread: each thread is its own conversation.)")

    def _agents_summary(self, scope: str | None = None) -> str:
        lines = []
        for b in BACKENDS:
            mark = "▶️" if b == self.agent.backend_of(scope) else "▫️"
            lines.append(f"{mark} {esc(self.agent.describe(b, scope))}")
        return "\n".join(lines)

    async def cmd_backend(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        scope = self._scope(update)
        if not ctx.args:
            names = ", ".join(code(b) for b in BACKENDS)
            await self.reply(update, f"Agents in {self._where(scope)}:\n{self._agents_summary(scope)}\n\nSwitch with "
                                     f"/backend &lt;name&gt; ({names}); change the model / effort with /model and /effort.")
            return
        try:
            self.agent.set_backend(ctx.args[0], scope)
        except ValueError as e:
            await self.reply(update, f"⛔ {esc(e)}")
            return
        await self.reply(update, f"Agent in {self._where(scope)} is now {code(self.agent.backend_of(scope))}. Its "
                                 "conversations continue with a fresh session on this backend (the job context is "
                                 "re-sent).\n\n" + self._agents_summary(scope))

    async def cmd_model(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        scope = self._scope(update)
        b = self.agent.backend_of(scope)
        if not ctx.args:
            examples = "opus, sonnet, claude-opus-5-5" if b == "claude_code" else "gpt-5.5-codex"
            await self.reply(update, f"{esc(self.agent.describe(b, scope))} ({self._where(scope)})\n\nSet with "
                                     f"/model &lt;name&gt; (e.g. {examples}), or /model default to use config.yaml / "
                                     "the CLI default.")
            return
        try:
            self.agent.set_model(ctx.args[0], project=scope)
        except ValueError as e:
            await self.reply(update, f"⛔ {esc(e)}")
            return
        await self.reply(update, f"{esc(self.agent.describe(b, scope))} ({self._where(scope)})")

    async def cmd_effort(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        scope = self._scope(update)
        b = self.agent.backend_of(scope)
        levels = " | ".join(self.agent.runner(b, scope).efforts)
        if not ctx.args:
            await self.reply(update, f"{esc(self.agent.describe(b, scope))} ({self._where(scope)})\n\nSet with "
                                     f"/effort &lt;level&gt; ({esc(levels)}), or /effort default.")
            return
        try:
            self.agent.set_effort(ctx.args[0], project=scope)
        except ValueError as e:
            await self.reply(update, f"⛔ {esc(e)}")
            return
        await self.reply(update, f"{esc(self.agent.describe(b, scope))} ({self._where(scope)})")

    async def cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        scope = self._scope(update)
        running = [r for r in self.agent.running_sessions() if scope is None or r.project == scope]
        if not running:
            await self.reply(update, "Nothing is running.")
            return
        for s in running:
            await self.agent.stop(s.id)

    async def cmd_compact(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        session = self._replied_session(update) or self._here(update)
        if session is None:
            await self.reply(update, "No conversation here to compact (reply to one of its messages, or use it in its "
                                     "thread).")
            return
        self._spawn(self.agent.compact(session, " ".join(ctx.args or []).strip()))

    async def cmd_update(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if self.agent.running_sessions():
            await self.reply(update, "The agent is working: /update replaces the CLI it runs, so wait until it's done "
                                     "(or /stop it).")
            return
        await self.reply(update, "⬆️ Updating the agent CLIs…")
        lines = await self.agent.update_clis()
        if lines is None:
            await self.reply(update, "An agent run started meanwhile, or an update is already running: try again later.")
        else:
            await self.reply(update, "\n".join(lines) or "No agent CLI is installed.")

    async def cmd_health(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        last = f"{int(time.time() - self.monitor.last_ok)}s ago" if self.monitor.last_ok else "never"
        err = f"\nlast error: {code(self.monitor.last_error[:300])}" if self.monitor.last_error else ""
        conn = "up" if await master_alive(self.cfg) else "down (reopens on the next poll)"
        running = len(self.agent.running_sessions())
        auto = self.store.list_autopilots(active_only=True)
        await self.reply(update, f"HPC user {code(self.hpc.user or '?')} · SSH connection {conn}\n"
                                 f"last successful poll: {last}{err}\n"
                                 f"agent runs in progress: {running} · autopilots on: {len(auto)}\n"
                                 f"{self._worker_line()}\n"
                                 f"{await self._versions_line()}\n"
                                 f"projects: {len(self.store.list_projects())} · topics "
                                 f"{'on' if getattr(self.notifier, 'topics', False) else 'off (enable threaded mode in BotFather)'}"
                                 f"\n\n{self._agents_summary(self._scope(update))}")

    async def _versions_line(self) -> str:
        parts = []
        for backend, name in (("claude_code", "Claude Code"), ("codex", "Codex")):
            r = self.agent.runner(backend)
            if r.installed():
                parts.append(f"{name} {esc(await r.version() or '?')}")
        return "CLIs: " + (" · ".join(parts) or "none installed") + " (/update)"

    def _worker_line(self) -> str:
        w = getattr(self.monitor, "worker", None)
        if not w:
            return "compute worker: none (hpc-run starts one when needed)"
        where = f"on {esc(w.reason)}" if w.state == "RUNNING" else f"{esc(w.state)} ({esc(w.reason)})"
        return f"compute worker: job {code(w.job_id)} {where} · {esc(w.elapsed)} of {esc(w.time_limit)}"

    # ------------------------------------------------------------------ free text & buttons
    def _replied_session(self, update: Update | None) -> Session | None:
        """The conversation of the message this one replies to (a job message counts as its job's)."""
        msg = update.effective_message if update else None
        if not msg or not msg.reply_to_message:
            return None
        job_id, sid = self.store.message_link(update.effective_chat.id, msg.reply_to_message.message_id)
        session = self.agent.get(sid)
        if session is None and job_id:
            session = self.agent.session_for_job(job_id)
        return session

    async def on_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        text = update.effective_message.text
        session = self._replied_session(update) or self._here(update)
        if session is None:  # a thread the user just started
            await self._new_thread(update, text)
            return
        proj = self.store.get_project(session.project)
        if proj and proj.archived:
            await self.reply(update, f"Project {code(proj.name)} is archived. /project restore {esc(proj.name)} "
                                     "brings it back.")
            return
        self._spawn(self._run_agent(update.effective_chat.id, session, text))

    # ------------------------------------------------------------------ threads
    async def _new_thread(self, update: Update, text: str) -> None:
        """Each Telegram thread is its own conversation. A new one asks which project it's about
        (unless the message starts with "<project>: …" or there are no projects)."""
        chat_id, thread = update.effective_chat.id, self._thread(update)
        projects = self.store.list_projects()
        m = re.match(r"^\s*([a-z0-9][a-z0-9_-]{0,31})\s*:\s*(\S.*)$", text or "", re.S | re.I)
        if m and self.store.get_project(m.group(1).lower()) and not self.store.get_project(m.group(1).lower()).archived:
            await self._start_thread(chat_id, thread, m.group(1).lower(), m.group(2))
            return
        if not projects:
            await self._start_thread(chat_id, thread, None, text)
            return
        self._pending[(chat_id, thread)] = text
        buttons = [(f"📁 {p.name}", f"thread:{p.name}") for p in projects[:8]] + [("💬 No project", "thread:")]
        rows = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
        markup = InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in row] for row in rows])
        await update.effective_message.reply_text("New conversation. Which project is it about?", reply_markup=markup)

    async def _start_thread(self, chat_id: int, thread: int | None, project: str | None, text: str) -> None:
        session = self.agent.new_conversation(project)
        if thread:
            self.store.bind_thread(chat_id, thread, session.id)
            if project and hasattr(self.notifier, "rename_topic"):  # Telegram named it after the first message
                await self.notifier.rename_topic(chat_id, thread, f"{project} · {' '.join(text.split())[:60]}")
        if text and text.strip():
            self._spawn(self._run_agent(chat_id, session, text))

    async def cmd_auto(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        ap = self.agent.autopilot
        if ap is None:
            await self.reply(update, "Autopilot isn't available.")
            return
        args = list(ctx.args or [])
        target = self._replied_session(update)
        if not args or args[0].lower() in ("status", "list"):
            await self.reply(update, ap.describe())
            return
        scope = self._scope(update)
        if args[0].lower() in ("off", "stop"):
            if target or scope is None:
                stopped = ap.stop(target.id if target else None)
            else:  # in a project's topic: that project's autopilots
                stopped = [x for a in self.store.list_autopilots(True)
                           if (sess := self.store.get_session(a.session_id)) and sess.project == scope
                           for x in ap.stop(a.session_id)]
            await self.reply(update, f"⏹ Autopilot off for {', '.join(code(x) for x in stopped)}. Jobs keep running "
                                     "and are still reported." if stopped else "Autopilot wasn't on.")
            return
        wakeups, every, hours, i = None, None, None, 0
        if args[0].isdigit():
            wakeups, i = int(args[0]), 1
        while i < len(args):
            word = args[i].lower()
            if "=" in word and word.split("=", 1)[0] in ("every", "checkin", "for"):
                key, value, step = *word.split("=", 1), 1
            elif word in ("every", "checkin", "for") and i + 1 < len(args):
                key, value, step = word, args[i + 1], 2
            else:
                break
            d = parse_duration(value)
            if d is None or (key == "for" and d <= 0):
                await self.reply(update, f"⛔ {code(value)} isn't a duration: use e.g. 30m, 2h, 1d"
                                         + (" or off." if key != "for" else "."))
                return
            if key == "for":
                hours = d / 3600
            else:
                every = d
            i += step
        goal = " ".join(args[i:]).strip()
        if wakeups is None and every is None and hours is None and not goal:
            await self.reply(update, "Usage: /auto [wake-ups] [every 2h] [for 48h] &lt;goal&gt;, /auto off, or /auto for "
                                     "progress.")
            return
        session = target or self._here(update)
        if session is None:
            session = self.agent.new_conversation(scope)
            if self._thread(update):
                self.store.bind_thread(update.effective_chat.id, self._thread(update), session.id)
        if wakeups is None and hours is None and not goal and ap.active(session.id):  # only the check-ins change
            await self.reply(update, ap.set_checkins(session.id, every))
            return
        text = await ap.start(session, wakeups or self.cfg.agent.autopilot.default_wakeups, goal, every, hours)
        await self.notifier.send(Outgoing(text, session_id=session.id, job_id=session.job_id))

    # ------------------------------------------------------------------ projects
    def _projects_summary(self) -> str:
        rows = self.store.list_projects(include_archived=True)
        if not rows:
            return ("No projects yet. /project new &lt;name&gt; &lt;folder on the HPC&gt; [what it's about] creates one "
                    "with its own topic, conversations and jobs (or add them under projects: in config.yaml).")
        lines = []
        for p in rows:
            if p.archived:
                lines.append(f"🗄 {code(p.name)} (archived) · {code(p.dir)}")
                continue
            running = [j for j in self.store.project_jobs(p.name, 50) if not j.ended_at]
            auto = [a for a in self.store.list_autopilots(True)
                    if (sess := self.store.get_session(a.session_id)) and sess.project == p.name]
            b = self.agent.backend_of(p.name)
            r = self.agent.runner(b, p.name)
            bits = [f"📁 <b>{esc(p.name)}</b> · {code(p.dir)}", f"{esc(b)} {esc(r.model or 'default model')}"]
            if running:
                bits.append(f"{len(running)} job(s) active")
            if auto:
                bits.append("🔁 autopilot on")
            lines.append(" · ".join(bits) + (f"\n    {esc(p.context[:160])}" if p.context else ""))
        return "\n".join(lines)

    async def cmd_project(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        args = list(ctx.args or [])
        scope = self._scope(update)
        sub = args[0].lower() if args else "list"
        try:
            if sub in ("list", "ls"):
                await self.reply(update, self._projects_summary())
            elif sub == "new":
                if len(args) < 3:
                    await self.reply(update, "Usage: /project new &lt;name&gt; &lt;folder on the HPC&gt; [what it's about]")
                    return
                await self._new_project(update, args[1], args[2], " ".join(args[3:]))
            elif sub in ("context", "dir", "notes", "archive", "restore"):
                name = scope
                rest = args[1:]
                if (sub in ("archive", "restore", "notes") and rest) or (not scope and rest):
                    name, rest = rest[0].lower(), rest[1:]
                if not name:
                    await self.reply(update, f"Which project? Use it in the project's topic, or /project {sub} &lt;name&gt; …")
                    return
                await self._project_sub(update, sub, name, " ".join(rest))
            else:
                await self.reply(update, "Usage: /project [list] · new &lt;name&gt; &lt;dir&gt; [about] · context [text] · "
                                         "dir &lt;path&gt; · notes · archive · restore &lt;name&gt;")
        except ProjectError as e:
            await self.reply(update, f"⛔ {esc(e)}")

    async def _new_project(self, update: Update, name: str, folder: str, context: str) -> None:
        thread = self._thread(update)
        if thread and self._scope(update):
            raise ProjectError(f"this topic already belongs to project {self._scope(update)}")
        existing = self.store.get_project(name.strip().lower())
        p = self.projects.create(name, folder, context, "telegram")
        chat_id = update.effective_chat.id if update else None
        if thread and chat_id is not None:  # sent inside an existing topic: that topic becomes the project's
            p.set_thread(chat_id, thread)
            self.store.upsert_project(p)
        if hasattr(self.notifier, "ensure_topic"):
            await self.notifier.ensure_topic(p.name)
        verb = "restored" if existing else "created"
        settings = self._agents_summary(p.name)
        where = "this topic" if thread else ("its topic" if self.store.get_project(p.name).thread_map else "the main chat")
        await self.reply(update, f"📁 Project {code(p.name)} {verb}: {code(p.dir)}. Messages in {where} go to its agent; "
                                 f"jobs running under that folder are reported there.\nAgent (copied from the main "
                                 f"chat; change it in the project with /backend, /model, /effort):\n{settings}")

    async def _project_sub(self, update: Update, sub: str, name: str, rest: str) -> None:
        p = self.store.get_project(name)
        if not p:
            raise ProjectError(f"no project {name!r}")
        if sub == "context":
            if rest:
                p = self.projects.update(name, context=rest)
                await self.reply(update, f"Context of {code(name)} updated. New conversations get it; ongoing ones from "
                                         "their next message.")
            else:
                await self.reply(update, f"Context of {code(name)}:\n{esc(p.context) or '(none)'}\n\nSet it with "
                                         "/project context &lt;text&gt;.")
        elif sub == "dir":
            if not rest:
                await self.reply(update, f"{code(name)} lives in {code(p.dir)}.")
                return
            p = self.projects.update(name, dir=rest)
            await self.reply(update, f"{code(name)} now lives in {code(p.dir)}.")
        elif sub == "notes":
            path = f"{p.dir.rstrip('/')}/OVERSEER.md"
            notes = await self.hpc.try_head(self.agent._remote_path(path), 200)
            await self.reply(update, f"📝 {code(path)}" + (pre(notes, 3500) if notes.strip() else
                                     "\n(empty - the agent creates it when it learns something worth keeping)"))
        elif sub == "archive":
            self.projects.archive(name)
            await self.reply(update, f"🗄 Project {code(name)} archived: its autopilot is off and new jobs in its folder "
                                     f"go to the main chat. /project restore {esc(name)} brings it back.")
        elif sub == "restore":
            if not p.archived:
                raise ProjectError(f"{name!r} isn't archived")
            await self._new_project(update, name, p.dir, p.context)

    async def cmd_assign(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        args = list(ctx.args or [])
        job_id = None
        msg = update.effective_message if update else None
        if msg is not None and msg.reply_to_message:
            job_id, _ = self.store.message_link(update.effective_chat.id, msg.reply_to_message.message_id)
        if args and args[0].isdigit():
            job_id, args = args[0], args[1:]
        thread = self._thread(update)
        conv = self.agent.get(self.store.thread_session(update.effective_chat.id, thread)) if thread else None
        if job_id and not args and conv:  # in a conversation's thread: the job becomes part of it
            self.agent._record_job(conv, job_id)
            ap = self.agent.autopilot
            auto = ap and ap.active(conv.id)
            rec = self.store.get_job(job_id)
            if auto and rec and not rec.ended_at:  # part of the loop: its end wakes the agent
                ap.add_watch(conv, [job_id])
            await self.reply(update, f"🔗 Job {code(job_id)} is now part of this conversation: its reports come here"
                                     + (" and autopilot wakes the agent when it ends." if auto else "."))
            return
        target = args[0].lower() if args else self._scope(update)
        if not job_id or not target:
            await self.reply(update, "Usage: /assign &lt;job id&gt; &lt;project|main&gt; - or reply to a job's message "
                                     "with /assign &lt;project|main&gt;. In a conversation's thread, /assign &lt;job id&gt; "
                                     "makes the job part of that conversation (with autopilot on, its end wakes the agent).")
            return
        await self._assign(job_id, target, update)

    async def _assign(self, job_id: str, target: str | None, update: Update | None) -> None:
        project = None if target in (None, "main", "none") else target
        p = self.store.get_project(project) if project else None
        if project and (not p or p.archived):
            await self.reply(update, f"⛔ No active project {code(project)}. /project lists them.")
            return
        rec = self.store.get_job(job_id)
        if not rec:
            await self.reply(update, f"⛔ Unknown job {code(job_id)}.")
            return
        old = rec.project
        if old == project:
            await self.reply(update, f"Job {code(job_id)} is already in {self._where(project)}.")
            return
        self.agent.move_job(job_id, project)
        await self.reply(update, f"📁 Job {code(job_id)} moved to {self._where(project)}.")
        # A card in its new place, so its updates and replies continue there.
        await self.notifier.send(Outgoing(
            f"📁 Job {code(job_id)} {esc(rec.name)} ({esc(rec.state)}) moved here from {self._where(old)}.",
            job_id=job_id, silent=True,
            buttons=[[("📜 Log", f"log:{job_id}"), ("📊 Status", f"status:{job_id}")]]))

    async def on_raw_update(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        stopped = (update.api_kwargs or {}).get("stopped_message_generation")
        if not stopped:
            return
        if (stopped.get("chat") or {}).get("id") not in self.cfg.telegram.allowed_chat_ids:
            return
        sid = self.notifier.draft_session(stopped.get("draft_id")) if hasattr(self.notifier, "draft_session") else None
        if sid and await self.agent.stop(sid):
            log.info("stopped %s from the draft's Stop button", sid)

    async def _edit(self, q, text: str, buttons: list[list[tuple[str, str]]] | None = None) -> None:
        markup = InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in row]
                                       for row in buttons]) if buttons else None
        try:
            await q.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=markup)
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                log.warning("could not edit the message (%s)", e)

    async def _approval(self, q, what: str, sid: str) -> None:
        ap = self.agent.autopilot
        a = self.store.get_autopilot(sid) if ap else None
        if not a or a.status != "requested":
            await self._edit(q, "This request is no longer open." + (f" (autopilot: {esc(a.status)})" if a else ""))
            return
        if what == "ok":
            text = await ap.approve(sid)
            await self._edit(q, "✅ <b>Approved.</b>\n" + (text or ""))
        elif what == "no":
            ap.decline(sid)
            await self._edit(q, f"❌ <b>Declined</b>: {esc(a.goal)}\nThe agent is told the next time you write to it.")
        else:
            a = ap.adjust(sid, what)
            if a:
                await self._edit(q, *ap.card(a))

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
            elif action == "thread":
                thread = getattr(q.message, "message_thread_id", None) if q.message else None
                text = self._pending.pop((chat.id, thread), "")
                project = arg or None
                if project and not self.store.get_project(project):
                    project = None
                try:
                    await q.edit_message_text(f"📁 {project}" if project else "💬 No project")
                except Exception:
                    pass
                await self._start_thread(chat.id, thread, project, text)
            elif action == "plot":
                await self._start_plot(chat.id, arg)
            elif action == "assign":
                job_id, _, name = arg.partition(":")
                await self._assign(job_id, name or None, update)
            elif action == "ap":  # the autopilot approval card
                await self._approval(q, *arg.partition(":")[::2])
            elif action == "uw":  # ✖ on a one-off wake-up
                w = self.agent.autopilot.cancel_watch(int(arg)) if self.agent.autopilot and arg.isdigit() else None
                await self._edit(q, "✖ Cancelled: the agent won't be woken for this." if w else
                                 "This wake-up has already happened or was cancelled.")
        except SSHError as e:
            await self.notifier.send(Outgoing(f"⚠️ {esc(e)}"))
