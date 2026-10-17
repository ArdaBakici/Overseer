"""Telegram front-end. Only chats listed in telegram.allowed_chat_ids can talk to the bot.

Note: the agent has full access to your HPC account, so whoever controls an allowed chat
effectively has a shell there. Keep allowed_chat_ids to your own chat and protect your
Telegram account (2FA).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from pathlib import Path

from telegram import (InlineKeyboardButton, InlineKeyboardMarkup, InputFile, InputMediaPhoto, Message, ReplyParameters,
                      Update)
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
from .projects import ProjectError, Projects
from .richtext import RICH_LIMIT, strip_html
from .config import BACKENDS, Config
from ..hpc.remote import HPC
from ..hpc.monitor import Monitor, progress_log
from .notify import TG_LIMIT, Outgoing, code, esc, pre, split_message
from ..hpc.slurm import STATE_EMOJI
from ..hpc.ssh import SSHError
from ..hpc.sshlink import master_alive
from .store import ProjectRecord, Store
from ..hpc import results as R
from ..hpc.plots import resolve_plot
from .subagents import ENDED
from .usage import usage_report

log = logging.getLogger(__name__)

CAPTION_LIMIT = 1000  # Telegram: a photo's caption holds 1024 characters

# Files you send the agent: a file without a caption waits this long for your message about it; with a caption,
# only for the rest of its album (whose pictures arrive as separate messages).
UPLOAD_WAIT = 30
ALBUM_WAIT = 2
MAX_DOWNLOAD = 20 * 1024 * 1024  # the largest file the Bot API lets bots download

# /auto <goal>: what the agent is asked
AUTO_PROMPT = """Work on this on your own, over as many runs as it takes: {goal}

Keep the research log up to date (in a project). Submit the runs, `ov watch` them and end your reply: you're woken \
when they finish (or set `ov wake` timers). Carry on until the goal is reached or nothing sensible is left, then \
tell me. Ask me when you need a decision."""

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
/auto &lt;goal&gt; — ask the agent to work on it on its own: it watches its jobs and wakes itself (as a reply: in that conversation)
/auto — pending wake-ups · /auto off — cancel them (here, or as a reply: in that conversation)
/results [project] [sort] [N] [csv] — experiments and what each showed · /findings — insights and dead ends
/feedback [all|review] — what agents reported about Overseer; review: an agent designs fixes · /skills — installed skills
/subagents [on|off] — planner mode: the agent plans and delegates to cheaper subagents, each in a thread
/usage [today|yesterday|week] — tokens and cost of the agent's runs
/cluster [partition] — free GPUs and CPUs per partition, when your queued jobs should start, your fairshare
/digest — what happened since the last digest (one comes every morning)
/search &lt;text&gt; — past conversations, results, findings and jobs
/dashboard — jobs, results, findings and a timeline in a Mini App (on your tailnet)
/health — connection status

<b>Projects</b> (each gets its own topic; commands in a topic apply to that project)
/project — list · /project new &lt;name&gt; &lt;dir on HPC&gt; [what it's about]
/project context|dir|notes|archive|restore …
/assign &lt;job&gt; &lt;project|main&gt; — move a job (or reply to its message: /assign &lt;project&gt;)

Reply to any job message to talk to the agent about that job.
Send pictures and files too (up to 20 MB): with a caption, or followed by a message about them.
The agent has full access to your HPC account; you'll see every command it runs and every job it submits."""


def _preview(rec: dict, limit: int = 120) -> str:
    """The start of a cut-off run's request, on one line (a wake-up: its header, not the reports under it)."""
    text = " ".join((rec.get("text") or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


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
        its project's topic, then the main chat (labelled with the project if there is one).
        Job news (msg.jobs) goes to the jobs topic of the job's project instead."""
        if msg.jobs:
            proj = self.project_of(msg)
            if self.topics:
                return self.jobs_thread(chat_id, proj.name if proj else None), "", False
            return None, (f"📁 <b>{esc(proj.name)}</b> · " if proj else ""), False
        if msg.topic == "feedback":
            return (self.feedback_thread(chat_id) if self.topics else None), ("" if self.topics else "🛠 "), False
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

    # ------------------------------------------------------------------ a message edited later; subagent threads
    async def post(self, msg: Outgoing) -> dict[str, int]:
        """Send a short message (with buttons) that is edited later (`edit`): {chat id: message id}. Never raises."""
        markup = InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in row]
                                       for row in msg.buttons]) if msg.buttons else None
        refs: dict[str, int] = {}
        for chat_id in self.chat_ids:
            thread, label, _ = self._target(msg, chat_id)
            try:
                sent = await self.app.bot.send_message(chat_id, (label + msg.text)[:TG_LIMIT], parse_mode=ParseMode.HTML,
                                                       message_thread_id=thread, reply_markup=markup,
                                                       disable_notification=msg.silent)
            except Exception:
                log.warning("could not send a message", exc_info=True)
                continue
            refs[str(chat_id)] = sent.message_id
            self.store.link_message(chat_id, sent.message_id, msg.job_id, msg.session_id)
        return refs

    async def edit(self, refs: dict[str, int], text: str, buttons: list[list[tuple[str, str]]] | None = None) -> None:
        """Replace a posted message's text (and its buttons: none unless given). Never raises."""
        markup = InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in row]
                                       for row in buttons]) if buttons else None
        for chat_id, mid in refs.items():
            try:
                await self.app.bot.edit_message_text(text[:TG_LIMIT], chat_id=int(chat_id), message_id=mid,
                                                     parse_mode=ParseMode.HTML, reply_markup=markup)
            except Exception:
                log.debug("could not edit message %s", mid, exc_info=True)

    async def open_thread(self, session_id: str, name: str) -> None:
        """A thread of its own for a conversation (a subagent). Without topics its messages go to its project's."""
        if not self.topics:
            return
        for chat_id in self.chat_ids:
            try:
                topic = await self.app.bot.create_forum_topic(chat_id, name=name[:128])
            except Exception:
                log.warning("could not create a thread for %s", session_id, exc_info=True)
                continue
            self.store.bind_thread(chat_id, topic.message_thread_id, session_id)

    async def close_thread(self, session_id: str, done_name: str) -> bool:
        """Delete a conversation's threads (a finished subagent's). Where Telegram refuses, the thread is renamed
        (e.g. "✅ cosine (done)") instead. True if all were deleted."""
        deleted = True
        for chat_id, thread in self.store.session_threads(session_id):
            try:
                await self.app.bot.delete_forum_topic(chat_id, thread)
                self.store.unbind_thread(chat_id, thread)
            except Exception:
                log.info("could not delete thread %s; renaming it", thread, exc_info=True)
                deleted = False
                await self.rename_topic(chat_id, thread, done_name)
        return deleted

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

    # ------------------------------------------------------------------ jobs topics and live job cards
    def jobs_thread(self, chat_id: int, project: str | None) -> int | None:
        """The topic for job cards and reports: one per project ("📊 dreamer · jobs") and one for the rest."""
        v = self.store.get(f"jobs_topic:{chat_id}:{project or ''}")
        return int(v) if v else None

    def jobs_topic_owner(self, chat_id: int, thread: int | None) -> tuple[bool, str | None]:
        """(is it a jobs topic, of which project): writing there talks to that project's agent."""
        if thread:
            for project in [None, *(p.name for p in self.store.list_projects(include_archived=True))]:
                if self.jobs_thread(chat_id, project) == thread:
                    return True, project
        return False, None

    async def ensure_jobs_topics(self, projects: list[str]) -> None:
        if not self.topics:
            return
        for chat_id in self.chat_ids:
            for project in [None, *projects]:
                if not self.jobs_thread(chat_id, project):
                    await self._create_jobs_topic(chat_id, project)

    async def _create_jobs_topic(self, chat_id: int, project: str | None) -> int | None:
        name = f"📊 {project} · jobs" if project else "📊 jobs"
        try:
            topic = await self.app.bot.create_forum_topic(chat_id, name=name[:128])
        except Exception:
            log.warning("could not create the jobs topic %r", name, exc_info=True)
            return None
        self.store.set(f"jobs_topic:{chat_id}:{project or ''}", str(topic.message_thread_id))
        intro = (f"📊 <b>{esc(project)} · jobs</b> - a live card per job of this project, updated in place, and a "
                 "short report when it ends." if project else
                 "📊 <b>jobs</b> - a live card per job outside every project, updated in place, and a short report "
                 "when it ends.")
        try:
            await self.app.bot.send_message(chat_id, intro, parse_mode=ParseMode.HTML,
                                            message_thread_id=topic.message_thread_id, disable_notification=True)
        except Exception:
            log.debug("jobs topic intro failed", exc_info=True)
        return topic.message_thread_id

    async def _jobs_topic_for(self, msg: Outgoing, chat_id: int) -> None:
        if msg.jobs and self.topics:
            proj = self.project_of(msg)
            if not self.jobs_thread(chat_id, proj.name if proj else None):
                await self._create_jobs_topic(chat_id, proj.name if proj else None)

    # ------------------------------------------------------------------ a job's card: text, or a photo with its graphs
    # A job's card is a text message until the agent's plotting command produced graphs (plots.py). Then it becomes a
    # photo message - the graph, the card's text as its caption, the buttons - and stays one: status refreshes edit the
    # caption, new graphs replace the picture, and ◀ n/N ▶ pages through the graphs, all in place.
    def _markup(self, buttons):
        return InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in row]
                                     for row in buttons]) if buttons else None

    def _card_state(self, job_id: str) -> dict:
        try:
            st = json.loads(self.store.get(f"cardstate:{job_id}") or "{}")
            return st if isinstance(st, dict) else {}
        except ValueError:
            return {}

    def _set_card_state(self, job_id: str, **kw) -> None:
        self.store.set(f"cardstate:{job_id}", json.dumps({**self._card_state(job_id), **kw}))

    def _view(self, job_id: str) -> tuple[str, list]:
        try:
            text, buttons = json.loads(self.store.get(f"cardview:{job_id}") or "null")
            return text, buttons
        except (ValueError, TypeError):
            return "", []

    def _photo_buttons(self, job_id: str, buttons: list) -> list:
        st = self._card_state(job_id)
        n, i = int(st.get("pages", 1)), int(st.get("page", 0))
        if n <= 1:
            return buttons
        nav = [("◀", f"pg:{job_id}:{(i - 1) % n}"), (f"📈 {i + 1}/{n}", "noop:"), ("▶", f"pg:{job_id}:{(i + 1) % n}")]
        return [nav, *buttons]

    @staticmethod
    def _caption(text: str, limit: int = CAPTION_LIMIT) -> str:
        """The card's text as a photo caption (Telegram allows 1024 characters): whole lines are dropped from the end,
        GPU and alert lines first, until it fits."""
        lines = text.split("\n")
        while len(strip_html("\n".join(lines))) > limit and len(lines) > 1:
            drop = next((k for k in range(len(lines) - 1, 0, -1) if lines[k].startswith(("🎮", "🚨"))), len(lines) - 1)
            lines.pop(drop)
        return "\n".join(lines)

    async def card(self, job_id: str, text: str, buttons: list[list[tuple[str, str]]]) -> None:
        """Show a job's live card: edit it where it is, or post it (in the jobs topic) if there's none yet or it's
        gone. Always silent. Never raises."""
        if not self.store.get_job(job_id):
            return
        self.store.set(f"cardview:{job_id}", json.dumps([text, buttons]))  # to post it again further down (move_card)
        if self._card_state(job_id).get("kind") == "photo":
            await self._edit_photo_caption(job_id, text, buttons)
            return
        refs = self.store.card_refs(job_id)
        msg = Outgoing(text, job_id=job_id, buttons=buttons, silent=True, jobs=True)
        markup = self._markup(buttons)
        changed = False
        for chat_id in self.chat_ids:
            mid = refs.get(str(chat_id))
            if mid:
                try:
                    await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=mid,
                                                         parse_mode=ParseMode.HTML, reply_markup=markup)
                    continue
                except BadRequest as e:
                    if "not modified" in str(e).lower():
                        continue
                    log.info("job %s: its card can't be edited (%s); posting a new one", job_id, e)
                except Exception:
                    log.warning("job %s: card update failed", job_id, exc_info=True)
                    continue
            try:
                new = await self._post_card(chat_id, msg, markup)
            except Exception:
                log.warning("job %s: could not post its card", job_id, exc_info=True)
                continue
            if new:
                refs[str(chat_id)] = new
                changed = True
        if changed:
            self.store.set_card_refs(job_id, refs)

    async def _edit_photo_caption(self, job_id: str, text: str, buttons: list) -> None:
        markup = self._markup(self._photo_buttons(job_id, buttons))
        for chat_id, mid in self.store.card_refs(job_id).items():
            try:
                await self.app.bot.edit_message_caption(chat_id=int(chat_id), message_id=mid, caption=self._caption(text),
                                                        parse_mode=ParseMode.HTML, reply_markup=markup)
            except BadRequest as e:
                if "not modified" not in str(e).lower():
                    log.info("job %s: its card's caption can't be edited (%s)", job_id, e)
            except Exception:
                log.warning("job %s: card update failed", job_id, exc_info=True)

    async def show_plots(self, job_id: str, images: list[tuple[str, bytes]], repost: bool = False) -> None:
        """A job's graphs, in its card: the first time (or with `repost`) the card is posted again as a photo card,
        then the picture is replaced in place (on the page it was on). Never raises."""
        if not images or not self.store.get_job(job_id):
            return
        st = self._card_state(job_id)
        page = min(int(st.get("page", 0)), len(images) - 1)
        self._set_card_state(job_id, pages=len(images), page=page)
        if st.get("kind") != "photo" or repost:
            await self._post_photo_card(job_id, images)
            return
        if not await self._edit_photo(job_id, images[page]):
            await self._post_photo_card(job_id, images)  # (the card is gone, or can't be edited: a new one)

    async def page(self, job_id: str, page: int, images: list[tuple[str, bytes]]) -> None:
        """◀ / ▶ on a photo card: show another of its graphs, in place."""
        if not images:
            return
        page %= len(images)
        self._set_card_state(job_id, pages=len(images), page=page)
        await self._edit_photo(job_id, images[page])

    async def _edit_photo(self, job_id: str, image: tuple[str, bytes]) -> bool:
        text, buttons = self._view(job_id)
        markup = self._markup(self._photo_buttons(job_id, buttons))
        ok = True
        for chat_id, mid in self.store.card_refs(job_id).items():
            name, data = image
            try:
                await self.app.bot.edit_message_media(
                    chat_id=int(chat_id), message_id=mid, reply_markup=markup,
                    media=InputMediaPhoto(media=data, filename=name, caption=self._caption(text),
                                          parse_mode=ParseMode.HTML))
            except BadRequest as e:
                if "not modified" not in str(e).lower():
                    log.info("job %s: its graph can't be replaced (%s)", job_id, e)
                    ok = False
            except Exception:
                log.warning("job %s: graph update failed", job_id, exc_info=True)
                ok = False
        return ok

    async def _retire(self, chat_id: int, mid: int, job_id: str) -> None:
        """Remove an old card (where Telegram won't delete it, say it moved)."""
        try:
            await self.app.bot.delete_message(chat_id, mid)
        except Exception:
            try:
                await self.app.bot.edit_message_text(f"↓ <b>{esc(job_id)}</b>: its card moved to the bottom",
                                                     chat_id=chat_id, message_id=mid, parse_mode=ParseMode.HTML)
            except Exception:
                log.debug("could not retire the old card of %s", job_id, exc_info=True)

    async def _post_photo_card(self, job_id: str, images: list[tuple[str, bytes]]) -> None:
        """Post the card as a photo card (at the bottom of its jobs topic) and remove the previous card - and graphs
        sent under it by an older Overseer."""
        text, buttons = self._view(job_id)
        st = self._card_state(job_id)
        page = min(int(st.get("page", 0)), len(images) - 1)
        name, data = images[page]
        msg = Outgoing(text, job_id=job_id, jobs=True, silent=True)
        markup = self._markup(self._photo_buttons(job_id, buttons))
        old, refs = self.store.card_refs(job_id), {}
        legacy = self._legacy_plot_refs(job_id)
        for chat_id in self.chat_ids:
            await self._jobs_topic_for(msg, chat_id)
            thread, _, _ = self._target(msg, chat_id)
            try:
                sent = await self.app.bot.send_photo(chat_id, photo=data, filename=name, caption=self._caption(text),
                                                     parse_mode=ParseMode.HTML, reply_markup=markup,
                                                     disable_notification=True, message_thread_id=thread)
            except Exception:
                log.warning("job %s: could not post its card with graphs", job_id, exc_info=True)
                if old.get(str(chat_id)):
                    refs[str(chat_id)] = old[str(chat_id)]  # keep the text card
                continue
            refs[str(chat_id)] = sent.message_id
            self.store.link_message(chat_id, sent.message_id, job_id, None)
            if old.get(str(chat_id)):
                await self._retire(chat_id, old[str(chat_id)], job_id)
            for mid in legacy.get(str(chat_id), []):
                await self._retire(chat_id, mid, job_id)
        self.store.set_card_refs(job_id, refs)
        self.store.unset(f"plotrefs:{job_id}")
        if any(refs.get(c) != old.get(c) for c in refs):
            self._set_card_state(job_id, kind="photo")

    def _legacy_plot_refs(self, job_id: str) -> dict[str, list[int]]:
        """Graphs an older Overseer sent as separate messages under the card."""
        try:
            return {str(k): [int(x) for x in v] for k, v in json.loads(self.store.get(f"plotrefs:{job_id}") or "{}").items()}
        except (ValueError, TypeError):
            return {}

    def last_message_ids(self, job_id: str) -> dict[str, int]:
        """Per chat, the lowest-down message of a job in its jobs topic."""
        out = dict(self.store.card_refs(job_id))
        for chat, mids in self._legacy_plot_refs(job_id).items():
            out[chat] = max([out.get(chat, 0), *mids])
        return out

    async def move_card(self, job_id: str, images: list[tuple[str, bytes]] | None = None) -> None:
        """Post a (running) job's card again at the bottom of its jobs topic - as a photo card when it is one - and
        remove the old one. Never raises."""
        if self._card_state(job_id).get("kind") == "photo" and images:
            await self._post_photo_card(job_id, images)
            return
        text, buttons = self._view(job_id)
        if not text:
            return
        self._set_card_state(job_id, kind="text")
        refs = self.store.card_refs(job_id)
        msg = Outgoing(text, job_id=job_id, buttons=buttons, silent=True, jobs=True)
        for chat_id in self.chat_ids:
            old = refs.get(str(chat_id))
            try:
                new = await self._post_card(chat_id, msg, self._markup(buttons))
            except Exception:
                log.warning("job %s: could not post its card again", job_id, exc_info=True)
                continue
            if new:
                refs[str(chat_id)] = new
                if old:
                    await self._retire(chat_id, old, job_id)
        self.store.set_card_refs(job_id, refs)

    async def _post_card(self, chat_id: int, msg: Outgoing, markup) -> int | None:
        await self._jobs_topic_for(msg, chat_id)
        thread, label, _ = self._target(msg, chat_id)
        for attempt in range(2):
            try:
                sent = await self.app.bot.send_message(chat_id, label + msg.text, parse_mode=ParseMode.HTML,
                                                       message_thread_id=thread, disable_notification=True,
                                                       reply_markup=markup)
                self.store.link_message(chat_id, sent.message_id, msg.job_id, None)
                return sent.message_id
            except BadRequest as e:
                if not (thread and _thread_error(e)) or attempt:
                    raise
                proj = self.project_of(msg)  # the jobs topic was deleted: make a new one
                thread = await self._create_jobs_topic(chat_id, proj.name if proj else None)
        return None

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

    # ------------------------------------------------------------------ the 🛠 feedback topic (feedback.py)
    def feedback_thread(self, chat_id: int) -> int | None:
        v = self.store.get(f"feedback_topic:{chat_id}")
        return int(v) if v else None

    async def ensure_feedback_topic(self) -> None:
        if not self.topics:
            return
        for chat_id in self.chat_ids:
            if not self.feedback_thread(chat_id):
                await self._create_feedback(chat_id)

    async def _create_feedback(self, chat_id: int) -> int | None:
        try:
            topic = await self.app.bot.create_forum_topic(chat_id, name="🛠 feedback")
        except Exception:
            log.warning("could not create the feedback topic", exc_info=True)
            return None
        self.store.set(f"feedback_topic:{chat_id}", str(topic.message_thread_id))
        try:
            await self.app.bot.send_message(
                chat_id, "🛠 <b>feedback</b> - what agents ran into with Overseer itself: limitations, bugs, ideas. "
                "/feedback lists them; /feedback review has an agent group them and design fixes, and you can talk "
                "to that reviewer here. Skills agents propose (/skills) are approved where they're proposed.",
                parse_mode=ParseMode.HTML, message_thread_id=topic.message_thread_id, disable_notification=True)
        except Exception:
            log.debug("feedback topic intro failed", exc_info=True)
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
            await self._jobs_topic_for(msg, chat_id)
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
                key = ("\0jobs:" if msg.jobs else "") + (proj.name if proj else "\0general")
                log.warning("topic %s unavailable (%s)", proj.name if proj else "general", e)
                if time.time() - self._recreated.get(key, 0) > 3600:
                    self._recreated[key] = time.time()
                    if msg.jobs:
                        new = await self._create_jobs_topic(chat_id, proj.name if proj else None)
                    else:
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
        if msg.reply_to.get(str(chat_id)):
            kwargs["reply_parameters"] = {"message_id": msg.reply_to[str(chat_id)], "allow_sending_without_reply": True}
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
        reply = msg.reply_to.get(str(chat_id))
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            kw = dict(disable_notification=msg.silent, reply_markup=markup if last else None, message_thread_id=thread)
            if reply and i == 0:  # e.g. a job's final report, under its card
                kw["reply_parameters"] = ReplyParameters(message_id=reply, allow_sending_without_reply=True)
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
        self._pending_files: dict[tuple[int, int | None], list[tuple[str, Path]]] = {}  # ...and the files sent with it
        self._uploads: dict[tuple[int, int | None], dict] = {}  # files received, not yet given to the agent
        self.digest = None  # digest.Digest (the morning digest), set by the app
        self._searcher = None  # search.Search (/search), made on first use
        self.dashboard = None  # dashboard.Dashboard (the Mini App), when dashboard.enabled

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
        add(CommandHandler("usage", self.cmd_usage, filters=allowed))
        add(CommandHandler("cluster", self.cmd_cluster, filters=allowed))
        add(CommandHandler("digest", self.cmd_digest, filters=allowed))
        add(CommandHandler("search", self.cmd_search, filters=allowed))
        add(CommandHandler("dashboard", self.cmd_dashboard, filters=allowed))
        add(CommandHandler("stop", self.cmd_stop, filters=allowed))
        add(CommandHandler("health", self.cmd_health, filters=allowed))
        add(CommandHandler("auto", self.cmd_auto, filters=allowed))
        add(CommandHandler("subagents", self.cmd_subagents, filters=allowed))
        add(CommandHandler("results", self.cmd_results, filters=allowed))
        add(CommandHandler("feedback", self.cmd_feedback, filters=allowed))
        add(CommandHandler("skills", self.cmd_skills, filters=allowed))
        add(CommandHandler("findings", self.cmd_findings, filters=allowed))
        add(CommandHandler(["project", "projects"], self.cmd_project, filters=allowed))
        add(CommandHandler("assign", self.cmd_assign, filters=allowed))
        # Updates python-telegram-bot doesn't know yet (e.g. the Stop button of a streaming draft).
        add(TypeHandler(Update, self.on_raw_update), group=-1)
        add(CallbackQueryHandler(self.on_button))
        add(MessageHandler(allowed & filters.TEXT & ~filters.COMMAND, self.on_text))
        add(MessageHandler(allowed & (filters.PHOTO | filters.Document.ALL), self.on_file))
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
        fb = getattr(self.agent, "feedback", None)
        if fb and hasattr(self.notifier, "feedback_thread") and self.notifier.feedback_thread(chat_id) == thread:
            return fb.reviewer()  # the 🛠 feedback topic: talk to the reviewer
        is_jobs, project = (self.notifier.jobs_topic_owner(chat_id, thread)
                            if hasattr(self.notifier, "jobs_topic_owner") else (False, None))
        if is_jobs:  # a jobs topic: its project's conversation
            return self.agent.general_session(project=project)
        return self.agent.get(self.store.thread_session(chat_id, thread))

    def _outside_topics(self, update: Update) -> bool:
        """Written in Telegram's "All" view rather than in a topic, while new threads are off: Telegram either
        makes the message a topic of its own (one nobody assigned yet) or sends it with no topic at all."""
        if self.cfg.telegram.new_threads or not getattr(self.notifier, "topics", False):
            return False
        if self._replied_session(update):  # a reply to one of Overseer's messages: that conversation
            return False
        thread = self._thread(update)
        return not thread or self._here(update) is None

    async def _refuse_outside_topics(self, update: Update, text: str, files: list[tuple[str, Path]]) -> None:
        for _, path in files:
            path.unlink(missing_ok=True)
        chat_id, thread = update.effective_chat.id, self._thread(update)
        quoted = (f"\nYour message:<blockquote expandable>{esc(text[:3000])}</blockquote>" if text and text.strip()
                  else "")
        notice = ("✋ Messages outside the topics don't go to the agent. Write in a project's topic, or in 💬 general "
                  "for anything else." + (f" ({len(files)} file(s) not sent.)" if files else "") + quoted)
        if thread:  # the topic Telegram made for the message: removed, so the chat keeps only Overseer's topics
            try:
                await self.app.bot.delete_forum_topic(chat_id, thread)
            except Exception:
                log.debug("could not delete topic %s", thread, exc_info=True)
                await self.reply(update, notice)
                return
        await self.notifier.send(Outgoing(notice))

    def _where(self, scope: str | None) -> str:
        return f"project {code(scope)}" if scope else "the main chat"

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_agent(self, chat_id: int, session: Session, text: str,
                         files: list[tuple[str, Path]] | None = None) -> None:
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
            await self.agent.handle(session, text, files=files)
        except Exception:
            if not self.agent.shutting_down:  # (cut off by Overseer stopping: offered again at the next start)
                raise
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
        await self.notifier.send(Outgoing(f"🔧 {esc(self.agent.session_backend(session))} is working on job {code(job_id)}…",
                                          job_id=job_id, session_id=session.id, silent=True))
        self._spawn(self._run_agent(chat_id, session, text))

    async def _start_plot(self, chat_id: int, job_id: str) -> None:
        """📈: the job's graphs now, from the agent's plotting command; without one, the agent writes one."""
        rec = self.store.get_job(job_id)
        if not rec:
            await self.notifier.send(Outgoing(f"Unknown job {code(job_id)}."))
            return
        if resolve_plot(self.store, rec):  # no model involved: the graphs appear (or are updated) under the card
            n, err = await self.monitor.plots(rec)
            if not n:
                await self.notifier.send(Outgoing(f"📈 No graphs for {code(job_id)}: {esc(err)}", job_id=job_id,
                                                  jobs=True, reply_to=self.store.card_refs(job_id), silent=True))
            return
        log_path = (rec.info_dict.get("progress_log") or rec.stdout) if rec else ""
        text = (f"The researcher pressed 📈 Plot on job {job_id} ({rec.name}, {rec.state}), and there's no plotting "
                "command for it yet. Write a small plotting script in the project folder on the HPC that reads its "
                f"metrics - its log {log_path}, or metric files (metrics.jsonl, CSV, TensorBoard events) in its output "
                "folder - and saves clear PNGs (the main metrics against step or time, a few panels if useful, the job "
                "id in the title) into the folder it's given. Register it with `ov plots set` for this kind of job "
                "(--pattern or --project, so later runs get graphs too), then `ov plots test` it and look at the "
                "result. Describe the trend in two or three sentences.")
        session = self.agent.session_for_job(job_id)
        await self.notifier.send(Outgoing(f"📈 {esc(self.agent.session_backend(session))} is plotting job "
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

    async def cmd_cluster(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        partition = (ctx.args or [None])[0]
        try:
            snap = await self.monitor.cluster.snapshot(fresh=True)
        except (SSHError, OSError) as e:
            await self.reply(update, f"Couldn't read the cluster: {code(str(e)[:300])}")
            return
        names = {r.job_id: r.name for r in self.store.active_jobs()}
        await self.reply(update, "🖥 " + self.monitor.cluster.text(snap, names, partition=partition, html=True))

    DASHBOARD_SETUP = (
        "The dashboard needs two settings in config.yaml, then a restart:\n"
        "1. <code>dashboard: {enabled: true}</code> - Overseer serves it on this machine (127.0.0.1:8765).\n"
        "2. On the homelab: <code>tailscale serve --bg 8765</code>. It prints an https://….ts.net address, reachable "
        "only on your tailnet (not the internet: that would be <i>funnel</i>). Put it in <code>dashboard.url</code>. "
        "(Or any HTTPS reverse proxy to 127.0.0.1:8765, e.g. Caddy: see the README.)\n"
        "Then /dashboard opens it, from any of your devices with Tailscale on.")

    async def cmd_dashboard(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        url = self.dashboard.url if self.dashboard else ""
        if not url:
            await self.reply(update, self.DASHBOARD_SETUP)
            return
        from telegram import WebAppInfo
        markup = InlineKeyboardMarkup([[InlineKeyboardButton("📊 Open the dashboard", web_app=WebAppInfo(url))]])
        await update.effective_message.reply_text(
            "Jobs, results, findings and a timeline (read-only; only on your tailnet).", reply_markup=markup)

    async def set_dashboard_button(self) -> None:
        """The chat's menu button opens the dashboard (when it's reachable)."""
        url = self.dashboard.url if self.dashboard else ""
        if not url:
            return
        from telegram import MenuButtonWebApp, WebAppInfo
        for chat_id in self.cfg.telegram.allowed_chat_ids:
            try:
                await self.app.bot.set_chat_menu_button(chat_id=chat_id, menu_button=MenuButtonWebApp(
                    "Dashboard", WebAppInfo(url)))
            except Exception as e:  # (not fatal: /dashboard still works)
                log.warning("dashboard menu button for chat %s: %s", chat_id, e)

    async def cmd_search(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Past work: conversations, results, findings, jobs (in a project's topic: that project's)."""
        from .search import Search, SearchError, render_hits
        text = " ".join(ctx.args or []).strip()
        if not text:
            await self.reply(update, "Usage: /search &lt;text&gt; — e.g. /search ratio 64 (\"a phrase\", OR, "
                                     "word*). In a project's topic: that project only.")
            return
        if self._searcher is None:
            self._searcher = Search(self.cfg, self.store)
        project = self._scope(update)
        try:
            hits, total = await self._searcher.asearch(text, project=project if project else False, limit=10)
        except SearchError as e:
            await self.reply(update, esc(str(e)))
            return
        await self.reply(update, render_hits(hits, total, text, html=True)
                         + (f"\n<i>in project {esc(project)}</i>" if project else ""))

    async def cmd_digest(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """The digest now: since the last one (or the last 24 hours); the morning one still comes."""
        if not self.digest:
            await self.reply(update, "The digest isn't available.")
            return
        now = time.time()
        since = max(float(self.store.get("digest_at") or 0) or now - 86400, now - 7 * 86400)
        if now - since < 3600:  # just sent: the last day instead
            since = now - 86400
        await self.reply(update, await self.digest.build(since, now))

    async def cmd_usage(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        period = (ctx.args or ["today"])[0].lower()
        await self.reply(update, usage_report(self.store, period))

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
        auto = {w.session_id for w in self.store.watches()}
        await self.reply(update, f"HPC user {code(self.hpc.user or '?')} · SSH connection {conn}\n"
                                 f"last successful poll: {last}{err}\n"
                                 f"agent runs in progress: {running} · conversations with wake-ups pending: {len(auto)}\n"
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
        files: list[tuple[str, Path]] = []
        up = self._uploads.pop((update.effective_chat.id, self._thread(update)), None)
        if up:  # files sent just before: they go with this message
            up["timer"].cancel()
            text, files = "\n\n".join([*up["captions"], text]), up["files"]
        await self._route(update, text, files)

    async def _route(self, update: Update, text: str, files: list[tuple[str, Path]]) -> None:
        if self._outside_topics(update):
            await self._refuse_outside_topics(update, text, files)
            return
        session = self._replied_session(update) or self._here(update)
        if session is None:  # a thread the user just started
            await self._new_thread(update, text, files)
            return
        proj = self.store.get_project(session.project)
        if proj and proj.archived:
            for _, path in files:
                path.unlink(missing_ok=True)
            await self.reply(update, f"Project {code(proj.name)} is archived. /project restore {esc(proj.name)} "
                                     "brings it back.")
            return
        if self.agent.subagents:  # writing while the agent's plan is open declines it: the agent gets your message
            text = await self.agent.subagents.reject_by_reply(session, text)
        self._spawn(self._run_agent(update.effective_chat.id, session, text, files))

    # ------------------------------------------------------------------ runs cut off by a restart
    async def offer_cut_off_runs(self) -> None:
        """At startup: tell each conversation about the run the last Overseer didn't finish (the agent's turn, a
        wake-up, or a message still waiting its turn), with ▶️ Continue / ✖ Dismiss."""
        for rec in self.agent.take_cut_off_runs():
            session = self.agent.get(rec.get("session"))
            if not session:
                self.agent.pop_cut_off_run(rec["id"])
                continue
            what = {"wakeup": "a wake-up it had set up", "checkin": "a check-in it had set up"}.get(
                rec.get("origin"), "your message")
            state = "was working on" if rec.get("started") else "had yet to start"
            await self.notifier.send(Outgoing(
                f"⚠️ Overseer restarted while the agent {state} {what}:"
                f"<blockquote expandable>{esc(_preview(rec, 600))}</blockquote>",
                session_id=session.id, buttons=[[("▶️ Continue", f"rerun:{rec['id']}"),
                                                 ("✖ Dismiss", f"rerunx:{rec['id']}")]]))

    # ------------------------------------------------------------------ files sent to the agent
    async def on_file(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """A picture or file for the agent. It's saved in the agent's ./inbox and goes with the caption, or with
        the next message (a picture without a caption, then a few words about it); after UPLOAD_WAIT without one,
        on its own. An album is one message."""
        msg = update.effective_message
        if self._outside_topics(update):
            await self._refuse_outside_topics(update, msg.caption or "", [])
            return
        if msg.photo:
            media, name = msg.photo[-1], f"photo-{msg.message_id}.jpg"  # the largest size
        elif msg.document:
            media, name = msg.document, msg.document.file_name or f"file-{msg.message_id}"
        else:
            return
        if (media.file_size or 0) > MAX_DOWNLOAD:
            await self.reply(update, f"📎 {esc(name)} is too big: Telegram lets bots download files up to "
                                     f"{MAX_DOWNLOAD >> 20} MB. Put it on the HPC and tell the agent where it is.")
            return
        staging = self.cfg.local.tmp("uploads")
        staging.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = staging / secrets.token_hex(8)
        try:
            await (await media.get_file()).download_to_drive(path)
        except Exception as e:
            log.warning("could not download %s", name, exc_info=True)
            await self.reply(update, f"📎 Could not download {esc(name)}: {esc(type(e).__name__)}: {esc(str(e)[:200])}")
            return
        key = (update.effective_chat.id, self._thread(update))
        up = self._uploads.setdefault(key, {"update": update, "files": [], "captions": [], "timer": None})
        up["files"].append((name, path))
        if msg.caption:
            up["captions"].append(msg.caption)
        if up["timer"]:
            up["timer"].cancel()
        wait = ALBUM_WAIT if up["captions"] else UPLOAD_WAIT
        if wait == UPLOAD_WAIT and len(up["files"]) == 1 and not msg.media_group_id:
            await self.reply(update, f"📎 Got it. Write what it's about, or in {UPLOAD_WAIT} s the agent looks at it "
                                     "as is.")
        up["timer"] = asyncio.create_task(self._send_uploads(key, wait))

    async def _send_uploads(self, key: tuple[int, int | None], wait: float) -> None:
        await asyncio.sleep(wait)
        up = self._uploads.pop(key, None)
        if up:
            await self._route(up["update"], "\n\n".join(up["captions"]), up["files"])

    # ------------------------------------------------------------------ threads
    async def _new_thread(self, update: Update, text: str, files: list[tuple[str, Path]] | None = None) -> None:
        """Each Telegram thread is its own conversation. A new one asks which project it's about
        (unless the message starts with "<project>: …" or there are no projects)."""
        chat_id, thread = update.effective_chat.id, self._thread(update)
        projects = self.store.list_projects()
        m = re.match(r"^\s*([a-z0-9][a-z0-9_-]{0,31})\s*:\s*(\S.*)$", text or "", re.S | re.I)
        if m and self.store.get_project(m.group(1).lower()) and not self.store.get_project(m.group(1).lower()).archived:
            await self._start_thread(chat_id, thread, m.group(1).lower(), m.group(2), files)
            return
        if not projects:
            await self._start_thread(chat_id, thread, None, text, files)
            return
        self._pending[(chat_id, thread)] = text
        self._pending_files[(chat_id, thread)] = files or []
        buttons = [(f"📁 {p.name}", f"thread:{p.name}") for p in projects[:8]] + [("💬 No project", "thread:")]
        rows = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
        markup = InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in row] for row in rows])
        await update.effective_message.reply_text("New conversation. Which project is it about?", reply_markup=markup)

    async def _start_thread(self, chat_id: int, thread: int | None, project: str | None, text: str,
                            files: list[tuple[str, Path]] | None = None) -> None:
        session = self.agent.new_conversation(project)
        if thread:
            self.store.bind_thread(chat_id, thread, session.id)
            if project and hasattr(self.notifier, "rename_topic"):  # Telegram named it after the first message
                await self.notifier.rename_topic(chat_id, thread, f"{project} · {' '.join((text or '').split())[:60] or (files[0][0] if files else '')}")
        if (text and text.strip()) or files:
            self._spawn(self._run_agent(chat_id, session, text, files))

    def _scope_sessions(self, update: Update | None) -> list[str] | None:
        """The conversations a command applies to: the replied-to one, this thread's, a project topic's, or all."""
        target = self._replied_session(update)
        if target:
            return [target.id]
        thread = self._thread(update)
        sid = self.store.thread_session(update.effective_chat.id, thread) if update and thread else None
        if sid:
            return [sid]
        scope = self._scope(update)
        if scope:
            return [r.id for r in self.store.list_sessions(10_000) if r.project == scope]
        return None

    def _results_scope(self, update: Update, args: list[str]) -> tuple[str | None | bool, list[str]]:
        """A project named in the args ("all": every project, "main": the main chat), else this topic's."""
        rest, scope = [], self._scope(update)
        for a in args:
            if a == "all":
                scope = False
            elif a == "main":
                scope = None
            elif self.store.get_project(a):
                scope = a
            else:
                rest.append(a)
        return scope, rest

    async def cmd_results(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """/results [project|all] [sort key, e.g. score] [asc] [N] [csv]: the experiments and what each showed."""
        scope, rest = self._results_scope(update, list(ctx.args or []))
        want_csv, asc, limit, sort, verdicts = False, None, 15, "", []
        for a in rest:
            if a == "csv":
                want_csv = True
            elif a in ("asc", "desc"):
                asc = a == "asc"
            elif a.isdigit():
                limit = int(a)
            elif a in R.VERDICTS:
                verdicts.append(a)
            else:
                sort = a
        rs = R.select(self.store.results(scope), sort=sort, asc=asc, verdicts=verdicts,
                      limit=0 if want_csv else limit)
        where = "all projects" if scope is False else (f"project {scope}" if scope else "the main chat")
        if not rs:
            await self.reply(update, f"No results yet in {esc(where)}. Every finished job gets one; the agent adds "
                                     "what it showed (`ov result`).")
            return
        title = f"📊 Results in {where}" + (f", by {sort}" if sort else ", newest first")
        if want_csv:
            await self.notifier.send(Outgoing(f"{esc(title)}: {len(rs)} rows", project=scope or None,
                                              files=[("results.csv", R.render(rs, "csv").encode())]))
            return
        total = len(self.store.results(scope))
        more = f"\n{len(rs)} of {total}: /results [project] [sort] [N] [verdict] [csv]" if total > len(rs) else ""
        await self.notifier.send(Outgoing(
            f"<b>{esc(title)}</b>\n<pre>{esc(R.render(rs, 'table'))}</pre>{esc(more)}",
            markdown=f"**{title}**\n\n{R.render(rs, 'md')}{more}", project=scope or None))

    async def cmd_findings(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """/findings [project|all]: the insights and dead ends the agents recorded."""
        scope, _ = self._results_scope(update, list(ctx.args or []))
        fs = self.store.findings(scope)
        where = "all projects" if scope is False else (f"project {scope}" if scope else "the main chat")
        await self.reply(update, f"<b>Findings in {esc(where)}</b>\n" + "\n".join(esc(x) for x in R.finding_lines(fs))
                         if fs else f"No findings yet in {esc(where)}. Agents record them with `ov finding`.")

    async def cmd_feedback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """/feedback [all] - what agents reported about Overseer; /feedback review [focus] - an agent groups it and
        designs fixes (in the 🛠 feedback topic, where you can talk to it)."""
        fb = getattr(self.agent, "feedback", None)
        if not fb:
            await self.reply(update, "Feedback isn't available in this Overseer.")
            return
        args = list(ctx.args or [])
        if args[:1] == ["review"]:
            s = fb.reviewer()
            n = len(fb.listing())
            await self.notifier.send(Outgoing(f"🛠 Reviewing {n} open feedback item(s)…", session_id=s.id, silent=True))
            chat_id = update.effective_chat.id if update else self.cfg.telegram.allowed_chat_ids[0]
            self._spawn(self._run_agent(chat_id, s, fb.review_prompt(" ".join(args[1:]))))
            return
        items = fb.listing(None if args[:1] == ["all"] else ("open", "noted"))
        if not items:
            await self.reply(update, "No open feedback. Agents report what limits them with `ov feedback`.")
            return
        await self.reply(update, "<b>🛠 Feedback</b> (most +1s first)\n" + "\n".join(fb.lines(items[:30], html=True))
                         + "\n\n/feedback review [focus]: an agent groups these and designs fixes.")

    async def cmd_skills(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """/skills: the skills agents proposed and you installed, with a 🗑 button each."""
        addons = getattr(self.agent, "addons", None)
        if not addons:
            await self.reply(update, "Skills aren't available in this Overseer.")
            return
        installed = self.store.skills()
        pending = self.store.skills(("proposed",))
        if not installed and not pending:
            await self.reply(update, "No skills yet. Agents propose them (`ov skill propose`) when they write down "
                                     "know-how worth keeping; you approve them where they're proposed.")
            return
        lines = ["<b>🧩 Skills</b>"]
        lines += [f"• <b>{esc(k.name)}</b> ({'every agent' if not k.project else 'project ' + esc(k.project)}): "
                  f"{esc(k.description)}" for k in installed]
        lines += [f"• waiting for your answer: {esc(k.name)}" for k in pending]
        buttons = [[(f"🗑 {k.name}" + (f" ({k.project})" if k.project else ""), f"skrm:{k.id}")] for k in installed]
        await self.notifier.send(Outgoing("\n".join(lines), buttons=buttons[:20], project=self._scope(update)))

    async def cmd_subagents(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """/subagents [on|off|planner <model>|workers [backend] <m1,m2>|clean]: planner mode here (a project, or the
        main chat), its models, and its subagents."""
        subs = self.agent.subagents
        if not subs:
            await self.reply(update, "Subagents aren't available in this Overseer.")
            return
        scope, args = self._scope(update), list(ctx.args or [])
        backend = self.agent.backend_of(scope)
        where = self._where(scope)
        if args[:1] == ["on"] or args[:1] == ["off"]:
            on = args[0] == "on"
            self.agent._set_setting("subagents", "on" if on else None, scope)
            if not on:  # also conversations switched on by an approved plan
                for sid in self.store.planner_sessions(scope):
                    s = self.agent.get(sid)
                    s.rec.planner = 0
                    self.store.upsert_session(s.rec)
            await self.reply(update, (
                f"🧭 Planner mode is <b>on</b> in {where}: its conversations plan on "
                f"{code(self.agent.planner_model(backend, scope) or '?')} and start subagents on "
                f"{esc(', '.join(self.agent.worker_models(backend, scope)))}." if on else
                f"Planner mode is <b>off</b> in {where}: conversations use your /model again. Running subagents "
                "finish their work and report back."))
            return
        if args[:1] == ["planner"] and len(args) == 2:
            self.agent._set_setting(f"sub_planner:{backend}", None if args[1] == "default" else args[1], scope)
            await self.reply(update, f"Planner model for {esc(backend)} in {where}: "
                                     f"{code(self.agent.planner_model(backend, scope) or '?')}")
            return
        if args[:1] == ["workers"] and len(args) in (2, 3):
            b = args[1] if len(args) == 3 else backend
            if b not in BACKENDS:
                await self.reply(update, f"Backend must be one of {', '.join(BACKENDS)}.")
                return
            self.agent._set_setting(f"sub_workers:{b}", None if args[-1] == "default" else args[-1], scope)
            await self.reply(update, f"Subagent models for {esc(b)} in {where}: "
                                     f"{esc(', '.join(self.agent.worker_models(b, scope)))}")
            return
        if args[:1] == ["clean"]:
            done = [r for r in self.store.subagents(project=scope, states=ENDED) if self.store.session_threads(r.id)]
            kept = [r.name for r in done if not await self.notifier.close_thread(r.id, f"✅ {r.name} (done)")]
            await self.reply(update, f"🗑 Deleted the threads of {len(done) - len(kept)} finished subagent(s)"
                                     + (f"; renamed (Telegram refused): {esc(', '.join(kept))}" if kept else "") + ".")
            return
        if args:
            await self.reply(update, "Usage: /subagents [on | off | planner &lt;model&gt; | workers [backend] "
                                     "&lt;model,model&gt; | clean]")
            return
        on = self.agent._setting("subagents", scope) == "on"
        here = self._here(update)
        lines = [f"🧭 <b>Planner mode</b> in {where}: <b>{'on' if on else 'off'}</b>"
                 + ("" if on or not (here and here.rec.planner) else " (on in this conversation: you approved a plan)")]
        for b in BACKENDS:
            lines.append(f"{esc(b)}: planner {code(self.agent.planner_model(b, scope) or '?')} · subagents "
                         f"{esc(', '.join(self.agent.worker_models(b, scope)) or '?')}")
        recs = self.store.subagents(project=scope)
        active = [r for r in recs if r.state not in ENDED]
        recent = [r for r in recs if r.state in ENDED][-5:]
        if active or recent:
            lines.append("")
            lines += [esc(x) for x in subs.describe(active + recent)]
        lines.append("\n/subagents on|off · planner &lt;model&gt; · workers [backend] &lt;models&gt; · clean (delete "
                     "finished subagents' threads). Agents can also propose it: you approve or decline their plan.")
        await self.reply(update, "\n".join(lines))

    async def cmd_auto(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        ap = self.agent.autopilot
        if ap is None:
            await self.reply(update, "Wake-ups aren't available.")
            return
        args = list(ctx.args or [])
        if not args or args[0].lower() in ("status", "list"):
            await self.reply(update, ap.describe(self._scope_sessions(update)))
            return
        if args[0].lower() in ("off", "stop") and len(args) == 1:
            cancelled = ap.cancel_all(self._scope_sessions(update))
            await self.reply(update, f"⏹ Cancelled {len(cancelled)} pending wake-up(s). Jobs keep running and are still "
                                     "reported." if cancelled else "Nothing was pending.")
            return
        goal = " ".join(args).strip()
        session = self._replied_session(update) or self._here(update)
        if session is None:
            session = self.agent.new_conversation(self._scope(update))
            if self._thread(update):
                self.store.bind_thread(update.effective_chat.id, self._thread(update), session.id)
        self._spawn(self._run_agent(update.effective_chat.id if update else self.cfg.telegram.allowed_chat_ids[0],
                                    session, AUTO_PROMPT.format(goal=goal)))

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
            auto = [w for w in self.store.watches()
                    if (sess := self.store.get_session(w.session_id)) and sess.project == p.name]
            b = self.agent.backend_of(p.name)
            r = self.agent.runner(b, p.name)
            bits = [f"📁 <b>{esc(p.name)}</b> · {code(p.dir)}", f"{esc(b)} {esc(r.model or 'default model')}"]
            if running:
                bits.append(f"{len(running)} job(s) active")
            if auto:
                bits.append(f"⏰ {len(auto)} wake-up(s) pending")
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
            if hasattr(self.notifier, "ensure_jobs_topics"):
                await self.notifier.ensure_jobs_topics([p.name])
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
            notes = await self.hpc.try_head(self.agent.domain.remote_path(path), 200)
            await self.reply(update, f"📝 {code(path)}" + (pre(notes, 3500) if notes.strip() else
                                     "\n(empty - the agent creates it when it learns something worth keeping)"))
        elif sub == "archive":
            self.projects.archive(name)
            await self.reply(update, f"🗄 Project {code(name)} archived: its pending wake-ups are cancelled and new jobs in its folder "
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
            self.agent.domain.record_job(conv, job_id)
            ap = self.agent.autopilot
            rec = self.store.get_job(job_id)
            wake = bool(ap and rec and not rec.ended_at)
            if wake:  # part of the conversation's work: its end wakes the agent
                ap.add_watch(conv, [job_id])
            await self.reply(update, f"🔗 Job {code(job_id)} is now part of this conversation: its reports come here"
                                     + (" and the agent is woken when it ends (/auto off cancels that)." if wake else "."))
            return
        target = args[0].lower() if args else self._scope(update)
        if not job_id or not target:
            await self.reply(update, "Usage: /assign &lt;job id&gt; &lt;project|main&gt; - or reply to a job's message "
                                     "with /assign &lt;project|main&gt;. In a conversation's thread, /assign &lt;job id&gt; "
                                     "makes the job part of that conversation (its end wakes the agent).")
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
        self.agent.domain.move_job(job_id, project)
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
            elif action == "status":  # on a job's card: refresh it in place
                rec = self.store.get_job(arg)
                if rec and not rec.ended_at:
                    await self.monitor.refresh_card_by_id(arg)
                else:
                    await self._send_status(arg)
            elif action in ("rerun", "rerunx"):  # a run cut off by a restart: ▶️ Continue / ✖ Dismiss
                rec = self.agent.pop_cut_off_run(arg)
                session = self.agent.get(rec.get("session")) if rec else None
                if not session:
                    await self._edit(q, "This was already continued or dismissed.")
                elif action == "rerunx":
                    for _, path in rec.get("files") or []:
                        Path(path).unlink(missing_ok=True)
                    await self._edit(q, f"✖ Dismissed: {esc(_preview(rec))}")
                    if session.rec.is_worker and session.rec.state == "running" and self.agent.subagents:
                        await self.agent.subagents.stop(session, by="researcher")  # its planner is told
                else:
                    await self._edit(q, f"▶️ Continuing: {esc(_preview(rec))}")
                    text, files = self.agent.continue_prompt(rec)
                    self._spawn(self._run_agent(chat.id, session, text, files))
            elif action in ("plan", "planx") and self.agent.subagents:  # a proposed plan: ✅ Approve / ✖ No
                subs = self.agent.subagents
                result = await (subs.approve(arg) if action == "plan" else subs.decline(arg))
                if result.startswith("This "):  # already answered: the card says so
                    await self._edit(q, result)
            elif action == "pg":  # ◀ / ▶ on a job's card: another of its graphs, in place
                job_id, _, page = arg.rpartition(":")
                if page.isdigit() and hasattr(self.notifier, "page"):
                    await self.notifier.page(job_id, int(page), self.monitor.cached_plots(job_id))
            elif action == "noop":
                pass
            elif action == "fb" and getattr(self.agent, "feedback", None):  # 👀 Noted / ✅ Fixed / ✖ Won't fix
                fid, _, status = arg.partition(":")
                if not (fid.isdigit() and await self.agent.feedback.set_status(int(fid), status)):
                    await self._edit(q, "This feedback no longer exists.")
            elif action in ("sk", "skx", "skrm") and getattr(self.agent, "addons", None):  # skill proposals
                addons = self.agent.addons
                if not arg.isdigit():
                    pass
                elif action == "skrm":
                    k = addons.remove(int(arg))
                    await self._edit(q, f"🗑 Removed skill {esc(k.name)}: agents don't get it from their next run."
                                     if k else "This skill isn't installed any more.")
                else:
                    result = await (addons.approve(int(arg)) if action == "sk" else addons.decline(int(arg)))
                    if result.startswith("This "):
                        await self._edit(q, result)
            elif action in ("subdel", "subkeep"):  # a finished subagent's thread: 🗑 Delete / Keep
                rec = self.store.get_session(arg)
                if not rec or not rec.is_worker:
                    await self._edit(q, "This subagent no longer exists.")
                elif action == "subkeep":
                    try:
                        await q.edit_message_reply_markup(reply_markup=None)
                    except Exception:
                        pass
                elif rec.state not in ENDED:
                    await self._edit(q, f"{esc(rec.name)} is working again: /stop it here first, or keep the thread.")
                elif not await self.notifier.close_thread(rec.id, f"✅ {rec.name} (done)"):
                    await self._edit(q, "Telegram didn't let me delete this thread, so I renamed it "
                                        f"“✅ {esc(rec.name)} (done)”. You can delete it from the chat's topic list.")
            elif action == "thread":
                thread = getattr(q.message, "message_thread_id", None) if q.message else None
                text = self._pending.pop((chat.id, thread), "")
                files = self._pending_files.pop((chat.id, thread), [])
                project = arg or None
                if project and not self.store.get_project(project):
                    project = None
                try:
                    await q.edit_message_text(f"📁 {project}" if project else "💬 No project")
                except Exception:
                    pass
                await self._start_thread(chat.id, thread, project, text, files)
            elif action == "plot":
                await self._start_plot(chat.id, arg)
            elif action == "assign":
                job_id, _, name = arg.partition(":")
                await self._assign(job_id, name or None, update)
            elif action == "ap":  # an approval card from before wake-ups needed no approval
                await self._edit(q, "This approval card is no longer used: the agent sets up its wake-ups itself now "
                                    "(/auto lists them).")
            elif action == "uw":  # ✖ on a wake-up's notice, or on a job's card (its watch)
                w = self.agent.autopilot.cancel_watch(int(arg)) if self.agent.autopilot and arg.isdigit() else None
                if w and w.kind == "jobs":  # the button is on the job's card: update the card
                    for jid in w.job_ids:
                        await self.monitor.refresh_card_by_id(jid)
                else:
                    await self._edit(q, "✖ Cancelled: the agent won't be woken for this." if w else
                                     "This wake-up has already happened or was cancelled.")
        except SSHError as e:
            await self.notifier.send(Outgoing(f"⚠️ {esc(e)}"))
