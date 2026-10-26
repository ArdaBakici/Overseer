"""Domains: what makes the platform one particular bot.

The platform (core/) runs agent conversations from Telegram: the Claude Code / Codex runners and their sandbox, the
Telegram layer, the agent's control command, wake-ups, transcripts and search, usage, the digest and the dashboard.
A domain (hpc/, the personal assistant, ...) plugs into it what is its own: the agent's instructions, the context it
gets at the start of a conversation, what its sandbox may reach, what it watches in the agent's commands, and what
happens after a run. Each domain runs as its own bot and process, with its own state and secrets.

Every hook has a neutral default, so a domain overrides only what it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .runners import Activity, Sandbox

if TYPE_CHECKING:
    from .agent import Agent, Session
    from .config import Config
    from .notify import Notifier
    from .store import Store


@dataclass
class RunNote:
    """What a domain adds to the agent's reply after a run (e.g. the notes it changed, with an Undo button)."""

    lines: list[str] = field(default_factory=list)  # Telegram HTML, after the reply
    buttons: list[list[tuple[str, str]]] = field(default_factory=list)  # rows of (label, callback data)


class Domain:
    name = "base"
    command = "ov"  # the agent's command for talking to this bot
    planner = True  # planner mode (subagents) is offered to its agents
    reset_prefixes: tuple[str, ...] = ()  # its flags in the store (kv), wiped by `overseer reset`

    def __init__(self, cfg: Config, store: Store, notifier: Notifier):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.agent: Agent | None = None

    def attach(self, agent: Agent) -> None:
        self.agent = agent

    # ------------------------------------------------------------------ the app (app.py)
    def prepare(self) -> None:
        """Before anything else starts (e.g. write the files the agent's tools need)."""

    def build(self) -> None:
        """The domain's own parts, once the agent exists (e.g. a monitor)."""

    def services(self) -> list[tuple[str, object]]:
        """Background loops to run while the bot runs: (name, coroutine)."""
        return []

    async def start(self, notifier, projects) -> None:
        """Once Telegram's topics are known: the domain's own topics (e.g. the HPC's jobs topics)."""

    async def project_created(self, notifier, name: str) -> None:
        """A project was created (or restored), with its topic: the domain's own topics for it (the HPC's jobs)."""

    async def check(self) -> list[tuple[str, bool, str]]:
        """`overseer check`: the domain's own checks, (what, ok, detail)."""
        return []

    # ------------------------------------------------------------------ the sandbox
    def sandbox(self) -> Sandbox:
        return Sandbox()

    # ------------------------------------------------------------------ the agent's instructions and context
    def system_prompt(self, s: Session | None) -> str:
        """The domain's part of the system prompt: who the agent is, its environment, its tools."""
        return ""

    async def context(self, s: Session, fresh: bool) -> list[str]:
        """Parts put before a request (`fresh`: the CLI session is new, so it knows nothing yet)."""
        return []

    async def wake_context(self, s: Session) -> list[str]:
        """Parts put into a wake-up's prompt, after what woke the agent."""
        return []

    # ------------------------------------------------------------------ a run
    def new_activity(self, s: Session) -> Activity:
        """What records the run's commands (a domain's subclass watches them, e.g. for submitted jobs)."""
        return Activity()

    async def before_run(self, s: Session) -> Any:
        """Right before the CLI starts (e.g. make sure a connection is up). Its result goes to after_run."""
        return None

    async def after_run(self, s: Session, activity: Activity, state: Any, reply: str) -> RunNote | None:
        """After the CLI answered, before the reply is sent: what to add to the reply, if anything."""
        return None

    # ------------------------------------------------------------------ the agent's control command (ctl.py)
    def ctl_commands(self, ctl):
        """The domain's commands for the agent: an object with help, errors, register(subparsers), dispatch(session,
        args) -> answer or None, and optionally preprocess(argv) and status_lines(session). None: only the platform's."""
        return None

    # ------------------------------------------------------------------ wake-ups (autopilot.py)
    def attach_autopilot(self, autopilot) -> None:
        """The wake-up engine exists: register the domain's watch kinds (autopilot.WatchKind)."""

    def wake_instructions(self, s: Session) -> str:
        """The last part of a wake-up's prompt: what to do now."""
        return ("Look at what woke you and decide the next step. If you need the researcher's decision, ask; if "
                "nothing is left to do, say so and set no more wake-ups.")

    def ledger_context(self, project: str | None, session_id: str | None = None) -> str:
        """What the domain keeps about a project's progress (e.g. the HPC's results and findings), for prompts."""
        return ""

    # ------------------------------------------------------------------ Telegram (bot.py)
    def bot_commands(self, bot):
        """The domain's Telegram commands and buttons: an object with help, menu [(command, description)],
        commands() -> [(names, handler)], on_button(update, q, chat_id, action, arg) -> handled,
        watch_cancelled(watch) -> handled, health_lines() -> [str], errors. None: only the platform's."""
        return None

    def auto_prompt(self, goal: str) -> str:
        """What /auto <goal> asks the agent."""
        return (f"Work on this on your own, over as many runs as it takes: {goal}\nSet wake-ups (`{self.command} wake`) "
                "for what you're waiting for and end your reply; carry on when woken until the goal is reached or "
                "nothing sensible is left, then tell me. Ask me when you need a decision.")

    # A project's folder: what it is, for /project new ("folder on the HPC": it must be given), and what having it
    # means (shown when a project is created). "": projects need none (it can be set later).
    project_folder = ""
    project_folder_note = ""

    def check_project_dir(self, path: str) -> str:
        """A project's folder as given: checked and normalised (raise ValueError if it isn't one)."""
        return (path or "").strip()

    async def project_notes_file(self, project) -> tuple[str, str] | None:
        """A project's notes file for /project notes: (where it is, its beginning); None if the domain keeps none."""
        return None

    # ------------------------------------------------------------------ the morning digest (digest.py)
    async def digest_sections(self, since: float, now: float) -> list[str]:
        """The domain's sections (HTML), first in the digest."""
        return []

    async def digest_closing(self) -> list[str]:
        """The domain's last lines in the digest."""
        return []

    # ------------------------------------------------------------------ the dashboard (dashboard.py)
    def dashboard_js(self) -> bytes:
        """The dashboard page's script: it builds the domain's tabs (nav) and renders them from dashboard_api."""
        return (b'document.getElementById("main").innerHTML = '
                b'"<div class=\\"empty\\">This bot has no dashboard yet.</div>";\n')

    async def dashboard_api(self, parts: list[str], query: dict[str, str]):
        """/api/<parts...>: (status, content type, body, extra headers), or None (404). Only called for the
        researcher (the platform checked Telegram's signed launch data)."""
        return None

    # ------------------------------------------------------------------ search (search.py)
    def search_rows(self) -> list[tuple]:
        """The domain's records to search besides conversations: (body, title, kind, ref, session, project, ts,
        what). Read in the event loop, so they may come from the store."""
        return []

    def search_line(self, hit, where: str, snippet: str, html: bool) -> str | None:
        """How a hit of the domain's kinds reads in the results (None: a plain line)."""
        return None

