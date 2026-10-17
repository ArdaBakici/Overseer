"""Domains: what makes the platform one particular bot.

The platform (core/) runs agent conversations from Telegram: the Claude Code / Codex runners and their sandbox, the
Telegram layer, the agent's control command, wake-ups, transcripts and search, usage, the digest and the dashboard.
A domain (hpc/, the personal assistant, ...) plugs into it what is its own: the agent's instructions, the context it
gets at the start of a conversation, what its sandbox may reach, what it watches in the agent's commands, and what
happens after a run. Each domain runs as its own bot and process, with its own state and secrets.

Every hook has a neutral default, so a domain overrides only what it needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .runners import Activity, Sandbox

if TYPE_CHECKING:
    from .agent import Agent, Session
    from .config import Config
    from .notify import Notifier
    from .store import Store


class Domain:
    name = "base"
    command = "ov"  # the agent's command for talking to this bot

    def __init__(self, cfg: Config, store: Store, notifier: Notifier):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier
        self.agent: Agent | None = None

    def attach(self, agent: Agent) -> None:
        self.agent = agent

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

    async def after_run(self, s: Session, activity: Activity, state: Any, reply: str) -> None:
        """After the CLI answered, before the reply is sent."""

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

    async def project_notes_file(self, project) -> tuple[str, str] | None:
        """A project's notes file for /project notes: (where it is, its beginning); None if the domain keeps none."""
        return None

