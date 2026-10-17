"""One bot: the platform wired around its domain (config.yaml's `domain`, hpc by default).

Each domain runs as its own process (`overseer run -c <its config>`), with its own Telegram bot, state folder and
secrets. The domain module (overseer/<domain>/domain.py) provides make_domain(cfg, store, notifier); the domain then
prepares what it needs (prepare), starts its own services (services) and sets up its topics (start).
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os

from .config import Config, ConfigError

log = logging.getLogger("overseer")

CORE_MENU = [
    ("ask", "talk to the agent"), ("new", "fresh agent conversation"),
    ("backend", "show/switch claude_code or codex"), ("model", "show/set the agent's model"),
    ("effort", "show/set the agent's effort"), ("stop", "stop the agent's current run"),
    ("compact", "compact the conversation (free context)"), ("update", "update Claude Code and Codex"),
    ("auto", "wake-ups: list, off, or a goal to work on alone"), ("usage", "tokens and cost used today"),
    ("digest", "what happened since the last digest"), ("search", "search past work"),
    ("dashboard", "the dashboard Mini App"), ("subagents", "planner mode: on/off, models, threads"),
    ("feedback", "agents' feedback about Overseer; review"), ("skills", "skills agents added"),
    ("project", "projects: list, new, context, notes"), ("health", "connection status"), ("help", "help"),
]


def prepare_local_dirs(cfg: Config) -> None:
    """Create the local state and tmp dirs (inside local.allowed_roots), private to this user."""
    for d in (cfg.local.state(), cfg.local.tmp()):
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        if d.is_symlink() or d.stat().st_uid != os.getuid():
            raise ConfigError(f"{d} must be a real directory owned by you")
        os.chmod(d, 0o700)


def domain_factory(cfg: Config):
    """overseer.<domain>.domain.make_domain (by name: the platform never imports a domain itself)."""
    return importlib.import_module(f"overseer.{cfg.domain}.domain").make_domain


def build(cfg: Config):
    from telegram.ext import Application

    from .addons import Addons
    from .agent import Agent
    from .autopilot import Autopilot
    from .bot import Bot, TelegramNotifier
    from .ctl import ControlServer
    from .digest import Digest
    from .feedback import Feedback
    from .projects import Projects
    from .store import Store
    from .subagents import Subagents

    prepare_local_dirs(cfg)
    store = Store(cfg.local.db_path)
    tasks: list[asyncio.Task] = []

    async def post_init(app: Application) -> None:
        if await notifier.detect_topics():  # a topic per project, and "general", in the chat with the bot
            await notifier.ensure_general_topic()
            for proj in projects.active():
                await notifier.ensure_topic(proj.name)
            await domain.start(notifier, projects)  # the domain's topics (e.g. the HPC's jobs topics)
            await notifier.ensure_feedback_topic()  # agents' feedback about Overseer
        elif projects.active():
            log.warning("projects are defined but the bot has no topics in private chats: their messages are "
                        "labelled in the main chat (enable threaded mode for the bot in @BotFather)")
        await ctl.start()  # the agent's control command
        await bot.offer_cut_off_runs()  # runs the last Overseer didn't finish: ▶️ Continue / ✖ Dismiss
        subagents.resume()  # queued subagents get their turn
        for name, coro in domain.services():  # e.g. the HPC's SLURM monitor
            tasks.append(asyncio.create_task(coro, name=name))
        tasks.append(asyncio.create_task(agent.update_loop(), name="cli-updates"))
        tasks.append(asyncio.create_task(digest.loop(), name="digest"))
        tasks.append(asyncio.create_task(autopilot.loop(), name="timers"))
        if dashboard:
            try:
                await dashboard.start()
            except OSError as e:
                log.error("dashboard: can't listen on %s:%s (%s)", cfg.dashboard.host, cfg.dashboard.port, e)
            await bot.set_dashboard_button()
        await app.bot.set_my_commands([*(bot.ext.menu if bot.ext else []), *CORE_MENU])

    async def post_stop(app: Application) -> None:
        agent.shutting_down = True  # runs cut off from here on are offered again at the next start
        from .runners import kill_running
        if kill_running():  # (their CLIs are in process groups of their own: Ctrl-C doesn't reach them)
            log.info("stopped the agent runs in progress; they're offered again at the next start")

    async def post_shutdown(app: Application) -> None:
        for t in tasks:
            t.cancel()
        await ctl.stop()
        if dashboard:
            await dashboard.stop()

    app = (
        Application.builder()
        .token(cfg.telegram.token)
        .concurrent_updates(True)
        .post_init(post_init)
        .post_stop(post_stop)
        .post_shutdown(post_shutdown)
        .build()
    )
    notifier = TelegramNotifier(app, cfg.telegram.allowed_chat_ids, store, rich=cfg.telegram.rich_messages,
                                streaming=cfg.telegram.streaming)
    domain = domain_factory(cfg)(cfg, store, notifier)  # this bot: the HPC (Overseer), ...
    domain.prepare()  # e.g. the HPC's ssh configs and the agent's ssh wrappers
    agent = Agent(cfg, store, notifier, domain)
    domain.build()  # its own parts, now that the agent exists (e.g. the HPC's monitor)
    autopilot = Autopilot(cfg, store, agent, notifier)  # wake-ups (the domain adds its watch kinds)
    subagents = Subagents(cfg, store, agent, notifier)  # planner mode (off unless turned on or approved)
    Feedback(cfg, store, agent, notifier)  # agents' feedback about Overseer, in the 🛠 feedback topic
    Addons(store, agent, notifier)  # skills agents proposed and the researcher approved
    projects = Projects(cfg, store, agent)
    for name in projects.sync_config():
        log.info("project %s created from config.yaml", name)
    ctl = ControlServer(cfg, agent, projects, notifier)
    digest = Digest(cfg, store, notifier, agent)  # the morning digest, in the general topic
    bot = Bot(cfg, app, store, agent, notifier, projects)
    bot.digest = digest
    dashboard = None
    if cfg.dashboard.enabled:  # the Mini App: on this machine, reachable on your tailnet (tailscale serve, Caddy)
        from .dashboard import Dashboard
        dashboard = Dashboard(cfg, store, cfg.telegram.token, domain)
    bot.dashboard = dashboard
    bot.register()
    return app
