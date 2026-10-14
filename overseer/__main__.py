"""Entry point.

    overseer ssh setup [--import KEY]   create Overseer's own SSH key/config in the Overseer folder
    overseer ssh connect                open the SSH connection interactively (for 2FA clusters)
    overseer ssh test                   check the connection works
    overseer login claude_code|codex    log the agent CLI in (stored in the state dir)
    overseer sessions                   list agent conversations
    overseer open [conversation]        open a conversation in Claude Code / Codex (default: latest)
    overseer update                     update Claude Code and Codex (the bot also does it once a day)
    overseer watch [conversation]       follow what the agent is doing live: reasoning, commands, output
                                        (default: every conversation; --full for complete outputs)
    overseer check                      verify everything without starting the bot
    overseer run                        start the bot
All take -c/--config (default: config.yaml).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys

from .config import BACKENDS, Config, ConfigError, load_config

log = logging.getLogger("overseer")


def prepare_local_dirs(cfg: Config) -> None:
    """Create the local state and tmp dirs (inside local.allowed_roots), private to this user."""
    for d in (cfg.local.state(), cfg.local.tmp()):
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        if d.is_symlink() or d.stat().st_uid != os.getuid():
            raise ConfigError(f"{d} must be a real directory owned by you")
        os.chmod(d, 0o700)


def build(cfg: Config):
    from telegram.ext import Application

    from .agent import Agent
    from .autopilot import Autopilot
    from .bot import Bot, TelegramNotifier
    from .ctl import ControlServer
    from .diagnose import Reporter
    from .hpc import HPC
    from .monitor import Monitor
    from .projects import Projects
    from .ssh import SSHRunner
    from .sshlink import write_configs
    from .store import Store

    prepare_local_dirs(cfg)
    write_configs(cfg)
    store = Store(cfg.local.db_path)
    hpc = HPC(SSHRunner(cfg), cfg.slurm.setup)
    tasks: list[asyncio.Task] = []

    async def post_init(app: Application) -> None:
        if await notifier.detect_topics():  # a topic per project, and "general", in the chat with the bot
            await notifier.ensure_general_topic()
            for proj in projects.active():
                await notifier.ensure_topic(proj.name)
            await notifier.ensure_jobs_topics([p.name for p in projects.active()])  # live job cards
            await notifier.ensure_feedback_topic()  # agents' feedback about Overseer
        elif projects.active():
            log.warning("projects are defined but the bot has no topics in private chats: their messages are "
                        "labelled in the main chat (enable threaded mode for the bot in @BotFather)")
        await ctl.start()  # the agent's `ov` command
        await bot.offer_cut_off_runs()  # runs the last Overseer didn't finish: ▶️ Continue / ✖ Dismiss
        subagents.resume()  # queued subagents get their turn
        tasks.append(asyncio.create_task(monitor.run_forever(), name="monitor"))
        tasks.append(asyncio.create_task(agent.update_loop(), name="cli-updates"))
        await app.bot.set_my_commands([
            ("jobs", "active and recent jobs"), ("job", "status of a job"), ("log", "tail a job's log"),
            ("fix", "ask the agent to fix a job"), ("ask", "talk to the agent"), ("new", "fresh agent conversation"),
            ("backend", "show/switch claude_code or codex"), ("model", "show/set the agent's model"),
            ("effort", "show/set the agent's effort"), ("stop", "stop the agent's current run"),
            ("compact", "compact the conversation (free context)"), ("update", "update Claude Code and Codex"),
            ("auto", "wake-ups: list, off, or a goal to work on alone"), ("usage", "tokens and cost used today"),
            ("subagents", "planner mode: on/off, models, threads"),
            ("results", "experiments and what each showed"), ("findings", "insights and dead ends"),
            ("feedback", "agents' feedback about Overseer; review"), ("skills", "skills agents added"), ("project", "projects: list, new, context, notes"),
            ("assign", "move a job to a project"), ("health", "connection status"), ("help", "help"),
        ])

    async def post_stop(app: Application) -> None:
        agent.shutting_down = True  # runs cut off from here on are offered again at the next start
        from .runners import kill_running
        if kill_running():  # (their CLIs are in process groups of their own: Ctrl-C doesn't reach them)
            log.info("stopped the agent runs in progress; they're offered again at the next start")

    async def post_shutdown(app: Application) -> None:
        for t in tasks:
            t.cancel()
        await ctl.stop()

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
    agent = Agent(cfg, store, notifier, hpc)

    async def report_text(system: str, prompt: str) -> str:
        return await agent.runner().text(system=system, prompt=prompt)

    reporter = Reporter(report_text, cfg.agent.enabled, cfg.agent.llm_status_updates)
    monitor = Monitor(cfg, hpc, store, notifier, reporter)
    agent.monitor = monitor
    monitor.autopilot = Autopilot(cfg, store, agent, notifier, hpc)
    from .subagents import Subagents
    subagents = Subagents(cfg, store, agent, notifier)  # planner mode (off unless turned on or approved)
    from .addons import Addons
    from .feedback import Feedback
    Feedback(cfg, store, agent, notifier)  # agents' feedback about Overseer, in the 🛠 feedback topic
    Addons(store, agent, notifier)  # skills agents proposed and the researcher approved
    projects = Projects(cfg, store, agent)
    for name in projects.sync_config():
        log.info("project %s created from config.yaml", name)
    ctl = ControlServer(cfg, agent, projects, notifier, hpc)
    bot = Bot(cfg, app, store, hpc, monitor, agent, notifier, projects)
    bot.register()
    return app


async def check(cfg: Config) -> int:
    """Verify SSH, SLURM, the agent CLI and credentials without starting the bot."""
    from .agent import Agent
    from .hpc import HPC
    from .ssh import SSHRunner
    from .sshlink import ensure_master, key_permissions_ok, write_configs
    from .store import Store

    ok = True

    def report(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    prepare_local_dirs(cfg)
    report("local folders", True, f"state={cfg.local.state()} tmp={cfg.local.tmp()}")
    write_configs(cfg)
    report("ssh key", key_permissions_ok(cfg), str(cfg.ssh_key) if key_permissions_ok(cfg)
           else f"missing or group/world-readable: {cfg.ssh_key} (run: overseer ssh setup)")
    try:
        await ensure_master(cfg)
        hpc = HPC(SSHRunner(cfg), cfg.slurm.setup)
        await hpc.connect()
        report("ssh connection", True, f"{cfg.ssh.host} -> {hpc.user}@{cfg.ssh.hostname} (home {hpc.home})")
    except Exception as e:
        report("ssh connection", False, str(e))
        return 1
    slurm_ok = True
    for name, coro in [("squeue", hpc.squeue()), ("sacct", hpc.sacct_recent(60))]:
        try:
            jobs = await coro
            report(name, True, f"{len(jobs)} job(s)")
        except Exception as e:
            slurm_ok = False
            report(name, False, str(e))
    if not slurm_ok:
        print("  SLURM isn't on PATH even in a login shell on the HPC. Log in and run `type squeue`;\n"
              "  if you normally need e.g. `module load slurm`, put that in slurm.setup in config.yaml.\n"
              f"  (login-shell PATH Overseer saw: {hpc.path or 'none - could not run bash -lc'})")
    try:
        cfg.telegram.token
        report("telegram token", True)
    except ConfigError as e:
        report("telegram token", False, str(e))
    if cfg.agent.enabled:
        store = Store(cfg.local.db_path)
        agent = Agent(cfg, store, None, hpc)  # type: ignore[arg-type]
        for backend in BACKENDS:
            runner = agent.runner(backend)
            cmd = runner.c.command
            if not shutil.which(cmd):
                print(f"[skip] {backend}: {cmd!r} not installed")
                continue
            try:
                reply = await asyncio.wait_for(
                    runner.text(system="You are a connectivity check.", prompt="Reply with exactly: OK"), 120)
                report(f"{backend} model call", "OK" in reply, reply[:200])
            except asyncio.TimeoutError:
                report(f"{backend} model call", False,
                       f"no answer within 2 minutes (not logged in?)\n  (log in with: overseer login {backend} -c <config>)")
            except Exception as e:
                report(f"{backend} model call", False,
                       f"{e}\n  (log in with: overseer login {backend} -c <config>)")
        print(f"active agent backend: {agent.backend}")
    return 0 if ok else 1


def ssh_command(cfg: Config, action: str, import_from: str | None) -> int:
    from .sshlink import connect_interactive, ensure_master, setup_key

    prepare_local_dirs(cfg)
    if action == "setup":
        pub = setup_key(cfg, import_from)
        print(f"Overseer's SSH files are in {cfg.ssh_dir} (config, known_hosts, {cfg.ssh.key_name}).")
        print("\nAdd this public key to ~/.ssh/authorized_keys on the HPC (one line):\n")
        print(f"  {pub}\n")
        print("For example, from a machine where you can already log in:")
        print(f"  echo '{pub}' | ssh {cfg.ssh.user + '@' if cfg.ssh.user else ''}{cfg.ssh.hostname} "
              "'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys'")
        print("\nThen run: overseer ssh test")
        return 0
    if action == "connect":
        return connect_interactive(cfg)
    if action == "test":
        try:
            asyncio.run(ensure_master(cfg))
        except ConfigError as e:
            print(f"FAIL: {e}")
            return 1
        agent_ssh = cfg.local.tmp("agent-bin", "ssh")
        out = subprocess.run([str(agent_ssh), cfg.ssh.host, "echo connected as $(whoami) on $(hostname)"],
                             capture_output=True, text=True, timeout=60)
        print(out.stdout.strip() or out.stderr.strip())
        return out.returncode
    print("usage: overseer ssh setup|connect|test", file=sys.stderr)
    return 2


def login(cfg: Config, backend: str, extra: list[str]) -> int:
    from .runners import make_runner

    if backend not in BACKENDS:
        print(f"usage: overseer login {'|'.join(BACKENDS)}", file=sys.stderr)
        return 2
    prepare_local_dirs(cfg)
    runner = make_runner(cfg, backend)
    argv, env = runner.login_command(extra)
    if not shutil.which(argv[0], path=env.get("PATH")):
        print(f"{argv[0]!r} not found on PATH (install it, or set agent.{backend}.command)", file=sys.stderr)
        return 1
    print(f"Running {' '.join(argv)} with its config in {runner.home()}")
    return subprocess.call(argv, env=env)


def _ago(ts: float) -> str:
    import time

    d = max(0, int(time.time() - (ts or 0)))
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if d >= n:
            return f"{d // n}{unit} ago"
    return f"{d}s ago"


def sessions_command(cfg: Config, limit: int = 20) -> int:
    from .store import Store

    rows = Store(cfg.local.db_path).list_sessions(limit)
    if not rows:
        print("No agent conversations yet.")
        return 0
    print(f"{'conversation':<20} {'project':<12} {'backend':<12} {'job':<10} {'last used':<10} cli session")
    for r in rows:
        print(f"{r.id:<20} {r.project or '-':<12} {r.backend or '-':<12} {r.job_id or '-':<10} {_ago(r.updated):<10} "
              f"{r.cli_session_id or '(none yet)'}")
    print("\nOpen one with: overseer open <conversation>   (no argument = the most recent)")
    print("Follow one live: overseer watch <conversation>   (no argument = all of them)")
    return 0


INTERACTIVE_TERM_ENV = ["COLORTERM", "TERM_PROGRAM", "TERM_PROGRAM_VERSION", "COLUMNS", "LINES"]


def open_command(cfg: Config, target: str | None) -> int:
    from .runners import make_runner
    from .sshlink import ensure_master, write_configs
    from .store import Store

    prepare_local_dirs(cfg)
    rows = [r for r in Store(cfg.local.db_path).list_sessions(500) if r.cli_session_id]
    if target:
        rows = [r for r in rows if r.id == target] or [r for r in rows if r.id.startswith(target)
                                                        or (r.cli_session_id or "").startswith(target)]
    if not rows:
        print("No matching conversation with a CLI session (see: overseer sessions).", file=sys.stderr)
        return 1
    if target and len(rows) > 1 and rows[0].id != target:
        print(f"{target!r} matches several conversations: {', '.join(r.id for r in rows)}", file=sys.stderr)
        return 1
    rec = rows[0]
    write_configs(cfg)
    try:
        asyncio.run(ensure_master(cfg))
    except ConfigError as e:
        print(f"note: SSH connection to the HPC is not available, `ssh hpc` won't work in this session: {e}")
    runner = make_runner(cfg, rec.backend)
    argv, env, cwd = runner.interactive_command(rec.cli_session_id, rec.id)
    # The bot runs the CLIs without a terminal (TERM=dumb, NO_COLOR); here they get the user's terminal.
    env.pop("NO_COLOR", None)
    env["TERM"] = os.environ.get("TERM") or "xterm-256color"
    env.update({k: os.environ[k] for k in INTERACTIVE_TERM_ENV if k in os.environ})
    if not shutil.which(argv[0], path=env.get("PATH")):
        print(f"{argv[0]!r} not found on PATH (install it, or set agent.{rec.backend}.command)", file=sys.stderr)
        return 1
    print(f"Opening {rec.id} in {rec.backend} (session {rec.cli_session_id}), in {cwd}.")
    print("Same sandbox and `ssh hpc` access as the bot. Anything you add here becomes part of the conversation,")
    print("so don't type while the bot is working on it.\n")
    return subprocess.call(argv, env=env, cwd=str(cwd))


async def update_command(cfg: Config) -> int:
    """Update Claude Code and Codex with their own updaters (the bot does it once a day by itself)."""
    from .runners import make_runner
    rc = 0
    for backend in BACKENDS:
        r = make_runner(cfg, backend)
        if not r.installed():
            print(f"{backend}: {r.c.command!r} not installed")
            continue
        before = await r.version()
        print(f"{backend}: {before or 'version unknown'} - updating...")
        ok, out = await r.update()
        print(out)
        print(f"{backend}: {'now ' + (await r.version() or '?') if ok else 'update FAILED'}\n")
        rc = rc or (0 if ok else 1)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="overseer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "check", "login", "ssh", "sessions", "open", "watch", "update"])
    ap.add_argument("target", nargs="?",
                    help="login: claude_code|codex; ssh: setup|connect|test; open: conversation (default: latest); "
                         "watch: conversation (default: all)")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--import", dest="import_key", help="ssh setup: copy this existing private key instead of generating one")
    ap.add_argument("--full", action="store_true", help="watch: show complete command output")
    ap.add_argument("-v", "--verbose", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:  # everything after `--` goes to the CLI's login command
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        cfg = load_config(args.config)
    except (OSError, ConfigError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    try:
        if args.command == "check":
            return asyncio.run(check(cfg))
        if args.command == "login":
            return login(cfg, args.target or cfg.agent.backend, extra)
        if args.command == "ssh":
            return ssh_command(cfg, args.target or "", args.import_key)
        if args.command == "sessions":
            return sessions_command(cfg)
        if args.command == "open":
            return open_command(cfg, args.target)
        if args.command == "watch":
            from .live import watch
            return watch(cfg, args.target, full=args.full)
        if args.command == "update":
            return asyncio.run(update_command(cfg))
        app = build(cfg)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    app.run_polling(allowed_updates=["message", "callback_query", "stopped_message_generation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
