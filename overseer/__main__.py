"""Entry point.

    overseer ssh setup [--import KEY]   (the HPC bot) create Overseer's own SSH key/config in the Overseer folder
    overseer ssh connect                open the SSH connection interactively (for 2FA clusters)
    overseer ssh test                   check the connection works
    overseer login claude_code|codex    log the agent CLI in (stored in the state dir)
    overseer sessions                   list agent conversations
    overseer open [conversation]        open a conversation in Claude Code / Codex (default: latest)
    overseer update                     update Claude Code and Codex (the bot also does it once a day)
    overseer watch [conversation]       follow what the agent is doing live: reasoning, commands, output
                                        (default: every conversation; --full for complete outputs)
    overseer check                      verify everything without starting the bot
    overseer config                     bring the config file up to date: what you changed, what's new, and
                                        <file>.new - the current example with your values (yours isn't touched)
    overseer reset [--yes]              wipe the bot's flags: its wake-ups, paused wake-ups, and the domain's own
                                        (the assistant's reminders, day logs, set-up flag); the bot must be stopped
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

import yaml

from .core.config import BACKENDS, Config, ConfigError, load_config

log = logging.getLogger("overseer")



from .core.app import build, prepare_local_dirs  # noqa: E402 (the app: core/app.py)


async def check(cfg: Config) -> int:
    """Verify the bot's setup (for the HPC: SSH and SLURM), the agent CLI and credentials without starting it."""
    from .core.agent import Agent
    from .core.app import domain_factory
    from .core.store import Store

    ok = True

    def report(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    prepare_local_dirs(cfg)
    report("local folders", True, f"state={cfg.local.state()} tmp={cfg.local.tmp()}")
    store = Store(cfg.local.db_path)
    if cfg.domain == "hpc":
        domain = await check_hpc(cfg, store, report)
        if domain is None:
            return 1
    else:
        domain = domain_factory(cfg)(cfg, store, None)
        domain.prepare()
        for name, good, detail in await domain.check():
            report(name, good, detail)
    try:
        cfg.telegram.token
        report("telegram token", True)
    except ConfigError as e:
        report("telegram token", False, str(e))
    if cfg.agent.enabled:
        agent = Agent(cfg, store, None, domain)  # type: ignore[arg-type]
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


async def check_hpc(cfg: Config, store, report):
    """SSH and SLURM; the HPC domain, or None if the HPC can't be reached."""
    from .hpc.domain import HpcDomain
    from .hpc.remote import HPC
    from .hpc.ssh import SSHRunner
    from .hpc.sshlink import ensure_master, key_permissions_ok, write_configs

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
        return None
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
    return HpcDomain(cfg, store, None, hpc)


def ssh_command(cfg: Config, action: str, import_from: str | None) -> int:
    if cfg.domain != "hpc":
        print(f"overseer ssh is for the HPC bot (this config's domain is {cfg.domain})", file=sys.stderr)
        return 2
    from .hpc.sshlink import connect_interactive, ensure_master, setup_key

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
    from .core.runners import make_runner

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
    from .core.store import Store

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
    from .core.runners import make_runner
    from .core.store import Store

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
    if cfg.domain == "hpc":
        from .hpc.domain import hpc_sandbox
        from .hpc.sshlink import ensure_master, write_configs
        write_configs(cfg)
        try:
            asyncio.run(ensure_master(cfg))
        except ConfigError as e:
            print(f"note: SSH connection to the HPC is not available, `ssh hpc` won't work in this session: {e}")
        sandbox = hpc_sandbox(cfg)
    else:
        from .core.app import domain_factory
        domain = domain_factory(cfg)(cfg, Store(cfg.local.db_path), None)
        domain.prepare()
        sandbox = domain.sandbox()
    runner = make_runner(cfg, rec.backend, sandbox)  # the bot's sandbox, exactly
    argv, env, cwd = runner.interactive_command(rec.cli_session_id, rec.id)
    # The bot runs the CLIs without a terminal (TERM=dumb, NO_COLOR); here they get the user's terminal.
    env.pop("NO_COLOR", None)
    env["TERM"] = os.environ.get("TERM") or "xterm-256color"
    env.update({k: os.environ[k] for k in INTERACTIVE_TERM_ENV if k in os.environ})
    if not shutil.which(argv[0], path=env.get("PATH")):
        print(f"{argv[0]!r} not found on PATH (install it, or set agent.{rec.backend}.command)", file=sys.stderr)
        return 1
    print(f"Opening {rec.id} in {rec.backend} (session {rec.cli_session_id}), in {cwd}.")
    print("Same sandbox" + (" and `ssh hpc` access" if cfg.domain == "hpc" else "") + " as the bot. Anything you add "
          "here becomes part of the conversation,")
    print("so don't type while the bot is working on it.\n")
    return subprocess.call(argv, env=env, cwd=str(cwd))


def config_command(path: str) -> int:
    """`overseer config`: the report, and <file>.new to look over and move into place."""
    from pathlib import Path

    from .core.configtool import update
    src = Path(path)
    try:
        report, text = update(src)
    except (OSError, ConfigError, yaml.YAMLError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    print(report)
    if text is None or text == src.read_text():
        print("\nYour file is up to date.")
        return 0
    new = src.with_name(src.name + ".new")
    new.write_text(text)
    shutil.copymode(src, new)
    print(f"\nWrote {new}: the current example with your values (and defaults for the rest) - it means the same "
          f"as your file. Look it over, then:\n  mv {new} {src}")
    return 0


PLATFORM_FLAGS = ("wakeups_paused:", "last_user_message:")  # (kv) per-conversation flags the platform keeps


def bot_running(cfg: Config) -> bool:
    """Is this bot running? (its control socket answers)"""
    import socket
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(1)
        s.connect(str(cfg.ctl_socket))
        return True
    except OSError:
        return False
    finally:
        s.close()


def reset_command(cfg: Config, yes: bool, ask=input) -> int:
    """Wipe the bot's flags (not its conversations, notes, settings or logins): the wake-ups its agent set, paused
    wake-ups, and the domain's own (e.g. the assistant's follow-ups, day logs and first-run flag)."""
    from .core.app import domain_factory
    from .core.store import Store

    if bot_running(cfg):
        print("The bot is running: stop it first (Ctrl-c in its window), then run this again.", file=sys.stderr)
        return 1
    if not cfg.local.db_path.exists():
        print("Nothing to reset (no state yet).")
        return 0
    store = Store(cfg.local.db_path)
    domain = domain_factory(cfg)(cfg, store, None)
    keys = [k for p in (*PLATFORM_FLAGS, *domain.reset_prefixes) for k in store.keys(p)]
    watches = store.watches()
    if not keys and not watches:
        print("Nothing to reset.")
        return 0
    print(f"The {cfg.domain} bot's flags ({cfg.local.db_path}):")
    for k in keys:
        print(f"  {k}")
    for w in watches:
        print(f"  wake-up w{w.id} ({w.kind}) of conversation {w.session_id}" + (f": {w.note}" if w.note else ""))
    print("Conversations, notes, settings and logins are kept.")
    if not yes and ask("Remove these? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing removed.")
        return 1
    for k in keys:
        store.unset(k)
    for w in watches:
        w.state = "cancelled"
        store.update_watch(w)
    print(f"Removed {len(keys)} flag(s) and {len(watches)} wake-up(s).")
    return 0


async def update_command(cfg: Config) -> int:
    """Update Claude Code and Codex with their own updaters (the bot does it once a day by itself)."""
    from .core.runners import make_runner
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
    ap.add_argument("command", choices=["run", "check", "login", "ssh", "sessions", "open", "watch", "update", "reset",
                                           "config"])
    ap.add_argument("target", nargs="?",
                    help="login: claude_code|codex; ssh: setup|connect|test; open: conversation (default: latest); "
                         "watch: conversation (default: all)")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--import", dest="import_key", help="ssh setup: copy this existing private key instead of generating one")
    ap.add_argument("--full", action="store_true", help="watch: show complete command output")
    ap.add_argument("--yes", action="store_true", help="reset: don't ask")
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
    if args.command == "config":  # (before loading it: an old file may not load any more)
        return config_command(args.config)
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
            from .core.live import watch
            return watch(cfg, args.target, full=args.full)
        if args.command == "reset":
            return reset_command(cfg, args.yes)
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
