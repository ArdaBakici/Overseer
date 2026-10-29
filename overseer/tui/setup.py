"""`overseer setup`: set a bot up step by step, in the terminal - the HPC bot (config.yaml) or the personal
assistant (`-c assistant.yaml`). Each step shows what's already done (detected every time, so it can be run again to
change something); steps that need the terminal (installing, logging in, approving a 2FA login) pause the screen,
run there, and come back. It writes config.yaml (the example, comments and all, with your values) and .env (the
secrets, private to you), and finally starts the bot in the background.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (Button, Checkbox, ContentSwitcher, Footer, Header, Input, Label, ListItem, ListView,
                             RichLog, Select, Static)

from ..core import daemon
from . import steps as S

PLACEHOLDERS = {"ssh.hostname": "login.cluster.edu", "ssh.user": "your-netid"}
EXAMPLE_CHAT = 123456789
ICON = {"ok": "✅", "warn": "⚠️ ", "todo": "⬜"}

HPC_STEPS = [("welcome", "Welcome"), ("prereq", "Install what's needed"), ("telegram", "Telegram bot"),
             ("hpc", "Connect to the HPC"), ("cluster", "Cluster settings"), ("agents", "AI agents"),
             ("prefs", "Preferences"), ("check", "Check everything"), ("start", "Start the bot")]
ASSISTANT_STEPS = [("welcome", "Welcome"), ("prereq", "Install what's needed"), ("telegram", "Telegram bot"),
                   ("vault", "Obsidian vault"), ("agents", "AI agents"), ("prefs", "Preferences"),
                   ("check", "Check everything"), ("start", "Start the bot")]


class Setup(App):
    TITLE = "Overseer setup"
    CSS = """
    #steps { width: 30; border-right: solid $primary-background; }
    #steps ListItem { padding: 0 1; }
    #panes { height: 1fr; }
    .pane { padding: 1 2; }
    .pane Input, .pane Select { width: 60; margin-bottom: 1; }
    .pane Button { margin: 0 1 1 0; }
    .row { height: auto; }
    .hint { color: $text-muted; margin-bottom: 1; }
    .title { text-style: bold; margin-bottom: 1; }
    #out { height: 12; border-top: solid $primary-background; }
    """
    BINDINGS = [Binding("ctrl+n", "next", "next step"), Binding("ctrl+q", "quit", "quit")]

    def __init__(self, config: Path):
        super().__init__()
        self.config = config.resolve()
        self.base = self.config.parent
        self.new = not self.config.exists()  # (then its values are the example's until you save yours)
        self.cluster_saved = False
        domain = "assistant" if "assistant" in config.name else "hpc"
        if self.config.exists():
            domain = str(S.read_yaml(self.config).get("domain") or "hpc")
        example = S.configtool.example_path(domain)
        self.draft = S.Draft(self.config if self.config.exists() else example, domain)
        self.draft.path = self.config  # (a new file starts as the example: then your values go in)
        self.steps = ASSISTANT_STEPS if self.draft.domain == "assistant" else HPC_STEPS
        self.env_path = self.base / ".env"
        self.ssh_ok = False
        self.check_ok: bool | None = None
        self.partitions: list[S.Partition] = []

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ListView(*[ListItem(Label(title), id=f"s-{sid}") for sid, title in self.steps], id="steps")
            with Vertical():
                with ContentSwitcher(id="panes", initial="welcome"):
                    for sid, _ in self.steps:
                        with VerticalScroll(id=sid, classes="pane"):
                            yield from getattr(self, f"pane_{sid}")()
                yield RichLog(id="out", wrap=True, markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        what = "the personal assistant" if self.draft.domain == "assistant" else "the HPC bot"
        self.sub_title = f"{what} · {self.config.name}"
        self.refresh_steps()

    def say(self, text: str, style: str = "") -> None:
        self.query_one("#out", RichLog).write(Text(text, style=style))

    def refresh_steps(self) -> None:
        for sid, title in self.steps:
            state = getattr(self, f"status_{sid}", lambda: "")()
            self.query_one(f"#s-{sid}", ListItem).query_one(Label).update(f"{ICON.get(state, '  ')} {title}")

    @on(ListView.Highlighted, "#steps")
    def go(self, event: ListView.Highlighted) -> None:
        if event.item and event.item.id:
            sid = event.item.id[2:]
            self.query_one("#panes", ContentSwitcher).current = sid
            getattr(self, f"show_{sid}", lambda: None)()

    def action_next(self) -> None:
        lv = self.query_one("#steps", ListView)
        lv.index = min((lv.index or 0) + 1, len(self.steps) - 1)

    def save(self) -> bool:
        backup = self.draft.save()
        if backup:
            self.say(f"Your previous {self.config.name} is kept as {backup.name}", "dim")
        self.say(f"Saved {self.config.name}", "green")
        self.refresh_steps()
        return True

    def in_terminal(self, cmds: list[list[str]], note: str = "") -> int:
        """Run commands in the terminal itself (they may ask for a password, a code, a browser login)."""
        rc = 0
        with self.suspend():
            print(f"\n{note}\n" if note else "")
            for cmd in cmds:
                print(f"$ {shlex.join(cmd)}")
                rc = subprocess.call(cmd, cwd=str(self.base))
                if rc:
                    print(f"\n(it ended with exit code {rc})")
                    break
            input("\nPress Enter to go back to the setup... ")
        self.refresh_steps()
        return rc

    # ------------------------------------------------------------------ 1. welcome
    def pane_welcome(self):
        bot = "your personal assistant" if self.draft.domain == "assistant" else "Overseer, the HPC bot"
        yield Static(f"Setting up {bot}.", classes="title")
        yield Static(
            "Go through the steps on the left (Ctrl+N: the next one). Each shows what's already done - ✅ done, "
            "⬜ to do - so you can come back any time to change something.\n\n"
            f"This writes {self.config.name} (your settings, with explanations) and .env (your secrets, private to "
            "you) in " + str(self.base) + ". Steps that need the terminal - installing, logging in - pause this "
            "screen and come back.\n\n" + ("The HPC bot is set up with `overseer setup`; the assistant with "
                                           "`overseer setup -c assistant.yaml`."), classes="hint")
        yield Button("Begin", id="begin", variant="primary")

    @on(Button.Pressed, "#begin")
    def begin(self) -> None:
        self.action_next()

    # ------------------------------------------------------------------ 2. prerequisites
    def pane_prereq(self):
        yield Static("What Overseer needs on this machine", classes="title")
        yield Static(id="prereq-list")
        yield Static("Which AI agents? (one is enough; with both you can switch between them)", classes="hint")
        yield Checkbox("Claude Code (a Claude subscription or an Anthropic API key)", True, id="want-claude")
        yield Checkbox("Codex (a ChatGPT subscription or an OpenAI API key)", False, id="want-codex")
        with Horizontal(classes="row"):
            yield Button("Install what's missing", id="install", variant="primary")
            yield Button("Check again", id="recheck")

    def show_prereq(self) -> None:
        lines = []
        for c in S.prerequisites():
            mark = "✅" if c.ok else ("⬜" if c.optional else "❌")
            lines.append(f"{mark} {c.name:<7} {c.detail}")
        self.query_one("#prereq-list", Static).update("\n".join(lines) + "\n")

    def status_prereq(self) -> str:
        checks = S.prerequisites()
        must = all(c.ok for c in checks if not c.optional)
        return "ok" if must and any(c.ok for c in checks if c.name in ("claude", "codex")) else "todo"

    @on(Button.Pressed, "#recheck")
    def recheck(self) -> None:
        self.show_prereq()
        self.refresh_steps()

    @on(Button.Pressed, "#install")
    def install(self) -> None:
        clis = [b for b, w in (("claude_code", "#want-claude"), ("codex", "#want-codex"))
                if self.query_one(w, Checkbox).value]
        cmds = S.install_commands(S.prerequisites(), clis)
        if not cmds:
            self.say("Nothing to install.", "green")
            return
        if S.package_manager() is None and any(c[0] == "sudo" for c in cmds):
            self.say("No known package manager (apt, dnf, pacman, zypper): install ssh, bubblewrap, socat and npm "
                     "yourself.", "red")
        self.in_terminal(cmds, "Installing what Overseer needs (sudo asks for your password):")
        self.show_prereq()

    # ------------------------------------------------------------------ 3. Telegram
    @property
    def token_env(self) -> str:
        default = "ASSISTANT_BOT_TOKEN" if self.draft.domain == "assistant" else "TELEGRAM_BOT_TOKEN"
        return str(self.draft.get("telegram.token_env") or default)

    def _chat_ids(self) -> list[int]:
        ids = [int(x) for x in self.draft.get("telegram.allowed_chat_ids") or [] if str(x).lstrip("-").isdigit()]
        return [x for x in ids if x != EXAMPLE_CHAT]

    def pane_telegram(self):
        yield Static("Your Telegram bot", classes="title")
        yield Static("1. In Telegram, open @BotFather and send /newbot. Pick a name and a username ending in 'bot'. "
                     "It answers with a token: paste it here.", classes="hint")
        yield Input(placeholder="the token from @BotFather (123456789:AA...)", password=True, id="token")
        yield Button("Check and save the token", id="save-token", variant="primary")
        yield Static(id="bot-info")
        yield Static("2. Your chat id: only you may talk to the bot. Press Detect, then send any message to your "
                     "bot in Telegram (open it, press Start).", classes="hint")
        yield Input(placeholder="your chat id, e.g. 123456789", id="chat-id")
        with Horizontal(classes="row"):
            yield Button("Detect my chat id", id="detect")
            yield Button("Save the chat id", id="save-chat")

    def show_telegram(self) -> None:
        if S.read_env(self.env_path).get(self.token_env):
            self.query_one("#token", Input).placeholder = f"saved in .env as {self.token_env} ✓ (paste to replace)"
        ids = self._chat_ids()
        if ids and not self.query_one("#chat-id", Input).value:
            self.query_one("#chat-id", Input).value = ", ".join(map(str, ids))

    def status_telegram(self) -> str:
        return "ok" if S.read_env(self.env_path).get(self.token_env) and self._chat_ids() else "todo"

    @on(Button.Pressed, "#save-token")
    def save_token(self) -> None:
        token = self.query_one("#token", Input).value.strip() or S.read_env(self.env_path).get(self.token_env, "")
        if not token:
            self.say("Paste the token from @BotFather first.", "red")
            return
        self._check_token(token)

    @work(thread=True, exclusive=True, group="telegram")
    def _check_token(self, token: str) -> None:
        try:
            info = S.bot_info(token)
        except S.TelegramError as e:
            self.call_from_thread(self.say, f"✗ {e}", "red")
            return
        S.set_env(self.env_path, self.token_env, token)
        if self.draft.domain == "assistant" and not self.draft.get("telegram.token_env"):
            self.draft.set("telegram.token_env", self.token_env)
        topics = ("threaded mode is on: projects get topics of their own" if info["topics"] else
                  "threaded mode is off: everything goes to one chat. For topics (recommended), in @BotFather open "
                  "your bot → Bot Settings → Threaded mode")

        def done() -> None:
            self.query_one("#token", Input).value = ""
            self.query_one("#bot-info", Static).update(f"✅ @{info['username']} ({info['name']}) · {topics}\n")
            self.say(f"Token saved in .env as {self.token_env}.", "green")
            self.show_telegram()
            self.refresh_steps()
        self.call_from_thread(done)

    @on(Button.Pressed, "#detect")
    def detect(self) -> None:
        token = S.read_env(self.env_path).get(self.token_env, "")
        if not token:
            self.say("Save the bot's token first.", "red")
            return
        cfg_state = self._state_dir()
        if cfg_state and daemon.state(cfg_state).running:
            self.say("The bot is running (it reads its own messages): stop it first (`overseer stop`), or type your "
                     "id.", "red")
            return
        self.say("Waiting for your message: send anything to your bot in Telegram now (60 s)...", "yellow")
        self._detect(token)

    @work(thread=True, exclusive=True, group="telegram")
    def _detect(self, token: str) -> None:
        try:
            found = S.wait_for_chat(token, wait=60)
        except S.TelegramError as e:
            self.call_from_thread(self.say, f"✗ {e}", "red")
            return
        if not found:
            self.call_from_thread(self.say, "No message arrived: press Detect and try again.", "red")
            return
        chat, name = found

        def done() -> None:
            self.query_one("#chat-id", Input).value = str(chat)
            self.say(f"Found you: {name} ({chat}).", "green")
            self.save_chat()
        self.call_from_thread(done)

    @on(Button.Pressed, "#save-chat")
    def save_chat(self) -> None:
        raw = self.query_one("#chat-id", Input).value
        ids = [int(x) for x in raw.replace(",", " ").split() if x.lstrip("-").isdigit()]
        if not ids:
            self.say("A chat id is a number, e.g. 123456789.", "red")
            return
        self.draft.set("telegram.allowed_chat_ids", ids)
        self.save()

    def _state_dir(self) -> Path | None:
        try:
            from .data import peek_config
            return peek_config(self.config).local.state() if self.config.exists() else None
        except Exception:
            return None

    # ------------------------------------------------------------------ 4. the HPC connection
    def pane_hpc(self):
        yield Static("Connecting to the HPC", classes="title")
        yield Static("Overseer uses its own SSH key (in ./state/ssh, never ~/.ssh). Fill in how you log in to the "
                     "cluster, then create the key and add it there.", classes="hint")
        yield Input(placeholder="the cluster's login node, e.g. login.cluster.edu", id="hostname")
        yield Input(placeholder="your username there", id="user")
        yield Input(placeholder="a gateway you ssh through first (optional), e.g. you@gateway.cluster.edu",
                    id="proxy")
        yield Input(placeholder="port (22)", id="port")
        yield Button("Save", id="save-hpc", variant="primary")
        yield Static("The key", classes="title")
        with Horizontal(classes="row"):
            yield Button("Create Overseer's key", id="make-key")
            yield Input(placeholder="or import an existing key, e.g. ~/.ssh/id_ed25519", id="import-key")
            yield Button("Import", id="import")
        yield Static(id="pubkey")
        with Horizontal(classes="row"):
            yield Button("Add the key to the cluster for me", id="copy-key")
            yield Button("Test the connection", id="ssh-test", variant="primary")
            yield Button("Log in with 2FA / Duo", id="ssh-connect")

    def show_hpc(self) -> None:
        for wid, key in (("hostname", "ssh.hostname"), ("user", "ssh.user"), ("proxy", "ssh.proxy_jump"),
                         ("port", "ssh.port")):
            v = self.draft.get(key)
            if v and v != PLACEHOLDERS.get(key) and not self.query_one(f"#{wid}", Input).value:
                self.query_one(f"#{wid}", Input).value = str(v)
        pub = self._pubkey()
        if pub:
            self._show_pub(pub)

    def _pubkey(self) -> str:
        for p in (self.base / "state" / "ssh").glob("*.pub") if (self.base / "state" / "ssh").is_dir() else []:
            return p.read_text().strip()
        return ""

    def _show_pub(self, pub: str) -> None:
        self.query_one("#pubkey", Static).update(
            f"Overseer's public key (it goes in ~/.ssh/authorized_keys on the cluster):\n{pub}\n")

    def _hpc_set(self) -> bool:
        return all(self.draft.get(k) and self.draft.get(k) != v for k, v in PLACEHOLDERS.items())

    def status_hpc(self) -> str:
        if self.ssh_ok:
            return "ok"
        return "warn" if self._hpc_set() and self._pubkey() else "todo"

    @on(Button.Pressed, "#save-hpc")
    def save_hpc(self) -> bool:
        host = self.query_one("#hostname", Input).value.strip()
        user = self.query_one("#user", Input).value.strip()
        if not host or not user:
            self.say("The login node and your username are needed.", "red")
            return False
        self.draft.set("ssh.hostname", host)
        self.draft.set("ssh.user", user)
        self.draft.set("ssh.proxy_jump", self.query_one("#proxy", Input).value.strip() or None)
        port = self.query_one("#port", Input).value.strip()
        self.draft.set("ssh.port", int(port) if port.isdigit() else 22)
        return self.save()

    @on(Button.Pressed, "#make-key")
    def make_key(self) -> None:
        if self.save_hpc():
            self._run_captured(["ssh", "setup"], then=self.show_hpc)

    @on(Button.Pressed, "#import")
    def import_key(self) -> None:
        path = self.query_one("#import-key", Input).value.strip()
        if path and self.save_hpc():
            self._run_captured(["ssh", "setup", "--import", str(Path(path).expanduser())], then=self.show_hpc)

    @on(Button.Pressed, "#copy-key")
    def copy_key(self) -> None:
        pub = self._pubkey()
        if not pub or not self._hpc_set():
            self.say("Save the connection and create the key first.", "red")
            return
        target = f"{self.draft.get('ssh.user')}@{self.draft.get('ssh.hostname')}"
        jump = ["-J", self.draft.get("ssh.proxy_jump")] if self.draft.get("ssh.proxy_jump") else []
        port = ["-p", str(self.draft.get("ssh.port"))] if self.draft.get("ssh.port") not in (None, 22) else []
        remote = ("mkdir -p ~/.ssh && chmod 700 ~/.ssh && grep -qxF " + shlex.quote(pub) + " ~/.ssh/authorized_keys "
                  "2>/dev/null || echo " + shlex.quote(pub) + " >> ~/.ssh/authorized_keys")
        self.in_terminal([["ssh", *jump, *port, target, remote]],
                         f"Adding Overseer's key on {target}: log in as you usually do (password, 2FA).")

    @on(Button.Pressed, "#ssh-test")
    def ssh_test(self) -> None:
        def done(rc: int, out: str) -> None:
            self.ssh_ok = rc == 0
            self.refresh_steps()
        self._run_captured(["ssh", "test"], then_rc=done)

    @on(Button.Pressed, "#ssh-connect")
    def ssh_connect(self) -> None:
        self.in_terminal([S.overseer_argv(self.config, "ssh", "connect")],
                         "Opening the connection: approve the login (2FA / Duo). It then stays open.")

    # ------------------------------------------------------------------ 5. cluster settings
    def pane_cluster(self):
        yield Static("Your cluster", classes="title")
        yield Static("Where the agent runs heavier commands (`hpc-run`: plots, scripts) instead of the login node, "
                     "and how your disk quota is read. Look them up, then pick.", classes="hint")
        yield Button("Look up the partitions", id="lookup", variant="primary")
        yield Static("Compute partition (CPU work)")
        yield Select([], id="cpu-part", prompt="look up the partitions first")
        yield Static("GPU partition (hpc-run --gpu)")
        yield Select([], id="gpu-part", prompt="look up the partitions first")
        yield Input(placeholder="GPU request, e.g. gpu:1 or gpu:a100:1", id="gres")
        yield Input(placeholder="your quota command (e.g. myquota; empty: don't check)", id="quota")
        yield Input(placeholder="an account for sbatch, if your cluster needs one (e.g. myproject)", id="account")
        yield Button("Save", id="save-cluster", variant="primary")

    def show_cluster(self) -> None:
        if not self.query_one("#quota", Input).value:
            self.query_one("#quota", Input).value = str(self.draft.get("slurm.quota_command", "myquota") or "")
        if not self.query_one("#gres", Input).value:
            self.query_one("#gres", Input).value = str(self.draft.get("slurm.compute.gpu_gres") or "gpu:1")

    def status_cluster(self) -> str:
        return "ok" if self.cluster_saved or (not self.new and self.draft.get("slurm.compute.partition")) else "todo"

    @on(Button.Pressed, "#lookup")
    def lookup(self) -> None:
        if not self._hpc_set():
            self.say("Set up the HPC connection first.", "red")
            return
        self.say("Asking the cluster (sinfo)...", "yellow")
        self._lookup()

    @work(thread=True, exclusive=True, group="cluster")
    def _lookup(self) -> None:
        parts, quota, err = S.cluster_info(self.config)

        def done() -> None:
            if err or not parts:
                self.say(f"✗ {err or 'sinfo listed no partitions'}", "red")
                return
            self.partitions = parts
            cpu = [p for p in parts if p.up and not p.gpus] or [p for p in parts if p.up]
            gpu = [p for p in parts if p.up and p.gpus]
            label = lambda p: f"{p.name}{' (default)' if p.default else ''} · {p.nodes} nodes · max {p.limit}"  # noqa: E731
            self.query_one("#cpu-part", Select).set_options([(label(p), p.name) for p in cpu])
            self.query_one("#gpu-part", Select).set_options([(label(p) + " · " + ",".join(p.gres), p.name) for p in gpu])
            pick = next((p for p in cpu if p.default), cpu[0] if cpu else None)
            if pick:
                self.query_one("#cpu-part", Select).value = pick.name
            if gpu:
                self.query_one("#gpu-part", Select).value = gpu[0].name
                self.query_one("#gres", Input).value = S.gpu_gres(gpu[0])
            if quota:
                self.query_one("#quota", Input).value = quota
            self.say(f"{len(parts)} partition(s) found" + (f"; quota command: {quota}" if quota else ""), "green")
        self.call_from_thread(done)

    @on(Select.Changed, "#gpu-part")
    def gpu_changed(self, event: Select.Changed) -> None:
        p = next((p for p in self.partitions if p.name == event.value), None)
        if p:
            self.query_one("#gres", Input).value = S.gpu_gres(p)

    @on(Button.Pressed, "#save-cluster")
    def save_cluster(self) -> None:
        cpu, gpu = self.query_one("#cpu-part", Select).value, self.query_one("#gpu-part", Select).value
        if isinstance(cpu, str):
            self.draft.set("slurm.compute.partition", cpu)
        elif not self.draft.get("slurm.compute.partition"):
            self.say("Pick a compute partition (look them up first).", "red")
            return
        if isinstance(gpu, str):
            self.draft.set("slurm.compute.gpu_partition", gpu)
        self.draft.set("slurm.compute.gpu_gres", self.query_one("#gres", Input).value.strip() or "gpu:1")
        self.draft.set("slurm.quota_command", self.query_one("#quota", Input).value.strip())
        account = self.query_one("#account", Input).value.strip()
        if account:
            self.draft.set("slurm.compute.sbatch_args", [f"--account={account}"])
        self.cluster_saved = True
        self.save()

    # ------------------------------------------------------------------ the assistant's vault
    def pane_vault(self):
        yield Static("Your Obsidian vault", classes="title")
        yield Static("The assistant's memory: the folder of your vault on this machine (e.g. the one Syncthing "
                     "keeps in sync). It must be outside Overseer's folder.", classes="hint")
        yield Input(placeholder="e.g. /home/me/Vault", id="vault-path")
        yield Button("Save", id="save-vault", variant="primary")

    def show_vault(self) -> None:
        v = self.draft.get("vault.path")
        if v and not self.query_one("#vault-path", Input).value:
            self.query_one("#vault-path", Input).value = str(v)

    def status_vault(self) -> str:
        v = self.draft.get("vault.path")
        return "ok" if v and Path(str(v)).expanduser().is_dir() else "todo"

    @on(Button.Pressed, "#save-vault")
    def save_vault(self) -> None:
        p = Path(self.query_one("#vault-path", Input).value.strip()).expanduser()
        if not p.is_dir():
            self.say(f"{p} isn't a folder.", "red")
            return
        if not (p / ".obsidian").is_dir():
            self.say(f"Note: {p} has no .obsidian folder - is it the vault itself?", "yellow")
        self.draft.set("vault.path", str(p))
        self.save()

    # ------------------------------------------------------------------ 6. the agents
    def pane_agents(self):
        yield Static("The AI agents", classes="title")
        yield Static("Which one answers by default (/backend switches any time), and their logins. Logins are "
                     "Overseer's own: your normal ~/.claude and ~/.codex aren't touched.", classes="hint")
        yield Select([("Claude Code", "claude_code"), ("Codex", "codex")], id="backend", allow_blank=False,
                     value="claude_code")
        yield Static("Several accounts per CLI (optional, e.g. personal and work): name=folder, separated by "
                     "commas. Empty: one login each.", classes="hint")
        yield Input(placeholder="Claude Code accounts, e.g. personal=./state/claude, work=./state/claude-work",
                    id="claude-accounts")
        yield Input(placeholder="Codex accounts, e.g. default=./state/codex", id="codex-accounts")
        yield Button("Save", id="save-agents", variant="primary")
        yield Static(id="logins")
        yield Vertical(id="login-buttons")

    def show_agents(self) -> None:
        self.query_one("#backend", Select).value = str(self.draft.get("agent.backend") or "claude_code")
        for wid, b in (("claude-accounts", "claude_code"), ("codex-accounts", "codex")):
            acc = self.draft.get(f"agent.{b}.accounts") or {}
            if acc and not self.query_one(f"#{wid}", Input).value:
                self.query_one(f"#{wid}", Input).value = ", ".join(f"{k}={v}" for k, v in acc.items())
        env = S.read_env(self.env_path)
        lines, box = [], self.query_one("#login-buttons", Vertical)
        box.remove_children()
        for b, (cmd, _) in S.CLI_PACKAGES.items():
            if not S.shutil.which(cmd):
                lines.append(f"⬜ {cmd} isn't installed (step 2)")
                continue
            accounts = S.accounts(self.draft, b, self.base)
            for name, home in accounts.items():
                ok = S.logged_in(b, home, env)
                who = f"{cmd} ({name})" if len(accounts) > 1 else cmd
                lines.append(f"{'✅' if ok else '⬜'} {who}: {'logged in' if ok else 'not logged in yet'}")
                box.mount(Button(f"Log {who} in", name=f"{b}|{name}", classes="login"))
        self.query_one("#logins", Static).update("\n".join(lines) + "\n")

    def status_agents(self) -> str:
        b = str(self.draft.get("agent.backend") or "claude_code")
        accounts = S.accounts(self.draft, b, self.base)
        home = next(iter(accounts.values()))
        return "ok" if S.shutil.which(S.CLI_PACKAGES[b][0]) and S.logged_in(b, home, S.read_env(self.env_path)) \
            else "todo"

    @on(Button.Pressed, "#save-agents")
    def save_agents(self) -> None:
        self.draft.set("agent.backend", self.query_one("#backend", Select).value)
        for wid, b in (("claude-accounts", "claude_code"), ("codex-accounts", "codex")):
            raw = self.query_one(f"#{wid}", Input).value
            acc = dict(x.strip().split("=", 1) for x in raw.split(",") if "=" in x)
            self.draft.set(f"agent.{b}.accounts", {k.strip(): v.strip() for k, v in acc.items()})
        problem = self.draft.validate()
        if problem and "account" in problem:
            self.say(f"✗ {problem}", "red")
            return
        self.save()
        self.show_agents()

    @on(Button.Pressed, ".login")
    def login_button(self, event: Button.Pressed) -> None:
        b, _, name = (event.button.name or "").partition("|")
        accounts = S.accounts(self.draft, b, self.base)
        args = ["login", b] + (["--account", name] if len(accounts) > 1 else [])
        how = ("Claude Code: open the link it prints on any device, sign in, and paste the code back here."
               if b == "claude_code" else "Codex: open the link on any device and enter the code it shows.")
        self.save()
        self.in_terminal([S.overseer_argv(self.config, *args)], how)
        self.show_agents()

    # ------------------------------------------------------------------ 7. preferences
    def pane_prefs(self):
        yield Static("Preferences", classes="title")
        yield Static("Your time zone (reminders, the digest, times in messages)", classes="hint")
        yield Input(placeholder="e.g. Asia/Dubai", id="tz")
        if self.draft.domain == "hpc":
            yield Static("The morning digest's time (empty: no digest)", classes="hint")
            yield Input(placeholder="08:00", id="digest")
            yield Static("Research notes in Obsidian (optional): a folder inside your vault, e.g. "
                         "/home/me/Vault/HPC", classes="hint")
            yield Input(placeholder="leave empty to keep the research logs on the HPC", id="obsidian")
        yield Checkbox("Restart the bot by itself if it crashes", False, id="restart")
        yield Checkbox("ADHD-friendly replies (lead with the next action, numbered steps, no tangents)", False,
                       id="adhd")
        yield Button("Save", id="save-prefs", variant="primary")

    def show_prefs(self) -> None:
        tz = self.query_one("#tz", Input)
        if not tz.value:
            tz.value = str(self.draft.get("local.timezone") or S.detect_timezone())
        if self.draft.domain == "hpc":
            d = self.query_one("#digest", Input)
            if not d.value:
                d.value = str(self.draft.get("digest.time") or "08:00") if self.draft.get("digest.enabled", True) \
                    is not False else ""
            o = self.query_one("#obsidian", Input)
            if not o.value and self.draft.get("obsidian.folder"):
                o.value = str(self.draft.get("obsidian.folder"))
        self.query_one("#restart", Checkbox).value = bool(self.draft.get("local.restart_on_crash", False))
        self.query_one("#adhd", Checkbox).value = bool(self.draft.get("agent.i_have_adhd", False))

    def status_prefs(self) -> str:
        return "ok" if self.draft.get("local.timezone") else "todo"

    @on(Button.Pressed, "#save-prefs")
    def save_prefs(self) -> None:
        self.draft.set("local.timezone", self.query_one("#tz", Input).value.strip())
        if self.draft.domain == "hpc":
            digest = self.query_one("#digest", Input).value.strip()
            self.draft.set("digest.enabled", bool(digest))
            if digest:
                self.draft.set("digest.time", digest)
            self.draft.set("obsidian.folder", self.query_one("#obsidian", Input).value.strip())
        self.draft.set("local.restart_on_crash", self.query_one("#restart", Checkbox).value)
        self.draft.set("agent.i_have_adhd", self.query_one("#adhd", Checkbox).value)
        problem = self.draft.validate()
        if problem and any(k in problem for k in ("timezone", "digest", "obsidian")):
            self.say(f"✗ {problem}", "red")
            return
        self.save()

    # ------------------------------------------------------------------ 8. check
    def pane_check(self):
        yield Static("Check everything", classes="title")
        yield Static("Overseer's own check: the config, " + ("SSH and SLURM, " if self.draft.domain == "hpc" else
                                                               "the vault, ") +
                     "the Telegram token, and a real test call to each agent (it takes up to a minute).",
                     classes="hint")
        yield Button("Run the check", id="run-check", variant="primary")

    def status_check(self) -> str:
        return {True: "ok", False: "warn"}.get(self.check_ok, "todo")

    @on(Button.Pressed, "#run-check")
    def run_check(self) -> None:
        problem = self.draft.validate()
        if problem:
            self.say(f"✗ the config isn't complete yet: {problem}", "red")
            return

        def done(rc: int, out: str) -> None:
            self.check_ok = rc == 0
            self.say("Everything works." if rc == 0 else "Something needs fixing: see the [FAIL] lines above.",
                     "bold green" if rc == 0 else "bold red")
            self.refresh_steps()
        self._run_captured(["check"], then_rc=done)

    # ------------------------------------------------------------------ 9. start
    def pane_start(self):
        yield Static("Start the bot", classes="title")
        yield Static("It runs in the background: closing this window, or logging out, doesn't stop it. "
                     "`overseer stop` (or the monitor) stops it.", classes="hint")
        yield Static(id="run-state")
        with Horizontal(classes="row"):
            yield Button("Start it in the background", id="start-bot", variant="primary")
            yield Button("Open the monitor", id="open-monitor")
        if self.draft.domain == "hpc":
            yield Button("Set up the personal assistant too", id="setup-assistant")

    def show_start(self) -> None:
        state = self._state_dir()
        st = daemon.state(state) if state else None
        text = (f"● running (pid {st.pid})" + (", restarted after a crash" if st.watched else "")
                if st and st.running else "○ not running")
        note = daemon.linger_note()
        self.query_one("#run-state", Static).update(text + (f"\n\n⚠ {note}" if note else "") + "\n")

    def status_start(self) -> str:
        state = self._state_dir()
        return "ok" if state and daemon.state(state).running else "todo"

    @on(Button.Pressed, "#start-bot")
    def start_bot(self) -> None:
        problem = self.draft.validate()
        if problem:
            self.say(f"✗ the config isn't complete yet: {problem}", "red")
            return
        self.say("Starting...", "yellow")
        self._start()

    @work(thread=True, exclusive=True, group="start")
    def _start(self) -> None:
        from ..core.config import load_config
        try:
            ok, msg = daemon.start(load_config(self.config))
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"

        def done() -> None:
            self.say(("✅ " if ok else "✗ ") + msg, "green" if ok else "red")
            self.show_start()
            self.refresh_steps()
        self.call_from_thread(done)

    @on(Button.Pressed, "#open-monitor")
    def open_monitor(self) -> None:
        self.exit("monitor")

    @on(Button.Pressed, "#setup-assistant")
    def setup_assistant(self) -> None:
        self.exit("assistant")

    # ------------------------------------------------------------------ Overseer's own commands, captured
    def _run_captured(self, args: list[str], then=None, then_rc=None) -> None:
        self.say("$ overseer " + " ".join(args), "bold")
        self._captured(args, then, then_rc)

    @work(thread=True, group="commands")
    def _captured(self, args: list[str], then, then_rc) -> None:
        try:
            res = S.overseer(self.config, *args)
            out, rc = (res.stdout + res.stderr).strip(), res.returncode
        except subprocess.TimeoutExpired:
            out, rc = "(it took too long)", 1

        def done() -> None:
            for line in out.splitlines():
                style = "green" if line.startswith("[ok]") else "red" if line.startswith(("[FAIL]", "FAIL")) else ""
                self.say(line, style)
            if then:
                then()
            if then_rc:
                then_rc(rc, out)
            self.refresh_steps()
        self.call_from_thread(done)


def run_setup(config: str = "config.yaml") -> int:
    path = Path(config).resolve()
    while True:
        result = Setup(path).run()
        if result == "assistant":
            path = path.parent / "assistant.yaml"
            continue
        if result == "monitor":
            from .monitor import run_monitor
            return run_monitor(str(path))
        return 0
