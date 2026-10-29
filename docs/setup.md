# Setting up Overseer

This guide takes you from nothing to a running bot: you chat with it on Telegram, it watches your SLURM jobs, and an
AI agent (Claude Code or Codex) can fix and resubmit them for you. It takes about 20-30 minutes.

> **The easy way:** after [step 2](#2-get-overseer) (getting Overseer), run `uv run overseer setup`. It walks you
> through the rest in the terminal: it installs what's missing, checks your bot's token, finds your chat id by itself
> (you just message the bot), sets up the SSH key and adds it to the cluster, lists the cluster's partitions to pick
> from, logs the agents in, runs the check and starts the bot in the background. Each step shows what's already done,
> so you can run it again any time to change something. For the personal assistant: `uv run overseer setup -c
> assistant.yaml`. The manual steps below do the same by hand.

**Contents:**
1. [What you need](#what-you-need)
2. [Install the system packages and the agent CLIs](#1-install-the-system-packages-and-the-agent-clis)
3. [Get Overseer](#2-get-overseer)
4. [Create your Telegram bot](#3-create-your-telegram-bot)
5. [Write config.yaml](#4-write-configyaml)
6. [Connect to the HPC](#5-connect-to-the-hpc)
7. [Log the agents in](#6-log-the-agents-in)
8. [Check and start it](#7-check-and-start-it)
9. [Keep it running](#8-keep-it-running)
10. [Optional extras](#optional-extras)
11. [Troubleshooting](#troubleshooting)
12. [Updating Overseer](#updating-overseer)

## What you need

- **A Linux machine that stays on**, e.g. a homelab server, a NAS or a spare laptop. Overseer and the agents run
  here, not on the HPC. Ubuntu/Debian, Fedora and Arch all work; macOS and Windows aren't supported (the agents'
  sandbox needs Linux's bubblewrap).
- **An account on a SLURM cluster** you can reach with SSH from that machine. Nothing is installed on the cluster.
- **Telegram** on your phone or computer.
- **At least one AI agent subscription or API key:**
  - Claude Code: a Claude Pro/Max subscription, or an Anthropic API key;
  - Codex: a ChatGPT Plus/Pro subscription, or an OpenAI API key.

  One is enough. With both you can switch between them, and with
  [several accounts](guide.md#several-accounts-per-cli) the bot moves to another when one runs out of usage.

> **Before you start:** the agent gets full access to your HPC account (that's how it fixes jobs), and your Telegram
> chat becomes the way to drive it. Read [security.md](security.md) for what it can and can't do. On the machine
> running Overseer it's sandboxed and can never read your SSH key or logins.

## 1. Install the system packages and the agent CLIs

**System packages:** an SSH client, plus `bubblewrap` and `socat` for the agents' sandbox:

```bash
sudo apt install openssh-client bubblewrap socat git       # Ubuntu / Debian
sudo dnf install openssh-clients bubblewrap socat git      # Fedora
sudo pacman -S openssh bubblewrap socat git                # Arch
```

**uv** (installs Python and Overseer's dependencies for you):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**The agent CLIs** (Node.js 18 or newer; install it from your distribution or with [nvm](https://github.com/nvm-sh/nvm)).
Install either or both:

```bash
npm install -g @anthropic-ai/claude-code      # Claude Code: the `claude` command
npm install -g @openai/codex                  # Codex: the `codex` command
```

Check they're on your PATH: `claude --version`, `codex --version`. Overseer keeps them up to date itself afterwards
(once a day, or `/update` in Telegram).

## 2. Get Overseer

```bash
git clone https://github.com/ArdaBakici/Overseer ~/Overseer
cd ~/Overseer
uv sync
```

`uv sync` creates `.venv` with everything Overseer needs. Every command below is run from `~/Overseer`, as
`uv run overseer …`. Overseer only writes inside this folder (in `./state`) and `/tmp/overseer`.

## 3. Create your Telegram bot

1. **Create the bot.** In Telegram, open [@BotFather](https://t.me/BotFather) and send `/newbot`. Pick a display
   name and a username ending in `bot` (e.g. `my_hpc_overseer_bot`). BotFather replies with a **token** like
   `123456789:AAH...`. Keep it secret: whoever has it controls the bot.
2. **Store the token.** In `~/Overseer`:

   ```bash
   cp .env.example .env && chmod 600 .env
   nano .env            # set TELEGRAM_BOT_TOKEN=123456789:AAH...
   ```

   `.env` holds every secret; the agents can never read it.
3. **Find your chat id.** Open [@userinfobot](https://t.me/userinfobot) and send it any message: it answers with
   your numeric **Id** (e.g. `123456789`). You'll put it in `config.yaml`, and the bot will answer only you.
4. **Recommended: topics.** Overseer works best with topics, one per project plus a 📊 jobs topic, a 💬 general one and
   so on. In @BotFather, open your bot → **Bot Settings** → **Threaded mode** and turn it on. Without it, everything
   arrives in one chat (still fine; project messages are labelled).
5. **Recommended: protect your Telegram account** with two-step verification (Settings → Privacy and Security), since
   this chat can drive an agent on your HPC account.

Then open a chat with your new bot and press **Start**, so it's allowed to message you.

## 4. Write config.yaml

```bash
cp config.example.yaml config.yaml
```

Every setting is explained in the file's comments. To get started you only need these:

```yaml
local:
  timezone: Europe/Istanbul     # your time zone ("" = this machine's)

ssh:
  hostname: login.cluster.edu   # the cluster's login node
  user: your-netid              # your username there
  proxy_jump: null              # if you first ssh to a gateway: your-netid@gateway.cluster.edu

telegram:
  allowed_chat_ids: [123456789] # your chat id from step 3

agent:
  backend: claude_code          # or codex: the one you installed (switch any time with /backend)
```

Worth checking for your cluster (ask the cluster's docs, or `sinfo -s` on the login node):

- `slurm.compute.partition` and `gpu_partition`, `gpu_gres`: where `hpc-run` runs the agent's heavier commands
  (plots, scripts) instead of the login node. `sbatch_args: ["--account=myproject"]` if your cluster needs an account.
- `slurm.quota_command`: a command that prints your disk quota (`myquota` on many clusters; `""` to turn it off).
- `slurm.setup`: only if `squeue` isn't found in a login shell, e.g. `"module load slurm"`.

If your config gets out of date after an update, `uv run overseer config` shows what's new and writes an updated copy
([details](guide.md#configuration-notes)).

## 5. Connect to the HPC

Overseer uses its **own** SSH key, kept in `./state/ssh`. Your `~/.ssh` is never used, and the agents can't read the
key.

```bash
uv run overseer ssh setup              # creates a new key and prints its public half
# or reuse a key you already use for the cluster:
uv run overseer ssh setup --import ~/.ssh/id_ed25519
```

With a new key, add the printed public key to `~/.ssh/authorized_keys` on the cluster. `ssh setup` prints a
ready-made command for it, to run from a machine where you can already log in. Then test the connection:

```bash
uv run overseer ssh test               # "connected as your-netid on login1"
```

**Clusters with 2FA / Duo:** run `uv run overseer ssh connect` and approve the login. The connection then stays open
(`ControlPersist`). If the network drops, run it again; the bot tells you when it can't reach the HPC.

## 6. Log the agents in

Overseer keeps its own logins for the agents, in `./state/claude` and `./state/codex`. Your normal `~/.claude` and
`~/.codex` aren't touched. Both logins work on a headless server:

```bash
uv run overseer login claude_code   # prints a URL: open it on any device, sign in, paste the code back
uv run overseer login codex         # prints a URL and a code: open the URL on any device, enter the code
```

Alternatives:
- **Claude Code with a token:** `uv run overseer login claude_code -- setup-token` prints a long-lived token. Put it in
  `.env` as `CLAUDE_CODE_OAUTH_TOKEN=...`.
- **API keys:** put `ANTHROPIC_API_KEY=...` in `.env`, or log Codex in with one:
  `printenv OPENAI_API_KEY | uv run overseer login codex -- --with-api-key`.
- **Several accounts per CLI** (e.g. personal and work): list them under `agent.claude_code.accounts` first, then log
  each in with `--account <name>` ([details](guide.md#several-accounts-per-cli)).

## 7. Check and start it

```bash
uv run overseer check
```

This checks SSH and SLURM, the Telegram token, and makes a real test call to each installed agent. Every line should
say `[ok]` (an agent you didn't install shows `[skip]`). Then start the bot in the background:

```bash
uv run overseer start
```

Send it a message on Telegram, e.g. *"which of my jobs are running?"* or *"how much quota do I have left?"*. Within
a minute it also posts a card for each of your running jobs. Watch it with `uv run overseer` (the TUI)
([the monitor](guide.md#the-terminal-monitor-overseer-tui)); `uv run overseer run` runs it in the terminal instead
(Ctrl-C stops it).

**First things to try:**
- `/project new myproj /scratch/me/myproj What this project is about`: a topic of its own for a project.
- Reply to a failed job's card, or tap **🔧 Ask agent to fix**.
- `/help` lists every command; the [guide](guide.md) explains them.

## 8. Keep it running

`uv run overseer start` runs the bot in the background: closing the terminal or logging out doesn't stop it.

```bash
uv run overseer status                # running? since when? restarted after a crash?
uv run overseer stop                  # a clean stop (runs in progress are offered again at the next start)
uv run overseer restart               # after changing config.yaml
uv run overseer                       # the TUI: watch it live; s / x / r start, stop and restart it there too
```

- **Restart after a crash (off by default)**: with `local.restart_on_crash: true` in the config, a small watcher
  process runs the bot and starts it again after a crash, after a pause (30 s, longer if it keeps crashing). It
  doesn't restart a bot you stopped, nor one that can't start (e.g. a config error). The monitor shows whether a
  bot is watched, and how often it was restarted.
- **Only one copy runs**: starting a bot that is already running (from here, the monitor or a terminal) is refused.
- **Its log** is in `state/logs/overseer.log` (the monitor's Log tab).
- **Logging out:** on systems that end a user's processes at logout (`KillUserProcesses=yes`), `overseer start` and the
  monitor tell you; `loginctl enable-linger $USER` keeps the bot running.
- **After a reboot** it stays off until you start it again. To start it at boot, use systemd instead:

### Or: systemd, to start at boot

To start Overseer at boot and restart it if it crashes, install the user service in `deploy/overseer.service`
(instead of `overseer start`):

```bash
mkdir -p ~/.config/systemd/user
cp deploy/overseer.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now overseer
loginctl enable-linger $USER          # keep it running when you're logged out
```

The service expects Overseer in `~/Overseer`; edit `WorkingDirectory` and `ExecStart` in the copied file if yours is
elsewhere. Useful commands:

```bash
systemctl --user status overseer      # is it running?
journalctl --user -u overseer -f      # its log
systemctl --user restart overseer     # after changing config.yaml
```

Restarting doesn't lose the agent's work: a run that was cut off is offered again with **▶️ Continue**
([details](guide.md#restarts-dont-lose-the-agents-work)).

## Optional extras

| | | |
|---|---|---|
| **Projects in config** | define projects up front instead of with `/project new` | `projects:` in `config.yaml` |
| **Research notes in Obsidian** | research logs, results, findings and graphs as notes in a folder of your vault | [guide](guide.md#research-notes-in-obsidian-optional) |
| **Several accounts** | more than one login per CLI, with automatic failover | [guide](guide.md#several-accounts-per-cli) |
| **ADHD-friendly replies** | `agent.i_have_adhd: true` | [guide](guide.md#replies-shaped-for-adhd-optional) |
| **The dashboard** | a read-only Mini App in Telegram, on your tailnet only | [below](#optional-the-dashboard) |
| **The personal assistant** | a second bot whose memory is your Obsidian vault | [assistant.md](assistant.md) |

### Optional: the dashboard

The dashboard is a Telegram Mini App, a web page that opens inside Telegram. Overseer serves it on this machine
(`127.0.0.1:8765`), and [Tailscale](https://tailscale.com) makes it reachable over HTTPS from your own devices only:

```bash
# config.yaml
dashboard:
  enabled: true
  url: https://homelab.tail1234.ts.net     # what `tailscale serve` prints

# on the homelab, once (it stays on across reboots)
tailscale serve --bg 8765
```

- **Not public:** `tailscale serve` shares it with your tailnet only. (`tailscale funnel` would put it on the internet: don't.) The page opens in Telegram on your phone or laptop, which needs Tailscale on; Telegram's servers never fetch it.
- **Only you:** every data request carries Telegram's signed launch data. Overseer checks the signature (made with the bot's token), its age (24 h) and that the user is in `telegram.allowed_chat_ids`. A browser outside Telegram, or another device on your tailnet, gets nothing.
- **Read-only:** nothing in it changes anything; actions stay in the chat, where each one is announced.
- **Menu button:** with `url` set, the chat's menu button opens it too.

**Behind Caddy instead of `tailscale serve`.** Any HTTPS reverse proxy to `127.0.0.1:8765` works; put its address in `dashboard.url`. The certificate must be one your devices already trust, because Telegram opens the page in its own web view:

```
# Tailscale's certificate (Caddy fetches it from tailscaled; set TS_PERMIT_CERT_UID=caddy for tailscaled)
homelab.tail1234.ts.net {
    bind 100.x.y.z                 # the homelab's Tailscale IP: not on the LAN or the internet
    reverse_proxy 127.0.0.1:8765
}

# or your own domain, pointed at the Tailscale IP, with a DNS-challenge plugin (no open ports)
overseer.example.com {
    bind 100.x.y.z
    tls { dns cloudflare {env.CF_API_TOKEN} }
    reverse_proxy 127.0.0.1:8765
}
```

`tls internal` (Caddy's own CA) only works where you install Caddy's root certificate: iOS (and enable full trust) and desktops yes, Android most likely not, since apps there ignore user-installed CAs.

## Troubleshooting

**`overseer check` says the sandbox can't start (Ubuntu 24.04 and newer).** bubblewrap needs unprivileged user
namespaces, which AppArmor restricts there; see Claude Code's sandboxing docs for the AppArmor profile to add.
Overseer never falls back to running the agent unsandboxed, so it won't run until this is fixed.

**`claude` / `codex` not found under systemd, though it works in your terminal.** The service doesn't load your
shell's PATH (common with nvm or a user-level npm prefix). Put the full path in `config.yaml`:
`agent.claude_code.command: /home/me/.nvm/versions/node/v22.0.0/bin/claude` (what `which claude` prints), and the
same for `agent.codex.command`.

**The agent's model call fails in `overseer check`** ("no answer within 2 minutes", "not logged in"). Run the login
again (step 6). With several accounts, each one is checked and named in the output.

**`ssh test` fails.** Check `ssh.hostname`, `ssh.user` and `proxy_jump`, and that the public key from `ssh setup` is
in `~/.ssh/authorized_keys` on the cluster (one line). On a 2FA cluster, run `uv run overseer ssh connect` first.

**`squeue` not found.** SLURM isn't on PATH in a login shell on the cluster: set `slurm.setup`, e.g.
`"module load slurm"` (`overseer check` shows the PATH it saw).

**The bot doesn't answer.** Check `telegram.allowed_chat_ids` holds your id (from @userinfobot) and that you pressed
Start in the chat with the bot. Messages from other chats are ignored on purpose.

**No topics appear.** Turn on threaded mode for the bot in @BotFather (step 3) and restart Overseer; `/health` says
whether topics are on, and why not.

**A config error after updating.** `uv run overseer config` lists the settings that changed and writes
`config.yaml.new` with your values carried over.

## Updating Overseer

```bash
cd ~/Overseer
git pull
uv sync
uv run overseer config                # anything new or renamed in the config?
systemctl --user restart overseer
```

The agent CLIs update themselves once a day (or `/update` in Telegram).
