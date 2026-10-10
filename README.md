# Overseer

Overseer is a Telegram bot for your SLURM jobs. It runs on your homelab and watches your jobs on the university HPC over SSH. When one fails, you hand it to **Claude Code** or **OpenAI Codex**: they get a real shell and full access to your HPC account, and fix and resubmit the job themselves.

- **Monitoring.** No LLM is involved, and checks run every minute.
  - **Job lifecycle:** you get messages when a job is 📥 **queued** and 🚀 **started** (which node), plus 📊 hourly **status** (elapsed time, latest metrics, log tail).
  - **Final report:** when a job ends you get ✅/❌, with the state, exit code, runtime against the limit and peak memory.
  - **Why it failed:** heuristics flag CUDA OOM, host OOM, time limits, NCCL errors, missing modules, NaNs and similar. An LLM root-cause summary is added on top.
- **Fixing.** Tap **🔧 Ask agent to fix**, reply to any job message, or just chat. The agent investigates with real commands (`ssh hpc 'sacct …'`, logs, code, quota, modules), edits, and runs `sbatch`.
  - **Visibility:** you see every command it ran, and you get a message the moment it submits a job.
  - **Follow-up:** that job is then monitored like any other, and replying to its report continues the same conversation.
- **Switching agents.** Change between Claude Code and Codex at any time with `/backend`.

```
 homelab (your machine)                                                    HPC login node
┌───────────────────────────────────────────────────────────┐           ┌────────────────┐
│ Telegram bot ◄──► you                                     │           │                │
│ Monitor ── squeue/sacct ──┐                               │           │                │
│                           ├─► Overseer's SSH master ──────┼── ssh ───►│ your account   │
│ Claude Code / Codex       │   (key in ./state/ssh)        │           │ (full access)  │
│  └─ sandboxed shell ──────┘   via /tmp/overseer/ssh/      │           │                │
│     `ssh hpc …`               master.sock                 │           │                │
└───────────────────────────────────────────────────────────┘           └────────────────┘
```

## What the agent can and can't do

**On the HPC: everything your account can.** That includes editing files, `sbatch`, `scancel`, `pip install` and so on. Nothing is restricted there, by design. In return you see all of it:
- **Commands:** after each request, Telegram gets the list of commands the agent ran. Set `report_commands: live` to see them while it works.
- **Submissions:** any job it submits is announced right away (`🚀 Agent submitted job 1234` plus the command), then tracked by the monitor.

**On your homelab: confined.** Claude Code and Codex each run their shell inside their own OS sandbox (bubblewrap on Linux), configured by Overseer:

| | agent's shell |
|---|---|
| write | only its own scratch dir, `/tmp/overseer/agents/<conversation>` |
| read | the system, except Overseer's state dir (which includes the SSH key), `~/.ssh`, `~/.gnupg`, `~/.docker`, `~/.aws`, gcloud credentials, docker/podman sockets, `/run/user/<uid>` and your ssh-agent socket |
| network | none, except `agent.allowed_domains`. The HPC is reached through the unix socket of Overseer's SSH connection |
| environment | scrubbed. Only `PATH`, locale and the API-key variables in `pass_env` are passed; the Telegram token is never passed |
| escape hatches | Claude Code: `failIfUnavailable` and `allowUnsandboxedCommands: false`, so it never runs a command unsandboxed. Codex: a named permission profile with approvals off |

**Your SSH key never leaves the Overseer folder:**
- **Overseer's own key setup:** `overseer ssh setup` creates (or imports) a key, an ssh config and known_hosts in `./state/ssh`. Your `~/.ssh` isn't used at all.
- **One shared connection:** Overseer opens a single SSH master connection. The agent's `ssh`, `scp`, `sftp` and `rsync` are wrappers that only talk to that connection's socket, so the agent never needs the key and its sandbox can't read it.
- **Clean shutdown:** if Overseer stops, the agent loses HPC access.

**Since the agent is effectively a shell on your HPC account, so is your Telegram chat.** Keep `telegram.allowed_chat_ids` to your own chat and turn on 2FA for your Telegram account.

All of this was tested end to end, using the real Claude Code (2.1) and Codex (0.160) binaries against a local sshd standing in for the HPC (`tests/test_e2e.py`):

- `ssh hpc` and `sbatch` work.
- Reading the key and `~/.ssh` fails.
- Writing outside the scratch dir fails.
- `curl` to the internet fails.

## Setup

**On the HPC:** nothing to install.

**On the homelab** (Linux): Python 3.10+, OpenSSH client, `bubblewrap` and `socat` (for Claude Code's sandbox; Codex brings its own), and the CLIs.

```bash
sudo apt install openssh-client bubblewrap socat
npm i -g @anthropic-ai/claude-code @openai/codex      # either or both

git clone <this repo> ~/Overseer && cd ~/Overseer
uv sync                                 # or: python3 -m venv .venv && .venv/bin/pip install -e .
cp config.example.yaml config.yaml      # set ssh.hostname/user and telegram.allowed_chat_ids

# 1. Overseer's own SSH key (kept in ./state/ssh)
uv run overseer ssh setup               # or: ssh setup --import ~/.ssh/id_ed25519 to copy an existing key
#    -> add the printed public key to ~/.ssh/authorized_keys on the HPC, then:
uv run overseer ssh test

# 2. Log the agents in (stored in ./state/claude and ./state/codex; your normal logins are untouched)
uv run overseer login claude_code       # or export ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN
uv run overseer login codex             # or export OPENAI_API_KEY

# 3. Telegram: create a bot with @BotFather, get your chat id (e.g. via @userinfobot)
export TELEGRAM_BOT_TOKEN=...
uv run overseer check                   # ssh, slurm, a real test call to each installed agent
uv run overseer run
```

Everything after `--` is passed to the CLI's login, e.g. `overseer login codex -- --device-auth` on a headless machine. For Claude Code on a server, `overseer login claude_code -- setup-token` prints a long-lived token, which you then export as `CLAUDE_CODE_OAUTH_TOKEN`.

**Clusters with 2FA/Duo.** Run `overseer ssh connect` once. It opens the master connection interactively so you can approve the login, and the connection then stays up (`ControlPersist`). If the network drops, run it again; the bot tells you when it can't reach the HPC.

**Ubuntu 24.04+.** bubblewrap may need user namespaces allowed by AppArmor; see Claude Code's sandboxing docs. `overseer check` and the first agent run will tell you if the sandbox can't start, and Overseer never falls back to running unsandboxed.

To keep it running, use `deploy/overseer.service` (a systemd user unit; instructions are inside the file).

## Using it

| | |
|---|---|
| *(automatic)* | 📥 queued · 🚀 started · 📊 hourly status · ✅/❌ final report with analysis |
| **🔧 Ask agent to fix** button | the agent investigates the failed job, fixes it and resubmits |
| *reply to any job message* | talk to the agent about that job ("lower the lr to 1e-4 and resubmit", "why is it slow?") |
| *plain message* | general conversation ("how much quota do I have left?", "start the sweep in ~/exp/sweep.sh") |
| `/jobs`, `/job <id>`, `/log <id> [n]` | status and log tails (long tails come as a file) |
| `/fix <id> [instructions]`, `/ask <text>`, `/new` | agent |
| `/backend [claude_code\|codex]` | show or switch the agent; conversations continue on the new one with a fresh session |
| `/stop` | stop the agent's current run (kills the whole process group) |
| `/health` | SSH connection, last poll, which agent is active |

Conversations are stored in `./state` (Overseer's conversation ↔ the CLI's session ID), so they survive restarts. A job the agent submits stays linked to the conversation that submitted it.

## Configuration notes

- `slurm.poll_interval` (default 60 s) sets how often jobs are checked. `slurm.track: agent` reports only on jobs the agent submitted.
- `agent.report_commands`: `summary` (default) sends one message per request, `live` batches commands every 5 s while the agent works, and `off` sends nothing.
- `agent.allowed_domains`: give the agent's local shell internet access to specific hosts (e.g. `pypi.org`). Its web search and fetch tools work regardless, because they run on the model provider's side.
- `agent.job_reports: false` skips the LLM root-cause summary, leaving only the heuristics.
- Anything the agent reads is sent to the model provider of the backend you use.
- Overseer itself writes only to `./state` and `/tmp/overseer` (`local.allowed_roots`).

## Development

```bash
uv sync --extra dev && uv run pytest
OVERSEER_TEST_CLI_DIR=/path/to/node_modules/.bin uv run pytest tests/test_e2e.py   # real CLIs + local sshd
```

- `tests/test_integration.py` covers the monitor, agent sessions, job detection, the runners' sandbox settings and the bot. It uses fake SLURM (`tests/fake_slurm/`) and fake CLIs (`tests/fake_cli/`).
- `tests/test_e2e.py` runs the real Claude Code and Codex against a scripted model (`tests/fake_model_server.py`) and a local sshd. It is skipped unless sshd, bubblewrap, socat and the CLIs are available.

| module | role |
|---|---|
| `config.py` | config schema; keeps local writes inside `local.allowed_roots` |
| `sshlink.py` | Overseer's own key/config/known_hosts, the master connection, the agent's ssh wrappers |
| `ssh.py`, `hpc.py` | Overseer's own read-only queries (squeue/sacct/scontrol, log tails) |
| `monitor.py`, `slurm.py`, `diagnose.py` | polling → queued/started/status/final-report messages |
| `runners.py` | Claude Code / Codex: sandbox settings, scrubbed env, event parsing, job detection |
| `agent.py` | conversations, backend switching, command reports, "agent submitted job" tracking |
| `bot.py` | Telegram commands, buttons, message → conversation routing |
