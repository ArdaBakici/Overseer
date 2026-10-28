# Overseer

**A Telegram bot that watches your SLURM jobs and lets an AI agent fix them.**

Overseer runs on a machine you own, such as a homelab server, and keeps an eye on your jobs on a university or company
HPC cluster over SSH. Every job gets a live card in Telegram. When one fails, you hand it to **Claude Code** or
**OpenAI Codex** with one tap: the agent gets a real shell on your HPC account, finds the problem, fixes it and
resubmits, and you see every command it runs.

It can also work on its own: give it a goal (*"get seed 1 above score 5, one change per run"*) and it submits a run,
sleeps until the job ends, reads the results, writes them down and submits the next one.

```
 your machine (homelab)                                                    HPC login node
┌───────────────────────────────────────────────────────────┐           ┌────────────────┐
│ Telegram bot ◄──► you                                     │           │                │
│ Monitor ── squeue/sacct ──┐                               │           │                │
│                           ├─► Overseer's SSH master ──────┼── ssh ───►│ your account   │
│ Claude Code / Codex       │   (key in ./state/ssh)        │           │ (full access)  │
│  └─ sandboxed shell ──────┘   via /tmp/overseer/ssh/      │           │                │
│     `ssh hpc …`               master.sock                 │           │                │
└───────────────────────────────────────────────────────────┘           └────────────────┘
```

## What it does

**Monitoring, with no AI involved**
- A **live card per job**, edited in place as it goes from queued to running to done: node, time left, progress
  and the metrics that matter, with graphs from your own plotting script.
- **Why it failed:** the decisive log line, plus checks for CUDA and host out-of-memory, time limits, NCCL errors,
  missing modules and NaNs.
- **Alerts** on numbers you care about (`score < 1 after 2h`, `loss is nan`), an **idle-GPU alarm**, disk **quota
  warnings**, a **morning digest**, and `/cluster` for free GPUs and when your queued jobs should start.

**An agent that fixes and runs experiments**
- **One tap to fix:** 🔧 on a failed job, or just reply to any job message ("lower the lr and resubmit"). While it
  works you see its reasoning and the command it's running; afterwards, every command it ran.
- **Works on its own:** it watches the jobs it submits and is woken when they end, keeps a research log, and records
  each experiment's result and what it showed, so it doesn't retry what already failed.
- **Projects:** each project gets its own Telegram topic, conversations, notes and settings.
- **Planner mode:** a strong model plans and delegates to subagents on cheaper models, each in its own thread.
- **Claude Code or Codex, several accounts each:** switch with `/backend` and `/account`. When an account runs out
  of usage, the bot moves to the next one by itself, and the conversation keeps its history.

**Built to be trusted with your account**
- On the HPC the agent can do anything you can, by design, and you see all of it.
- On your machine it's **sandboxed**: it can't read your SSH key, your logins or your secrets, and it can only write
  its own scratch folder. Overseer uses its own SSH key and never touches `~/.ssh`. See [security.md](docs/security.md).

**Optional extras:** research notes in your **Obsidian** vault, a read-only **dashboard** in Telegram (on your
tailnet only), and a second bot, a **personal assistant** whose memory is your Obsidian vault
([assistant.md](docs/assistant.md)).

## Getting started

You need a Linux machine that stays on, an SSH login to a SLURM cluster, Telegram, and a Claude or ChatGPT
subscription (or an API key). Get Overseer, then let the setup walk you through the rest:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh            # uv: Python and Overseer's dependencies
git clone https://github.com/ArdaBakici/Overseer ~/Overseer && cd ~/Overseer
uv sync
uv run overseer setup
```

The setup runs in the terminal, step by step. It does the following:
- installs what's missing (the agents' sandbox, Node.js, Claude Code / Codex);
- checks your Telegram bot's token and finds your chat id when you message the bot;
- creates Overseer's own SSH key and adds it to the cluster, then lists the cluster's partitions to pick from;
- logs the agents in, checks everything, and starts the bot in the background.

Run it again any time to change something. The [setup guide](docs/setup.md) explains each step, and how to do it
by hand.

```bash
uv run overseer tui        # watch the bots live: jobs, what the agents are doing, wake-ups, usage, the log
uv run overseer stop       # (and start, restart, status) - the bot runs in the background, restarted after a crash
```

## Everyday use

Just talk to it: *"which of my jobs are running?"*, *"why did 18746955 fail?"*, *"plot the reward for seeds 0-2"*.
Reply to a job's message to talk about that job.

| Command | |
|---|---|
| **🔧 Ask agent to fix** | on a failed job's card: the agent investigates, fixes and resubmits |
| `/jobs`, `/job <id>`, `/log <id>` | your jobs, one job's card, a log tail |
| `/ask <text>`, `/new`, `/stop` | talk to the agent, start a fresh conversation, stop its run |
| `/auto <goal>` | let the agent work on a goal on its own; `/auto off` cancels its wake-ups |
| `/project new <name> <dir> [about]` | a project with its own topic |
| `/results`, `/findings` | what each experiment showed |
| `/backend`, `/account`, `/model`, `/effort` | which agent, which login, which model and effort |
| `/usage`, `/digest`, `/search <text>`, `/cluster` | tokens and cost, the daily summary, past work, free GPUs |
| `/health`, `/update`, `/help` | status, update the agent CLIs, every command |

The **[guide](docs/guide.md)** explains every command and feature in detail.

## Documentation

| | |
|---|---|
| [docs/setup.md](docs/setup.md) | **Setting up**: `overseer setup`, or each step by hand; troubleshooting |
| [docs/guide.md](docs/guide.md) | **Using it:** commands, job cards, projects, working on its own, results, planner mode, accounts, the terminal monitor, configuration |
| [docs/security.md](docs/security.md) | What the agent can and can't do, the sandbox, your SSH key |
| [docs/assistant.md](docs/assistant.md) | The personal assistant bot: Obsidian, tasks, reminders, calendars |
| [docs/development.md](docs/development.md) | Running the tests, the architecture |

## Development

```bash
uv sync --extra dev && uv run pytest
```

Overseer is a platform (`overseer/core`) with domains plugged into it: the HPC bot (`overseer/hpc`) and the personal
assistant (`overseer/assistant`). See [development.md](docs/development.md) for the architecture and the end-to-end
tests with the real Claude Code and Codex.
