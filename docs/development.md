# Development

## Running the tests

```bash
uv sync --extra dev && uv run pytest
OVERSEER_TEST_CLI_DIR=/path/to/node_modules/.bin uv run pytest tests/test_e2e.py   # real CLIs + local sshd
```

- `tests/test_integration.py` covers the monitor, agent sessions, job detection, the runners' sandbox settings and the bot. It uses fake SLURM (`tests/fake_slurm/`) and fake CLIs (`tests/fake_cli/`).
- `tests/test_assistant.py` covers the assistant: tasks, reminders, the vault's history and undo, its config and sandbox, a whole turn.
- `tests/test_calendar.py` covers the assistant's calendars; `tests/test_hpc_vault.py` the HPC bot's Obsidian folder; `tests/test_accounts.py` several accounts per CLI, usage limits and failover.
- `tests/test_e2e.py` runs the real Claude Code and Codex against a scripted model (`tests/fake_model_server.py`) and a local sshd. It is skipped unless sshd, bubblewrap, socat and the CLIs are available.

## Architecture: a platform and its domains

Overseer is a platform (`overseer/core`) with domains plugged into it (`overseer/hpc`, the SLURM jobs;
`overseer/assistant`, the personal assistant). The platform
runs agent conversations from Telegram: the Claude Code / Codex runners and their sandbox, the Telegram layer,
the agent's control command, wake-ups, transcripts and search, usage, the digest and the dashboard. A domain adds
what is its own through the hooks in `core/domain.py`: the agent's instructions and context, its sandbox additions
and skills, what it watches in the agent's commands, its watch kinds, `ov` commands, Telegram commands and buttons,
digest sections, dashboard tabs and searchable records. `config.yaml`'s `domain:` picks it (`hpc` by default), and
its own config sections (`ssh`, `slurm`) live in `overseer/<domain>/config.py`.

Each domain runs as its own process and Telegram bot, with its own config, state folder and secrets
(`overseer run -c <its config>`). The platform never imports a domain's code (a test enforces it): it loads the
domain by name.

**The platform (`overseer/core`)**

| module | role |
|---|---|
| `domain.py` | the Domain interface: every hook a domain can fill, with neutral defaults |
| `app.py` | one bot: the platform wired around its domain (`overseer run`) |
| `config.py` | config schema, the domain's sections, `.env` loading; keeps local writes inside `local.allowed_roots` |
| `runners.py` | Claude Code / Codex: sandbox settings (plus the domain's), scrubbed env, event parsing |
| `agent.py` | conversations, backend switching, command reports, the domain's hooks around each run |
| `live.py` | transcripts, the self-updating progress message, `overseer watch` |
| `autopilot.py` | timers and the domain's watch kinds that wake conversations, pausing after a failed wake-up |
| `ctl.py` | the agent's control command over a unix socket (projects, wake-ups, feedback, skills, subagents, history), plus the domain's commands |
| `bot.py` | Telegram: the platform's commands and buttons, message → conversation routing, the domain's commands |
| `subagents.py`, `feedback.py`, `addons.py` | planner mode; agents' feedback about Overseer; skills as addons |
| `search.py` | `/search`, `ov history`: a full-text index (SQLite FTS5) of transcripts and the domain's records |
| `digest.py` | the morning digest: the domain's sections, the agents, tokens, what waits for you |
| `dashboard.py`, `dashboard.html` | the Mini App's server (Telegram initData auth) and page; the domain gives its tabs |
| `usage.py`, `projects.py`, `store.py`, `notify.py`, `richtext.py` | `/usage`; projects; the database; messages; Markdown → Telegram |

**The HPC domain (`overseer/hpc`)**

| module | role |
|---|---|
| `domain.py` | HpcDomain: the hooks, filled in for SLURM jobs on the cluster |
| `config.py` | the `ssh` and `slurm` sections and the paths derived from them |
| `prompts.py` | the agent's instructions for the HPC |
| `activity.py` | jobs the agent submits (sbatch), cancels (scancel), heavy work on the login node |
| `sshlink.py`, `ssh.py`, `remote.py` | Overseer's own key/config/known_hosts, the master connection, the agent's ssh wrappers; Overseer's read-only queries |
| `monitor.py`, `slurm.py`, `diagnose.py` | polling → live job cards, final reports, alerts, idle GPUs |
| `watches.py` | job watches (`ov watch`), check-ins, alert and idle-GPU wake-ups |
| `commands.py`, `telegram.py` | the HPC's `ov` commands; its Telegram commands and job-card buttons |
| `metrics.py`, `plots.py`, `results.py` | what matters in a job, its graphs, results and findings |
| `compute.py`, `cluster.py`, `quota.py` | `hpc-run` and the compute worker; `ov cluster`; disk quota |
| `digest.py`, `dashboard.py`, `dashboard.js`, `search.py` | the HPC's digest sections, dashboard tabs and searchable records |

**The assistant domain (`overseer/assistant`)**

| module | role |
|---|---|
| `domain.py` | AssistantDomain: the vault-writing sandbox, persona and context, the vault's history around each reply (↩️ Undo) |
| `config.py` | the `vault` section; its own state, tmp folders and token required |
| `prompts.py`, `skills/assistant/SKILL.md` | the agent's instructions and its skill |
| `vault.py` | the vault: its own notes (persona, about me, writing style, personality), daily notes, Syncthing conflicts, the history (git outside the vault) and undo |
| `tasks.py`, `when.py` | Obsidian Tasks: parsing, selecting, adding, completing, recurrences; days and times as people write them |
| `routines.py`, `scheduler.py` | Routines.md: parsing, times, follow-up policies; running them, the first hello |
| `media.py` | `ov fetch` (public addresses only, media only, size-limited) and Giphy search; `ov capabilities` |
| `reminders.py` | follow-ups: reminders, `say:` routines, check-backs on commitments, focus sessions (with follow-up policies) |
| `calendar.py` | Google Calendar and iCal links: the login, reading events (every few minutes, two weeks ahead), free time, changes behind ✅ cards or with ↩️ Undo, reminders before events |
| `commands.py`, `telegram.py` | `ov task`, `ov remind`, `ov cal`; /tasks, /task, /remind, /reminders, /undo, /vault, /persona and their buttons |
| `digest.py`, `dashboard.py`, `dashboard.js`, `search.py` | its digest sections, dashboard tabs and the notes in search |
