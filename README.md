# Overseer

Overseer is a Telegram bot for your SLURM jobs. It runs on your homelab and watches your jobs on the university HPC over SSH. When one fails, you hand it to **Claude Code** or **OpenAI Codex**: they get a real shell and full access to your HPC account, and fix and resubmit the job themselves.

- **Monitoring.** No LLM is involved, and checks run every minute.
  - **Job lifecycle:** you get messages when a job is 📥 **queued** and 🚀 **started** (which node), plus 📊 hourly **status** (elapsed time, latest metrics, log tail).
  - **Final report:** when a job ends you get ✅/❌, with the state, exit code, runtime against the limit and peak memory.
  - **Why it failed:** heuristics flag CUDA OOM, host OOM, time limits, NCCL errors, missing modules, NaNs and similar. An LLM root-cause summary is added on top.
- **Fixing.** Tap **🔧 Ask agent to fix**, reply to any job message, or just chat. The agent investigates with real commands (`ssh hpc 'sacct …'`, logs, code, quota, modules), edits, and runs `sbatch`.
  - **Visibility:** while it works, one message updates itself with its latest reasoning and the command it is running. Afterwards you see every command it ran, and you get a message the moment it submits a job.
  - **Follow-up:** that job is then monitored like any other, and replying to its report continues the same conversation.
- **Switching agents.** Change between Claude Code and Codex at any time with `/backend`.
- **Projects.** Each project (e.g. dreamer, chemistry) gets its own Telegram topic, conversations, jobs, agent settings and notes.
- **Rich messages.** Agent replies are formatted (bold, lists, code, tables), logs and command lists fold away, and the agent can send plots it makes on the HPC.
- **Autopilot.** `/auto 10 every 2h <goal>` lets the agent run experiments on its own. It is woken when its jobs finish, reads the results, keeps a research log and submits the next run, with optional check-ins while jobs run.

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
  - **Script submissions:** a job submitted through a script or CLI (no `sbatch` in the command itself) is found at the end of the request. Overseer asks SLURM which of your jobs were submitted during it, and links those whose id appears in the agent's output or reply.

**Heavy work goes to compute nodes, not the login node.** The agent has `hpc-run`, which runs a command on a compute node and returns its output:
```bash
hpc-run --dir /scratch/me/rf-dreamer -- python plot_rewards.py      # on the shared worker node
hpc-run --cpus 32 --mem 128G --time 120 -- python preprocess.py    # on a node of its own
hpc-run --gpu -- python eval.py                                    # on a GPU node of its own
hpc-run --status | --stop                                          # the shared worker
```
- **Shared worker:** the first `hpc-run` submits a small job (`overseer-worker`) on the compute partition that just holds a node (8 cores, 32 GB, 8 h by default). Later commands run inside it as job steps (`srun --jobid … --overlap`), so they start at once instead of queueing.
  - **Ending:** it ends itself after 45 minutes unused, before clusters cancel idle jobs.
  - **Not reported:** it isn't announced as a job; `/health` shows it.
- **Environment:** commands run in a login shell (modules and PATH as usual, plus `slurm.setup`), with `MPLBACKEND=Agg` so plots save without a display.
- **Instructions:** the agent is told to use `hpc-run` for anything that computes (scripts, plots, data processing, tests) and `sbatch` for real runs. The 📈 Plot button asks for plots made this way.
- **Guardrail:** commands the agent runs directly on the login node get 120 CPU seconds (`ulimit -t`, set by its `ssh` wrapper). A heavy script there is stopped with a message pointing to `hpc-run`.
  - **Exemptions:** commands that hand work to SLURM (`sbatch`, `srun`, `salloc`) aren't limited, because SLURM would pass the limit on to the job.
  - **Not a security boundary:** it guards against mistakes only.
- **Visibility:** a command that ran on the login node for 2 minutes or more is flagged in the command summary (`⚠️ ran 3m 10s on the login node: …`).

**On your homelab: confined.** Claude Code and Codex each run their shell inside their own OS sandbox (bubblewrap on Linux), configured by Overseer:

| | agent's shell |
|---|---|
| write | only its own scratch dir, `/tmp/overseer/agents/<conversation>` |
| read | the system, except Overseer's state dir (which includes the SSH key), `~/.ssh`, `~/.gnupg`, `~/.docker`, `~/.aws`, gcloud credentials, docker/podman sockets, `/run/user/<uid>` and your ssh-agent socket |
| network | none, except `agent.allowed_domains`. The HPC is reached through the unix socket of Overseer's SSH connection. **With Codex**, commands can still connect to services listening on this machine's localhost (127.0.0.1), because Codex's sandbox doesn't give them a network of their own; Claude Code's does. If your homelab runs services on localhost that the agent mustn't reach, use Claude Code. Proxy settings passed to Codex for its own API traffic aren't handed to the agent's commands |
| environment | scrubbed. Only `PATH`, locale and the API-key variables in `pass_env` are passed; the Telegram token is never passed, and the `.env` file can't be read |
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
uv run overseer login codex             # device-code login: open the printed URL on your phone/laptop

# 3. Telegram: create a bot with @BotFather, get your chat id (e.g. via @userinfobot)
cp .env.example .env && chmod 600 .env  # put TELEGRAM_BOT_TOKEN=... in it
uv run overseer check                   # ssh, slurm, a real test call to each installed agent
uv run overseer run
```

Both logins work on a headless server.

- **Codex:** `overseer login codex` uses Codex's device-code flow. It prints a URL and a code; open the URL on any device and enter the code. To use an API key instead, run `printenv OPENAI_API_KEY | overseer login codex -- --with-api-key`.
- **Claude Code:** `overseer login claude_code` prints a URL to open elsewhere and asks you to paste back the code. Alternatively, `overseer login claude_code -- setup-token` prints a long-lived token, which you then export as `CLAUDE_CODE_OAUTH_TOKEN`.
- **Other flags:** anything after `--` is passed to the CLI's login command in place of these defaults.

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
| **📈 Plot** button | the agent finds the job's metrics, plots them on the HPC and sends the picture with a short description of the trend |
| **📊 Status** / **📜 Log** buttons | Status: time used and left, latest metrics, GPU use, problems on stderr and the last lines of the log. Log: the log itself. If the job's output names its own log file (e.g. `output: …/train_<jobid>.log`), Overseer follows that file instead of SLURM's stdout, for status, final reports and autopilot check-ins too |
| `/fix <id> [instructions]`, `/ask <text>`, `/new` | agent |
| `/backend [claude_code\|codex]` | show both agents with their model and effort, or switch the active one; conversations continue on the new one with a fresh session |
| `/model [name\|default]`, `/effort [level\|default]` | show or set the active agent's model and effort without restarting (`default` goes back to `config.yaml`). Claude Code accepts low, medium, high, xhigh and max; Codex accepts minimal, low, medium, high and xhigh |
| *(while the agent works)* | a live ⏳ preview that updates every 5 s: elapsed time, number of commands, the agent's latest reasoning summaries (💭) and notes (💬), and the command running right now. Its **Stop** button stops the run. It's a Telegram streaming draft, which vanishes when the answer arrives; where drafts aren't available (`telegram.streaming: false`, group chats) it's a message that is edited in place and then removed |
| `/stop` | stop the agent's current run (kills the whole process group) |
| `/auto [rounds] [every 2h] <goal>` | autopilot (below). As a reply to a message, it applies to that conversation. `/auto` shows progress, `/auto off` stops it |
| `/health` | SSH connection, last poll, agents with their model and effort |

Every agent reply ends with a footer such as `claude_code · claude-opus-5-5 · effort high · 14.2k tokens (11.8k cached) · $0.04`.

- **Tokens:** every model call re-sends the CLI's instructions, its tool definitions and the conversation so far, so the total grows with each command the agent runs. Most of that is served from the provider's prompt cache, which costs roughly a tenth of fresh input.
  - **Codex counts:** Codex reports a running total for the whole conversation, so Overseer subtracts the previous total and shows this request's share. A conversation started with an older Overseer shows its total once, labelled "in this conversation so far".

- **Model and effort:** the footer shows what actually ran, even when nothing is configured.
  - **Claude Code** reports the model it used.
  - **Codex** doesn't report it in its output, so Overseer reads it from Codex's session log in `./state/codex/sessions`. If no effort was set, the model's default effort comes from Codex's built-in model catalog (`codex debug models`) and is shown as e.g. `effort medium (model default)`.
  - `/backend` and `/health` show the configured values plus what the last run actually used, e.g. `model CLI default (last run: gpt-5.5)`.
- **Cost:** only Claude Code reports it.

**Rich messages.**
- **Formatted replies:** the agent writes Markdown, and the reply is sent as one of Telegram's native rich messages (Bot API 10.1). That gives real tables, task lists, code blocks with syntax highlighting, formulas, collapsible sections, and pictures and files inside the reply itself, up to 32,768 characters per message.
  - **Escaping:** outside code, `$`, `||`, `==` and `<` are escaped, so `$HOME` or `a == b` doesn't turn into a formula, highlight or tag.
  - **Fallback:** if rich messages aren't available or Telegram rejects one, the regular formatted message goes out instead. Bold, lists, code and tables become aligned text, long answers keep their first paragraph and fold the rest, and pictures and files follow as separate messages. `telegram.rich_messages: false` always uses that path.
  - **No library change:** python-telegram-bot doesn't wrap the new methods yet, so Overseer calls them directly through its generic API call.
- **Folding:** the list of commands the agent ran, and the log tail or traceback in job reports, show their first lines and expand on tap.
- **Files and plots:** anything the agent puts in `./outbox/` in its working folder is sent with its reply. Pictures arrive as photos (an album if several, up to 10 per reply, 10 MB each); other files as documents (up to 50 MB).
  - **Where plots are made:** on the HPC, where the data is (e.g. "plot reward vs step for seeds 0-2"). The agent writes a small plotting script, runs it there and copies the PNG back with `scp`, so Overseer needs no plotting libraries.
  - **Safety:** Overseer only sends regular files from the outbox. Symlinks and hard links are refused, so the agent can't get Overseer to send something its sandbox can't read.
- **Fallback:** if Telegram ever rejects the formatting, the message is resent as plain text.

**Threads: each one is a conversation.** With topics enabled, the chat with the bot works like an AI chat app:
- **New thread:** a message from the ✏️ composer starts a new thread, which is its own conversation. If you have projects, Overseer asks which one it's about (📁 dreamer / 📁 chem / 💬 no project) and renames the thread, e.g. "dreamer · plot the reward curves". Starting the message with `dreamer: …` skips the question.
- **Continuing:** write in an existing thread to continue that conversation.
- **Commands:** `/new` gives the thread a fresh conversation, and `/ask`, `/auto` and `/stop` act on the thread they're sent in.
- **Job reports:** reports on jobs the agent submitted come back to the thread that submitted them.
- **What a thread inherits:** everything in a project shares its notes (`OVERSEER.md`), research log, agent settings and jobs. The project's own topic stays its home for jobs you submit by hand.
- **Deleted threads:** if you delete a thread, its messages continue in the project's topic.
- **Turning it off:** to stop users creating threads, use the setting in the @BotFather Mini App; then only Overseer's project topics exist.

**Projects: separate contexts, each in its own topic.**

```
/project new dreamer /scratch/ab11578/rf-dreamer DreamerV3 benchmarks on crafter and minigrid
/project new chem /scratch/ab11578/chem DFT with ORCA 6 on the cpu partition
```

Each project gets a topic in your chat with the bot, so the main chat stays uncluttered. Projects can also be defined in `config.yaml` under `projects:`.
- **Conversations:** a message in a project's topic goes to that project's agent conversation. `/new`, `/ask`, `/auto` and `/stop` in a topic apply to that project only.
- **Context:**
  - **The agent's instructions:** the project's folder and description (`/project context <text>` changes it).
  - **Project notes:** `OVERSEER.md` in the project folder on the HPC. The agent reads it at the start of every conversation and keeps it up to date with durable facts: environments, how to run things, known issues. `/project notes` shows it.
- **Agent settings:** a new project copies the main chat's backend, model and effort. After that, `/backend`, `/model` and `/effort` in its topic change only that project.
- **Jobs:**
  - **Agent jobs:** jobs the agent submits in a project are reported in its topic.
  - **Your own jobs:** jobs you submit by hand go to the project whose folder contains their working directory (the deepest one if folders nest). Others stay in the main chat, with one-tap "📁 project" buttons.
  - **Moving a job:** `/assign <job> <project|main>`, or reply to a job's message with `/assign <project>`. Its later updates and its conversation follow.
  - **Linking a job to a conversation:** `/assign <job>` with no project, sent in a conversation's thread, makes the job part of that conversation. Its reports come there, and autopilot wakes the agent when it ends.
- **Other commands:** `/project` lists projects; `/project archive <name>` and `/project restore <name>`; `/project dir <path>`.

Topics in private chats need the bot's topic mode ("threaded mode"), which you switch on in @BotFather. `/health` shows whether it's on. Without it, project messages still work, labelled `📁 project ·` in the main chat. If a project's topic gets deleted, Overseer creates a new one, or falls back to the labelled main-chat message.

**Autopilot: letting the agent iterate on its own.**

```
/auto 10 every 2h Get crafter seed 1 above score 5 in /scratch/me/rf-dreamer. One change per run, a100 or h100, whichever starts first.
```

1. **Kickoff:** the agent looks at the project, creates `RESEARCH_LOG.md` in the project folder on the HPC, submits the first experiment and stops. Nothing runs while the job waits in the queue.
2. **Rounds:** when the job finishes, Overseer resumes the same conversation with its final report: state, exit code, runtime, peak memory, findings and the end of the log. The agent reads the results, updates the log and submits the next run. That is one round.
   - **Batches:** jobs submitted together are reported together. The round starts when all of them have finished, or as soon as one fails.
3. **Check-ins** (`every 2h`; off unless you ask): while its jobs run, the agent sees their progress (metrics and the end of the log). If a run is clearly broken or diverging, it cancels it, notes why in the log and resubmits. Healthy check-ins come silently and don't count as rounds. There are at most 12 per job.
4. **Ending:** every reply ends with a status line, which Overseer removes from the message:
   - **continue:** keep the loop going;
   - **done:** 🏁 the goal is reached or nothing sensible is left, and autopilot turns off;
   - **need help:** 🙋 you get pinged with the agent's question. Reply to answer; the loop continues once the agent submits again.

Guardrails:
- **Hard round limit:** the default is 10 and the maximum 50. The agent is told when it's on its last round and should wrap up.
- **Pauses on any problem:** a round that fails or is `/stop`ped pauses the loop. Send `/auto <rounds>` as a reply to retry.
- **Its own jobs only:** it only wakes for jobs the agent submitted in that conversation. Jobs it cancels itself don't trigger a new round.
- **Visible at every step:** you see every round, command and submitted job in Telegram as usual. `overseer watch` shows the reasoning live.
- **Cost:** each round re-sends the conversation so far. Rounds are usually hours apart, so the provider's prompt cache has expired and long autopilot runs get pricier per round. The research log keeps the important history in a file you can read any time.

**Watching the agent live from the homelab.** `overseer watch` follows what the agent is doing in the terminal: its reasoning summaries, what it writes between steps, every command with its output, and the final answer.

```bash
uv run overseer watch                 # every conversation: what is running now, then anything new
uv run overseer watch chat-a7b233     # one conversation, from its first logged request (a unique prefix is enough)
uv run overseer watch chat-a7 --full  # complete command output instead of the last 12 lines
```

- **Read-only:** unlike `overseer open`, it's safe to use while the bot is working.
- **Where it reads from:** Overseer records every request in `./state/transcripts/<conversation>.jsonl`, starting with this version. Older conversations have no live log; `overseer open` still shows their history.
- **Reasoning is a summary:** neither OpenAI nor Anthropic returns the model's raw chain of thought. Codex passes on OpenAI's reasoning summaries, and Claude Code passes on Claude's thinking summaries. How much you see depends on the model and the effort level.

**Opening a conversation in the CLI itself.** To read the full log (every message, command and output) or keep working in the terminal:

```bash
uv run overseer sessions              # conversations, backend, job, last used
uv run overseer open                  # open the most recent one
uv run overseer open chat-b920aa      # or a specific one (a unique prefix is enough)
```

- **What it does:** this resumes the session in Codex's or Claude Code's normal interactive UI, with the same sandbox, the same `ssh hpc` access through Overseer's connection, and Overseer's isolated config. Your own `~/.codex` and `~/.claude` aren't touched.
- **First use of Claude Code:** it shows its one-time welcome and theme setup the first time Overseer's config is opened interactively.
- **Shared history:** anything you add there becomes part of the conversation, so avoid typing while the bot is working on it.

Conversations are stored in `./state` (Overseer's conversation ↔ the CLI's session ID, saved as soon as a run starts), so they survive restarts. A job the agent submits stays linked to the conversation that submitted it.

## Configuration notes

- `slurm.poll_interval` (default 60 s) sets how often jobs are checked. `slurm.track: agent` reports only on jobs the agent submitted.
- `agent.report_commands`: `summary` (default) sends one message per request, `live` batches commands every 5 s while the agent works, and `off` sends nothing.
- `agent.live_progress: false` turns off the self-updating ⏳ message (the transcript for `overseer watch` is always written).
- `slurm.compute`: the partition, cores, memory and hours of the shared worker, how long it may stay idle, how long `hpc-run` waits for a node, the GPU partition and gres for `--gpu`, extra `sbatch` flags (e.g. an account), the login-node CPU limit (`login_cpu_seconds`, 0 = off) and the warning threshold.
- `agent.autopilot`: default and maximum rounds, a default check-in interval (`checkin_minutes`, 0 = off), the shortest interval allowed, and the check-in cap per job.
- `agent.allowed_domains`: give the agent's local shell internet access to specific hosts (e.g. `pypi.org`). Its web search and fetch tools work regardless, because they run on the model provider's side.
- `agent.job_reports: false` skips the LLM root-cause summary, leaving only the heuristics.
- Anything the agent reads is sent to the model provider of the backend you use.
- Overseer itself writes only to `./state` and `/tmp/overseer` (`local.allowed_roots`).
- Secrets live in `.env` next to `config.yaml` (`local.env_file`). It is loaded at startup, and variables already set in the environment take precedence. Overseer warns if the file is readable by other users, and the agent's sandbox blocks reading it.

## Development

```bash
uv sync --extra dev && uv run pytest
OVERSEER_TEST_CLI_DIR=/path/to/node_modules/.bin uv run pytest tests/test_e2e.py   # real CLIs + local sshd
```

- `tests/test_integration.py` covers the monitor, agent sessions, job detection, the runners' sandbox settings and the bot. It uses fake SLURM (`tests/fake_slurm/`) and fake CLIs (`tests/fake_cli/`).
- `tests/test_e2e.py` runs the real Claude Code and Codex against a scripted model (`tests/fake_model_server.py`) and a local sshd. It is skipped unless sshd, bubblewrap, socat and the CLIs are available.

| module | role |
|---|---|
| `config.py` | config schema, `.env` loading; keeps local writes inside `local.allowed_roots` |
| `sshlink.py` | Overseer's own key/config/known_hosts, the master connection, the agent's ssh wrappers |
| `ssh.py`, `hpc.py` | Overseer's own read-only queries (squeue/sacct/scontrol, log tails) |
| `monitor.py`, `slurm.py`, `diagnose.py` | polling → queued/started/status/final-report messages |
| `runners.py` | Claude Code / Codex: sandbox settings, scrubbed env, event parsing, job detection |
| `agent.py` | conversations, backend switching, command reports, "agent submitted job" tracking |
| `live.py` | transcripts, the self-updating progress message, `overseer watch` |
| `autopilot.py` | `/auto`: waking conversations when their jobs finish, check-ins, round limits, status lines |
| `projects.py` | projects from config and Telegram, folder matching for jobs |
| `compute.py` | `hpc-run` and the shared compute worker, the agent's ssh wrapper with the login-node CPU limit |
| `richtext.py` | Markdown → Telegram HTML (fallback) and Rich Markdown, tag-safe message splitting, collapsible sections |
| `bot.py` | Telegram commands, buttons, message → conversation routing |
