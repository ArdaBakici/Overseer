# Overseer

Overseer is a Telegram bot for your SLURM jobs. It runs on your homelab and watches your jobs on the university HPC over SSH. When one fails, you hand it to **Claude Code** or **OpenAI Codex**: they get a real shell and full access to your HPC account, and fix and resubmit the job themselves.

- **Monitoring.** No LLM is involved, and checks run every minute.
  - **A live card per job:** each job gets one message in its project's "📊 jobs" topic, edited in place and silently as it goes from queued to running. It shows the node, time used and left, progress, and the metrics that matter.
  - **Final report:** when a job ends, the card shows the end and a short report is posted under it: ✅/❌, runtime, the key result and a one-line verdict, with exit code, memory and the log tail folded away. Only failures make a sound.
  - **Why it failed:** heuristics flag CUDA OOM, host OOM, time limits, NCCL errors, missing modules, NaNs and similar. An LLM root-cause summary is added on top.
- **Job arrays.** An array is one job, under the id `sbatch` printed. It's running while any task is, and its report gives the task states ("📦 job array: 9 COMPLETED, 2 FAILED"). It ends with the worst task's state once all its tasks have ended.
- **Fixing.** Tap **🔧 Ask agent to fix**, reply to any job message, or just chat. The agent investigates with real commands (`ssh hpc 'sacct …'`, logs, code, quota, modules), edits, and runs `sbatch`.
  - **Visibility:** while it works, one message updates itself with its latest reasoning and the command it is running. Afterwards you see every command it ran, and you get a message the moment it submits a job.
  - **Follow-up:** that job is then monitored like any other, and replying to its report continues the same conversation.
- **Switching agents.** Change between Claude Code and Codex at any time with `/backend`.
- **Projects.** Each project (e.g. dreamer, chemistry) gets its own Telegram topic, conversations, jobs, agent settings and notes. A "general" topic holds everything that isn't about a project.
- **The agent talks to Overseer.** With its `ov` command it registers the jobs it submits, creates projects and moves or stops tracking jobs when you ask it to.
- **Rich messages.** Agent replies are formatted (bold, lists, code, tables), logs and command lists fold away, and the agent can send plots it makes on the HPC.
- **Working on its own.** The agent watches the jobs it submits and is woken when they end, or by timers it sets. It then reads the results, updates the project's research log and submits the next run. It needs no approval, and every wake-up is announced with a Cancel button. `/auto <goal>` hands it a goal, and `/usage` shows the tokens and cost of the day.

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
  - **Registration:** the agent registers every job it submits with `ov job add <id>` (below), whichever way it submitted it. Overseer checks the id with SLURM first.
  - **Fallback detection:** Overseer also spots submissions itself, in case the agent forgets to register one. It reads `sbatch` output as the agent works. At the end of each request it asks SLURM which of your jobs were submitted during it, and links those whose id appears in the agent's output or reply. This catches jobs submitted through a script or CLI.

**The agent talks to Overseer: `ov`.** The agent's shell has an `ov` command that sends requests to the running Overseer over a local unix socket and gets an answer at once:
```bash
ov job add 18746955 18746956          # "I submitted these": checked with SLURM, then tracked in this conversation
ov job list [--all] | show <id>
ov job move 18746955 dreamer          # reassign to a project (or "main")
ov job delete 18746955                # stop tracking it - the job keeps running (scancel cancels it)
ov project new chem --dir /scratch/me/chem --about "DFT with ORCA" [--move-here]
ov project set <name> --dir … --about … | archive <name> | list
ov conversation move <project|main>   # this conversation belongs to a project
ov notify "epoch 3 done, loss 0.41"   # a message to you right away
ov status                             # project, its jobs, what will wake it
ov watch 18746955 [--every 2h] [--note "…"]   # wake me when this job ends (below)
ov wake in 90m "…"  |  ov wake every 2h [--for 12h] "…"  |  ov watches  |  ov unwatch w3
```
So you can just tell the agent "make a project for my chemistry runs in /scratch/me/chem", "this job belongs to dreamer" or "stop tracking 18746955".
- **No confirmations:** every change is announced in Telegram (`📁 The agent created project chem`, `🗑 Overseer no longer tracks job …`), and all of them are easy to undo with `/project restore` or `/assign`.
- **Verified:** `ov job add` only accepts jobs that SLURM knows as yours: from `scontrol`, or from `sacct` for jobs that finished a while ago. Ids that don't exist, or that belong to another user, are refused.
- **Stop tracking:** `ov job delete` only stops Overseer from tracking a job: no more reports, and nothing waits for it any more. The agent can cancel a job with `scancel`.
- **Scoped to the run:** each run of the agent gets a one-time token (`$OVERSEER_CTL`) that tells Overseer which conversation is asking. The token stops working when the run ends.
- **Sandbox:** the socket sits in Overseer's tmp dir, and both sandboxes allow connecting to it (as with the SSH socket).
- **The agents know how to use it:**
  - **System prompt:** it has the essential rules (register every job, use `ov` for project and job requests, how to be woken and work on its own over many runs).
  - **`overseer` skill:** a full reference with workflows (`overseer/skills/overseer/SKILL.md`). Overseer installs it for both CLIs:
    - Claude Code: in its config dir.
    - Codex: in each conversation's scratch dir as `.agents/skills`, because Codex has the model read the file with a command, and the sandbox can't read Overseer's state dir.
  - **`ov help`:** prints the same commands.

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

**Storage and Python on the HPC.**
- **Files (inodes):** on clusters the file-count quota usually runs out long before space does. When it's full, jobs, installs and even logins fail.
  - **The agent's instructions** include your quotas, read with `slurm.quota_command` (e.g. `myquota`). For example: `/home` 50GB and 500k files, `/scratch` 5000GB and 500k files.
  - **Rules for the agent:** few large files instead of many small ones (sharded datasets, one metrics file per run, rotating checkpoints). The code it writes and the job scripts it submits must do the same. Clean up caches and temporary files, and ask before deleting your files.
- **Warnings:** Overseer reads the quota at startup and once a day.
  - When a filesystem's space or files pass 80%, you get one Telegram message ("💾 /scratch is at 85% of its quota: 425,000 of 500,000 files"). It warns again only after usage has dropped back below and risen again.
  - The agent is told at the start of every request while it stays above 80%.
- **Python:** the agent's instructions say to use uv (`uv venv`, `uv add`, `uv run`, in job scripts too). They also say to keep uv's cache on the same filesystem as the environments, so packages are hard-linked rather than copied into new files.

**On your homelab: confined.** Claude Code and Codex each run their shell inside their own OS sandbox (bubblewrap on Linux), configured by Overseer:

| | agent's shell |
|---|---|
| write | only its own scratch dir, `/tmp/overseer/agents/<conversation>` |
| read | the system, except Overseer's state dir (which includes the SSH key), `~/.ssh`, `~/.gnupg`, `~/.docker`, `~/.aws`, gcloud credentials, docker/podman sockets, `/run/user/<uid>` and your ssh-agent socket |
| network | none, except `agent.allowed_domains`. The HPC is reached through the unix socket of Overseer's SSH connection, and Overseer itself through the `ov` socket. **With Codex**, commands can still connect to services listening on this machine's localhost (127.0.0.1), because Codex's sandbox doesn't give them a network of their own; Claude Code's does. If your homelab runs services on localhost that the agent mustn't reach, use Claude Code. Proxy settings passed to Codex for its own API traffic aren't handed to the agent's commands |
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
- `curl` to the internet fails, while the models' own web search works: Claude Code's WebSearch and WebFetch, and Codex's live `web_search` (`agent.codex.web_search: live | cached | disabled`).
- `hpc-run`, `scp` into the outbox and `ov` work from inside both sandboxes.

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
| *(automatic)* | a live card per job in its jobs topic (queued → running → ended, edited in place) · ✅/❌ short final report under it |
| **🔧 Ask agent to fix** button | the agent investigates the failed job, fixes it and resubmits |
| *reply to any job message* | talk to the agent about that job ("lower the lr to 1e-4 and resubmit", "why is it slow?") |
| *plain message* | general conversation ("how much quota do I have left?", "start the sweep in ~/exp/sweep.sh") |
| `/jobs`, `/job <id>`, `/log <id> [n]` | status and log tails (long tails come as a file) |
| **📈 Plot** button | the agent finds the job's metrics, plots them on the HPC and sends the picture with a short description of the trend |
| **📊 Status** / **📜 Log** buttons | Status (on a job's card): refreshes the card now; `/job <id>` shows the card's content plus the last lines of the log. Log: the log itself. If the job's output names its own log file (e.g. `output: …/train_<jobid>.log`), Overseer follows that file instead of SLURM's stdout, for status, final reports and check-ins too |
| `/fix <id> [instructions]`, `/ask <text>`, `/new` | agent |
| `/backend [claude_code\|codex]` | show both agents with their model and effort, or switch the active one; conversations continue on the new one with a fresh session |
| `/model [name\|default]`, `/effort [level\|default]` | show or set the active agent's model and effort without restarting (`default` goes back to `config.yaml`). Claude Code accepts low, medium, high, xhigh and max; Codex accepts minimal, low, medium, high and xhigh |
| *(while the agent works)* | a live ⏳ preview that updates every 5 s: elapsed time, number of commands, the agent's latest reasoning summaries (💭) and notes (💬), and the command running right now. Its **Stop** button stops the run. It's a Telegram streaming draft, which vanishes when the answer arrives; where drafts aren't available (`telegram.streaming: false`, group chats) it's a message that is edited in place and then removed |
| `/stop` | stop the agent's current run (kills the whole process group) |
| `/update` | update Claude Code and Codex now, with their own updaters; `/health` shows their versions. Overseer also does it once a day while no agent run is in progress (`agent.auto_update`), and tells you when a version changed. On the homelab: `uv run overseer update` |
| `/compact [focus]` | compact the conversation now (below). As a reply to a message, it applies to that conversation |
| `/auto <goal>` | ask the agent to work on a goal on its own (below). As a reply to a message, it applies to that conversation. `/auto` lists pending wake-ups, `/auto off` cancels them (in this thread, topic, or replied-to conversation) |
| `/usage [today\|yesterday\|week\|month]` | tokens and cost of the agent's runs: total, runs from you, wake-ups and check-ins, by backend, the top conversations |
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
  - **Linking a job to a conversation:** `/assign <job>` with no project, sent in a conversation's thread, makes the job part of that conversation. Its reports come there, and its end wakes the agent.
- **Other commands:** `/project` lists projects; `/project archive <name>` and `/project restore <name>`; `/project dir <path>`.

**The "general" topic: anything that isn't about a project.** With topics on, Overseer also creates a 💬 general topic. Use it for questions like "how much quota do I have left?" or "which partitions have free GPUs?". It is the main chat's conversation, so `/new`, `/ask`, `/auto` and `/stop` there act on the main conversation. Jobs that belong to no project are reported there too, and if the topic is deleted, Overseer makes a new one.

Topics in private chats need the bot's topic mode ("threaded mode"), which you switch on in @BotFather. `/health` shows whether it's on. Without it, project messages still work, labelled `📁 project ·` in the main chat. If a project's topic gets deleted, Overseer creates a new one, or falls back to the labelled main-chat message.

**Job cards and the jobs topics.** Job news lives apart from your conversations, so the agent's replies don't get buried:
- **Where:** with topics on, each project has a "📊 dreamer · jobs" topic, and one "📊 jobs" topic holds the jobs that belong to no project.
- **One card per job:** each job has a single card there, edited in place and silently: queued, running, and refreshed every 10 minutes (`slurm.status_interval_minutes`). 📊 Status refreshes it now.
  ```
  🏃 18756118 bench-dreamerv3-mixed-s1 · cn015
  ⏱ 16:08 / 24:00 (67%, 7h51m left) · step 2.85M / 5M (57%)
  🎯 score 4.1
  📈 value-loss 0.82 · dyn-loss 3.67
  ⏰ agent watching (check-ins every 2h)
  ```
- **When it ends:** the card shows the final state, and a short report is posted as a reply to it:
  ```
  ❌ 18771337 pka-ft OUT_OF_MEMORY after 2h03m (peak RAM 31.8G of 32G)
  🧠 Host OOM while loading the full split; ask for --mem 64G.
  <the decisive log line>
  ▸ details (exit, memory, metrics, log tail - folded)
  ```
- **In the conversation:** when a job the agent submitted ends and the agent isn't watching it, you get a one-line pointer: "✅ Job 123 … finished". Watched jobs need no pointer, because the agent's reply follows.

**What matters in a job: the agent decides, and sets alerts.** The agent tells Overseer which numbers matter and how to get them, per job, per job-name pattern (a sweep) or as a project default:
```bash
ov metrics set --pattern "bench-dreamerv3-*" --critical "crl/train_episode/score=score" \
    --important "train/loss/value=value-loss,train/loss/dyn=dyn-loss" --progress "step/5000000" \
    --command 'tail -n1 {workdir}/runs/seed1/metrics.jsonl' \
    --alert "score < 1 after 2h" --wake-on "train/loss/value is nan" --wake-on "step stalled 30m"
```
- **The card's lines:** the critical values are its 🎯 line (also in the report's headline), the important ones its 📈 line, and progress the steps done out of the total.
- **Where values come from:** the job's log by default. `--command` is a quick command on the HPC that prints `name=value` lines or a JSON object, for values the log doesn't show.
- **Without a setting:** Overseer shows the 3 most informative metrics it finds in the log.
- **Alerts:**
  - forms: comparisons (`score < 1 after 2h`), `is nan`, and `stalled 30m`;
  - when: checked at every card refresh;
  - how often: each fires once per job.
  - `--alert` pings you with a 🚨 reply under the card. `--wake-on` also wakes the agent, with the job's details, so it can stop or fix the run.

**Working on its own: watches and wake-ups.** The agent only runs when you write to it, or when something it set up wakes it. It needs no approval:
- `ov watch <jobs>`: woken when those jobs have ended (all of them, or as soon as one fails), with their final reports: state, exit code, runtime, peak memory, findings and the end of the log. `--every 30m` adds check-ins while they run.
- `ov wake in 90m` / `ov wake every 2h [--for 12h]`: woken by a timer, once or repeatedly.

This is how it runs experiments over many hours: submit, watch, end its reply. When woken it reads the results, updates the research log, submits the next run and watches it. It stops when the goal is reached or nothing sensible is left, and says so; when it needs your decision, it asks. `/auto <goal>` hands it a goal to work on this way, e.g.
```
/auto Get crafter seed 1 above score 5 in /scratch/me/rf-dreamer. One change per run, a100 or h100, whichever starts first.
```
- **You see every one:** a watched job's card says "⏰ agent watching", with a ✖ watch button that cancels it; a timer is announced quietly in the conversation's topic, with a ✖ Cancel button. Replies end with what the agent is waiting for, and `/auto` lists everything pending.
- **Turning it off:** `/auto off` cancels the pending wake-ups (in that thread or topic, or as a reply: that conversation), and the agent is told. `/stop` stops a run.
- **Limits:** check-ins and repeating timers are at least 15 minutes apart, and a repeating timer ends (after 24 hours unless the agent sets `--for`). There is no limit on the number of wake-ups. `/usage` shows what they cost.
- **Pauses on problems:** a wake-up that fails or is `/stop`ped pauses that conversation's wake-ups until you write to it there.
- **Research log:** in every project, the agent keeps `RESEARCH_LOG.md` in the project folder.
  - **Contents:** the current goal at the top (rewritten when the goal changes), then an entry per experiment: what changed and why, the job ids, the results, the next idea.
  - **What the agent sees:** Overseer shows it the log's beginning and latest entries when a conversation starts and at every wake-up, so it keeps the thread across long gaps and compaction.
  - **Size:** the agent keeps it well under 200k tokens; past about 600 KB, Overseer asks it to condense the older entries.
- **Cost:** each wake-up re-sends the conversation so far, and wake-ups are often hours apart, after the provider's prompt cache has expired. The agent is told to wake itself only when there is something to look at.

**Long conversations: context compaction.** When a conversation's context fills up, the CLIs compact it on their own: older turns are replaced by a summary, and the work goes on. Overseer leaves this on and shows it:
- **When it happens:**
  - **Claude Code:** when the context is almost full (about 97% of the model's window in tests).
  - **Codex:** at the model's auto-compact limit. To compact earlier, set it with `agent.codex.extra_config: ["model_auto_compact_token_limit=200000"]`.
- **Where you see it:**
  - **In the reply:** `🗜 The conversation was compacted automatically, as the context was full (970.2k → 12.4k tokens): older turns are now a summary.` Codex doesn't report token counts.
  - **While it works:** the live progress message says "🗜 compacting the conversation…".
  - **Afterwards:** the transcript and `overseer watch` record it too.
- **Doing it yourself:** `/compact`, in the conversation's thread or topic, or as a reply to one of its messages, compacts it now. Use it before a long run of experiments, or when old details keep confusing the agent.
  - **Claude Code** runs its own `/compact` in the same session and takes a focus: `/compact keep the lr sweep results and the baseline numbers`.
  - **Codex** compacts through its app server (`thread/compact/start` on the same thread). It doesn't take a focus.
- **What survives:** the summary is lossy, so the important facts are kept in files the agent re-reads: `OVERSEER.md` for project notes and `RESEARCH_LOG.md` for the experiments. `/new` starts a clean conversation instead.

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
- `slurm.quota_command`, `quota_check_hours`, `quota_warn_percent`: the disk quota check (above). The command's output must be a table like `myquota`'s: size, quota (%), files, quota (%). Set it to `""` to turn the check off.
- `slurm.compute`: the partition, cores, memory and hours of the shared worker, how long it may stay idle, how long `hpc-run` waits for a node, the GPU partition and gres for `--gpu`, extra `sbatch` flags (e.g. an account), the login-node CPU limit (`login_cpu_seconds`, 0 = off) and the warning threshold.
- `slurm.status_interval_minutes`: how often a running job's card is refreshed and the agent's alerts on it checked (10).
- `agent.autopilot`: the shortest interval for check-ins and repeating timers (`min_checkin_minutes`, 15), and how long a repeating timer lasts when the agent doesn't say (`repeat_hours`, 24). Settings from the old approval-based autopilot are ignored.
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
| `autopilot.py` | watches and timers that wake conversations, check-ins, pausing after a failed wake-up |
| `metrics.py` | what matters in a job (`ov metrics`): critical/important metrics, progress, the extraction command, alerts |
| `usage.py` | `/usage`: tokens and cost per day, by origin, backend and conversation |
| `projects.py` | projects from config and Telegram, folder matching for jobs |
| `ctl.py` | `ov`: the agent's requests to Overseer over a unix socket (jobs, projects, watches and timers, notifications); installs the `overseer` skill |
| `compute.py` | `hpc-run` and the shared compute worker, the agent's ssh wrapper with the login-node CPU limit |
| `richtext.py` | Markdown → Telegram HTML (fallback) and Rich Markdown, tag-safe message splitting, collapsible sections |
| `bot.py` | Telegram commands, buttons, message → conversation routing |
