# Overseer

Overseer is a Telegram bot that keeps an eye on your SLURM jobs. It runs on your homelab and reaches the HPC login node over SSH (pubkey auth).

- Tells you when a job is **queued**, **starts** (which node), and sends **status updates** while it runs (elapsed time, latest metrics, log tail, and optionally GPU utilisation).
- When a job **ends**, it sends a **report**: final state, exit code, runtime against the limit, peak memory, and the last metrics it found. For a failed job it also says **why it failed**, using built-in heuristics (CUDA OOM, host OOM, time limit, NCCL, missing modules, NaNs, …) plus a Claude analysis.
- You can then say *"fix it"* or reply with instructions. A Claude agent reads the logs and code and changes files only in folders you allowed. It then **proposes** a resubmission. You get the full script and a diff of every changed file in Telegram, and nothing reaches `sbatch` until you tap **Approve**.

```
 homelab                                         HPC login node
┌───────────────────────────────────┐   ssh   ┌──────────────────────────────────────────┐
│ Telegram bot  ◄──► you            │ ──────► │ squeue / sacct / scontrol (read)         │
│ Monitor (polls every 60 s)        │         │ read_roots      → read only              │
│ Claude agent (no shell)           │         │ write_roots     → create/modify/submit   │
│ ./state: DB, audit log, backups   │         │ + folders you grant from Telegram        │
│ /tmp/overseer: ssh socket         │         │ everything else → untouched              │
└───────────────────────────────────┘         └──────────────────────────────────────────┘
```

## Where Overseer writes

**On the machine it runs on:** only inside `local.allowed_roots`. By default that is Overseer's own folder (the one containing `config.yaml`) and `/tmp`. Both working directories must sit inside those roots, which is checked when the config loads:

- `local.state_dir` (default `./state`): the SQLite database (jobs, proposals, folder grants), the audit log, and local backups of every HPC file the agent modifies. It lives in Overseer's folder, so grants and backups survive reboots.
- `local.tmp_dir` (default `/tmp/overseer`): the SSH control socket and other scratch files.

Both are created with mode `700`, and Overseer refuses to start if someone else owns them.

**On the HPC:** only inside the folders in `hpc.write_roots` (config file), plus folders you grant at runtime. When the agent needs another folder, it calls `request_access`:

> 🔐 **Access request**: the agent wants to write to (create/modify/delete files in) `/home/you/projects/llm-ft`
> Why: fix the OOM by lowering the batch size in train.py
> [✅ Grant write] [❌ Deny]

The agent then pauses. Your tap resumes it with the answer. You can also manage access yourself with `/access`, `/grant <path> [write|read]` and `/revoke <path>`.

## Safeguards

Every layer below is covered by tests.

1. **No shell for the agent.** Claude only gets these tools: read/list/search, `copy_path`, `write_file`, `edit_file`, `delete_path`, `request_access`, `propose_submission` and `propose_cancel`. It cannot run commands.
2. **Command allowlist on the wire** (`policy.check_remote_argv`). Every SSH command is an argv list, shell-quoted and checked before it leaves the homelab.
   - Read-only tools (`squeue`, `sacct`, `cat`, `grep`, …) are allowed.
   - Tools that change things (`mkdir`, `cp`, `rm`, `dd`, `sbatch`) only run with absolute targets inside a write root and a fixed set of options. `sbatch` also needs a `--chdir` inside a write root.
   - `scancel` takes one job id, and `scontrol` is limited to `show` and printing a batch script.
   - Anything else is refused, including `bash -c` and `python -c`. Until the write roots are resolved at startup, every modifying command is refused.
3. **Symlink-proof paths.** Every path is resolved with `realpath` on the cluster, and the resolved path is what gets checked and used.
4. **Deny lists that override access:**
   - Secrets (`~/.ssh`, `~/.netrc`, `*.pem`, `.env`, cloud credentials, HF token) are never readable.
   - Files that would make something *run* later (`~/.bashrc` and other login scripts, `.git/hooks`, autostart/systemd files) are never writable, even inside a granted folder.
   - `/` and your whole home directory can't be granted for writing.
5. **Backup before every modification.** The first time the agent changes a file, its original is saved in `state_dir` on the homelab (mode `600`). This drives the exact diff in proposals and a **↩️ Revert all changes** button: restore the originals, delete new files, remove copies. Files over 2 MB or non-text files can't be modified.
6. **Batch script policy** (`policy.check_script`), checked when a job is proposed and again at approval:
   - **Resource limits**: GPUs, nodes, `--time` (required), partitions, and array size.
   - **Blocked outright**: writes outside the write roots (`>`/`>>` redirects, `rm`/`mv`/`cp`/`tee`/… targets, `#SBATCH --output/--error/--chdir`), wiping an entire write root, `sudo`, `curl | sh`, and jobs that call `sbatch`/`scancel`.
   - **Shown as warnings**: references to other paths (for example a dataset path), `pip install`, and `cd` outside the write roots.
7. **Human approval** for every `sbatch` and `scancel`. The proposal shows the script, its sha256, resources, warnings and the diff. The script is re-read at approval, and the submission is refused if it changed. Each proposal can be approved only once. Only jobs Overseer submitted can be cancelled.
8. **Audit log.** Every remote command, with its exit code, goes to `state/audit.log`. After each request the agent's actions (read, copied, edited, blocked, requested access) are posted to Telegram.

**What these layers don't cover.** An approved job runs as *you* on a compute node, and Python code inside it can do anything your account can do. The static check can't fully analyse that code. The approval step and the diff are what protect you there, so read the diff.

## Setup

**On the HPC:** nothing to install. Your login shell should be bash or zsh.

**On the homelab:**

```bash
git clone <this repo> ~/Overseer && cd ~/Overseer
python3 -m venv .venv && .venv/bin/pip install -e .

# 1. SSH must work non-interactively. Put the host in ~/.ssh/config:
#      Host hpc
#        HostName login.cluster.edu
#        User <netid>
#        IdentityFile ~/.ssh/id_ed25519
#        # ProxyJump gateway.cluster.edu   (if needed)
ssh -o BatchMode=yes hpc true        # must succeed without any prompt

# 2. Telegram: create a bot with @BotFather, then message it once and get your chat id
#    (e.g. via @userinfobot).
# 3. Configure
cp config.example.yaml config.yaml   # set ssh.host, telegram.allowed_chat_ids, hpc.write_roots
export TELEGRAM_BOT_TOKEN=...  ANTHROPIC_API_KEY=...

.venv/bin/python -m overseer check -c config.yaml   # verifies ssh, slurm, folder access, credentials
.venv/bin/python -m overseer run   -c config.yaml
```

To run it permanently, use `deploy/overseer.service` (systemd user unit; instructions are inside the file).

If your cluster needs 2FA/Duo on every login, keep a multiplexed connection open: Overseer uses `ControlMaster`/`ControlPersist`, so approving one login can keep the socket alive. Some sites forbid automated logins entirely, so check your cluster's acceptable-use policy. Overseer polls once a minute over a single multiplexed connection, which is light on the login node.

## Using it

| | |
|---|---|
| *(automatic)* | 📥 queued · 🚀 started · 📊 hourly status · ✅/❌ final report with analysis |
| **🔧 Diagnose & fix** button | the agent investigates the failed job and proposes a fixed resubmission |
| *reply to any job message* | talk to the agent about that job ("lower the lr to 1e-4 and resubmit", "why is it slow?") |
| *plain message* | general conversation with the agent ("what should I try next for job 1234?") |
| `/jobs`, `/job <id>`, `/log <id> [n]` | status, log tails (long tails come as a file) |
| `/fix <id> [instructions]`, `/ask <text>`, `/new` | agent |
| `/pending`, `/cancel <id>`, `/health` | proposals, cancel an Overseer job, connection status |
| `/access`, `/grant <path> [write\|read]`, `/revoke <path>` | see and manage which HPC folders Overseer may use |
| **↩️ Revert all changes** button | undo every file change the agent made in that conversation |

A typical fix goes like this. The agent reads the failed job's log and batch script, then works in a writable folder. It either edits the files in place (if their folder is writable or you grant it), or copies the project to a new folder (e.g. `/scratch/you/overseer/fix-oom`) and edits the copy. It writes a batch script there and proposes it. The job runs with `--chdir` set to that folder, and its log is `slurm-<jobid>.out` unless the script sets `--output` (which must also be inside a writable folder).

A job submitted from a session stays linked to that conversation. When it finishes, replying to its report continues the same thread, so you can iterate: *fail → fix → resubmit → report → tweak*.

## Configuration notes

- `slurm.track: overseer` reports only on jobs Overseer submitted. The default `all` reports on every job of your user.
- `llm.enabled: false` turns Overseer into a pure monitor with heuristic failure analysis (no API key needed).
- `llm.model` defaults to `claude-opus-5-5` with adaptive thinking. Set `adaptive_thinking: false` and `*_effort: null` for models that don't support those parameters. `fallbacks: true` enables the API's server-side refusal fallback.
- Anything the agent reads (logs, code inside `read_roots`) is sent to the Anthropic API. Narrow `read_roots` if that matters.
- Agent conversations live in memory and are lost on restart. Job state, proposals, grants and message links are kept in SQLite in `local.state_dir`.
- The HPC login shell should be bash or zsh, since commands are quoted for a POSIX shell.

## Development

```bash
pip install -e '.[dev]'
pytest
```

`tests/test_integration.py` runs the real HPC layer, grants, backups/revert, monitor, agent tools and approval flow against a local shell, with fake `squeue`/`sacct`/`scontrol`/`sbatch`/`scancel` scripts (`tests/fake_slurm/`).

| module | role |
|---|---|
| `policy.py` | command allowlist, path rules, batch-script analysis (pure, heavily tested) |
| `config.py` | config schema; keeps local writes inside `local.allowed_roots` |
| `ssh.py` | multiplexed `ssh` transport, output caps, audit log |
| `hpc.py` | policy-enforcing file ops, write/read roots and grants, SLURM calls |
| `changes.py` | local backups of modified files → diff and revert |
| `slurm.py` | `squeue`/`sacct`/`scontrol` parsing |
| `monitor.py` | polling loop → queued/started/status/final-report events |
| `diagnose.py` | failure heuristics, traceback/metric extraction, LLM reports |
| `agent.py` | Claude tool-use loop with the policy-checked tool set |
| `proposals.py` | proposal → approval → `sbatch` / `scancel` / folder grant |
| `bot.py` | Telegram commands, buttons, message → job routing |
