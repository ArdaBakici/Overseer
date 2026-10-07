# Overseer

Overseer is a Telegram bot that keeps an eye on your SLURM jobs. It runs on your homelab and reaches the HPC login node over SSH (pubkey auth).

- Tells you when a job is **queued**, **starts** (which node), and sends **status updates** while it runs (elapsed time, latest metrics, log tail, and optionally GPU utilisation).
- When a job **ends**, it sends a **report**: final state, exit code, runtime against the limit, peak memory, and the last metrics it found. For a failed job it also says **why it failed**, using built-in heuristics (CUDA OOM, host OOM, time limit, NCCL, missing modules, NaNs, …) plus a Claude analysis.
- You can then say *"fix it"* or reply with instructions. A Claude agent reads the logs and code, copies the code into a **sandbox**, edits it there, and writes a batch script. It then **proposes** the submission. You get the full script and a diff of every code change in Telegram, and nothing reaches `sbatch` until you tap **Approve**.

```
 homelab                                    HPC login node
┌───────────────────────────────┐   ssh   ┌──────────────────────────────────────┐
│ Telegram bot  ◄──► you        │ ──────► │ squeue / sacct / scontrol (read)     │
│ Monitor (polls every 60 s)    │         │ your project dirs (read only)        │
│ Claude agent (no shell)       │         │ /tmp/overseer  ← only writable place │
│ SQLite state + audit.log      │         │   sessions/<id>/{orig,work,runs,logs}│
└───────────────────────────────┘         └──────────────────────────────────────┘
```

## Safeguards

You asked that Overseer never runs or modifies anything outside `/tmp`. That is enforced in layers, and every layer is covered by tests:

1. **No shell for the agent.** Claude only gets these tools: read/list/search files, `stage` (copy code into the sandbox), write/edit/delete *inside the sandbox*, and `propose_submission` / `propose_cancel`. It cannot run commands.
2. **Command allowlist on the wire** (`policy.check_remote_argv`). Every SSH command is an argv list, shell-quoted and checked before it leaves the homelab. Read-only tools (`squeue`, `sacct`, `cat`, `grep`, …) are allowed. Tools that change things (`mkdir`, `cp`, `rm`, `dd`, `sbatch`, `scancel`) only run with absolute target paths inside the sandbox and a fixed set of options. `scancel` takes one job id, and `scontrol` is limited to `show` and `write batch_script <id> -` (written to stdout). Anything else is refused, including `bash -c` and `python -c`.
3. **Symlink-proof paths.** Before a write, the path is resolved with `realpath` on the cluster, so a symlink in the sandbox can't point a write elsewhere. The sandbox root is created with mode `700`. Overseer refuses to start if the root is owned by someone else (an attacker could pre-create a directory in `/tmp`), is a symlink, or is readable by others.
4. **Reads are limited too.** Reads stay inside `sandbox.read_roots`, plus the log and work directories of the jobs being monitored. A denylist blocks `~/.ssh`, `~/.netrc`, `*.pem`, `.env`, cloud credentials and HF tokens. Staging refuses directories that contain such files, as well as your whole home directory.
5. **Batch script policy** (`policy.check_script`), checked when a job is proposed and again at approval:
   - **Resource limits**: GPUs, nodes, `--time` (required), partitions, and array size.
   - **Blocked outright**: writes outside the sandbox (`>`/`>>` redirects, `rm`/`mv`/`cp`/`tee`/… with targets in `~`, `$HOME` or absolute paths), `sudo`, `curl | sh`, and jobs that call `sbatch`/`scancel`.
   - **Shown as warnings**: anything that only *references* paths outside the sandbox (for example a dataset path), `pip install`, and the like.
   - **Forced output paths**: `--chdir`, `--output` and `--error` are always set on the command line to locations inside the sandbox, which overrides any `#SBATCH` lines.
6. **Human approval.** The proposal message contains the script, its sha256, the resource line, warnings, and the diff as a file. On approval, Overseer re-reads the script and refuses to submit if it changed after you reviewed it. Each proposal can be approved only once.
7. **Only Overseer's own jobs** can be cancelled, and only after you confirm.
8. **Audit log.** Every remote command, with its exit code, goes to `state/audit.log`. The agent's file actions (staged, edited, wrote, blocked) are posted to Telegram after each request.

**What these layers don't cover.** An approved job runs as *you* on a compute node. Python code inside the job can do anything your account can do, and the static check can't fully analyse it. The approval step and the diff are what protect you there, so read the diff. To keep the agent's code changes contained, the jobs it writes get `$OVERSEER_RUN_DIR` (and `TMPDIR`) set inside the sandbox, and the agent is told to put every output there.

### Where should the sandbox live?

On most clusters `/tmp` is **local to each node**. Files written to the login node's `/tmp` are not visible on a compute node. Jobs that Overseer submits run from the sandbox (`--chdir`, logs and outputs live there), so the compute nodes must be able to see it.

- If your cluster's `/tmp` is node-local (most are, including the big university clusters), set `sandbox.root` to a dedicated directory on **shared scratch**, for example `/scratch/<netid>/overseer`. The rules are the same; only the root changes.
- `/tmp/overseer` works when the login and compute nodes share `/tmp`, or if you only use Overseer for monitoring and diagnosis.

Also remember that scratch filesystems are often purged on a schedule. Copy results you want to keep.

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
cp config.example.yaml config.yaml   # set ssh.host, telegram.allowed_chat_ids, sandbox.root
export TELEGRAM_BOT_TOKEN=...  ANTHROPIC_API_KEY=...

.venv/bin/python -m overseer check -c config.yaml   # verifies ssh, slurm, sandbox, credentials
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

The agent works in `sandbox/sessions/<session>/`:

- `orig/` holds pristine copies of whatever it staged. You review the diff between this and `work/`.
- `work/` holds the edited code and the batch script. It is the job's working directory.
- `runs/<proposal>/` is `$OVERSEER_RUN_DIR`, where outputs and checkpoints go.
- `logs/slurm-<jobid>.{out,err}` hold the job's stdout and stderr.

A job submitted from a session stays linked to that conversation. When it finishes, replying to its report continues the same thread, so you can iterate: *fail → fix → resubmit → report → tweak*.

## Configuration notes

- `slurm.track: overseer` reports only on jobs Overseer submitted. The default `all` reports on every job of your user.
- `llm.enabled: false` turns Overseer into a pure monitor with heuristic failure analysis (no API key needed).
- `llm.model` defaults to `claude-opus-5-5` with adaptive thinking. Set `adaptive_thinking: false` and `*_effort: null` for models that don't support those parameters. `fallbacks: true` enables the API's server-side refusal fallback.
- Anything the agent reads (logs, code inside `read_roots`) is sent to the Anthropic API. Narrow `read_roots` if that matters.
- Agent conversations live in memory and are lost on restart. Job state, proposals and message links are kept in SQLite.

## Development

```bash
pip install -e '.[dev]'
pytest
```

`tests/test_integration.py` runs the real HPC layer, monitor, agent tools and approval flow against a local shell, with fake `squeue`/`sacct`/`scontrol`/`sbatch`/`scancel` scripts (`tests/fake_slurm/`).

| module | role |
|---|---|
| `policy.py` | command allowlist, path rules, batch-script analysis (pure, heavily tested) |
| `ssh.py` | multiplexed `ssh` transport, output caps, audit log |
| `hpc.py` | policy-enforcing file ops and SLURM calls |
| `slurm.py` | `squeue`/`sacct`/`scontrol` parsing |
| `monitor.py` | polling loop → queued/started/status/final-report events |
| `diagnose.py` | failure heuristics, traceback/metric extraction, LLM reports |
| `agent.py` | Claude tool-use loop with the sandboxed tool set |
| `proposals.py` | proposal → approval → `sbatch`/`scancel` |
| `bot.py` | Telegram commands, buttons, message → job routing |
