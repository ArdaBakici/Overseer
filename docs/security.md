# Security: what the agent can and can't do

Overseer gives an AI agent a real shell on your HPC account, so it's worth knowing exactly where the lines are. In
short: **on the HPC it can do anything your account can, and you see all of it; on the machine running Overseer it is
locked in a sandbox, and it never sees your SSH key, your logins or your secrets.**

## On the HPC: everything your account can

That includes editing files, `sbatch`, `scancel`, `pip install` and so on. Nothing is restricted there, by design. In
return you see all of it:

- **Commands:** after each request, Telegram gets the list of commands the agent ran. Set `report_commands: live` to see them while it works.
- **Submissions:** any job it submits is announced right away (`🚀 Agent submitted job 1234` plus the command), then tracked by the monitor.
  - **Registration:** the agent registers every job it submits with [`ov job add <id>`](guide.md#the-agent-talks-to-overseer-ov), whichever way it submitted it. Overseer checks the id with SLURM first.
  - **Fallback detection:** Overseer also spots submissions itself, in case the agent forgets to register one. It reads `sbatch` output as the agent works. At the end of each request it asks SLURM which of your jobs were submitted during it, and links those whose id appears in the agent's output or reply. This catches jobs submitted through a script or CLI.

## On your homelab: confined

Claude Code and Codex each run their shell inside their own OS sandbox (bubblewrap on Linux), configured by Overseer:

| | agent's shell |
|---|---|
| write | only its own scratch dir, `/tmp/overseer/agents/<conversation>` |
| read | the system, except Overseer's state dir (which includes the SSH key), `~/.ssh`, `~/.gnupg`, `~/.docker`, `~/.aws`, gcloud credentials, docker/podman sockets, `/run/user/<uid>` and your ssh-agent socket |
| network | none, except `agent.allowed_domains`. The HPC is reached through the unix socket of Overseer's SSH connection, and Overseer itself through the `ov` socket. **With Codex**, commands can still connect to services listening on this machine's localhost (127.0.0.1), because Codex's sandbox doesn't give them a network of their own; Claude Code's does. If your homelab runs services on localhost that the agent mustn't reach, use Claude Code. Proxy settings passed to Codex for its own API traffic aren't handed to the agent's commands |
| environment | scrubbed. Only `PATH`, locale and the API-key variables in `pass_env` are passed; the Telegram token is never passed, and the `.env` file can't be read |
| escape hatches | Claude Code: `failIfUnavailable` and `allowUnsandboxedCommands: false`, so it never runs a command unsandboxed. Codex: a named permission profile with approvals off |

## Your SSH key never leaves the Overseer folder

- **Overseer's own key setup:** `overseer ssh setup` creates (or imports) a key, an ssh config and known_hosts in `./state/ssh`. Your `~/.ssh` isn't used at all.
- **One shared connection:** Overseer opens a single SSH master connection. The agent's `ssh`, `scp`, `sftp` and `rsync` are wrappers that only talk to that connection's socket, so the agent never needs the key and its sandbox can't read it.
- **Clean shutdown:** if Overseer stops, the agent loses HPC access.

## Your Telegram chat is a shell on your HPC account

Since the agent is effectively a shell on your HPC account, so is your Telegram chat. Keep
`telegram.allowed_chat_ids` to your own chat, turn on two-step verification for your Telegram account, and don't add
the bot to groups. Messages from any other chat are ignored.

## Tested end to end

All of this was tested end to end, using the real Claude Code (2.1) and Codex (0.160) binaries against a local sshd standing in for the HPC (`tests/test_e2e.py`):

- `ssh hpc` and `sbatch` work.
- Reading the key and `~/.ssh` fails.
- Writing outside the scratch dir fails.
- `curl` to the internet fails, while the models' own web search works: Claude Code's WebSearch and WebFetch, and Codex's live `web_search` (`agent.codex.web_search: live | cached | disabled`).
- `hpc-run`, `scp` into the outbox and `ov` work from inside both sandboxes.

## The personal assistant and the HPC bot are kept apart

Each bot runs as its own process, with its own Telegram bot, config, state folder and logins. Every bot's agent is
denied the other bots' state and tmp folders (the HPC bot's SSH connection socket is in its tmp folder), all of
`./state`, and every `.env*` file. See [assistant.md](assistant.md#kept-apart-from-the-hpc-bot).

## Logins and accounts

The agent CLIs' logins live in their home folders (`./state/claude`, `./state/codex`, or each account's folder with
[several accounts](guide.md#several-accounts-per-cli)). Agents can never read them, wherever you put them.

## Reporting a problem

If you find a way for an agent to get past these limits, please open an issue.
