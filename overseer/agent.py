"""The fixing agent: Claude with a small, sandboxed tool set.

The agent has no shell. Its only tools are: read/list/search files (within read roots),
copy code into its sandbox workspace, write/edit/delete files *inside the sandbox*, and
*propose* a job submission or cancellation, which the user approves in Telegram.
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .hpc import HPC
from .llm import LLM, response_text
from .notify import Notifier, Outgoing, code, esc
from .policy import PolicyError
from .proposals import Submitter
from .ssh import SSHError
from .store import Store

log = logging.getLogger(__name__)

TOOL_RESULT_LIMIT = 60_000

SYSTEM_PROMPT = """You are Overseer, an assistant that manages a researcher's ML training jobs on a university SLURM cluster.
The researcher talks to you through Telegram. They will ask you to diagnose failed jobs, fix them, and resubmit,
or ask for guidance on next steps.

## Hard rules (enforced by the system — violating them just produces errors)
- You can READ files under: {read_roots} (secrets such as ~/.ssh are blocked).
- You can only CREATE, MODIFY or DELETE files inside the sandbox: {sandbox}. The researcher's original code is never modified.
- You cannot run commands. The only way to run anything is `propose_submission`: the researcher sees the full script and
  the diff of your code changes in Telegram and must approve it before it is passed to sbatch.
- Limits for jobs you propose: max {max_gpus} GPUs, {max_nodes} node(s), max --time {max_time}{partitions}.
  Every script must set `#SBATCH --time`. Job arrays allowed up to {max_array} tasks.

## Workspace
Each conversation has a workspace directory. Relative paths in tools resolve against `work/` of the workspace.
- `stage` copies a file or directory from the researcher's project into BOTH `orig/` (pristine) and `work/` (yours to edit).
  The diff between them is what the researcher reviews, so stage before you edit, and edit only in `work/`.
- Write the batch script into `work/` too (e.g. work/run.sbatch).
- When the job runs: cwd = `work/`; these env vars are set:
  `$OVERSEER_RUN_DIR` (empty per-run output dir), `$OVERSEER_WORKSPACE` (= work/), `$TMPDIR` (inside the run dir).
  stdout/stderr are captured by Overseer automatically — do not set #SBATCH --output/--error/--chdir.
- All outputs the job WRITES (checkpoints, logs, results, wandb dirs, caches you control) must go under $OVERSEER_RUN_DIR
  or relative paths. Reading datasets/pretrained weights from their original locations is fine.
  Never redirect output to, or rm/mv/cp into, paths outside the sandbox — the policy check will reject the script.
- Staged code runs from `work/`, not from the original project directory: relative paths to data or checkpoints in the
  original project won't resolve. Pass absolute paths for inputs (reading them in place is fine).
- Reproduce the original job's environment setup (module loads, conda/venv activation, partition/account/gres) from
  its batch script unless that setup is the bug.

## Working style
- Diagnose from evidence: read the job's logs and batch script first, then the relevant source. Quote the decisive log line.
- Make the smallest change that fixes the problem; don't refactor unrelated code.
- If the researcher only asks a question or for guidance, answer it — don't propose a job unless asked or clearly implied.
- If a fix needs something you can't do (installing packages into their env, changing data outside the sandbox),
  explain exactly what the researcher should run themselves.
- Your final message is sent to Telegram: keep it short and concrete (what was wrong, what you changed, what's pending).
  Plain text, no Markdown tables or headings."""


def _schema(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


S = {"type": "string"}
I = {"type": "integer"}

TOOLS: list[dict[str, Any]] = [
    {"name": "list_jobs", "description": "List recently seen SLURM jobs (active and finished) with state and log paths.",
     "input_schema": _schema({}, [])},
    {"name": "job_info", "description": "Details of one job: state, workdir, log paths, batch script, previous analysis.",
     "input_schema": _schema({"job_id": S}, ["job_id"])},
    {"name": "job_log", "description": "Tail of a job's stdout or stderr log.",
     "input_schema": _schema({"job_id": S, "stream": {"type": "string", "enum": ["stdout", "stderr"]}, "lines": I}, ["job_id"])},
    {"name": "read_file", "description": "Read a text file (readable roots or sandbox). Use tail_lines for the end of large logs.",
     "input_schema": _schema({"path": S, "tail_lines": I}, ["path"])},
    {"name": "list_dir", "description": "List a directory (ls -la).", "input_schema": _schema({"path": S}, ["path"])},
    {"name": "search", "description": "Recursive grep -E for a regex under a directory. Optional include glob like '*.py'.",
     "input_schema": _schema({"pattern": S, "path": S, "include": S}, ["pattern", "path"])},
    {"name": "stage", "description": "Copy a file or directory from the researcher's project into the workspace (orig/ and work/). Returns the path in work/. Code only — no datasets/checkpoints (size-limited).",
     "input_schema": _schema({"src": S, "subdir": {"type": "string", "description": "optional subdirectory of work/ to copy into"}}, ["src"])},
    {"name": "write_file", "description": "Create or overwrite a file in the sandbox (relative paths are under work/).",
     "input_schema": _schema({"path": S, "content": S}, ["path", "content"])},
    {"name": "edit_file", "description": "Replace an exact, unique snippet in a sandbox file.",
     "input_schema": _schema({"path": S, "old_string": S, "new_string": S}, ["path", "old_string", "new_string"])},
    {"name": "delete_path", "description": "Delete a file or directory inside the sandbox.",
     "input_schema": _schema({"path": S}, ["path"])},
    {"name": "show_diff", "description": "Unified diff of work/ against the staged originals.", "input_schema": _schema({}, [])},
    {"name": "propose_submission", "description": "Ask the researcher to approve submitting a batch script from the sandbox with sbatch. They will see the script and the diff.",
     "input_schema": _schema({"script_path": S, "summary": {"type": "string", "description": "1-3 sentences: what changed and why, what to expect"}}, ["script_path", "summary"])},
    {"name": "propose_cancel", "description": "Ask the researcher to approve cancelling a job that Overseer submitted.",
     "input_schema": _schema({"job_id": S, "reason": S}, ["job_id", "reason"])},
]


@dataclass
class Session:
    id: str
    root: str
    job_id: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # system events to tell the agent next turn
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    initialized: bool = False

    @property
    def work(self) -> str:
        return posixpath.join(self.root, "work")

    @property
    def orig(self) -> str:
        return posixpath.join(self.root, "orig")


class Agent:
    def __init__(self, cfg: Config, hpc: HPC, store: Store, llm: LLM, notifier: Notifier, submitter: Submitter):
        self.cfg = cfg
        self.hpc = hpc
        self.store = store
        self.llm = llm
        self.notifier = notifier
        self.submitter = submitter
        self.sessions: dict[str, Session] = {}
        self.by_job: dict[str, str] = {}
        self.general_id: str | None = None

    # ------------------------------------------------------------------ sessions
    def system_prompt(self) -> str:
        lim = self.cfg.limits
        return SYSTEM_PROMPT.format(
            read_roots=", ".join(self.cfg.sandbox.read_roots), sandbox=self.hpc.root,
            max_gpus=lim.max_gpus_per_job, max_nodes=lim.max_nodes_per_job, max_time=lim.max_time,
            partitions=f", partitions {', '.join(lim.allowed_partitions)}" if lim.allowed_partitions else "",
            max_array=lim.max_array_tasks,
        )

    def _new_session(self, job_id: str | None) -> Session:
        sid = (f"j{job_id}-" if job_id else "chat-") + uuid.uuid4().hex[:6]
        s = Session(id=sid, root=posixpath.join(self.hpc.root, "sessions", sid), job_id=job_id)
        self.sessions[sid] = s
        return s

    def session_for_job(self, job_id: str) -> Session:
        rec = self.store.get_job(job_id)
        # A follow-up job submitted by a session continues that session's conversation.
        sid = self.by_job.get(job_id) or (rec.session_id if rec else None)
        if sid and sid in self.sessions:
            self.by_job[job_id] = sid
            return self.sessions[sid]
        s = self._new_session(job_id)
        self.by_job[job_id] = s.id
        return s

    def general_session(self, reset: bool = False) -> Session:
        if reset or not self.general_id or self.general_id not in self.sessions:
            self.general_id = self._new_session(None).id
        return self.sessions[self.general_id]

    def get(self, sid: str | None) -> Session | None:
        return self.sessions.get(sid) if sid else None

    def note(self, sid: str, text: str) -> None:
        s = self.sessions.get(sid)
        if s:
            s.notes.append(text)

    async def _job_context(self, job_id: str) -> str:
        rec = self.store.get_job(job_id)
        if not rec:
            return f"(Job {job_id} is not known to Overseer.)"
        tail = await self.hpc.try_tail(rec.stdout, 80)
        err = await self.hpc.try_tail(rec.stderr, 60) if rec.stderr and rec.stderr != rec.stdout else ""
        return "\n".join(x for x in [
            f"<job id='{rec.job_id}' name='{rec.name}' state='{rec.state}' workdir='{rec.workdir}' "
            f"stdout='{rec.stdout}' stderr='{rec.stderr}' submitted_by_overseer='{bool(rec.overseer)}'>",
            f"<previous_analysis>\n{rec.report}\n</previous_analysis>" if rec.report else "",
            f"<batch_script>\n{rec.batch_script[:8000]}\n</batch_script>" if rec.batch_script else "<batch_script>unavailable</batch_script>",
            f"<stdout_tail>\n{tail[-8000:]}\n</stdout_tail>",
            f"<stderr_tail>\n{err[-6000:]}\n</stderr_tail>" if err else "",
            "</job>",
        ] if x)

    # ------------------------------------------------------------------ main loop
    async def handle(self, s: Session, user_text: str) -> None:
        if not self.llm.enabled:
            await self.notifier.send(Outgoing("LLM is disabled in the config; the agent is unavailable."))
            return
        async with s.lock:
            actions: list[str] = []
            try:
                await self._turn(s, user_text, actions)
            except (SSHError, PolicyError) as e:
                await self.notifier.send(Outgoing(f"⚠️ Agent stopped: {esc(e)}", session_id=s.id, job_id=s.job_id))
            except Exception as e:
                log.exception("agent turn failed")
                await self.notifier.send(Outgoing(f"⚠️ Agent error: {esc(type(e).__name__)}: {esc(str(e)[:500])}",
                                                  session_id=s.id, job_id=s.job_id))
            finally:
                if actions:
                    await self.notifier.send(Outgoing(
                        "🛠 <b>Agent actions</b>\n" + "\n".join(f"• {a}" for a in actions[-40:]),
                        session_id=s.id, job_id=s.job_id, silent=True))

    def _append_user(self, s: Session, blocks: list[dict[str, Any]]) -> None:
        if s.messages and s.messages[-1]["role"] == "user":
            prev = s.messages[-1]["content"]
            if isinstance(prev, str):
                prev = [{"type": "text", "text": prev}]
            s.messages[-1]["content"] = prev + blocks
        else:
            s.messages.append({"role": "user", "content": blocks})

    async def _turn(self, s: Session, user_text: str, actions: list[str]) -> None:
        text_parts = []
        if not s.initialized:
            await self.hpc.mkdir(s.work)
            await self.hpc.mkdir(s.orig)
            await self.hpc.mkdir(posixpath.join(s.root, "runs"))
            s.initialized = True
            text_parts.append(f"<workspace root='{s.root}' work='{s.work}' orig='{s.orig}'/>")
            if s.job_id:
                text_parts.append(await self._job_context(s.job_id))
        if s.notes:
            text_parts.append("<events>\n" + "\n".join(s.notes) + "\n</events>")
            s.notes.clear()
        text_parts.append(f"<request>\n{user_text}\n</request>")
        self._append_user(s, [{"type": "text", "text": "\n\n".join(text_parts)}])

        for _ in range(self.cfg.llm.max_agent_steps):
            resp = await self.llm.create(system=self.system_prompt(), messages=s.messages,
                                         tools=TOOLS, effort=self.cfg.llm.agent_effort)
            s.messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "refusal":
                await self.notifier.send(Outgoing("The model declined this request.", session_id=s.id))
                return
            tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                final = response_text(resp)
                if resp.stop_reason == "max_tokens":
                    final += "\n(…output truncated)"
                await self.notifier.send(Outgoing(f"🤖 {esc(final or '(no answer)')}", session_id=s.id, job_id=s.job_id))
                return
            results = []
            for b in tool_uses:
                content, is_error = await self._run_tool(s, b.name, b.input, actions)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": is_error})
            s.messages.append({"role": "user", "content": results})
        await self.notifier.send(Outgoing(
            f"⏸ Agent stopped after {self.cfg.llm.max_agent_steps} steps. Reply to continue.", session_id=s.id, job_id=s.job_id))

    async def _run_tool(self, s: Session, name: str, args: Any, actions: list[str]) -> tuple[str, bool]:
        if not isinstance(args, dict):
            return "invalid tool input", True
        try:
            out = await self._dispatch(s, name, args, actions)
            if len(out) > TOOL_RESULT_LIMIT:
                out = out[:TOOL_RESULT_LIMIT] + "\n<truncated>"
            return out or "(empty)", False
        except PolicyError as e:
            actions.append(f"⛔ blocked {esc(name)}: {esc(str(e)[:200])}")
            return f"POLICY: {e}", True
        except (SSHError, KeyError, ValueError, TypeError) as e:
            return f"ERROR: {type(e).__name__}: {e}", True

    @staticmethod
    def _str(args: dict[str, Any], key: str, required: bool = True) -> str:
        v = args.get(key)
        if v is None and not required:
            return ""
        if not isinstance(v, str) or (required and not v):
            raise ValueError(f"'{key}' must be a non-empty string")
        return v

    async def _dispatch(self, s: Session, name: str, a: dict[str, Any], actions: list[str]) -> str:
        h = self.hpc
        if name == "list_jobs":
            rows = self.store.recent_jobs(20)
            return "\n".join(
                f"{r.job_id}\t{r.name}\t{r.state}\toverseer={bool(r.overseer)}\tstdout={r.stdout}" for r in rows
            ) or "no jobs seen yet"
        if name == "job_info":
            return await self._job_context(self._str(a, "job_id"))
        if name == "job_log":
            rec = self.store.get_job(self._str(a, "job_id"))
            if not rec:
                return "unknown job"
            path = rec.stderr if a.get("stream") == "stderr" else rec.stdout
            lines = min(int(a.get("lines") or 200), 2000)
            return (await h.read_file(path, tail_lines=lines)).text
        if name == "read_file":
            tl = a.get("tail_lines")
            ft = await h.read_file(self._str(a, "path"), base=s.work, tail_lines=int(tl) if tl else None)
            actions.append(f"📖 read {code(ft.path)}")
            return ft.text + ("\n<truncated: file is larger; use tail_lines or search>" if ft.truncated else "")
        if name == "list_dir":
            return await h.list_dir(self._str(a, "path"), base=s.work)
        if name == "search":
            return await h.search(self._str(a, "pattern"), self._str(a, "path"), base=s.work,
                                  include=self._str(a, "include", required=False) or None)
        if name == "stage":
            sub = self._str(a, "subdir", required=False).strip("/")
            if ".." in sub.split("/"):
                raise PolicyError("subdir may not contain '..'")
            src = self._str(a, "src")
            dest_work = posixpath.join(s.work, sub) if sub else s.work
            dest_orig = posixpath.join(s.orig, sub) if sub else s.orig
            await h.stage(src, dest_orig, base=s.work)
            path = await h.stage(src, dest_work, base=s.work)
            actions.append(f"📦 staged {code(src)} → {code(path)}")
            return f"staged to {path} (pristine copy under {dest_orig})"
        if name == "write_file":
            content = a.get("content")
            if not isinstance(content, str):
                raise ValueError("'content' must be a string")
            path = await h.write_file(self._str(a, "path"), content, base=s.work)
            actions.append(f"✏️ wrote {code(path)} ({len(content.splitlines())} lines)")
            return f"wrote {path}"
        if name == "edit_file":
            path = await h.check_write(self._str(a, "path"), base=s.work)
            old, new = self._str(a, "old_string"), a.get("new_string")
            if not isinstance(new, str):
                raise ValueError("'new_string' must be a string")
            text = (await h.read_file(path, max_bytes=2_000_000)).text
            n = text.count(old)
            if n != 1:
                raise ValueError(f"old_string found {n} times in {path}; it must match exactly once")
            await h.write_file(path, text.replace(old, new, 1))
            actions.append(f"✏️ edited {code(path)} (−{len(old.splitlines())}/+{len(new.splitlines())} lines)")
            return f"edited {path}"
        if name == "delete_path":
            path = await h.delete(self._str(a, "path"), base=s.work)
            actions.append(f"🗑 deleted {code(path)}")
            return f"deleted {path}"
        if name == "show_diff":
            return await h.diff(s.orig, s.work) or "no differences"
        if name == "propose_submission":
            result = await self.submitter.propose_submit(s, self._str(a, "script_path"), self._str(a, "summary"))
            actions.append(f"📝 proposed submission of {code(a['script_path'])}")
            return result
        if name == "propose_cancel":
            result = await self.submitter.propose_cancel(s, self._str(a, "job_id"), self._str(a, "reason"))
            actions.append(f"🛑 proposed cancelling job {code(a['job_id'])}")
            return result
        raise ValueError(f"unknown tool {name}")
