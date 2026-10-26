"""The HPC domain's Telegram commands and buttons: /jobs, /job, /log, /fix, /results, /findings, /assign, /cluster,
and the buttons on job cards (📜 Log, 📊 Status, 📈 Plot, 🔧 Fix, ◀ ▶ through graphs, 📁 move to a project).
The platform's bot (core/bot.py) registers them for the HPC bot."""

from __future__ import annotations

import time

from telegram import Update
from telegram.ext import ContextTypes

from ..core.notify import Outgoing, code, esc, pre
from . import results as R
from .monitor import progress_log
from .plots import resolve_plot
from .slurm import STATE_EMOJI
from .ssh import SSHError
from .sshlink import master_alive

HELP = """<b>Overseer</b> — your SLURM jobs, from Telegram.

/jobs — active and recent jobs
/job &lt;id&gt; — status + log tail of a job
/log &lt;id&gt; [lines] — tail of a job's log
/fix &lt;id&gt; [instructions] — ask the agent to diagnose/fix a job
/ask &lt;text&gt; — talk to the agent (or just send a message)
/new — start a fresh agent conversation
/backend [claude_code|codex] — show or switch the agent
/model [name|default] — show or set the active agent's model
/effort [level|default] — show or set the active agent's effort
/stop — stop the agent's current run
/update — update Claude Code and Codex now (Overseer also does it once a day, while nothing runs)
/compact [focus] — compact the conversation now: older turns become a summary, which frees context (as a reply: that conversation). Claude Code takes a focus, e.g. /compact keep the lr sweep results
/auto &lt;goal&gt; — ask the agent to work on it on its own: it watches its jobs and wakes itself (as a reply: in that conversation)
/auto — pending wake-ups · /auto off — cancel them (here, or as a reply: in that conversation)
/results [project] [sort] [N] [csv] — experiments and what each showed · /findings — insights and dead ends
/feedback [all|review] — what agents reported about Overseer; review: an agent designs fixes · /skills — installed skills
/subagents [on|off] — planner mode: the agent plans and delegates to cheaper subagents, each in a thread
/usage [today|yesterday|week] — tokens and cost of the agent's runs
/cluster [partition] — free GPUs and CPUs per partition, when your queued jobs should start, your fairshare
/digest — what happened since the last digest (one comes every morning)
/search &lt;text&gt; — past conversations, results, findings and jobs
/dashboard — jobs, results, findings and a timeline in a Mini App (on your tailnet)
/health — connection status

<b>Projects</b> (each gets its own topic; commands in a topic apply to that project)
/project — list · /project new &lt;name&gt; &lt;dir on HPC&gt; [what it's about]
/project context|dir|notes|archive|restore …
/assign &lt;job&gt; &lt;project|main&gt; — move a job (or reply to its message: /assign &lt;project&gt;)

Reply to any job message to talk to the agent about that job.
Send pictures and files too (up to 20 MB): with a caption, or followed by a message about them.
The agent has full access to your HPC account; you'll see every command it runs and every job it submits."""

AUTO_PROMPT = """Work on this on your own, over as many runs as it takes: {goal}
Keep the research log up to date (in a project). Submit the runs, `ov watch` them and end your reply: you're woken \
when they finish (or set `ov wake` timers). Carry on until the goal is reached or nothing sensible is left, then \
tell me. Ask me when you need a decision."""

MENU = [("jobs", "active and recent jobs"), ("job", "status of a job"), ("log", "tail a job's log"),
        ("fix", "ask the agent to fix a job"), ("results", "experiments and what each showed"),
        ("findings", "insights and dead ends"), ("cluster", "free GPUs, when queued jobs start"),
        ("assign", "move a job to a project")]


class HpcBot:
    """Registered with the platform's bot by HpcDomain. The bot's helpers (reply, scope, the agent, ...) are
    reached through it."""

    help = HELP
    menu = MENU
    errors = (SSHError,)

    def __init__(self, bot, domain):
        self.bot = bot
        self.domain = domain

    def __getattr__(self, name):  # reply, _scope, _thread, _where, store, agent, notifier, cfg, _run_agent, ...
        return getattr(self.bot, name)

    @property
    def hpc(self):
        return self.domain.hpc

    @property
    def monitor(self):
        return self.domain.monitor

    def commands(self) -> list[tuple[str | list[str], object]]:
        return [("jobs", self.cmd_jobs), ("job", self.cmd_job), ("log", self.cmd_log), ("fix", self.cmd_fix),
                ("results", self.cmd_results), ("findings", self.cmd_findings), ("assign", self.cmd_assign),
                ("cluster", self.cmd_cluster)]

    async def on_button(self, update: Update, q, chat_id: int, action: str, arg: str) -> bool:
        """A button of the HPC's (on job cards); False when it isn't one."""
        if action == "fix":
            await self._start_fix(chat_id, arg)
        elif action == "log":
            await self._send_log(arg, 200)
        elif action == "status":  # on a job's card: refresh it in place
            rec = self.store.get_job(arg)
            if rec and not rec.ended_at:
                await self.monitor.refresh_card_by_id(arg)
            else:
                await self._send_status(arg)
        elif action == "pg":  # ◀ / ▶ on a job's card: another of its graphs, in place
            job_id, _, page = arg.rpartition(":")
            if page.isdigit() and hasattr(self.notifier, "page"):
                await self.notifier.page(job_id, int(page), self.monitor.cached_plots(job_id))
        elif action == "plot":
            await self._start_plot(chat_id, arg)
        elif action == "assign":
            job_id, _, name = arg.partition(":")
            await self._assign(job_id, name or None, update)
        else:
            return False
        return True

    async def watch_cancelled(self, w) -> bool:
        """✖ on a job's card (its watch): the card shows it's no longer watched."""
        if w.kind != "jobs":
            return False
        for jid in w.job_ids:
            await self.monitor.refresh_card_by_id(jid)
        return True

    async def health_lines(self) -> list[str]:
        m = self.monitor
        last = f"{int(time.time() - m.last_ok)}s ago" if m and m.last_ok else "never"
        err = f"\nlast error: {code(m.last_error[:300])}" if m and m.last_error else ""
        conn = "up" if await master_alive(self.cfg) else "down (reopens on the next poll)"
        return [f"HPC user {code(self.hpc.user or '?')} · SSH connection {conn}",
                f"last successful poll: {last}{err}", self._worker_line()]

    async def cmd_jobs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        scope = self._scope(update)
        rows = self.store.project_jobs(scope, 15) if scope else self.store.recent_jobs(20)
        if not rows:
            await self.reply(update, "No jobs seen yet.")
            return
        lines = []
        for r in rows:
            tag = " 🤖" if r.by_agent else ""
            proj = f" · 📁{esc(r.project)}" if r.project and not scope else ""
            lines.append(f"{STATE_EMOJI.get(r.state, '•')} {code(r.job_id)} {esc(r.name)} — {esc(r.state)}{tag}{proj}")
        title = f"Jobs of {esc(scope)}" if scope else "Recent jobs"
        await self.reply(update, f"<b>{title}</b> (🤖 = submitted by the agent)\n" + "\n".join(lines))

    async def cmd_job(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.args:
            await self.reply(update, "Usage: /job &lt;id&gt;")
            return
        await self._send_status(ctx.args[0])

    async def _send_status(self, job_id: str) -> None:
        rec = self.store.get_job(job_id)
        if not rec:
            await self.notifier.send(Outgoing(f"Unknown job {code(job_id)}."))
            return
        live = {j.job_id: j for j in await self.hpc.squeue()}
        msg = await self.monitor.status_message(rec, live.get(job_id), gpu=True)
        msg.silent = False
        if rec.report:
            msg.text += f"\n\n🧠 {esc(rec.report[:1500])}"
        await self.notifier.send(msg)

    async def cmd_log(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.args:
            await self.reply(update, "Usage: /log &lt;id&gt; [lines]")
            return
        lines = int(ctx.args[1]) if len(ctx.args) > 1 and ctx.args[1].isdigit() else 100
        await self._send_log(ctx.args[0], lines)

    async def _send_log(self, job_id: str, lines: int = 100) -> None:
        rec = self.store.get_job(job_id)
        if not rec:
            await self.notifier.send(Outgoing(f"Unknown job {code(job_id)}."))
            return
        lines = max(1, min(lines, 5000))
        log_path = await progress_log(self.hpc, rec)
        self.store.upsert_job(rec)
        out = await self.hpc.try_tail(rec.stdout, lines)
        files = [(f"job-{job_id}-stdout.log", out.encode())]
        if rec.stderr and rec.stderr != rec.stdout:
            files.append((f"job-{job_id}-stderr.log", (await self.hpc.try_tail(rec.stderr, lines)).encode()))
        if log_path:  # the program writes its real output here; SLURM's stdout only has the preamble
            main = await self.hpc.try_tail(log_path, lines)
            files.insert(0, (log_path.rsplit("/", 1)[-1], main.encode()))
            text = f"📜 Job {code(job_id)} {code(log_path)}" + pre(main, 3000) + f"\nSLURM stdout: {code(rec.stdout)}"
            files = files if lines > 60 else []
            await self.notifier.send(Outgoing(text, job_id=job_id, files=files))
            return
        await self.notifier.send(Outgoing(f"📜 Job {code(job_id)} {code(rec.stdout)}" + pre(out, 3000),
                                          job_id=job_id, files=files if lines > 60 else []))

    async def cmd_fix(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.args:
            await self.reply(update, "Usage: /fix &lt;job id&gt; [instructions]")
            return
        await self._start_fix(update.effective_chat.id, ctx.args[0], " ".join(ctx.args[1:]))

    async def _start_fix(self, chat_id: int, job_id: str, instr: str = "") -> None:
        session = self.agent.session_for_job(job_id)
        text = instr or (f"Job {job_id} failed. Find out why, fix it, and resubmit it if the fix is clear. "
                         "If it needs a decision from me, explain the options instead.")
        await self.notifier.send(Outgoing(f"🔧 {esc(self.agent.session_backend(session))} is working on job {code(job_id)}…",
                                          job_id=job_id, session_id=session.id, silent=True))
        self._spawn(self._run_agent(chat_id, session, text))

    async def _start_plot(self, chat_id: int, job_id: str) -> None:
        """📈: the job's graphs now, from the agent's plotting command; without one, the agent writes one."""
        rec = self.store.get_job(job_id)
        if not rec:
            await self.notifier.send(Outgoing(f"Unknown job {code(job_id)}."))
            return
        if resolve_plot(self.store, rec):  # no model involved: the graphs appear (or are updated) under the card
            n, err = await self.monitor.plots(rec)
            if not n:
                await self.notifier.send(Outgoing(f"📈 No graphs for {code(job_id)}: {esc(err)}", job_id=job_id,
                                                  jobs=True, reply_to=self.store.card_refs(job_id), silent=True))
            return
        log_path = (rec.info_dict.get("progress_log") or rec.stdout) if rec else ""
        text = (f"The researcher pressed 📈 Plot on job {job_id} ({rec.name}, {rec.state}), and there's no plotting "
                "command for it yet. Write a small plotting script in the project folder on the HPC that reads its "
                f"metrics - its log {log_path}, or metric files (metrics.jsonl, CSV, TensorBoard events) in its output "
                "folder - and saves clear PNGs (the main metrics against step or time, a few panels if useful, the job "
                "id in the title) into the folder it's given. Register it with `ov plots set` for this kind of job "
                "(--pattern or --project, so later runs get graphs too), then `ov plots test` it and look at the "
                "result. Describe the trend in two or three sentences.")
        session = self.agent.session_for_job(job_id)
        await self.notifier.send(Outgoing(f"📈 {esc(self.agent.session_backend(session))} is plotting job "
                                          f"{code(job_id)}…", job_id=job_id, session_id=session.id, silent=True))
        self._spawn(self._run_agent(chat_id, session, text))

    async def cmd_cluster(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        partition = (ctx.args or [None])[0]
        try:
            snap = await self.monitor.cluster.snapshot(fresh=True)
        except (SSHError, OSError) as e:
            await self.reply(update, f"Couldn't read the cluster: {code(str(e)[:300])}")
            return
        names = {r.job_id: r.name for r in self.store.active_jobs()}
        await self.reply(update, "🖥 " + self.monitor.cluster.text(snap, names, partition=partition, html=True))

    def _results_scope(self, update: Update, args: list[str]) -> tuple[str | None | bool, list[str]]:
        """A project named in the args ("all": every project, "main": the main chat), else this topic's."""
        rest, scope = [], self._scope(update)
        for a in args:
            if a == "all":
                scope = False
            elif a == "main":
                scope = None
            elif self.store.get_project(a):
                scope = a
            else:
                rest.append(a)
        return scope, rest

    async def cmd_results(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """/results [project|all] [sort key, e.g. score] [asc] [N] [csv]: the experiments and what each showed."""
        scope, rest = self._results_scope(update, list(ctx.args or []))
        want_csv, asc, limit, sort, verdicts = False, None, 15, "", []
        for a in rest:
            if a == "csv":
                want_csv = True
            elif a in ("asc", "desc"):
                asc = a == "asc"
            elif a.isdigit():
                limit = int(a)
            elif a in R.VERDICTS:
                verdicts.append(a)
            else:
                sort = a
        rs = R.select(self.store.results(scope), sort=sort, asc=asc, verdicts=verdicts,
                      limit=0 if want_csv else limit)
        where = "all projects" if scope is False else (f"project {scope}" if scope else "the main chat")
        if not rs:
            await self.reply(update, f"No results yet in {esc(where)}. Every finished job gets one; the agent adds "
                                     "what it showed (`ov result`).")
            return
        title = f"📊 Results in {where}" + (f", by {sort}" if sort else ", newest first")
        if want_csv:
            await self.notifier.send(Outgoing(f"{esc(title)}: {len(rs)} rows", project=scope or None,
                                              files=[("results.csv", R.render(rs, "csv").encode())]))
            return
        total = len(self.store.results(scope))
        more = f"\n{len(rs)} of {total}: /results [project] [sort] [N] [verdict] [csv]" if total > len(rs) else ""
        await self.notifier.send(Outgoing(
            f"<b>{esc(title)}</b>\n<pre>{esc(R.render(rs, 'table'))}</pre>{esc(more)}",
            markdown=f"**{title}**\n\n{R.render(rs, 'md')}{more}", project=scope or None))

    async def cmd_findings(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """/findings [project|all]: the insights and dead ends the agents recorded."""
        scope, _ = self._results_scope(update, list(ctx.args or []))
        fs = self.store.findings(scope)
        where = "all projects" if scope is False else (f"project {scope}" if scope else "the main chat")
        await self.reply(update, f"<b>Findings in {esc(where)}</b>\n" + "\n".join(esc(x) for x in R.finding_lines(fs))
                         if fs else f"No findings yet in {esc(where)}. Agents record them with `ov finding`.")

    async def cmd_assign(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        args = list(ctx.args or [])
        job_id = None
        msg = update.effective_message if update else None
        if msg is not None and msg.reply_to_message:
            job_id, _ = self.store.message_link(update.effective_chat.id, msg.reply_to_message.message_id)
        if args and args[0].isdigit():
            job_id, args = args[0], args[1:]
        thread = self._thread(update)
        conv = self.agent.get(self.store.thread_session(update.effective_chat.id, thread)) if thread else None
        if job_id and not args and conv:  # in a conversation's thread: the job becomes part of it
            self.agent.domain.record_job(conv, job_id)
            ap = self.agent.autopilot
            rec = self.store.get_job(job_id)
            wake = bool(ap and rec and not rec.ended_at)
            if wake:  # part of the conversation's work: its end wakes the agent
                ap.add_watch(conv, "jobs", [job_id])
            await self.reply(update, f"🔗 Job {code(job_id)} is now part of this conversation: its reports come here"
                                     + (" and the agent is woken when it ends (/auto off cancels that)." if wake else "."))
            return
        target = args[0].lower() if args else self._scope(update)
        if not job_id or not target:
            await self.reply(update, "Usage: /assign &lt;job id&gt; &lt;project|main&gt; - or reply to a job's message "
                                     "with /assign &lt;project|main&gt;. In a conversation's thread, /assign &lt;job id&gt; "
                                     "makes the job part of that conversation (its end wakes the agent).")
            return
        await self._assign(job_id, target, update)

    async def _assign(self, job_id: str, target: str | None, update: Update | None) -> None:
        project = None if target in (None, "main", "none") else target
        p = self.store.get_project(project) if project else None
        if project and (not p or p.archived):
            await self.reply(update, f"⛔ No active project {code(project)}. /project lists them.")
            return
        rec = self.store.get_job(job_id)
        if not rec:
            await self.reply(update, f"⛔ Unknown job {code(job_id)}.")
            return
        old = rec.project
        if old == project:
            await self.reply(update, f"Job {code(job_id)} is already in {self._where(project)}.")
            return
        self.agent.domain.move_job(job_id, project)
        await self.reply(update, f"📁 Job {code(job_id)} moved to {self._where(project)}.")
        # A card in its new place, so its updates and replies continue there.
        await self.notifier.send(Outgoing(
            f"📁 Job {code(job_id)} {esc(rec.name)} ({esc(rec.state)}) moved here from {self._where(old)}.",
            job_id=job_id, silent=True,
            buttons=[[("📜 Log", f"log:{job_id}"), ("📊 Status", f"status:{job_id}")]]))

    def _worker_line(self) -> str:
        w = getattr(self.monitor, "worker", None)
        if not w:
            return "compute worker: none (hpc-run starts one when needed)"
        where = f"on {esc(w.reason)}" if w.state == "RUNNING" else f"{esc(w.state)} ({esc(w.reason)})"
        return f"compute worker: job {code(w.job_id)} {where} · {esc(w.elapsed)} of {esc(w.time_limit)}"
