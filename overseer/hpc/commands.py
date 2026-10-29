"""The HPC domain's `ov` commands: jobs, what matters in them (metrics, alerts), their graphs, watching them, the
results and findings, and the cluster. The platform's control server (core/ctl.py) runs them for the agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import time

from ..core.autopilot import human_interval
from ..core.ctl import CtlError
from ..core.notify import code, esc
from ..core.store import MetricSpec
from . import results as R
from .compute import is_helper
from .metrics import SpecError, check_progress, parse_alert
from .results import ResultError
from .slurm import TERMINAL_STATES, array_parent

AREAS = ("job", "metrics", "plots", "watch", "result", "results", "finding", "findings", "cluster")

HELP = """ov - talk to Overseer (the system that runs you and reports to the researcher on Telegram)

  ov job add <id> [<id>...]        register jobs you submitted (any way: sbatch, a script, a CLI). Overseer
                                   checks them with SLURM and reports on them to the researcher.
  ov job list [--all]              jobs of this conversation (--all: every job Overseer tracks)
  ov job show <id>
  ov job move <id> <project|main>  reassign a job to a project
  ov job delete <id> [<id>...]     stop tracking jobs (they keep running; use scancel to cancel them)
  ov project list
  ov project new <name> --dir <folder on the HPC> [--about "what it is"] [--move-here]
  ov project set <name> [--dir ...] [--about ...]
  ov project archive <name>
  ov conversation move <project|main>
  ov notify <text>                 send the researcher a message now
  ov react <emoji>                 react to the researcher's last message (👍 👀 🎉 ...) instead of a message
  ov history search <text> [--project p|main] [--kind conversation|result|finding|job] [--since 30d] [--limit N]
                                   search past work: every conversation's requests, replies and commands, the
                                   results, findings and jobs. "quoted phrase", OR, the last word as a prefix
  ov history show <conversation> [--last 5] [--grep text]   read a past conversation's turns
  ov history day [today|yesterday|YYYY-MM-DD]   every conversation of a day, condensed
  ov cluster [-p <partition>] [--fresh]   what's free now per partition (GPUs, CPUs, RAM, time limits), SLURM's
                                   estimated start of your queued jobs, and your fairshare. Check it before
                                   choosing a partition and a job size.

Graphs under a job's card (no model involved: your plotting script, run by Overseer on the HPC):
  ov plots set <job | --pattern '<name glob>' | --project> --command "python {project}/plots/x.py --log {stdout} --out {out}"
                                   writes PNGs (up to 4) into {out}; also {job_id} {name} {workdir}. Run when the job
                                   ends, hourly while it runs, and on the researcher's 📈 button.
  ov plots test <job>              run it now (graphs go under the card; copies into ./inbox for you to look at)
  ov plots show <job> | clear ...

Results and findings (each experiment's outcome; the project's lasting lessons):
  ov result <job> [--verdict V] [--hypothesis ...] [--conclusion ...] [--tags a,b] [--metric name=value] [--dead-end]
                                   V: baseline | promising | no-gain | worse | broken | dead-end | inconclusive
  ov result add --name "..." [same options]   a result that isn't a job (an evaluation by eye, a comparison)
  ov result show <id>
  ov results [--sort <metric>|name|verdict|elapsed] [--asc] [--verdict promising,baseline] [--name 'glob']
             [--tag t] [--since 7d] [--search text] [--todo] [--limit N] [--format table|md|csv|json]
             [--columns id,name,score,...] [--project p | --all-projects]
  ov finding insight|dead-end "..." [--job ID]   ov findings   ov finding rm F3

Feedback about Overseer itself (what limited you; shown in the researcher's 🛠 feedback topic):
  ov feedback "<what limited you>" [--kind limitation|bug|idea] [--context "what you were doing"]
  ov feedback list [--all]         ov feedback +1 FB7    (same problem: add a +1 instead of a new item)
Skills (know-how for agents, installed once the researcher approves):
  ov skill propose skill.md [--all]   (--all: every agent, not just this project)   ov skill list | show <name>

Subagents (planner mode, off unless the researcher turned it on or approved your plan):
  ov propose plan.json             propose planner mode: {"goal", "why", "subagents": [{"name", "task", "model"?,
                                   "backend"?}]}. The researcher approves or declines it (then end your reply).
  ov spawn <name> <task...> [--model M] [--backend claude_code|codex]   (planner mode) start a subagent
  ov agents                        your subagents and their state
  ov agent show|stop <name>        its task and report / stop it
  ov agent send <name> <text...>   a message to a subagent
  ov planner off                   end planner mode (the goal is reached)
  ov status                        this conversation: project, its jobs, what will wake you

What matters in a job (shown on its live card in Telegram; alerts ping the researcher or wake you):
  ov metrics set <job id> | --pattern '<job name glob>' | --project [name]
      [--critical "metric=label,..."] [--important "metric=label,..."] [--progress step/5000000]
      [--command "<quick command printing name=value lines or a JSON object>"]
      [--alert "score < 1 after 2h"] [--wake-on "loss is nan"] [--wake-on "step stalled 30m"] [--clear-alerts]
                                   {job_id} {workdir} {stdout} {name} in --command are filled in
  ov metrics show <job id>         what its card shows now (and whether --command works)
  ov metrics list | clear <job id> | --pattern ... | --project ...

Being woken (you only run when someone writes to you - or when something you set up wakes you; no
permission needed, but each wake-up costs a run, and the researcher sees each one and can cancel it):
  ov watch <id> [<id>...] [--every 30m] [--note "what to do then"]
                                   wake me when these jobs have ended (all of them, or as soon as one fails);
                                   --every: check-ins while they run (at least 15 minutes apart)
  ov wake in 90m ["what to do then"]           wake me once, later
  ov wake every 2h [--for 12h] ["..."]         wake me repeatedly (at least 15 minutes apart; 24h unless --for)
  ov watches                       what will wake me (ids like w12)
  ov unwatch <w12 | job id | all>
"""


class HpcCommands:
    """Registered with the control server by HpcDomain. Shared helpers (announcing, durations, the wake-up engine)
    are the server's."""

    help = HELP
    errors = (SpecError, ResultError)

    def __init__(self, ctl, domain):
        self.ctl = ctl
        self.domain = domain
        self._cluster = None  # cluster.Cluster, made on first use

    def __getattr__(self, name):  # store, agent, notifier, cfg, _announce, _ap, _duration, _bg, ...
        return getattr(self.ctl, name)

    @property
    def hpc(self):
        return self.domain.hpc

    def register(self, sub) -> None:
        job = sub.add_parser("job", add_help=False).add_subparsers(dest="action")
        a = job.add_parser("add", add_help=False)
        a.add_argument("ids", nargs="+")
        a = job.add_parser("list", add_help=False)
        a.add_argument("--all", action="store_true")
        job.add_parser("show", add_help=False).add_argument("id")
        a = job.add_parser("move", add_help=False)
        a.add_argument("id")
        a.add_argument("project")
        job.add_parser("delete", add_help=False).add_argument("ids", nargs="+")
        met = sub.add_parser("metrics", add_help=False).add_subparsers(dest="action")
        for name in ("set", "clear"):
            a = met.add_parser(name, add_help=False)
            a.add_argument("job", nargs="?")
            a.add_argument("--pattern")
            a.add_argument("--project", nargs="?", const="")
            if name == "set":
                for opt in ("--critical", "--important", "--progress", "--command"):
                    a.add_argument(opt)
                a.add_argument("--alert", action="append", default=[])
                a.add_argument("--wake-on", dest="wake_on", action="append", default=[])
                a.add_argument("--clear-alerts", action="store_true")
        met.add_parser("show", add_help=False).add_argument("job")
        met.add_parser("list", add_help=False)
        pl = sub.add_parser("plots", add_help=False).add_subparsers(dest="action")
        for name in ("set", "clear"):
            a = pl.add_parser(name, add_help=False)
            a.add_argument("job", nargs="?")
            a.add_argument("--pattern")
            a.add_argument("--project", nargs="?", const="")
            if name == "set":
                a.add_argument("--command", required=True)
        pl.add_parser("test", add_help=False).add_argument("job")
        pl.add_parser("show", add_help=False).add_argument("job")
        a = sub.add_parser("watch", add_help=False)
        a.add_argument("ids", nargs="+")
        a.add_argument("--every")
        a.add_argument("--note", default="")
        res = sub.add_parser("result", add_help=False).add_subparsers(dest="action")
        for name in ("set", "add"):
            a = res.add_parser(name, add_help=False)
            if name == "set":
                a.add_argument("id")
            a.add_argument("--name")
            a.add_argument("--verdict")
            a.add_argument("--hypothesis")
            a.add_argument("--conclusion")
            a.add_argument("--tags")
            a.add_argument("--metric", action="append", default=[])
            a.add_argument("--dead-end", action="store_true", dest="dead_end")
            if name == "set":
                a.add_argument("--hide", action="store_true")
                a.add_argument("--unhide", action="store_true")
        res.add_parser("show", add_help=False).add_argument("id")
        a = sub.add_parser("results", add_help=False)
        a.add_argument("--project")
        a.add_argument("--all-projects", action="store_true", dest="all_projects")
        a.add_argument("--sort", default="")
        a.add_argument("--asc", action="store_true")
        a.add_argument("--desc", action="store_true")
        a.add_argument("--verdict", default="")
        a.add_argument("--name", default="")
        a.add_argument("--tag", default="")
        a.add_argument("--state", default="")
        a.add_argument("--since", default="")
        a.add_argument("--search", default="")
        a.add_argument("--todo", action="store_true")
        a.add_argument("--hidden", action="store_true")
        a.add_argument("--limit", type=int, default=0)
        a.add_argument("--format", default="table", choices=["table", "md", "csv", "json"])
        a.add_argument("--columns", default="")
        fnd = sub.add_parser("finding", add_help=False).add_subparsers(dest="action")
        for kind in ("insight", "dead-end"):
            a = fnd.add_parser(kind, add_help=False)
            a.add_argument("text", nargs="+")
            a.add_argument("--job", default="")
        fnd.add_parser("rm", add_help=False).add_argument("id")
        a = sub.add_parser("findings", add_help=False)
        a.add_argument("--project")
        a.add_argument("--all-projects", action="store_true", dest="all_projects")
        a = sub.add_parser("cluster", add_help=False)
        a.add_argument("--partition", "-p")
        a.add_argument("--fresh", action="store_true")

    def preprocess(self, argv: list[str]) -> list[str]:
        if argv[:1] == ["result"] and len(argv) > 1 and argv[1] not in ("set", "add", "show"):
            return ["result", "set", *argv[1:]]  # ov result <id> --verdict ...
        return argv

    async def dispatch(self, session, args) -> str | None:
        """The answer, or None when the command isn't the HPC's."""
        area = args.area
        if area not in AREAS:
            return None
        if area in ("job", "metrics", "plots", "result") and args.action:
            return await getattr(self, f"{area}_{args.action}")(session, args)
        if area == "finding" and args.action:
            return await self.finding(session, args)
        if area in ("watch", "results", "findings", "cluster"):
            return await getattr(self, area)(session, args)
        return None

    def status_lines(self, session) -> list[str]:
        jobs = [j for j in self.store.session_jobs(session.id) if not j.ended_at]
        return ["running jobs of this conversation: " + (", ".join(f"{j.job_id} ({j.state})" for j in jobs) or "none")]

    async def cluster(self, session, args) -> str:
        if self._cluster is None:
            from .cluster import Cluster
            self._cluster = Cluster(self.hpc)
        snap = await self._cluster.snapshot(fresh=args.fresh)
        names = {r.job_id: r.name for r in self.store.active_jobs()}
        return self._cluster.text(snap, names, partition=args.partition)

    def _scope(self, session, args) -> str | None | bool:
        """Which project's results: this conversation's, --project (main = the main chat), or all."""
        if getattr(args, "all_projects", False):
            return False
        p = getattr(args, "project", None)
        if p:
            if p == "main":
                return None
            if not self.store.get_project(p):
                raise CtlError(f"no project {p!r} (ov project list)")
            return p
        return session.project

    async def result_set(self, session, args) -> str:
        rid = array_parent(args.id) if args.id[:1].isdigit() else args.id
        job = self.store.get_job(rid) if rid.isdigit() else None
        if job is None and self.store.get_result(rid) is None:
            raise CtlError(f"Overseer doesn't track a job {rid} (ov job add {rid} first)" if rid.isdigit() else
                           f"no result {rid!r}: `ov results` lists them; `ov result add --name ...` records a new one")
        return self._annotated(session, args, rid, job)

    async def result_add(self, session, args) -> str:
        if not args.name:
            raise CtlError("a result without a job needs --name (what was evaluated)")
        return self._annotated(session, args, self.store.next_result_id(), None)

    def _annotated(self, session, args, rid: str, job) -> str:
        verdict = "dead-end" if args.dead_end else args.verdict
        r = R.annotate(self.store, rid, session_id=session.id, project=session.project, job=job,
                       hypothesis=args.hypothesis, conclusion=args.conclusion, verdict=verdict, tags=args.tags,
                       metrics=R.parse_metric_args(args.metric), name=args.name,
                       hidden=True if getattr(args, "hide", False) else False if getattr(args, "unhide", False)
                       else None)
        out = f"✓ result {r.id} ({r.name or 'unnamed'}): " + (f"{r.verdict}" if r.verdict else "no verdict yet")
        if args.dead_end:
            text = r.conclusion or r.hypothesis or r.name
            f = R.add_finding(self.store, r.project, "dead-end", f"{r.name}: {text}" if r.name else text, session.id,
                              job_id=r.job_id or "")
            out += f"; dead end F{f.id} added to the project's findings"
        return out

    async def result_show(self, session, args) -> str:
        r = self.store.get_result(args.id)
        if not r:
            raise CtlError(f"no result {args.id!r} (ov results lists them)")
        m = R._metrics(r)
        return "\n".join(x for x in [
            f"{r.id}: {r.name} · {r.state or 'no job'}" + (f" · {r.elapsed}" if r.elapsed else "")
            + (f" · project {r.project}" if r.project else ""),
            f"verdict: {R.VERDICTS.get(r.verdict, '')} {r.verdict}" if r.verdict else "verdict: (none yet)",
            f"hypothesis: {r.hypothesis}" if r.hypothesis else "",
            f"conclusion: {r.conclusion}" if r.conclusion else "",
            f"tags: {r.tags}" if r.tags else "",
            "metrics: " + ", ".join(f"{k}={v}" for k, v in m.items()) if m else "",
        ] if x)

    async def results(self, session, args) -> str:
        from ..core.autopilot import parse_duration
        scope = self._scope(session, args)
        since = None
        if args.since:
            d = parse_duration(args.since)
            if not d:
                raise CtlError(f"--since {args.since!r}: use e.g. 3d, 12h")
            since = time.time() - d
        asc = True if args.asc else False if args.desc else None
        rs = R.select(self.store.results(scope, hidden=args.hidden), sort=args.sort, asc=asc,
                      verdicts=[v for v in args.verdict.split(",") if v], name=args.name, tag=args.tag,
                      state=args.state, since=since, text=args.search, annotated=False if args.todo else None,
                      limit=args.limit or (40 if args.format in ("table", "md") else 0))
        if not rs:
            return "no results match" + ("" if args.all_projects else
                                         f" in {'project ' + scope if scope else 'the main chat'}") + "."
        cols = [c.strip() for c in args.columns.split(",") if c.strip()] or None
        return R.render(rs, args.format, cols)

    async def finding(self, session, args) -> str:
        if args.action == "rm":
            fid = int(args.id.lstrip("Ff")) if args.id.lstrip("Ff").isdigit() else -1
            f = self.store.get_finding(fid)
            if not f or f.project != session.project:
                raise CtlError(f"no finding {args.id} in this project (ov findings)")
            self.store.remove_finding(fid)
            return f"✓ removed F{fid}"
        f = R.add_finding(self.store, session.project, args.action, " ".join(args.text), session.id, args.job)
        await self._announce(session, f"{R.FINDING_KINDS[f.kind]} <b>{'Dead end' if f.kind == 'dead-end' else 'Insight'}"
                                      f"</b> recorded: {esc(f.text)}")
        return f"✓ F{f.id} ({f.kind}) recorded: shown to every conversation in this project from now on"

    async def findings(self, session, args) -> str:
        fs = self.store.findings(self._scope(session, args))
        return "\n".join(R.finding_lines(fs)) or "no findings yet (ov finding insight|dead-end \"...\")"

    async def _verify(self, job_id: str) -> tuple[str, str, str]:
        """(name, state, workdir) of one of your jobs, according to SLURM; raises if it isn't one."""
        if not job_id.isdigit():
            raise CtlError(f"{job_id!r} isn't a job id")
        await self.hpc.ensure_connected()
        info = await self.hpc.scontrol_job(job_id)
        if info:
            owner = info.get("UserId", "").split("(")[0]
            if owner and self.hpc.user and owner != self.hpc.user:
                raise CtlError(f"job {job_id} belongs to {owner}, not to {self.hpc.user}")
            return info.get("JobName", ""), info.get("JobState", ""), info.get("WorkDir", "")
        job = await self.hpc.sacct_job(job_id)  # finished a while ago: only accounting still knows it
        if job:
            return job.name, job.state, job.workdir
        raise CtlError(f"SLURM doesn't know a job {job_id} of yours")

    async def job_add(self, session, args) -> str:
        out = []
        for jid in dict.fromkeys(array_parent(x) for x in args.ids):  # a job array is tracked by its id
            try:
                name, state, workdir = await self._verify(jid)
            except CtlError as e:
                out.append(f"✗ {e}")
                continue
            if is_helper(name):
                out.append(f"✗ job {jid} is hpc-run's own (its compute worker, or a run's node), not one of your jobs: "
                           "don't register or watch it (hpc-run starts and ends it)")
                continue
            rec = self.store.get_job(jid)
            alive = bool(state) and state.split()[0] not in TERMINAL_STATES
            already = bool(rec and rec.by_agent and rec.session_id == session.id and not rec.ignored
                           and not (rec.ended_at and alive))
            if rec and (rec.ignored or (rec.ended_at and alive)):  # deleted, or wrongly taken as ended: track again
                rec.ignored, rec.missing_polls = 0, 0
                if alive:
                    rec.ended_at, rec.state = None, state.split()[0]
                self.store.upsert_job(rec)
            if not already:
                self.agent.domain.record_job(session, jid)
                rec = self.store.get_job(jid)
                if not rec.name:  # what SLURM said, until the monitor's next poll
                    rec.name, rec.state, rec.workdir = name, state or rec.state, workdir
                    self.store.upsert_job(rec)
                await self._announce(session, f"🚀 Agent submitted job {esc(jid)} {esc(name)}",
                                     job_id=jid)
            out.append(f"✓ job {jid} ({name or '?'}, {state or '?'}) is tracked in this conversation"
                       + (" (already was)" if already else "") + ": Overseer reports on it"
                       + (f" (ov watch {jid} to be woken when it ends)" if not self._watched(session, jid) else ""))
        return "\n".join(out)

    async def job_list(self, session, args) -> str:
        rows = self.store.recent_jobs(40) if args.all else self.store.session_jobs(session.id)
        if not rows:
            return "No jobs." if args.all else "No jobs in this conversation (register yours with: ov job add <id>)."
        return "\n".join(f"{r.job_id}  {r.state or '?':<10} {r.name or ''}"
                         + (f"  [project {r.project}]" if r.project else "")
                         + ("  (this conversation)" if r.session_id == session.id else "") for r in rows)

    async def job_show(self, session, args) -> str:
        rec = self.store.get_job(args.id)
        if not rec:
            raise CtlError(f"Overseer doesn't track job {args.id} (ov job add {args.id} to track it)")
        return "\n".join(x for x in [
            f"job {rec.job_id}: {rec.name} - {rec.state}" + (" (ended)" if rec.ended_at else ""),
            f"project: {rec.project or 'main'}" + (f"; conversation: {rec.session_id}" if rec.session_id else ""),
            f"workdir: {rec.workdir}" if rec.workdir else "",
            f"stdout: {rec.stdout}" if rec.stdout else "",
            (f"job array: each task writes {rec.info_dict['stdout_tpl']} (%a = the task id; sacct -j {rec.job_id} "
             "lists them)") if rec.info_dict.get("stdout_tpl") else "",
            f"report: {rec.report[:1500]}" if rec.report else "",
        ] if x)

    async def job_move(self, session, args) -> str:
        project = None if args.project.lower() in ("main", "none", "general") else args.project.lower()
        if project and not self.store.get_project(project):
            raise CtlError(f"no project {project!r} (ov project list)")
        rec = self.store.get_job(args.id) or None
        if not rec:
            raise CtlError(f"Overseer doesn't track job {args.id}")
        old = rec.project
        self.agent.domain.move_job(args.id, project)
        where = f"project {code(project)}" if project else "the main chat"
        await self._announce(None, f"🔀 Job {code(args.id)} {esc(rec.name)} moved to {where} by the agent "
                                   f"(was: {esc(old or 'main')}).", job_id=args.id)
        return f"✓ job {args.id} moved to {project or 'main'}"

    async def job_delete(self, session, args) -> str:
        out = []
        for jid in args.ids:
            rec = self.store.get_job(jid)
            if not rec or rec.ignored:
                out.append(f"✗ Overseer doesn't track job {jid}")
                continue
            rec.ignored = 1
            self.store.upsert_job(rec)
            if getattr(self.agent.domain, "jobs", None):
                self.agent.domain.jobs.forget_job(jid)
            await self._announce(session, f"🗑 Overseer no longer tracks job {code(jid)} {esc(rec.name)} "
                                          "(the job itself wasn't touched).")
            out.append(f"✓ job {jid} is no longer tracked (it keeps running; scancel {jid} would cancel it)")
        return "\n".join(out)

    def _spec_scope(self, session, args) -> tuple[str, str]:
        given = [x for x in (args.job, args.pattern, args.project) if x is not None]
        if len(given) != 1:
            raise CtlError("say what it's for: a job id, --pattern '<job name glob>' or --project [name]")
        if args.job is not None:
            jid = array_parent(args.job)
            if not jid.isdigit():
                raise CtlError(f"{args.job!r} isn't a job id")
            return "job", jid
        if args.pattern is not None:
            if not args.pattern.strip():
                raise CtlError("--pattern needs a glob on job names, e.g. 'bench-dreamerv3-*'")
            return "pattern", args.pattern.strip()
        project = (args.project or session.project or "").strip().lower()
        if not project or not self.store.get_project(project):
            raise CtlError("--project: this conversation has no project; give its name (ov project list)")
        return "project", project

    def _affected(self, scope: str, key: str) -> list[str]:
        import fnmatch
        out = []
        for rec in self.store.active_jobs():
            if (scope == "job" and rec.job_id == key) or (scope == "pattern" and fnmatch.fnmatch(rec.name or "", key)) \
                    or (scope == "project" and rec.project == key):
                out.append(rec.job_id)
        return out

    async def metrics_set(self, session, args) -> str:
        scope, key = self._spec_scope(session, args)
        spec = self.store.get_metric_spec(scope, key) or MetricSpec(scope=scope, key=key)
        for field in ("critical", "important", "command"):
            value = getattr(args, field)
            if value is not None:
                setattr(spec, field, value.strip())
        if args.progress is not None:
            spec.progress = check_progress(args.progress)
        alerts = [] if args.clear_alerts else json.loads(spec.alerts or "[]")
        for text, wake in [(t, False) for t in args.alert] + [(t, True) for t in args.wake_on]:
            a = parse_alert(text, wake)
            alerts = [x for x in alerts if x.get("when") != a.text] + [{"when": a.text, "wake": wake}]
        spec.alerts = json.dumps(alerts)
        spec.session_id = session.id
        self.store.set_metric_spec(spec)
        affected = self._affected(scope, key)
        self._refresh_cards(affected)
        what = {"job": f"job {key}", "pattern": f"jobs named {key}", "project": f"project {key}'s jobs"}[scope]
        lines = [f"✓ metrics for {what}" + (f" (cards updated: {', '.join(affected[:8])})" if affected else "")]
        lines += [f"  {k}: {v}" for k, v in (("critical", spec.critical), ("important", spec.important),
                                              ("progress", spec.progress), ("command", spec.command)) if v]
        for a in alerts:
            lines.append(f"  alert: {a['when']}" + ("  (wakes you)" if a.get("wake") else "  (pings the researcher)"))
        if scope == "job" or affected:
            lines.append(f"Check what the card shows: ov metrics show {key if scope == 'job' else affected[0]}")
        return "\n".join(lines)

    async def metrics_show(self, session, args) -> str:
        jid = array_parent(args.job)
        rec = self.store.get_job(jid)
        if not rec:
            raise CtlError(f"Overseer doesn't track job {jid} (ov job add {jid})")
        monitor = getattr(self.agent.domain, "monitor", None)
        if not monitor:
            raise CtlError("the monitor isn't running")
        from .metrics import lines as metric_lines
        view = await monitor._view(rec)
        spec = view["spec"]
        critical, important, progress = metric_lines(spec, view["values"], view["auto"])
        out = [f"job {jid} {rec.name}: " + (f"metrics from {spec.scope} {spec.key}" if spec else
                                             "no metrics set (ov metrics set): the card shows what the log has")]
        out += [f"card 🎯 {critical}" if critical else "", f"card 📈 {important}" if important else "",
                f"progress: {progress}" if progress else ""]
        if spec and spec.command:
            out.append(f"command: {spec.command}")
            out.append(f"  error: {view['err']}" if view["err"] else "  ok")
        out.append("values: " + (", ".join(f"{k}={v}" for k, v in list(view["values"].items())[:30]) or "(none found)"))
        if spec and json.loads(spec.alerts or "[]"):
            from .metrics import fired_alerts
            fired = fired_alerts(self.store, rec)
            for a in json.loads(spec.alerts):
                out.append(f"alert: {a['when']}" + (" (wakes you)" if a.get("wake") else "")
                           + (" - FIRED" if a["when"] in fired else ""))
        return "\n".join(x for x in out if x)

    async def metrics_list(self, session, args) -> str:
        specs = self.store.metric_specs()
        if not specs:
            return "No metrics set (ov metrics set <job id> | --pattern '<glob>' | --project ...)."
        return "\n".join(f"{s.scope} {s.key}: critical={s.critical or '-'} important={s.important or '-'}"
                         + (f" progress={s.progress}" if s.progress else "") + (" command" if s.command else "")
                         + (f" alerts={len(json.loads(s.alerts or '[]'))}" if json.loads(s.alerts or "[]") else "")
                         for s in specs[:40])

    async def metrics_clear(self, session, args) -> str:
        scope, key = self._spec_scope(session, args)
        spec = self.store.get_metric_spec(scope, key)
        if not spec:
            raise CtlError(f"no metrics set for {scope} {key}")
        if spec.plot:  # its plotting command stays (ov plots clear removes it)
            spec.critical = spec.important = spec.progress = spec.command = ""
            spec.alerts = "[]"
            self.store.set_metric_spec(spec)
        else:
            self.store.delete_metric_spec(scope, key)
        self._refresh_cards(self._affected(scope, key))
        return f"✓ metrics for {scope} {key} removed"

    def _monitor(self):
        monitor = getattr(self.agent.domain, "monitor", None)
        if not monitor:
            raise CtlError("the monitor isn't running")
        return monitor

    def _tracked(self, job: str):
        jid = array_parent(job)
        rec = self.store.get_job(jid)
        if not rec:
            raise CtlError(f"Overseer doesn't track job {jid} (ov job add {jid})")
        return rec

    async def plots_set(self, session, args) -> str:
        scope, key = self._spec_scope(session, args)
        if "{out}" not in args.command:
            raise CtlError("the command must write its PNGs into {out} (Overseer fills in an empty folder), e.g. "
                           "--command 'python {project}/plots/curves.py --log {stdout} --out {out}'")
        spec = self.store.get_metric_spec(scope, key) or MetricSpec(scope=scope, key=key)
        spec.plot, spec.session_id = args.command.strip(), session.id
        self.store.set_metric_spec(spec)
        what = {"job": f"job {key}", "pattern": f"jobs named {key}", "project": f"project {key}'s jobs"}[scope]
        affected = self._affected(scope, key)
        return (f"✓ graphs for {what}: run when a job ends, every {self.cfg.slurm.plot_interval_minutes} min while it "
                "runs, and on the researcher's 📈 button. Check it now: ov plots test "
                + (key if scope == "job" else affected[0] if affected else "<job id>"))

    async def plots_clear(self, session, args) -> str:
        scope, key = self._spec_scope(session, args)
        spec = self.store.get_metric_spec(scope, key)
        if not spec or not spec.plot:
            raise CtlError(f"no plotting command set for {scope} {key}")
        spec.plot = ""
        if spec.critical or spec.important or spec.progress or spec.command or spec.alerts not in ("", "[]"):
            self.store.set_metric_spec(spec)
        else:
            self.store.delete_metric_spec(scope, key)
        return f"✓ no more graphs for {scope} {key}"

    async def plots_test(self, session, args) -> str:
        """Run the job's plotting command now: the graphs go under its card, and copies into ./inbox to look at."""
        rec = self._tracked(args.job)
        monitor = self._monitor()
        n, err = await monitor.plots(rec)
        if not n:
            raise CtlError(err or "no graphs")
        images = monitor.cached_plots(rec.job_id)
        from ..core.agent import deliver_uploads
        staging = self.cfg.local.tmp("uploads")
        staging.mkdir(mode=0o700, parents=True, exist_ok=True)
        staged = []
        for name, data in images:
            path = staging / secrets.token_hex(8)
            path.write_bytes(data)
            staged.append((f"plot-{rec.job_id}-{name}", path))
        items = deliver_uploads(self.agent.session_runner(session).workdir(session.id), staged)
        return (f"✓ {n} graph(s) shown under job {rec.job_id}'s card. Copies to look at: "
                + ", ".join(p for p, _ in items) + (f"\n⚠️ {err}" if err else ""))

    async def plots_show(self, session, args) -> str:
        from .plots import resolve_plot
        rec = self._tracked(args.job)
        spec = resolve_plot(self.store, rec)
        if not spec:
            return f"job {rec.job_id}: no plotting command (ov plots set)"
        err = self.store.get(f"plot_err:{rec.job_id}") or ""
        return (f"job {rec.job_id}: graphs from {spec.scope} {spec.key}\ncommand: {spec.plot}"
                + (f"\nlast run: {err}" if err else ""))

    def _watched(self, session, job_id: str) -> bool:
        return bool(self.agent.autopilot) and any(
            w.session_id == session.id for w in self.store.job_watches(job_id))

    async def watch(self, session, args) -> str:
        ap = self._ap()
        every = self._duration(args.every, "--every") if args.every else 0.0
        ids = list(dict.fromkeys(array_parent(x) for x in args.ids))  # a job array is watched by its id
        for jid in ids:
            rec = self.store.get_job(jid)
            if rec and is_helper(rec.name):
                raise CtlError(f"job {jid} is hpc-run's own, not one of your jobs: there's nothing to watch")
            if rec and rec.ended_at and not rec.ignored and rec.state != "UNKNOWN":
                raise CtlError(f"job {jid} has already ended ({rec.state}): look at it now (ov job show {jid})")
            if not rec or rec.ignored or rec.ended_at:  # not tracked (yet), or its end wasn't confirmed: ask SLURM
                name, state, _ = await self._verify(jid)
                if is_helper(name):
                    raise CtlError(f"job {jid} is hpc-run's own, not one of your jobs: there's nothing to watch")
                if state and state.split()[0] in TERMINAL_STATES:
                    raise CtlError(f"job {jid} has already ended ({state}): look at it now")
                await self.job_add(session, argparse.Namespace(ids=[jid]))
        w, remark = ap.add_watch(session, "jobs", ids, every, args.note)
        what = f"job{'s' if len(ids) > 1 else ''} {', '.join(ids)}"
        self._refresh_cards(ids)  # the cards show "⏰ agent watching", with a ✖ to cancel it
        return (f"✓ w{w.id}: you'll be woken when {what} {'have' if len(ids) > 1 else 'has'} ended"
                + (" (or as soon as one fails)" if len(ids) > 1 else "")
                + (f", with check-ins every {human_interval(w.every)} while running" if w.every else "") + remark)

    def _refresh_cards(self, job_ids: list[str]) -> None:
        monitor = getattr(self.agent.domain, "monitor", None)
        if monitor and hasattr(monitor, "refresh_card_by_id"):
            for jid in job_ids:
                task = asyncio.get_running_loop().create_task(monitor.refresh_card_by_id(jid))
                self._bg.add(task)
                task.add_done_callback(self._bg.discard)
