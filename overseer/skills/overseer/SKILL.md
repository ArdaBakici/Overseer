---
name: overseer
description: How to talk to Overseer with the `ov` command - register SLURM jobs you submit, record and query experiment results and findings (insights, dead ends), report feedback about Overseer and propose skills, create and manage projects, move or stop tracking jobs, message the researcher, have yourself woken later (when jobs end, on a timer, or with check-ins) to work on your own over many runs, with a research log, and propose or run planner mode (you plan, subagents on cheaper models do the work). Use it when you are a planner or a subagent, and whenever you submit a job, when the researcher asks about projects or job tracking, when you want to follow up on something later, or for multi-run experiments.
---

# Overseer and `ov`

Overseer is the system that runs you. It keeps the SSH connection to the HPC, monitors SLURM jobs,
reports on them to the researcher on Telegram, and starts your runs. You only run when the
researcher writes to you, or when something you set up wakes you.

`ov` is a local command that sends requests to Overseer. It answers at once. Every change it makes
is announced to the researcher, so you don't need to ask before using it. It only works inside a run
Overseer started (`$OVERSEER_CTL` holds this run's token).

## Jobs

```bash
ov job add 18746955 18746956     # after EVERY submission: Overseer checks the ids with SLURM, then reports on them
ov job list                      # jobs of this conversation (--all: every job Overseer tracks)
ov job show 18746955             # what Overseer knows: state, folders, logs, findings
ov job move 18746955 dreamer     # the job belongs to project "dreamer" (or "main")
ov job delete 18746955           # stop tracking it. The job keeps running: `scancel` cancels it
```

- Register every job you submit, however you submitted it: sbatch, a script, a Python CLI, an array.
  Overseer also tries to spot submissions on its own, but it can miss some.
- "Delete job X" from the researcher means `ov job delete X`. Cancel it with `scancel` only if they
  also want it stopped.
- Ids are checked: jobs that SLURM doesn't know as the researcher's are refused.
- Job arrays are one job to Overseer, under the id sbatch printed (`123`, not `123_4`). It's running while any
  task is, and it ends when all tasks have ended. Its report gives the task states (e.g. "9 COMPLETED,
  2 FAILED"); each task has its own log. Watching an array wakes you when all its tasks have ended, with
  each task's state and log path and the end of each log (failed tasks first). Its card and check-ins show the
  running task's log; `ov job show <id>` gives the tasks' log pattern (%a = the task id).
- `hpc-run`'s shared worker job (`overseer-worker`) is not one of your jobs. Don't register or watch
  it: `hpc-run` starts, reuses and ends it (`hpc-run --stop` ends it early).

## What matters in a job: its card, and alerts

Every job has a live card in the project's "📊 jobs" topic. Overseer keeps editing that card until the
job ends; the card's final version is the job's result. Tell Overseer what the card should show and what's
worth an alert:

```bash
ov metrics set 18756118 --critical "crl/train_episode/score=score" \
    --important "train/loss/value=value-loss,train/loss/dyn=dyn-loss" --progress "step/5000000"
ov metrics set --pattern "bench-dreamerv3-*" ...     # every run of a sweep
ov metrics set --project ...                         # the default for this project's jobs
ov metrics set 18756118 --command 'tail -n1 {workdir}/runs/seed1/metrics.jsonl'
ov metrics set 18756118 --alert "score < 1 after 2h" --wake-on "train/loss/value is nan" --wake-on "step stalled 30m"
ov metrics show 18756118     # what the card shows now, and whether the command works
ov metrics list | clear ...
```

- **`--critical`**: the 1–2 numbers that tell whether the run works. They're the card's first line
  and appear in the report's headline.
- **`--important`**: up to ~4 more, on the card's second line.
- **Names and labels:** use `metric=label` for a short label. A name matches the metric exactly, or by
  its last part ("score" finds "crl/train_episode/score").
- **`--progress metric/total`**: shows progress as steps done out of the total, which is more meaningful
  than time used.
- **What Overseer reads from the log by itself:**
  - `name value`, `name=value` and `name: value` for common metric names (step, epoch, loss, reward, score, ...);
  - every number in JSON records (or printed Python dicts), whatever their names, with nested keys joined with
    `/`. Printing `{"achievement_pct": 7.9, "return_mean": 2.3}` is enough to use those names.
  - It skips parameter counts (`reward: 656,129`, `... params`).
- **`--command`** is for when the values aren't in the log. It must be quick, because it runs on the
  login node at every card refresh (about every 10 minutes) with a 20 s limit.
  - It prints `name=value` lines or one JSON object; nested keys are joined with `/`.
  - Placeholders: `{job_id}`, `{workdir}`, `{stdout}`, `{name}`.
- **Alerts:**
  - forms: `<metric> < 1 after 2h` (also `<=`, `>`, `>=`, `==`, `!=`), `<metric> is nan`, and
    `<metric> stalled 30m` (unchanged for that long);
  - when: checked at every card refresh;
  - how often: each fires once per job;
  - who hears: `--alert` pings the researcher, and `--wake-on` also wakes you with the job's details.
- **Which setting applies:** a job's own setting, else the newest matching `--pattern`, else the project
  default. Set it once per kind of experiment.

## Graphs: your plotting script, run by Overseer

The researcher sees graphs under each job's card. They come from a plotting command you write once and register for a
job, a sweep (name pattern) or the whole project. No model is involved when they're made.

```bash
# 1. A small script in the project folder that reads the job's metrics (log, metrics.jsonl, CSV, TensorBoard events)
#    and saves PNGs into the folder it's given.
# 2. Register the command that runs it:
ov plots set --pattern "ne-crl-*" --command "python {project}/plots/crl.py --log {stdout} --workdir {workdir} --out {out}"
ov plots test 18801933     # run it now: the graphs go under the card, and copies land in ./inbox for you to check
ov plots show 18801933 | clear ...
```

- **When Overseer runs it:** on the HPC, when the job ends, every hour while it runs, and when the researcher presses
  📈 Plot.
- **What the command must do:**
  - write PNGs into `{out}` (an empty folder): up to 4, each under 10 MB;
  - finish within 90 s, because it runs on the login node at low priority.
  - **Placeholders:** `{job_id}`, `{name}`, `{workdir}`, `{stdout}` and `{project}` (the project's folder).
- **How they're shown:** the graphs appear inside the job's card, one at a time, and the researcher pages through them
  with ◀ ▶. The first one by file name is shown first, so put the most telling one first (e.g. `00-overview.png`).
- **Good graphs:** the main metrics against step or time, the job id in the title, readable on a phone (e.g. dpi 150).
  Prefer one clear multi-panel figure, or a few (at most 4). Look at the `ov plots test` copies before you're done.
- **If the researcher presses 📈 Plot and there's no command yet,** you're asked to write one. Register it for the kind
  of job (`--pattern` or `--project`), not just the one job, so later runs get graphs too.

## Projects

```bash
ov project list
ov project new chem --dir /scratch/me/chem --about "DFT with ORCA 6 on the cpu partition" [--move-here]
ov project set chem --dir /scratch/me/chem2 --about "..."
ov project archive chem          # the researcher can bring it back with /project restore chem
ov conversation move chem        # this conversation now belongs to project chem (or "main")
```

- Each project has its own Telegram topic, conversations, jobs and agent settings.
- `--dir` is the project's folder on the HPC. `~/...` refers to the HPC home.
- `--move-here` moves this conversation into the new project. Its folder, description and notes
  apply from your next run.

## Messages and status

```bash
ov notify "epoch 3 done, val loss 0.41 - continuing"   # a message to the researcher right now
ov status                        # this conversation: project, autopilot, jobs, what will wake you
```

Your final reply is sent to the researcher anyway. Use `ov notify` for news in the middle of a
long run.

## Feedback about Overseer, and skills

**Feedback.** When Overseer itself limits you, report it. The researcher sees each item in a 🛠 feedback topic, and an
agent there turns them into designs for improving Overseer.

```bash
ov feedback list                 # what's already reported (open items, most +1s first)
ov feedback +1 FB7               # the same problem: add a +1 rather than a new item
ov feedback "ov results can't filter by array task" --context "comparing 40 sweep tasks" [--kind bug|idea]
```

- **What to report:**
  - an `ov` command or option you needed that doesn't exist;
  - something Overseer should have told you but didn't;
  - the sandbox blocking something legitimate;
  - Overseer behaving wrongly (`--kind bug`);
  - a clear improvement (`--kind idea`).
- **What not to report:** problems with the cluster, your code, or decisions that are the researcher's. Never include
  secrets or tokens.
- **Limits:** at most 3 new items per conversation per day, and +1s are unlimited.
- **Then carry on:** work around the limitation if you can. You're told when the researcher fixes it or decides not to.

**Skills.** When you work out know-how other agents will need (how to do something on this cluster, a tool's quirks, a
procedure that took trial and error), write it as a skill and propose it. Once the researcher approves it, every agent
in the project (or every agent, with `--all`) gets it from its next run.

```bash
cat > skill.md <<'SKILL'
---
name: cluster-long-jobs
description: How to run jobs longer than 24 h on this cluster (the qos, checkpointing, requeue). Use when a job needs more than 24 h.
---
Jobs over 24 h need `--qos long` (max 7 days) ...
SKILL
ov skill propose skill.md        # this project's agents; --all: every agent
ov skill list                    # installed skills, and proposals waiting for the researcher
ov skill show cluster-long-jobs
```

- **The frontmatter holds only `name` and `description`.** Any other field is refused, because it could change what
  agents may do.
- **Keep it focused:** one subject, under 30 KB, written for an agent that has never seen this cluster.
- **Proposing a skill again** with the same name, once approved, replaces the old version.

## Results and findings

**Results** are the project's record of experiments: one row per finished job, plus any result you add by hand.
Overseer fills in what it knows when a job ends: state, runtime, and its key metrics (your `ov metrics` critical and
important ones first, then others from the log). You add what it means:

```bash
ov result 18756118 --hypothesis "cosine schedule beats step decay"        # when you submit it
ov result 18756118 --verdict promising --conclusion "+0.4 score over baseline, stable across seeds"
ov result 18756119 --verdict no-gain --conclusion "warmup 5k = 1k" --tags warmup,ablation
ov result add --name "sample quality, run 7 vs 5" --verdict worse --conclusion "more artifacts at high cfg" \
    --metric "artifacts=more"                      # a result that isn't a job: an evaluation by eye, a comparison
ov result 18756120 --dead-end --conclusion "fp16 diverges with this optimizer"   # also records a dead end
ov result show 18756118
```

- **Verdicts:** `baseline`, `promising`, `no-gain`, `worse`, `broken`, `dead-end`, `inconclusive`. A verdict needs no
  number: it's your judgement, and qualitative work is recorded the same way. Give every result you're responsible
  for one; your prompt lists those still missing.
- **A run of results without gain** (3 or more `no-gain`/`worse`/... since the last `promising` or `baseline`) is
  pointed out to you. Treat it as a cue to try something materially different, check the findings, or ask.

```bash
ov results                                   # this project's, newest first
ov results --sort score                      # by a metric, highest first (--asc for loss-like ones)
ov results --verdict promising,baseline --since 7d
ov results --name "bench-*" --tag ablation --search cosine
ov results --todo                            # finished, no verdict or conclusion yet
ov results --format md                       # a Markdown table, ready to give the researcher
ov results --format csv > outbox/results.csv # a file sent with your reply
ov results --columns id,name,score,value-loss,verdict --limit 10
ov results --all-projects | --project chem
```

When the researcher asks how experiments went, which run was best or what was tried, read them with
`ov results`, sort and filter them for the question, and answer with the table (or a CSV) plus your reading of it.

**Findings** are the project's lasting lessons. Unlike the research log, which gets condensed, they are kept whole
and shown to every conversation in the project when it starts or is woken.

```bash
ov finding dead-end "lr above 3e-4 diverges with the 1B model, any schedule" --job 18756120
ov finding insight "eval score tracks train/loss/dyn closely: watch it instead of waiting for evals"
ov findings                                  # all of them (F1, F2, ...)
ov finding rm F3                             # no longer true
```

- **Dead ends:** record one whenever an approach failed in a way that shouldn't be retried.
- **Insights:** record one when something holds beyond a single run.
- **Before you try something,** check the findings for it.

## Being woken later

Nothing runs between your turns. Don't sleep, poll in a loop or wait for jobs: have Overseer wake you.
You don't need permission. Each watch or timer is announced to the researcher, who can cancel it.

```bash
ov watch 18746955 --note "compare with the baseline in runs/base"   # when it ends
ov watch 18746955 18746956                    # when both have ended (or as soon as one fails)
ov watch 18746955 --every 1h                  # ...plus check-ins while it runs (at least 15 min apart)
ov wake in 90m "check that eval started"      # once, at a time
ov wake every 2h --for 12h "summarise the sweep"   # repeatedly (at least 15 min apart; 24h without --for)
ov watches                                    # what will wake you (ids like w12)
ov unwatch w12                                # or a job id, or "all"
```

When you are woken, the message contains:
- the final report of each finished job (state, exit code, runtime, memory, issues found, the end of
  its log);
- your timer's or watch's note;
- which jobs are still running;
- in a project, the research log's beginning and latest entries.

A check-in shows a running job's progress: metrics and the end of its log.
- **Healthy:** change nothing and reply in a sentence or two.
- **Clearly broken, diverging or wasting GPU time:** `scancel` it, note why in the research log,
  resubmit, and watch the new job.

Every wake-up re-reads the whole conversation and costs real money. Wake yourself when there's
something to look at, and prefer job ends over frequent check-ins.

## Working on your own over many runs

For tuning, ablations, sweeps or "keep going until the score is above X", carry on by yourself. No
mode needs to be switched on.

1. **Start:**
   - look at the project and earlier results (the research log);
   - write the goal at the top of the log;
   - submit the first experiment, then `ov job add` and `ov watch` it;
   - end your reply.
2. **Each time you're woken:**
   - read the results;
   - give each finished job its verdict and conclusion (`ov result`), and record a finding if it taught
     something lasting (`ov finding`);
   - add an entry to the log: what changed and why, the job ids, the key results, the next idea;
   - submit the next experiment and watch it;
   - end your reply.
3. **Stop** when the goal is reached or nothing sensible is left to try. Say so, with the conclusion,
   and set no more wake-ups (`ov unwatch all` if any are left).
4. **Ask** when you need the researcher's decision: in your reply, or `ov notify` for something urgent.
   Don't keep spending runs meanwhile.

If a wake-up fails, the conversation's wake-ups pause until the researcher writes. They can also cancel
them all (`/auto off`). You're told when they cancel something.

Research log (`RESEARCH_LOG.md` in the project folder; in planner mode only the planner writes it):
- **Structure:** the current goal at the top, then one entry per experiment or wake-up.
- **Goal changes:** rewrite the goal, keeping a one-line note of the old one.
- **Size:** keep it compact, well under 200k tokens (~800 KB). When it grows long, condense the old
  entries into a short summary section.

Rules:
- Change one thing at a time unless the goal says otherwise.
- Give every run its own name and output folder, and never delete or overwrite earlier results.
- Only cancel or change jobs you submitted in this conversation.

## Planner mode: subagents

Planner mode is off unless the researcher turns it on (`/subagents on`) or approves a plan you propose. When it's on,
you run on a strong planner model and delegate the hands-on work to subagents on cheaper models. Each subagent is a
conversation of its own, in its own Telegram thread, with the same sandbox, HPC access and `ov`.

| Backend | Planner | Subagents |
|---|---|---|
| Claude Code | `claude-opus-5-5` | `claude-sonnet-5-5` |
| Codex | `gpt-6-astra` | `gpt-6.1-sol` (default), `gpt-6-luna` (simple, mechanical tasks) |

The researcher can change these; your prompt and `ov agents` show the current ones.

### Proposing it

Propose planner mode when the work has several independent lines (ablations, competing ideas, a sweep plus its
analysis) or is a long experiment that separate workers would keep focused. Don't propose it for a single job, a quick
fix or a question, and propose at most once per task unless the researcher asks for a revision.

```bash
cat > plan.json <<'PLAN'
{"goal": "find an LR schedule that beats the baseline (score 5.2), 3 seeds each",
 "why": "three independent schedules can run side by side",
 "subagents": [
   {"name": "cosine", "task": "Self-contained brief: goal, paths, constraints, output folder, what to report."},
   {"name": "warmup", "task": "...", "model": "claude-sonnet-5-5"},
   {"name": "grid", "task": "...", "backend": "codex", "model": "gpt-6-luna"}]}
PLAN
ov propose plan.json
```

- **The planner model** is the one for your current backend. You keep your backend, and with it this conversation.
- **The researcher sees a card** with the goal, the planner and each subagent, and answers in one of three ways:
  - **Approve:** planner mode goes on for this conversation, and the subagents start exactly as listed. You're told in
    a note, and woken with their reports.
  - **No:** you're told it was declined. Carry on in normal mode.
  - **They write to you instead:** that declines the plan, and you get their message. Revise and propose again if it
    still makes sense.
- **After proposing, end your reply.** Set no wake-ups for it: the researcher's answer brings you back.

### As the planner

```bash
ov spawn cosine "brief..." [--model claude-sonnet-5-5] [--backend codex]   # start a subagent (or queue it)
ov agents                     # your subagents and their state
ov agent show cosine          # its task and report
ov agent send cosine "also try warmup 1k"   # it works on that and reports again
ov agent stop cosine
ov planner off                # the goal is reached: back to normal mode
```

- **Briefs are self-contained.** A subagent doesn't see your conversation: give the goal, context, exact paths,
  constraints, its own output folder (so subagents never overwrite each other's results), and what to report.
- **Pick the cheapest model that can do the task.**
- **At most 3 subagents run at once**, across all conversations. More wait their turn.
- **You're woken with their reports** once all the subagents you're waiting for are done, or as soon as one fails or
  is stopped. Subagents watch their own jobs: don't watch theirs.
- **The research log is yours alone.** Record each subagent's results and link its `NOTES.md`.
- **When the goal is reached,** say so and run `ov planner off`. Running subagents still finish and report.

### As a subagent

Your prompt says when you are one, and what your task and notes file are.

- **Do the task on your own:** submit jobs, watch them, wake yourself.
- **Notes:** keep your detailed notes (commands, intermediate results, dead ends) in your `NOTES.md`.
- **Use your own output folder.** Don't write the research log; your planner records your results.
- **Your final reply is your report to the planner:** what you did, the key numbers, where the results are (at most
  ~15 lines). If you're blocked or the task turns out wrong, say so there rather than guessing.
- **You can't start subagents or propose planner mode.** The researcher may write to you in your thread: answer them
  there.
