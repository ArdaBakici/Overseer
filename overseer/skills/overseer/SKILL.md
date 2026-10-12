---
name: overseer
description: How to talk to Overseer with the `ov` command - register SLURM jobs you submit, create and manage projects, move or stop tracking jobs, message the researcher, and have yourself woken later (when jobs end, on a timer, or with check-ins) to work on your own over many runs, with a research log. Use it whenever you submit a job, when the researcher asks about projects or job tracking, when you want to follow up on something later, or for multi-run experiments.
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
ov job show 18746955             # what Overseer knows: state, folders, logs, final report
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
  2 FAILED"); each task has its own log. Watching an array wakes you when all its tasks have ended.
- `hpc-run`'s shared worker job (`overseer-worker`) is not one of your jobs. Don't register or watch
  it: `hpc-run` starts, reuses and ends it (`hpc-run --stop` ends it early).

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
   - add an entry to the log: what changed and why, the job ids, the key results, the next idea;
   - submit the next experiment and watch it;
   - end your reply.
3. **Stop** when the goal is reached or nothing sensible is left to try. Say so, with the conclusion,
   and set no more wake-ups (`ov unwatch all` if any are left).
4. **Ask** when you need the researcher's decision: in your reply, or `ov notify` for something urgent.
   Don't keep spending runs meanwhile.

If a wake-up fails, the conversation's wake-ups pause until the researcher writes. They can also cancel
them all (`/auto off`). You're told when they cancel something.

Research log (`RESEARCH_LOG.md` in the project folder):
- **Structure:** the current goal at the top, then one entry per experiment or wake-up.
- **Goal changes:** rewrite the goal, keeping a one-line note of the old one.
- **Size:** keep it compact, well under 200k tokens (~800 KB). When it grows long, condense the old
  entries into a short summary section.

Rules:
- Change one thing at a time unless the goal says otherwise.
- Give every run its own name and output folder, and never delete or overwrite earlier results.
- Only cancel or change jobs you submitted in this conversation.
